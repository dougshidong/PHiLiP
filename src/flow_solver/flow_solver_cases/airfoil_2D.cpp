#include "airfoil_2D.h"
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include <deal.II/base/table_handler.h>
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/gmsh_reader.hpp"
#include "functional/lift_drag.hpp"
#include "functional/extraction_functional.hpp"
#include "functional/amiet_model.hpp"

namespace PHiLiP {
namespace FlowSolver {
//=========================================================
// Airfoil 2D
//=========================================================
template <int dim, int nstate>
Airfoil2D<dim, nstate>::Airfoil2D(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , airfoil_length(this->all_param.flow_solver_param.airfoil_length)
        , height(this->all_param.flow_solver_param.height)
        , length_b2(this->all_param.flow_solver_param.length_b2)
        , incline_factor(this->all_param.flow_solver_param.incline_factor)
        , bias_factor(this->all_param.flow_solver_param.bias_factor)
        , refinements(this->all_param.flow_solver_param.refinements)
        , n_subdivision_x_0(this->all_param.flow_solver_param.n_subdivision_x_0)
        , n_subdivision_x_1(this->all_param.flow_solver_param.n_subdivision_x_1)
        , n_subdivision_x_2(this->all_param.flow_solver_param.n_subdivision_x_2)
        , n_subdivision_y(this->all_param.flow_solver_param.n_subdivision_y)
        , airfoil_sampling_factor(this->all_param.flow_solver_param.airfoil_sampling_factor)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{}

template <int dim, int nstate>
void Airfoil2D<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    using Model_enum = Parameters::AllParameters::ModelType;
    const PDE_enum pde_type = this->all_param.pde_type;
    const Model_enum model_type = this->all_param.model_type;
    if (pde_type == PDE_enum::navier_stokes || (pde_type == PDE_enum::physics_model && model_type == Model_enum::reynolds_averaged_navier_stokes)){
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    }
    this->pcout << "- - Courant-Friedrich-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    const double pi = atan(1.0) * 4.0;
    this->pcout << "- - Angle of attack [deg]: " << this->all_param.euler_param.angle_of_attack*180/pi << std::endl;
    this->pcout << "- - Side-slip angle [deg]: " << this->all_param.euler_param.side_slip_angle*180/pi << std::endl;
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> Airfoil2D<dim,nstate>::generate_grid() const
{
    std::shared_ptr<dealii::parallel::distributed::Triangulation<2> > grid = std::make_shared<dealii::parallel::distributed::Triangulation<2> > (
#if PHILIP_DIM!=1
    this->mpi_communicator
#endif
    );
    dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    airfoil_data.airfoil_type = "NACA";
    airfoil_data.naca_id      = "0012";
    airfoil_data.airfoil_length = airfoil_length;
    airfoil_data.height         = height;
    airfoil_data.length_b2      = length_b2;
    airfoil_data.incline_factor = incline_factor;
    airfoil_data.bias_factor    = bias_factor; 
    airfoil_data.refinements    = refinements;

    airfoil_data.n_subdivision_x_0 = n_subdivision_x_0;
    airfoil_data.n_subdivision_x_1 = n_subdivision_x_1;
    airfoil_data.n_subdivision_x_2 = n_subdivision_x_2;
    airfoil_data.n_subdivision_y = n_subdivision_y;
    airfoil_data.airfoil_sampling_factor = airfoil_sampling_factor; 

    dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 1 || current_id == 4 || current_id == 5) {
                    cell->face(face)->set_boundary_id (1004); // farfield
                } else {
                    cell->face(face)->set_boundary_id (1001); // wall
                }
            }
        }
    }

#if PHILIP_DIM==1
    std::shared_ptr<Triangulation> grid_1D;
    return grid_1D;  
#endif  
#if PHILIP_DIM==2
    std::shared_ptr<Triangulation> grid_2D = grid;
    return grid_2D;  
#endif
#if PHILIP_DIM==3
    std::shared_ptr<dealii::parallel::distributed::Triangulation<2> > grid_2D = grid;
    std::shared_ptr<Triangulation> grid_3D = std::make_shared<Triangulation>(
    #if PHILIP_DIM!=1
        this->mpi_communicator
    #endif
    );
    const unsigned int n_slices = 17;
    const double height = 0.2;
    const bool copy_manifold_ids = false;
    //const std::vector<types::manifold_id> manifold_priorities = {};
    dealii::GridGenerator::extrude_triangulation(*grid_2D, n_slices, height, *grid_3D, copy_manifold_ids);


    //Loop through cells to define boundary id's (periodic) on the z plane. 
    for (typename dealii::parallel::distributed::Triangulation<3>::active_cell_iterator cell = grid_3D->begin_active(); cell != grid_3D->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<3>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 1005) { //Dealii automatically assigns the next available number to the new boundaries when creating the 3D mesh. Thus, since the largest number used is 1005, it assigns 1006 and 1007 to the new boundaries. 
                    cell->face(face)->set_boundary_id (1006); //(2005); // z = 0 boundaries
                } else if(current_id == 1006){
                    cell->face(face)->set_boundary_id (1006); //(2006); // z = height boundaries
                }
            }
        }
    }

    // Periodic boundary parameters
    const bool periodic_x = false;
    const bool periodic_y = false;
    const bool periodic_z = true;
    const int x_periodic_1 = 0; 
    const int x_periodic_2 = 0;
    const int y_periodic_1 = 0; 
    const int y_periodic_2 = 0;
    const int z_periodic_1 = 2005; 
    const int z_periodic_2 = 2006;

    //Check for periodic boundary conditions and apply
    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
    
    if (periodic_x) {
        dealii::GridTools::collect_periodic_faces(*grid_3D, x_periodic_1, x_periodic_2, 0, matched_pairs);
    }

    if (periodic_y) {
        dealii::GridTools::collect_periodic_faces(*grid_3D, y_periodic_1, y_periodic_2, 1, matched_pairs);
    }

    if (periodic_z) {
        dealii::GridTools::collect_periodic_faces(*grid_3D, z_periodic_1, z_periodic_2, 2, matched_pairs);
    }

    if (periodic_x || periodic_y || periodic_z) {
        grid_3D->add_periodicity(matched_pairs);
    }
    return grid_3D;
#endif
}

template <int dim, int nstate>
double Airfoil2D<dim,nstate>::compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> lift_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::lift);
    const double lift = lift_functional.evaluate_functional();
    return lift;
}

template <int dim, int nstate>
double Airfoil2D<dim,nstate>::compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> drag_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::drag);
    const double drag = drag_functional.evaluate_functional();
    return drag;
}


template <int dim, int nstate>
void Airfoil2D<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    if constexpr(nstate!=1){
        LiftDragFunctional<dim,dim+2,double,Triangulation> lift_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::lift);
        double lift = lift_functional.evaluate_functional();

        LiftDragFunctional<dim,dim+2,double,Triangulation> drag_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::drag);
        double drag = drag_functional.evaluate_functional();

        this->pcout << " Resulting lift : " << lift << std::endl;
        this->pcout << " Resulting drag : " << drag << std::endl;
    }
}

template <int dim, int nstate>
void Airfoil2D<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    // Compute aerodynamic values
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg);

    if(this->mpi_rank==0) {
        // Add values to data table
        this->add_value_to_data_table(current_time,"time",unsteady_data_table);
        this->add_value_to_data_table(lift,"lift",unsteady_data_table);
        this->add_value_to_data_table(drag,"drag",unsteady_data_table);
        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }
    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Lift: " << lift
                << "    Drag: " << drag;
    this->pcout << std::endl;

    // Abort if energy is nan
    if(std::isnan(lift) || std::isnan(drag)) {
        this->pcout << " ERROR: Lift or drag at time " << current_time << " is nan." << std::endl;
        this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }
}

#if PHILIP_DIM!=1
    //template class Airfoil2D <PHILIP_DIM,1>;
    template class Airfoil2D <PHILIP_DIM,PHILIP_DIM+2>;
    //template class Airfoil2D <PHILIP_DIM,PHILIP_DIM+3>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace
