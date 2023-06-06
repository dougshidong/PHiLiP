#include "naca0012.h"
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

namespace PHiLiP {
namespace FlowSolver {
//=========================================================
// NACA0012
//=========================================================
template <int dim, int nstate>
NACA0012<dim, nstate>::NACA0012(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
{}

template <int dim, int nstate>
void NACA0012<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    if (pde_type == PDE_enum::navier_stokes){
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    }
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    const double pi = atan(1.0) * 4.0;
    this->pcout << "- - Angle of attack [deg]: " << this->all_param.euler_param.angle_of_attack*180/pi << std::endl;
    this->pcout << "- - Side-slip angle [deg]: " << this->all_param.euler_param.side_slip_angle*180/pi << std::endl;
    this->pcout << "- - Farfield conditions: " << std::endl;
    const dealii::Point<dim> dummy_point;
    for (int s=0;s<nstate;s++) {
        this->pcout << "- - - State " << s << "; Value: " << this->initial_condition_function->value(dummy_point, s) << std::endl;
    }
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NACA0012<dim,nstate>::generate_grid() const
{
    //Dummy triangulation
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
    );
    dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    airfoil_data.airfoil_type = "NACA";
    airfoil_data.naca_id = "0012";
    airfoil_data.airfoil_length = 1.0;
    const double farfield_length = 500.0;
    airfoil_data.height = farfield_length;
    airfoil_data.length_b2  = farfield_length;
    airfoil_data.n_subdivision_x_0 = 2; // in front of leading edge.
    airfoil_data.n_subdivision_x_1 = 1; // between top of the airfoil and trailing edge.
    airfoil_data.n_subdivision_x_2 = 2; // vertical division behind trailing edge.
    airfoil_data.n_subdivision_y = 3; // horizontal division of cmesh.
    airfoil_data.refinements = 3;
    airfoil_data.bias_factor = 4;
    airfoil_data.airfoil_sampling_factor = 4;

    dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
    
    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) 
    {
        if(! cell->is_locally_owned()) {continue;}
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) 
        {
            if (cell->face(face)->at_boundary()) 
            {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 1 || current_id == 4 || current_id == 5) 
                {
                    cell->face(face)->set_boundary_id (1004); // farfield
                } 
                else 
                {
                    cell->face(face)->set_boundary_id (1001); // wall bc
                }
            }
        }
    }
    return grid;
}
/*
template <int dim, int nstate>
void NACA0012<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    const bool use_mesh_smoothing = false;
    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename, 0, use_mesh_smoothing);
    dg->set_high_order_grid(naca0012_mesh);
    for (int i=0; i<this->all_param.flow_solver_param.number_of_mesh_refinements; ++i) {
        dg->high_order_grid->refine_global();
    }
}
*/
template <int dim, int nstate>
void NACA0012<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> lift_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::lift);
    double lift = lift_functional.evaluate_functional();

    LiftDragFunctional<dim,dim+2,double,Triangulation> drag_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::drag);
    double drag = drag_functional.evaluate_functional();

    this->pcout << " Resulting lift : " << lift << std::endl;
    this->pcout << " Resulting drag : " << drag << std::endl;
}

#if PHILIP_DIM==2
    template class NACA0012<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

