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
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    if (pde_type == PDE_enum::navier_stokes){
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    }
    this->pcout << "- - Courant-Friedrich-Lewy number: " << this->all_param.flow_solver_param.courant_friedrich_lewy_number << std::endl;
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
    if constexpr(dim==2) {
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );
        dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
        dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
        grid->refine_global();
        return grid;
    } 
    else if constexpr(dim==3) {
        const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
        const bool use_mesh_smoothing = false;
        std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename, 0, use_mesh_smoothing);
        return naca0012_mesh->triangulation;
    }
    
    // TO DO: Avoid reading the mesh twice (here and in set_high_order_grid -- need a default dummy triangulation)
}

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

template <int dim, int nstate>
double NACA0012<dim,nstate>::compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double> lift_functional(dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::lift);
    const double lift = lift_functional.evaluate_functional();
    return lift;
}

template <int dim, int nstate>
double NACA0012<dim,nstate>::compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double> drag_functional(dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::drag);
    const double drag = drag_functional.evaluate_functional();
    return drag;
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg);

    this->pcout << " Resulting lift : " << lift << std::endl;
    this->pcout << " Resulting drag : " << drag << std::endl;
}

template <int dim, int nstate>
void NACA0012<dim, nstate>::compute_unsteady_data_and_write_to_table(
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
    template class NACA0012<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

