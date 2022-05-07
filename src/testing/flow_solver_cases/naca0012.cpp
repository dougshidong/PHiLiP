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
namespace Tests {
//=========================================================
// NACA0012
//=========================================================
template <int dim, int nstate>
NACA0012<dim, nstate>::NACA0012(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
{
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::display_additional_flow_case_specific_parameters(std::shared_ptr<InitialConditionFunction<dim,nstate,double>> initial_condition) const
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
        this->pcout << "- - - State " << s << "; Value: " << initial_condition->value(dummy_point, s) << std::endl;
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
    dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
    grid->refine_global();
    return grid;
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename);
    dg->set_high_order_grid(naca0012_mesh);
    for (unsigned int i=0; i<this->all_param.grid_refinement_study_param.num_refinements; ++i) {
        dg->high_order_grid->refine_global();
    }
    this->pcout << "Dimension: " << dim << "\t Polynomial degree p: " << this->all_param.grid_refinement_study_param.poly_degree << std::endl
          << ". Number of active cells: " << dg->triangulation->n_global_active_cells()
          << ". Number of degrees of freedom: " << dg->dof_handler.n_dofs()
          << std::endl;
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double> lift_functional(dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::lift);
    double lift = lift_functional.evaluate_functional();

    LiftDragFunctional<dim,dim+2,double> drag_functional(dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::drag);
    double drag = drag_functional.evaluate_functional();

    this->pcout << " Resulting lift : " << lift << std::endl;
    this->pcout << " Resulting drag : " << drag << std::endl;
}

#if PHILIP_DIM==2
    template class NACA0012<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

