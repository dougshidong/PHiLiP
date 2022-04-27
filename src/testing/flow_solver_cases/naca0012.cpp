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
void NACA0012<dim,nstate>::display_flow_solver_setup() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::euler)                {pde_string = "euler";}
    if (pde_type == PDE_enum::navier_stokes)        {pde_string = "navier_stokes";}
    this->pcout << "- PDE Type: " << pde_string << std::endl;
    this->pcout << "- Polynomial degree: " << this->all_param.grid_refinement_study_param.poly_degree << std::endl;
    this->pcout << "- Courant-Friedrich-Lewy number: " << this->all_param.flow_solver_param.courant_friedrich_lewy_number << std::endl;
    this->pcout << "- Final time: " << this->all_param.flow_solver_param.final_time << std::endl;
    this->pcout << "- Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    this->pcout << "- Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    this->pcout << "- Angle of attack: " << this->all_param.euler_param.angle_of_attack << std::endl;
    this->pcout << "- Side-slip angle: " << this->all_param.euler_param.side_slip_angle << std::endl;
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NACA0012<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    airfoil_data.airfoil_type = "NACA";
    airfoil_data.naca_id      = "0012";
    airfoil_data.airfoil_length = 1.0;
    airfoil_data.height         = 150.0; // Farfield radius.
    airfoil_data.length_b2      = 150.0;
    airfoil_data.incline_factor = 0.0;
    airfoil_data.bias_factor    = 5.0;
    airfoil_data.refinements    = 0;
    airfoil_data.n_subdivision_x_0 = 4;
    airfoil_data.n_subdivision_x_1 = 4;
    airfoil_data.n_subdivision_x_2 = 4;
    airfoil_data.n_subdivision_y = 4;

    airfoil_data.airfoil_sampling_factor = 100000; // default 2

    Grids::naca_airfoil(reinterpret_cast<dealii::parallel::distributed::Triangulation<2> &>(*grid), airfoil_data);

    grid->refine_global();

    return grid;
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename);
    dg->set_high_order_grid(naca0012_mesh);
    dg->high_order_grid->refine_global();
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double> lift_functional(dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::lift);
    double lift = lift_functional.evaluate_functional();

    LiftDragFunctional<dim,dim+2,double> drag_functional(dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::drag);
    double drag = drag_functional.evaluate_functional();

    std::cout << " Resulting lift : " << lift << std::endl;
    std::cout << " Resulting drag : " << drag << std::endl;
}


#if PHILIP_DIM==2
template class NACA0012<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

