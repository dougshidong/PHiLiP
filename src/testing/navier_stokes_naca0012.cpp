#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <fenv.h>

#include <fstream>

#include "dg/dg.h"
#include "functional/lift_drag.hpp"
#include "mesh/gmsh_reader.hpp"
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "physics/physics_factory.h"
#include <Sacado.hpp>

#include "navier_stokes_naca0012.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_base.h"
#include "ode_solver/ode_solver_factory.h"
#include "ode_solver/runge_kutta_methods/rk_tableau_base.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "flow_solver/flow_solver_cases/naca0012.h"
#include "ode_solver/explicit_ode_solver.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
NavierStokesNACA0012<dim, nstate> :: NavierStokesNACA0012(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}


template<int dim, int nstate>
void NavierStokesNACA0012<dim,nstate> :: set_p_degree_and_interpolate_solution(const unsigned int poly_degree, DGBase<dim,double> &dg) const
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; 
    using DoFHandlerType   = typename dealii::DoFHandler<dim>;
    using MeshType = dealii::parallel::distributed::Triangulation<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    assert(dg.get_min_fe_degree() == dg.get_max_fe_degree());
    const unsigned int current_poly_degree = dg.get_min_fe_degree();
    pcout<<"Changing poly degree from "<< current_poly_degree << " to "<<poly_degree<<" and interpolating solution."<<std::endl;
    VectorType solution_coarse = dg.solution;
    solution_coarse.update_ghost_values();

    SolutionTransfer solution_transfer(dg.dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_coarse);

    dg.set_all_cells_fe_degree(poly_degree);
    dg.allocate_system();
    dg.solution.zero_out_ghosts();

    if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>,
                                 decltype(solution_transfer)>) {
        solution_transfer.interpolate(solution_coarse, dg.solution);
    } else {
        solution_transfer.interpolate(dg.solution);
    }

    dg.solution.update_ghost_values();
    pcout<<"\nSolution successfully interpolated to poly degree "<<poly_degree<<std::endl;
}

template<int dim, int nstate>
int NavierStokesNACA0012<dim,nstate>
::run_test () const
{
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    //Define starting and ending poly degrees
    const unsigned int p_start             = param.manufactured_convergence_study_param.degree_start;
    const unsigned int p_end               = param.manufactured_convergence_study_param.degree_end;
    param.flow_solver_param.poly_degree = p_start;
    param.flow_solver_param.max_poly_degree_for_adaptation = p_end;

    //Create flow solver and run steady state implicit solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();

    //Interpolate solution to p_end
    set_p_degree_and_interpolate_solution(param.flow_solver_param.max_poly_degree_for_adaptation, *(flow_solver->dg));

    //Explicit Solver
    std::shared_ptr<PHiLiP::ODE::RKTableauBase<dim, double>> rk_tableau = PHiLiP::ODE::ODESolverFactory<dim, double>::create_RKTableau(flow_solver->dg);
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_explicit = std::make_shared<PHiLiP::ODE::RungeKuttaODESolver<dim,double,4,dealii::parallel::distributed::Triangulation<dim>>>(flow_solver->dg,rk_tableau);
    double finalTime = 0.1;
    ode_solver_explicit->current_iteration = 0;
    ode_solver_explicit->advance_solution_time(finalTime);

    return 0;
}

#if PHILIP_DIM!=1
template class NavierStokesNACA0012 <PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
