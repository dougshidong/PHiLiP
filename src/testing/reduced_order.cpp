#include <iostream>

#include "reduced_order.h"
#include "parameters/all_parameters.h"
#include "reduced_order/pod_basis.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "ode_solver/pod_galerkin_ode_solver.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
ReducedOrder<dim, nstate>::ReducedOrder(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int ReducedOrder<dim, nstate>::run_test() const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
    /*SET UP GRID, PARAMETERS AND INITIAL CONDITIONS*/

    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>();

    double left = param.grid_refinement_study_param.grid_left;
    double right = param.grid_refinement_study_param.grid_right;
    const bool colorize = true;
    int n_refinements = param.grid_refinement_study_param.num_refinements;
    unsigned int poly_degree = param.grid_refinement_study_param.poly_degree;
    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);

    grid->refine_global(n_refinements);
    pcout << "Grid generated and refined" << std::endl;

    pcout << "Implement initial conditions" << std::endl;
    dealii::FunctionParser<1> initial_condition;
    std::string variables = "x";
    std::map<std::string,double> constants;
    constants["pi"] = dealii::numbers::PI;
    std::string expression = "1";
    initial_condition.initialize(variables, expression, constants);

    double finalTime = param.flow_solver_param.final_time;

    /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
    /* FULL SOLUTION WITH IMPLICIT SOLVER */

    pcout << "Running full-order implicit ODE solver for Burgers Rewienski with parameter a: "
          << param.reduced_order_param.rewienski_a
          << " and parameter b: "
          << param.reduced_order_param.rewienski_b
          << std::endl;

    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg_implicit = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
    pcout << "dg implicit created" <<std::endl;
    dg_implicit->allocate_system ();

    //will use all basis functions
    std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod = std::make_shared<ProperOrthogonalDecomposition::POD<dim>>(dg_implicit);

    dealii::VectorTools::interpolate(dg_implicit->dof_handler,initial_condition,dg_implicit->solution);

    pcout << "Create implicit solver" << std::endl;
    // Create ODE solver using the factory and providing the DG object
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::implicit_solver;
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_implicit = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, dg_implicit);

    /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
    /*POD GALERKIN SOLUTION*/

    pcout << "Running POD-Galerkin ODE solver for Burgers Rewienski with parameter a: "
          << param.reduced_order_param.rewienski_a
          << " and parameter b: "
          << param.reduced_order_param.rewienski_b
          << std::endl;

    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg_pod_galerkin = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
    pcout << "dg reduced_order-galerkin created" <<std::endl;
    dg_pod_galerkin->allocate_system ();

    dealii::VectorTools::interpolate(dg_pod_galerkin->dof_handler,initial_condition,dg_pod_galerkin->solution);

    pcout << "Create POD-Galerkin ODE solver" << std::endl;
    // Create ODE solver using the factory and providing the DG object
    ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_galerkin_solver;
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_galerkin = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, dg_pod_galerkin, pod);

    /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
    /*POD PETROV-GALERKIN SOLUTION*/

    pcout << "Running POD-Petrov-Galerkin ODE solver for Burgers Rewienski with parameter a: "
          << param.reduced_order_param.rewienski_a
          << " and parameter b: "
          << param.reduced_order_param.rewienski_b
          << std::endl;

    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg_pod_petrov_galerkin = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
    pcout << "dg reduced_order-petrov-galerkin created" <<std::endl;
    dg_pod_petrov_galerkin->allocate_system ();

    dealii::VectorTools::interpolate(dg_pod_petrov_galerkin->dof_handler,initial_condition,dg_pod_petrov_galerkin->solution);

    pcout << "Create POD-Petrov-Galerkin ODE solver" << std::endl;
    // Create ODE solver using the factory and providing the DG object
    ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_petrov_galerkin = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, dg_pod_petrov_galerkin, pod);

    /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
    /*Time-averaged relative error, E = 1/n_t * sum_{n=1}^{n_t} (||U_FOM(t^{n}) - U_ROM(t^{n})||_L2 / ||U_FOM(t^{n})||_L2 )
     *Refer to section 6.1 in "The GNAT method for nonlinear model reduction: Effective implementation and application to computational ﬂuid dynamics and turbulent ﬂows"
     *Authors: Kevin Carlberg, Charbel Farhat, ulien Cortial,  David Amsallem
     *Journal of Computational Physics, 2013
     */

    const unsigned int number_of_time_steps = static_cast<int>(ceil(finalTime/param.ode_solver_param.initial_time_step));
    const double constant_time_step = finalTime/number_of_time_steps;

    pcout << " Advancing solution by " << finalTime << " time units, using "
          << number_of_time_steps << " iterations of size dt=" << constant_time_step << " ... " << std::endl;

    ode_solver_implicit->allocate_ode_system();
    ode_solver_galerkin->allocate_ode_system();
    ode_solver_petrov_galerkin->allocate_ode_system();

    double galerkin_error_norm_sum = 0;
    double petrov_galerkin_error_norm_sum = 0;

    unsigned int current_iteration = 0;

    while (current_iteration < number_of_time_steps)
    {
        pcout << " ********************************************************** "
              << std::endl
              << " Iteration: " << current_iteration + 1
              << " out of: " << number_of_time_steps
              << std::endl;

        dg_implicit->assemble_residual(false);
        dg_pod_galerkin->assemble_residual(false);
        dg_pod_petrov_galerkin->assemble_residual(false);

        const bool pseudotime = false;
        ode_solver_implicit->step_in_time(constant_time_step, pseudotime);
        ode_solver_galerkin->step_in_time(constant_time_step, pseudotime);
        ode_solver_petrov_galerkin->step_in_time(constant_time_step, pseudotime);

        dealii::LinearAlgebra::distributed::Vector<double> pod_galerkin_solution(dg_pod_galerkin->solution);
        dealii::LinearAlgebra::distributed::Vector<double> pod_petrov_galerkin_solution(dg_pod_petrov_galerkin->solution);
        dealii::LinearAlgebra::distributed::Vector<double> implicit_solution(dg_implicit->solution);

        galerkin_error_norm_sum = galerkin_error_norm_sum + ((pod_galerkin_solution.operator-=(implicit_solution)).l2_norm()/implicit_solution.l2_norm());
        petrov_galerkin_error_norm_sum = petrov_galerkin_error_norm_sum + (((pod_petrov_galerkin_solution.operator-=(implicit_solution)).l2_norm())/implicit_solution.l2_norm());

        current_iteration++;
    }

    double pod_galerkin_error = (1/(double)number_of_time_steps) * galerkin_error_norm_sum;

    double pod_petrov_galerkin_error = (1/(double)number_of_time_steps) * petrov_galerkin_error_norm_sum;

    pcout << "POD-Galerkin error: " << pod_galerkin_error << std::endl;
    pcout << "POD-Petrov-Galerkin error: " << pod_petrov_galerkin_error << std::endl;

    if (pod_galerkin_error < 1E-12 && pod_petrov_galerkin_error < 1E-12){
        pcout << "Passed!";
        return 0;
    }else{
        pcout << "Failed!";
        return -1;
    }
}
#if PHILIP_DIM==1
        template class ReducedOrder<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
