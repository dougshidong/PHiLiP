#include "unsteady_reduced_order.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/pod_basis_online.h"
namespace PHiLiP {
namespace Tests {

template<int dim, int nstate>
UnsteadyReducedOrder<dim,nstate>::UnsteadyReducedOrder(const Parameters::AllParameters *const parameters_input,
                                                       const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
int UnsteadyReducedOrder<dim,nstate>::run_test() const 
{
    pcout << "Starting unsteady reduced-order test..." << std::endl;
    int testfail = 0;

    // Creating FOM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_full_order = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    flow_solver_full_order->run();

    // Change Parameters to ROM
    Parameters::AllParameters ROM_param = *(TestsBase::all_parameters);
    ROM_param.ode_solver_param.ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_galerkin_runge_kutta_solver;
    ROM_param.ode_solver_param.allocate_matrix_dRdW = true;
    const Parameters::AllParameters ROM_param_const = ROM_param;

    // Create ROM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&ROM_param_const, parameter_handler);
    const int modes = flow_solver_galerkin->ode_solver->pod->getPODBasis()->n();
    flow_solver_galerkin->run();
    
    dealii::LinearAlgebra::distributed::Vector<double> full_order_solution(flow_solver_full_order->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> galerkin_solution(flow_solver_galerkin->dg->solution);

    const double galerkin_solution_error = ((galerkin_solution-=full_order_solution).l2_norm()/full_order_solution.l2_norm());
    
    pcout << "Galerkin solution error: " << galerkin_solution_error << std::endl;
    if (std::abs(galerkin_solution_error) > 2.5E-5) testfail = 1;

    // Hard coding expected_modes based on past test results
    if (constexpr int expected_modes = 30; modes != expected_modes) testfail = 1;
    return testfail;
}

template class UnsteadyReducedOrder<PHILIP_DIM, PHILIP_DIM+2>;


} // Tests namespace
} // PHiLiP namespace