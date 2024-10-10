#include "unsteady_reduced_order.h"

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
    
    // Creating flowsolver objects
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_full_order = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);

    
    flow_solver_full_order->run();

    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_galerkin = flow_solver_full_order->time_pod;
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_galerkin_runge_kutta_solver;
    flow_solver_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_galerkin->dg, pod_galerkin);
    flow_solver_galerkin->ode_solver->allocate_ode_system();

    flow_solver_galerkin->run();
    
    dealii::LinearAlgebra::distributed::Vector<double> full_order_solution(flow_solver_full_order->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> galerkin_solution(flow_solver_galerkin->dg->solution);

    double galerkin_solution_error = ((galerkin_solution-=full_order_solution).l2_norm()/full_order_solution.l2_norm());

    pcout << "Galerkin solution error: " << galerkin_solution_error << std::endl;

    if (std::abs(galerkin_solution_error) > 1E-10) testfail = 1;
    return testfail;
}

template class ReducedOrder<PHILIP_DIM, PHILIP_DIM>;
template class ReducedOrder<PHILIP_DIM, PHILIP_DIM+2>;

} // Tests namespace
} // PHiLiP namespace