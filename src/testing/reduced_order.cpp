#include "reduced_order.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include "functional/functional.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <deal.II/base/numbers.h>
#include "ode_solver/ode_solver_factory.h"
#include <iostream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
ReducedOrder<dim, nstate>::ReducedOrder(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int ReducedOrder<dim, nstate>::run_test() const
{
    pcout << "Starting reduced-order test..." << std::endl;

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_implicit = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto functional_implicit = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_implicit->dg);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_galerkin_solver;
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_galerkin->dg);
    flow_solver_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_galerkin->dg, pod_galerkin);
    flow_solver_galerkin->ode_solver->allocate_ode_system();
    auto functional_galerkin = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_galerkin->dg);


    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_petrov_galerkin->dg);
    flow_solver_petrov_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_petrov_galerkin->dg, pod_petrov_galerkin);
    flow_solver_petrov_galerkin->ode_solver->allocate_ode_system();
    auto functional_petrov_galerkin = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_petrov_galerkin->dg);

    flow_solver_implicit->run();
    flow_solver_galerkin->ode_solver->steady_state();
    flow_solver_petrov_galerkin->ode_solver->steady_state();

    dealii::LinearAlgebra::distributed::Vector<double> implicit_solution(flow_solver_implicit->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> galerkin_solution(flow_solver_galerkin->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> petrov_galerkin_solution(flow_solver_petrov_galerkin->dg->solution);

    double galerkin_solution_error = ((galerkin_solution-=implicit_solution).l2_norm()/implicit_solution.l2_norm());
    double petrov_galerkin_solution_error = ((petrov_galerkin_solution-=implicit_solution).l2_norm()/implicit_solution.l2_norm());

    double galerkin_func_error = functional_galerkin->evaluate_functional(false,false) - functional_implicit->evaluate_functional(false,false);
    double petrov_galerkin_func_error = functional_petrov_galerkin->evaluate_functional(false,false) - functional_implicit->evaluate_functional(false,false);

    pcout << "Galerkin solution error: " << galerkin_solution_error << std::endl
          << "Petrov-Galerkin solution error: " << petrov_galerkin_solution_error << std::endl
          << "Galerkin functional error: " << galerkin_func_error << std::endl
          << "Petrov-Galerkin functional error: " << petrov_galerkin_func_error << std::endl;

    if (std::abs(galerkin_solution_error) < 1E-10 && std::abs(petrov_galerkin_solution_error) < 1E-11 && std::abs(galerkin_func_error) < 1E-10 && std::abs(petrov_galerkin_func_error) < 1E-11){
        pcout << "Passed!";
        return 0;
    }else{
        pcout << "Failed!";
        return -1;
    }
}

#if PHILIP_DIM==1
        template class ReducedOrder<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class ReducedOrder<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
