#include "reduced_order.h"

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
    auto functional_implicit = functionalFactory(flow_solver_implicit->dg);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_galerkin_solver;
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_galerkin->dg);
    flow_solver_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_galerkin->dg, pod_galerkin);
    flow_solver_galerkin->ode_solver->allocate_ode_system();
    auto functional_galerkin = functionalFactory(flow_solver_galerkin->dg);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_petrov_galerkin->dg);
    flow_solver_petrov_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_petrov_galerkin->dg, pod_petrov_galerkin);
    flow_solver_petrov_galerkin->ode_solver->allocate_ode_system();
    auto functional_petrov_galerkin = functionalFactory(flow_solver_petrov_galerkin->dg);

    flow_solver_implicit->ode_solver->steady_state();
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

    if (galerkin_solution_error < 1E-11 && petrov_galerkin_solution_error < 1E-11 && galerkin_func_error < 1E-11 && petrov_galerkin_func_error < 1E-11){
        pcout << "Passed!";
        return 0;
    }else{
        pcout << "Failed!";
        return -1;
    }
}

template <int dim, int nstate>
std::shared_ptr<Functional<dim,nstate,double>> ReducedOrder<dim, nstate>::functionalFactory(std::shared_ptr<DGBase<dim, double>> dg) const
{
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if constexpr (dim==1 && nstate==dim){
            std::shared_ptr< DGBaseState<dim,nstate,double>> dg_state = std::dynamic_pointer_cast< DGBaseState<dim,nstate,double>>(dg);
            return std::make_shared<BurgersRewienskiFunctional<dim,nstate,double>>(dg,dg_state->pde_physics_fad_fad,true,false);
        }
    }
    else if (flow_type == FlowCaseEnum::naca0012){
        if constexpr (dim==2 && nstate==dim+2){
            return std::make_shared<LiftDragFunctional<dim,nstate,double>>(dg, LiftDragFunctional<dim,nstate,double>::Functional_types::lift);
        }
    }
    else{
        this->pcout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return nullptr;
}

#if PHILIP_DIM==1
        template class ReducedOrder<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class ReducedOrder<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
