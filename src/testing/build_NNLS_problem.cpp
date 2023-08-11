#include "build_NNLS_problem.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_factory.h"
#include "hyper_reduction/assemble_problem_ECSW.h"
#include "pod_adaptive_sampling.cpp"
#include <iostream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BuildNNLSProblem<dim, nstate>::BuildNNLSProblem(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int BuildNNLSProblem<dim, nstate>::run_test() const
{
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_petrov_galerkin->dg);
    std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);

    parameter_sampling->configureInitialParameterSpace();
    // parameter_sampling->placeInitialSnapshots();
    MatrixXd snapshot_parameters = parameter_sampling->snapshot_parameters;
    double *params = new double[snapshot_parameters.rows()];
    for (int i = 0; i < snapshot_parameters.rows(); i++){
        params[i] = snapshot_parameters(i, 0);
    }
    std::sort(params, params+snapshot_parameters.rows());
    std::cout << "Sorted Array looks like this." << std::endl;
    for (int i = 0; i <snapshot_parameters.rows(); i++){
        std::cout << params[i] << " ";}

    std::cout << "Construct instance of Assembler..."<< std::endl;
    HyperReduction::AssembleECSW<dim,nstate> constructer_NNLS_problem(flow_solver_petrov_galerkin->dg, pod_petrov_galerkin, params, ode_solver_type);
    std::cout << "Build Problem..."<< std::endl;
    constructer_NNLS_problem.build_problem();

    return 0;
}

#if PHILIP_DIM==1
        template class BuildNNLSProblem<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class BuildNNLSProblem<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
