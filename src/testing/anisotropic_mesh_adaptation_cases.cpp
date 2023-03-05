#include <stdlib.h>
#include <iostream>
#include "anisotropic_mesh_adaptation_cases.h"
#include "flow_solver/flow_solver_factory.h"
#include "mesh/mesh_adaptation/anisotropic_mesh_adaptation.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
AnisotropicMeshAdaptationCases<dim, nstate> :: AnisotropicMeshAdaptationCases(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int AnisotropicMeshAdaptationCases<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();
    const bool use_goal_oriented_approach = false;
    const double complexity = 300;
    double normLp = 2.0;
    if(use_goal_oriented_approach) {normLp = 1.0;}

    std::unique_ptr<AnisotropicMeshAdaptation<dim, nstate, double>> anisotropic_mesh_adaptation =
                        std::make_unique<AnisotropicMeshAdaptation<dim, nstate, double>> (flow_solver->dg, normLp, complexity, use_goal_oriented_approach);

    anisotropic_mesh_adaptation->compute_optimal_metric();

    return 0;
}

#if PHILIP_DIM==1
template class AnisotropicMeshAdaptationCases <PHILIP_DIM,PHILIP_DIM>;
#endif

#if PHILIP_DIM==2
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, 1>;
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, PHILIP_DIM + 2>;
#endif
} // namespace Tests
} // namespace PHiLiP 
    
