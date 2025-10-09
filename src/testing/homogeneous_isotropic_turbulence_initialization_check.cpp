#include "homogeneous_isotropic_turbulence_initialization_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_turbulence.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include <deal.II/base/table_handler.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nspecies, int nstate>
HomogeneousIsotropicTurbulenceInitializationCheck<dim, nspecies, nstate>::HomogeneousIsotropicTurbulenceInitializationCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
        , kinetic_energy_expected(parameters_input->flow_solver_param.expected_kinetic_energy_at_final_time)
{}

template <int dim, int nspecies, int nstate>
int HomogeneousIsotropicTurbulenceInitializationCheck<dim, nspecies, nstate>::run_test() const
{
    // copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nspecies,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nspecies,nstate>::select_flow_case(&parameters, parameter_handler);
    
    this->pcout << "Outputting solution files at initialization... " << std::flush;
    flow_solver->dg->output_results_vtk(9999);
    this->pcout << "done." << std::endl;

    std::unique_ptr<FlowSolver::PeriodicTurbulence<dim,nspecies,nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicTurbulence<dim,nspecies,nstate>>(this->all_parameters);
    flow_solver_case->output_velocity_field(flow_solver->dg,0,0.0);

    return 0;
}

#if PHILIP_DIM==3 && PHILIP_SPECIES==1
    template class HomogeneousIsotropicTurbulenceInitializationCheck<PHILIP_DIM,PHILIP_SPECIES,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace