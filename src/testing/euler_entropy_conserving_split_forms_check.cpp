#include "euler_entropy_conserving_split_forms_check.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_turbulence.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerSplitEntropyCheck<dim, nstate>::EulerSplitEntropyCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int EulerSplitEntropyCheck<dim, nstate>::run_test() const
{
    int testfail = 0;
    const unsigned int n_fluxes = 4;

    using TwoPtFluxEnum = Parameters::AllParameters::TwoPointNumericalFlux;
    const std::array<TwoPtFluxEnum, n_fluxes> two_point_fluxes{{TwoPtFluxEnum::IR, TwoPtFluxEnum::CH, TwoPtFluxEnum::Ra, TwoPtFluxEnum::KG}};
    const std::array<double, n_fluxes> tols{{5E-15, 5E-15, 5E-15, 1E-10}};
    const std::array<std::string, n_fluxes> flux_names{{"Ismail-Roe", "Chandrashekar", "Ranocha", "Kennedy-Gruber"}};

    for (unsigned int i = 0; i < n_fluxes; ++i){
        pcout << "-----------------------------------------------------------------------" << std::endl;
        pcout << "   Using " << flux_names[i] << " two-point flux" << std::endl;
        pcout << "-----------------------------------------------------------------------" << std::endl;
        
        const TwoPtFluxEnum flux = two_point_fluxes[i];
        const double tol = tols[i];

        // Copying parameters and modifying flux type
        PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
        parameters.two_point_num_flux_type = flux;

        // Initialize flow_solver
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate,1>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate,1>::select_flow_case(&parameters, parameter_handler);

        // Compute  initial and final entropy
        std::shared_ptr<FlowSolver::PeriodicTurbulence<dim, nstate>> flow_solver_case = std::dynamic_pointer_cast<FlowSolver::PeriodicTurbulence<dim, nstate>>(flow_solver->flow_solver_case);
        flow_solver_case->compute_and_update_integrated_quantities(*flow_solver->dg);
        const double initial_KE = flow_solver_case->get_integrated_kinetic_energy();

        static_cast<void>(flow_solver->run());
        const double final_entropy = flow_solver_case->get_numerical_entropy(flow_solver->dg); 
        flow_solver_case->compute_and_update_integrated_quantities(*flow_solver->dg);
        const double final_KE = flow_solver_case->get_integrated_kinetic_energy();

        //Compare initial and final entropy to confirm entropy preservation
        
        pcout << "Final numerical entropy: " << final_entropy << std::endl;
        pcout << "Initial kinetic energy: " << std::setprecision(16) << initial_KE << std::endl
              << "Final:                  " << final_KE << std::endl
              << "Scaled difference:      " << abs((initial_KE-final_KE)/initial_KE)
              << std::endl << std::endl;


        if (abs(final_entropy) > tol){
            pcout << "Entropy change is not within allowable tolerance. Test failing." << std::endl;
            testfail = 1;
        } else pcout << "Entropy change is allowable." << std::endl;
    }

    return testfail;
}

#if PHILIP_DIM==3
    template class EulerSplitEntropyCheck<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
