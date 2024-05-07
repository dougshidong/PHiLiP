#include "naca0012_unsteady_check_quick.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/naca0012.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
NACA0012UnsteadyCheckQuick<dim, nstate>::NACA0012UnsteadyCheckQuick(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters NACA0012UnsteadyCheckQuick<dim,nstate>::reinit_params(const bool use_weak_form_input, const bool use_two_point_flux_input) const
{
     PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
     
     parameters.use_weak_form = use_weak_form_input;
     using ConvFluxEnum = Parameters::AllParameters::ConvectiveNumericalFlux;
     parameters.conv_num_flux_type = (use_two_point_flux_input) ? ConvFluxEnum::two_point_flux : ConvFluxEnum::roe;
     
     return parameters;
}

template <int dim, int nstate>
int NACA0012UnsteadyCheckQuick<dim, nstate>::run_test() const
{
    const int n_runs = 3;
    double lift_calculated[n_runs] = {0};
    double drag_calculated[n_runs] = {0};
    
    const bool use_weak_form[3] = {true, false, false}; // choose weak or strong
    const bool use_two_point_flux[3] = {false, false, true}; // choose two point flux or roe flux
    for (unsigned int irun = 0; irun < n_runs; ++irun) {
        
        this->pcout << "=====================================================" << std::endl;
        // Make new parameters according to the current run
        const Parameters::AllParameters params_loop = reinit_params(use_weak_form[irun], use_two_point_flux[irun]);

        if (use_weak_form[irun]){
            this->pcout << "Initialized parameters with weak form." << std::endl;
        } else{
            this->pcout << "Initialized parameters with strong form." << std::endl;
        }

        if (use_two_point_flux[irun]){
            this->pcout << "Using two-point flux." << std::endl;
        } else{
            this->pcout << "Using roe flux." << std::endl;
        }
        this->pcout << std::endl;
        
        // Initialize flow_solver
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate,1>> flow_solver_loop = FlowSolver::FlowSolverFactory<dim,nstate,1>::select_flow_case(&params_loop, parameter_handler);
        
        static_cast<void>(flow_solver_loop->run());
        std::unique_ptr<FlowSolver::NACA0012<dim, nstate>> flow_solver_case_loop = std::make_unique<FlowSolver::NACA0012<dim,nstate>>(this->all_parameters);
        lift_calculated[irun] = flow_solver_case_loop->compute_lift((flow_solver_loop->dg));
        drag_calculated[irun] = flow_solver_case_loop->compute_drag((flow_solver_loop->dg));
        this->pcout << "Finished run." << std::endl;
        this->pcout << "Calculated lift value was :  " << lift_calculated[irun] << std::endl
                    << "Calculated drag value was :  " << drag_calculated[irun] << std::endl;
        this->pcout << "=====================================================" << std::endl;
    }

    const double acceptable_tolerance = 0.00001;
    int testfail = 0;

    // For now, allow split form to have a different end value. Leaving as a condition so we can reevaluate this choice later.
    const bool allow_strong_split_to_be_different = true;

    this->pcout << std::endl 
                << "Finished runs. Summary of results: " << std::endl
                << "Scheme        |    Lift    |    Drag" << std::endl
                << "------------------------------------" << std::endl
                << "Weak, roe     | " << lift_calculated[0] << " | " << drag_calculated[0] << std::endl
                << "Strong, roe   | " << lift_calculated[1] << " | " << drag_calculated[1] << std::endl
                << "Strong, split | " << lift_calculated[2] << " | " << drag_calculated[2] << std::endl;

    if (allow_strong_split_to_be_different) {
        if ((abs(lift_calculated[0]-lift_calculated[1]) > acceptable_tolerance)
                || (abs(drag_calculated[0]-drag_calculated[1]) > acceptable_tolerance)){
            testfail = 1;
        }
    } else{
        const double lift_max = *std::max_element(lift_calculated, lift_calculated+n_runs);
        const double lift_min = *std::min_element(lift_calculated, lift_calculated+n_runs);
        const double drag_max = *std::max_element(drag_calculated, drag_calculated+n_runs);
        const double drag_min = *std::min_element(drag_calculated, drag_calculated+n_runs);
        if ((abs(lift_max - lift_min) > acceptable_tolerance)
                    || (abs(drag_max - drag_min) > acceptable_tolerance)){
            testfail = 1;
        }

    }
    if (testfail == 1){
        this->pcout << "Test failing: difference is not within the allowable tolerance of " << acceptable_tolerance << std::endl;
        this->pcout << "If this test is failing, please check the *vtu output." << std::endl;
    } else {
        this->pcout << "Test passing: lift and drag close to the expected value." << std::endl;
    }

    return testfail; 
}

#if PHILIP_DIM==2
    template class NACA0012UnsteadyCheckQuick<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
