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
int NACA0012UnsteadyCheckQuick<dim, nstate>::run_test() const
{
    double lift_expected;
    double drag_expected;
    const double acceptable_tolerance = 0.05;

    // These values are hard-coded from the results of running the test case.
    // It is expected that changes such as boundary implementation,
    // flux choice etc could change these values, so they may need to be updated
    // in the future.
    if (this->all_parameters->use_weak_form) {
        // Weak, Roe flux
        lift_expected = 0.110977;
        drag_expected = 0.123985;
    }else if (this->all_parameters->use_split_form) {
        // Strong, Ra two-point flux
        lift_expected = 0.127556;
        drag_expected = 0.135251;
    } else {
        // Strong, Roe flux
        lift_expected = 0.110977;
        drag_expected = 0.123985;
    }

    // Initialize flow_solver
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, parameter_handler);

    static_cast<void>(flow_solver->run());

    //Get lift and drag from flow solver case
    // Placeholders for now
    std::unique_ptr<FlowSolver::NACA0012<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::NACA0012<dim,nstate>>(this->all_parameters);
    const double lift_calculated = flow_solver_case->compute_lift((flow_solver->dg));
    const double drag_calculated = flow_solver_case->compute_drag((flow_solver->dg));

    this->pcout << "Expected lift value was   :  " << lift_expected << std::endl
                << "Calculated lift value was :  " << lift_calculated << std::endl
                << "Expected drag value was   :  " << drag_expected << std::endl
                << "Calculated drag value was :  " << drag_calculated << std::endl;
    int testfail = 0;
    if ((abs(lift_calculated-lift_expected) > acceptable_tolerance) 
            || (abs(drag_calculated-drag_expected) > acceptable_tolerance)) {
        testfail = 1;
        this->pcout << "Test failing: difference is not within the allowable tolerance of " << acceptable_tolerance << std::endl;
        this->pcout << "If this test is failing, please check the *vtu output " << std::endl 
                    << "and/or check whether changes to physics have changed the expected values " << std::endl
                    << "which are hard-coded in src/testing/naca0012_unsteady_check_quick.cpp." <<std::endl;
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
