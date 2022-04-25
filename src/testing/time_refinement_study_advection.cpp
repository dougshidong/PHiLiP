#include "time_refinement_study_advection.h"
#include "flow_solver.h"
#include "flow_solver_cases/periodic_1D_flow.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TimeRefinementStudyAdvection<dim, nstate>::TimeRefinementStudyAdvection(const PHiLiP::Parameters::AllParameters *const parameters_input)  //CHECK NAME
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int TimeRefinementStudyAdvection<dim, nstate>::run_test() const
{

    const int n_time_calculations = 1;
    //const double refine_ratio = 0.5;
    //Construct flow_solver with initial_time_step

    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(this->all_parameters);
    for (int refinement = 0; refinement < n_time_calculations; ++refinement){
        static_cast<void>(flow_solver->run_test());
        //reset time to zero
        //refine timestep
    }

    //PASS/FAIL CHECK

    return 0;
}

#if PHILIP_DIM==1
    template class TimeRefinementStudyAdvection<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
