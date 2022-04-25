#ifndef __TIME_REFINEMENT_STUDY_ADVECTION__
#define __TIME_REFINEMENT_STUDY_ADVECTION__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Taylor Green Vortex
template <int dim, int nstate>
class TimeRefinementStudyAdvection: public TestsBase
{
public:
    /// Constructor
    TimeRefinementStudyAdvection(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~TimeRefinementStudyAdvection() {};
    
    /// Run test
    int run_test () const override;
protected:

    /// Number of times to solve for convergence summary
    const int n_time_calculations = 2;

    /// Ratio to refine by
    const double refine_ratio = 0.5;
    
    /// Data table for convergence summary
    //dealii::ConvergenceTable convergence_table;

    /// Write to data table at the end of a computation
    //void write_convergence_summary();

    /// L2 error
   // dealii::Vector<double> L2_error_conv_rate();


    /// Reinitialize parameters while refining the timestep. Necessary because all_parameters is constant.
    Parameters::AllParameters reinit_params_and_refine_timestep(int refinement) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
