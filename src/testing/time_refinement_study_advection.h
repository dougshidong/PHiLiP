#ifndef __TIME_REFINEMENT_STUDY_ADVECTION__
#define __TIME_REFINEMENT_STUDY_ADVECTION__

#include "tests.h"
#include <deal.II/base/convergence_table.h>

namespace PHiLiP {
namespace Tests {

/// Advection time refinement study 
template <int dim, int nstate>
class TimeRefinementStudyAdvection: public TestsBase
{
public:
    /// Constructor
    TimeRefinementStudyAdvection(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~TimeRefinementStudyAdvection() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
protected:

    /// Number of times to solve for convergence summary
    const int n_time_calculations = 4;

    /// Ratio to refine by
    const double refine_ratio = 0.5;
    
    /// Reinitialize parameters while refining the timestep. Necessary because all_parameters is constant.
    Parameters::AllParameters reinit_params_and_refine_timestep(int refinement) const;


    /// Parameters for the current refinement (i.e., duplicate of .prm file except for timestep size)
    //Parameters::AllParameters params;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
