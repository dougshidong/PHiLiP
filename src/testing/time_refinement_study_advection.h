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
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
