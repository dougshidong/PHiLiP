#ifndef __DUALWEIGHTEDRESIDUALMESHADAPTATION_H__ 
#define __DUALWEIGHTEDRESIDUALMESHADAPTATION_H__ 

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Test to check the goal-oriented mesh adaptation locations for various manufactured solutions.
template <int dim, int nstate>
class DualWeightedResidualMeshAdaptation : public TestsBase
{
public:
    /// Constructor of DualWeightedResidualConvergence.
    DualWeightedResidualMeshAdaptation(const Parameters::AllParameters *const parameters_input);

    /// Runs the test to check 2p+1 convergence of the dual weighted residual and goal-oriented mesh adaptation locations.
    int run_test() const;
}; 

} // Tests namespace
} // PHiLiP namespace

#endif

