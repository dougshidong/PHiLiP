#ifndef __DUALWEIGHTEDRESIDUALCONVERGENCE_H__ 
#define __DUALWEIGHTEDRESIDUALCONVERGENCE_H__ 

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Test to check the 2p+1 convergence of the dual weighted residual and goal-oriented mesh adaptation locations.
template <int dim, int nstate>
class DualWeightedResidualConvergence : public TestsBase
{
public:
    /// Constructor of DualWeightedResidualConvergence.
    DualWeightedResidualConvergence(const Parameters::AllParameters *const parameters_input);

    /// Runs the test to check 2p+1 convergence of the dual weighted residual and goal-oriented mesh adaptation locations.
    int run_test() const;

}; // Tests namespace

} // namespace Tests
} // PHiLiP namespace

#endif

