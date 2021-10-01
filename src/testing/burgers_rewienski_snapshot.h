#ifndef __BURGERS_REWIENSKI_TEST_H__
#define __BURGERS_REWIENSKI_TEST_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers Rewienski snapshot
template <int dim, int nstate>
class BurgersRewienskiSnapshot: public TestsBase
{
public:
    /// Constructor.
    BurgersRewienskiSnapshot(const Parameters::AllParameters *const parameters_input);

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
