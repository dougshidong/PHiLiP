#ifndef __BURGERS_LIMITER_H__
#define __BURGERS_LIMITER_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers' periodic unsteady test
template <int dim, int nstate>
class BurgersLimiter: public TestsBase
{
public:
    /// Constructor
    BurgersLimiter(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~BurgersLimiter() {};

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif

