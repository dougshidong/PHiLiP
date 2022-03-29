#ifndef __REDUCED_ORDER_H__
#define __REDUCED_ORDER_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// POD reduced order
template <int dim, int nstate>
class ReducedOrder: public TestsBase
{
public:
    /// Constructor.
    ReducedOrder(const Parameters::AllParameters *const parameters_input);

    /// Run POD reduced order
    int run_test () const override;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
