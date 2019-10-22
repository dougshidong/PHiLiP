#ifndef __ADVECTION_EXPLICIT_PERIODIC_H__
#define __ADVECTION_EXPLICIT_PERIODIC_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Linear advection through periodic boundary conditions.
template <int dim, int nstate>
class AdvectionPeriodic: public TestsBase
{
public:
    /// Constructor.
	AdvectionPeriodic(const Parameters::AllParameters *const parameters_input);

    /// Currently passes no matter what.
    /** Since it is linear advection, the exact solution about time T is known. Convergence orders can/should be checked.
     *  TO BE FIXED.
     */
    int run_test () const override;
private:
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
