#ifndef __STABILITY_FR_PARAMETERS_RANGE_H__
#define __STABILITY_FR_PARAMETERS_RANGE_H__

#include "tests.h"
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers' periodic unsteady test
template <int dim, int nstate>
class StabilityFRParametersRange: public TestsBase
{
public:
    /// Constructor
    explicit StabilityFRParametersRange(const Parameters::AllParameters *const parameters_input);

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
