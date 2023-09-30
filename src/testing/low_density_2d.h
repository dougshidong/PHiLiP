#ifndef __LOW_DENSITY_H__
#define __LOW_DENSITY_H__

#include "tests.h"
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// 2D Low Density Euler Test
template <int dim, int nstate>
class LowDensity2D: public TestsBase
{
public:
    /// Constructor
    explicit LowDensity2D(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~LowDensity2D() = default;

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif

