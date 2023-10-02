#ifndef __BURGERS_LIMITER_H__
#define __BURGERS_LIMITER_H__

#include "tests.h"
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Linear advection through periodic boundary conditions.
template <int dim, int nstate>
class BurgersLimiter : public TestsBase
{
public:
    /// Constructor.
    explicit BurgersLimiter(const Parameters::AllParameters* const parameters_input,
        const dealii::ParameterHandler& parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler& parameter_handler;

    /// Currently passes no matter what.
    int run_test() const override;

private:
    int run_burgers_lim() const;

    int run_burgers_lim_conv() const;

    void set_initial_time_step(const unsigned int n_global_active_cells, const int poly_degree) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
