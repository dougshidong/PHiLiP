#ifndef __PARAMETERS_LIMITER_H__
#define __PARAMETERS_LIMITER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to the limiter
class LimiterParam
{
public:
    /// Flag to perform convergence analysis for Limiter Tests (ie. burgers_limiter, advection_limiter, low_density_2d)
    bool use_OOA;

    /// Limiter type to be applied on the solution.
    enum LimiterType {
        none,
        maximum_principle,
        positivity_preservingZhang2010,
        positivity_preservingWang2012
    };
    LimiterType bound_preserving_limiter;

    // Epsilon value for Positivity-Preserving Limiter
    double pos_eps;

    /// Flag for applying TVB Limiter
    bool use_tvb_limiter;

    /// Maximum delta_x for TVB Limiter
    double tvb_h;

    /// Tuning parameters for TVB Limiter
    /** TVB Limiter can only be run for 1D, so max length is max nstate = 4 **/
    dealii::Tensor<1, 4, double> tvb_M;

    /// Constructor
    LimiterParam();

    /// Function to declare parameters.
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Function to parse parameters.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
