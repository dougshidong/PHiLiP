#ifndef __PARAMETERS_P_POISSON_H__
#define __PARAMETERS_P_POISSON_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to the p-Poisson equation
class PPoissonParam
{
public:
    double factor_p; ///< The factor p in p-Poisson equation.
    double stable_factor; ///< The additional factor for numerical stability in p-Poisson equation.

    PPoissonParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif