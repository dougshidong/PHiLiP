#ifndef __PARAMETERS_BURGERS__
#define __PARAMETERS_BURGERS__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to reduced-order model
class BurgersParam
{
public:
    /* Parameter a for eq.(18) in Carlberg 2011
     * Carlberg, K., Amsallem, D., Avery, P., Zahr, M., & Farhat, C. (2011).
     * The GNAT nonlinear model reduction method and its application to fluid dynamics problems.
     * In 6th AIAA Theoretical Fluid Mechanics Conference (p. 3112).
     */
    double rewienski_a;

    /* Parameter b for eq.(18) in Carlberg 2011
     * Carlberg, K., Amsallem, D., Avery, P., Zahr, M., & Farhat, C. (2011).
     * The GNAT nonlinear model reduction method and its application to fluid dynamics problems.
     * In 6th AIAA Theoretical Fluid Mechanics Conference (p. 3112).
     */
    double rewienski_b;

    /// Parameter for diffusion coefficient
    double diffusion_coefficient;

    /** Set as true for running a manufactured solution for Burgers Rewienski.
    *  Adds the manufactured solution source term to the PDE source term
    */
    bool rewienski_manufactured_solution;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // Parameters namespace
} // PHiLiP namespace
#endif
