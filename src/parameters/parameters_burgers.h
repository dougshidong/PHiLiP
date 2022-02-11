#ifndef __PARAMETERS_BURGERS__
#define __PARAMETERS_BURGERS__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to reduced-order model
class BurgersParam
{
public:
    /// Parameter a for eq.(18) in Carlberg 2011
    double rewienski_a;

    /// Parameter b for eq.(18) in Carlberg 2011
    double rewienski_b;

    /// Parameter for diffusion coefficient
    double diffusion_coefficient;

    /** Set as true for running a manufactured solution.
    *  Adds the manufactured solution source term to the PDE source term
    */
    bool rewienski_manufactured_solution;


    BurgersParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // Parameters namespace
} // PHiLiP namespace
#endif
