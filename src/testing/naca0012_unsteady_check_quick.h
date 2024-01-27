
#ifndef __NACA0012_UNSTEADY_CHECK_QUICK__
#define __NACA0012_UNSTEADY_CHECK_QUICK__

#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// NACA 0012 Unsteady Check 
/** Runs a short time interval for unsteady Euler flow over NACA0012 airfoil
 * Uses GMSH reader for the mesh
 * Compares against a hard-coded expectation value for lift
 * NOTE: it has not been verified that the results are physically meaningful;
 *       this is a verification test that weak and strong give relatively consistent results.
 */
template <int dim, int nstate>
class NACA0012UnsteadyCheckQuick: public TestsBase
{
public:
    /// Constructor
    NACA0012UnsteadyCheckQuick(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
protected:

    /// Reinit parameters based on specified weak/strong form and convective num flux
    Parameters::AllParameters reinit_params(const bool use_weak_form_input, const bool use_two_point_flux_input) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
