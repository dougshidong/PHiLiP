#ifndef __BURGERS_STABILITY_H__
#define __BURGERS_STABILITY_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers sine wave to shock.
/** Ensure that the kinetic energy is bounded.
 *  Gassner 2017.
 */
template <int dim, int nstate>
class BurgersEnergyStability: public TestsBase
{
public:
    /// Constructor.
	BurgersEnergyStability(const Parameters::AllParameters *const parameters_input);
    /// Ensure that the kinetic energy is bounded.
    /** If the kinetic energy increases about its initial value, then the test should fail.
     *  Gassner 2017.
     */
    int run_test () const override;
private:
    /// Computes an integral of the kinetic energy (solution squared) in the entire domain.
    /** Uses Inverse of inverse mass matrix (?) to evaluate integral of u^2.
     */
	double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
