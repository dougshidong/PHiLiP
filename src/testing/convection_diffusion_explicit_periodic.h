#ifndef __CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_H__
#define __CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Convection Diffusion periodic unsteady test (currently only diffusion)
template <int dim, int nstate>
class ConvectionDiffusionPeriodic: public TestsBase
{
public:
    /// Constructor
    ConvectionDiffusionPeriodic(const Parameters::AllParameters *const parameters_input);
    
    /// Destructor
    ~ConvectionDiffusionPeriodic() {};

    /// Run test
    int run_test () const override;
private:
    /// Function computes the energy bound.
    /** Note that for convection-diffusion, we have a proof on the norm of the time rate of
    * change of the energy with respect to the norm of the auxiliary variable.
    * Explicitly, \f$ \frac{1}{2}\frac{d}{dt}\| U\|^2 + b\|Q\|^2 \leq 0\f$. 
    * The energy dissipation is related to the auxiliary variable. Thus, for a central flux,
    * we can prove the above is equal to 0 within machine precision. This is not the same
    * as the energy being conserved.
    */
	double compute_energy_derivative_norm(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
    
    /// Function computes the conservation
    double compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const;
};

} // Tests namespace
} // PHiLiP namespace
#endif
