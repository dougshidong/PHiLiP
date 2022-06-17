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
    /// Function computes the energy
    double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
    /// Function computes the conservation
    double compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const;
};

} // Tests namespace
} // PHiLiP namespace
#endif
