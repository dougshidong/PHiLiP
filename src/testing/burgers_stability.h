#ifndef __BURGERS_STABILITY_H__
#define __BURGERS_STABILITY_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
/// Burgers' periodic unsteady test
class BurgersEnergyStability: public TestsBase
{
public:
        /// Constructor
	BurgersEnergyStability(const Parameters::AllParameters *const parameters_input);
        /// Run the testcase
        int run_test () const override;
private:
    /// Function computes the energy
	double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
    /// Function computes the conservation
        double compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
