#ifndef __BURGERS_STABILITY_H__
#define __BURGERS_STABILITY_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class BurgersEnergyStability: public TestsBase
{
public:
	BurgersEnergyStability(const PHiLiP::Parameters::AllParameters *const parameters_input);
    int run_test () const override;
private:
	double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
