#ifndef __BURGERS_STABILITY_H__
#define __BURGERS_STABILITY_H__

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Burgers' periodic unsteady test
template <int dim, int nstate>
class BurgersEnergyStability: public TestsBase
{
public:
    /// Constructor
    explicit BurgersEnergyStability(const Parameters::AllParameters *const parameters_input);

    /// Run test
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
