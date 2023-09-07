#ifndef __STABILITY_FR_PARAMETERS_RANGE_H__
#define __STABILITY_FR_PARAMETERS_RANGE_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers' periodic unsteady test
template <int dim, int nstate>
class StabilityFRParametersRange: public TestsBase
{
public:
    /// Constructor
    StabilityFRParametersRange(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~StabilityFRParametersRange() {};

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
