#ifndef __ADVECTION_EXPLICIT_PERIODIC_H__
#define __ADVECTION_EXPLICIT_PERIODIC_H__

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Advection periodic unsteady test
template <int dim, int nstate>
class AdvectionPeriodic: public TestsBase
{
public:
    /// Constructor
    AdvectionPeriodic(const Parameters::AllParameters *const parameters_input);

    /// Run test
    int run_test () const override;
private:
    /// Function computes the energy
    double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
    /// Function computes the conservation
    double compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const;
    /// Warping for nonlinear manifold (see CurvManifold above)
    static dealii::Point<dim> warp (const dealii::Point<dim> &p);
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
