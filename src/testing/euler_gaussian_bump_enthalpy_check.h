#ifndef __EULER_GAUSSIAN_BUMP_ENTHALPY_CHECK_H__
#define __EULER_GAUSSIAN_BUMP_ENTHALPY_CHECK_H__
#include "euler_gaussian_bump.h"

namespace PHiLiP {
namespace Tests {
/// Checks if enthalpy is conserved with enthalpy laplacian artificial dissipation.
template <int dim, int nstate>
class EulerGaussianBumpEnthalpyCheck: public TestsBase
{
    public:
    /// Constructor
    EulerGaussianBumpEnthalpyCheck(const Parameters::AllParameters *const parameters_input);
    
    /// Checks if enthalpy is conserved by comparing errors in subsonic and transonic runs.
    int run_test() const;
};

} // Tests namespace
} // PHiLiP namespace

#endif
