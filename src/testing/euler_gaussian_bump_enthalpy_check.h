 #ifndef __EULER_GAUSSIAN_BUMP_ENTHALPY_CHECK_H__
 #define __EULER_GAUSSIAN_BUMP_ENTHALPY_CHECK_H__
 #include "euler_gaussian_bump.h"

 namespace PHiLiP {
 namespace Tests {

 template <int dim, int nstate>
 class EulerGaussianBumpEnthalpyCheck: public TestsBase
 {

 public:
    EulerGaussianBumpEnthalpyCheck(const Parameters::AllParameters *const parameters_input);
    double run_euler_gaussian_bump(const Parameters::AllParameters parameters_input) const;
    int run_test() const;
 };

 } // Tests namespace
 } // PHiLiP namespace

 #endif
