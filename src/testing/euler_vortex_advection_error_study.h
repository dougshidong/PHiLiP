#ifndef __EULER_VORTEX_ADVECTION_ERROR_STUDY_H__
#define __EULER_VORTEX_ADVECTION_ERROR_STUDY_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerVortexAdvectionErrorStudy: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    EulerVortexAdvectionErrorStudy () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerVortexAdvectionErrorStudy(const Parameters::AllParameters *const parameters_input,
                      const dealii::ParameterHandler &parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Returns either the order of convergence or enthalpy, depending on the test type.
    double run_euler_gaussian_bump () const;

    /// Grid convergence on Euler Gaussian Bump
    /** Will run the a grid convergence test for various p
     *  on multiple grids to determine the order of convergence.
     *
     *  Expecting the solution to converge at p+1. and output to converge at 2p+1.
     *  Note that the output solution currently convergens slightly suboptimally
     *  depending on the case (around 2p). The implementation of the boundary conditions
     *  play a large role on this adjoint consistency.
     *  
     *  Want to see entropy/enthalpy go to 0.
     */
    int run_test () const;

};

//   /// Manufactured grid convergence
//   /** Currently the main function as all my test cases simply
//    *  check for optimal convergence of the solution
//    */
//   template<int dim>
//   int manufactured_grid_convergence (Parameters::AllParameters &parameters);

} // Tests namespace
} // PHiLiP namespace
#endif

