#ifndef __EULER_CYLINDER_ADJOINT_H__
#define __EULER_CYLINDER_ADJOINT_H__

#include <deal.II/grid/manifold_lib.h>

#include "dg/dg_base.h"
#include "parameters/all_parameters.h"
#include "physics/physics.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerCylinderAdjoint: public TestsBase
{
public:
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerCylinderAdjoint(const Parameters::AllParameters *const parameters_input);

    ~EulerCylinderAdjoint() {}; ///< Destructor.

    /// Grid convergence on Euler Gaussian Bump
    /** Will run the a grid convergence test for various p
     *  on multiple grids to determine the order of convergence.
     *
     *  Expecting the solution to converge at p+1. and output to converge at 2p+1.
     *  Note that the output solution currently convergens slightly suboptimally
     *  depending on the case (around 2p). The implementation of the boundary conditions
     *  play a large role on this adjoint consistency.
     *  
     *  Want to see entropy go to 0.
     */
    int run_test () const override;

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

