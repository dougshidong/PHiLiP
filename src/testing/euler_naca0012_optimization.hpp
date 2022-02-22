#ifndef __EULER_NACA_OPTIMIZATION_H__
#define __EULER_NACA_OPTIMIZATION_H__

#include <deal.II/grid/manifold_lib.h>

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerNACAOptimization: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    EulerNACAOptimization () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerNACAOptimization(const Parameters::AllParameters *const parameters_input);

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
    int run_test () const;

private:
    /// Actual test for which the number of design variables can be inputted.
    int optimize (const unsigned int nx_ffd, const unsigned int poly_degree) const;

};



} // Tests namespace
} // PHiLiP namespace
#endif


