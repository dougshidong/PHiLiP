#ifndef __EULER_ENTROPY_WAVES_H__
#define __EULER_ENTROPY_WAVES_H__

#include "tests.h"
#include "dg/dg.h"
#include "physics/euler.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Exact entropy waves solution.
/** Masatsuka2018 section 6.3 & section 7.13.3
 */
template <int dim, typename real>
class EulerEntropyWavesFunction : public dealii::Function<dim,real>
{
public:
    /// Constructor that initializes base_values, amplitudes, frequencies.
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     */
    EulerEntropyWavesFunction (const Physics::Euler<dim, dim+2, real> euler_physics, const real dimensional_density_inf);

    const Physics::Euler<dim, dim+2, real> euler_physics; ///< Euler physics.

    const real dimensional_density_inf; ///< Dimensional density at infinity.
    real Q_inf; ///< Velocity at infinity.

    /// Destructor
    ~EulerEntropyWavesFunction() {};
  
    /// Manufactured solution exact value
    /** Given A, density_inf, u_inf, v_inf, w_inf, and p_inf
     *  \code
     *  Q_inf = u_inf + v_inv + w_inf
     *  density = density_inf + A*sin( pi * (x+y+z-Q_inf*t) )
     *  u = u_inf
     *  v = u_inf
     *  w = u_inf
     *  p = p_inf
     *  \endcode
     *
     *  The non-dimensionalization of the problem with A = 1.0 gives
     *  \code
     *  Q_inf* = Q_inf/V_ref = (u_inf + v_inv + w_inf) / V_ref = (u*_inf + v*_inv + w*_inf) * V_ref / V_ref = (u*_inf + v*_inv + w*_inf)
     *  density*density_ref = density_inf + A*sin( pi * ((x*+y*+z*)*L - Q_inf * (t*)*L/V_ref) )
     *  density* = 1.0 + 1.0/density_inf * sin( pi*L * ((x*+y*+z*) - (Q_inf*) * (t*)) )
     *  u = u_inf*
     *  v = u_inf*
     *  w = u_inf*
     *  p = p_inf
     *  \endcode
     */
    real value (const dealii::Point<dim> &point, const unsigned int istate = 0) const;

private:
    /// Exact solution using the current \p time provided by the dealii::Function class
    dealii::Point<2> advected_location(const dealii::Point<2> old_location) const;

};

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerEntropyWaves: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    EulerEntropyWaves () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerEntropyWaves(const Parameters::AllParameters *const parameters_input);

    /// Manufactured grid convergence
    /** Will run the a grid convergence test for various p
     *  on multiple grids to determine the order of convergence.
     *
     *  Expecting the solution to converge at p+1. and output to converge at 2p+1.
     *  Note that the output solution currently convergens slightly suboptimally
     *  depending on the case (around 2p). The implementation of the boundary conditions
     *  play a large role on this adjoint consistency.
     *  
     *  The exact solution is given by the Physics module.
     */
    int run_test () const;

};


} // Tests namespace
} // PHiLiP namespace
#endif

