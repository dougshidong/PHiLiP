#ifndef __EULER_VORTEX_H__
#define __EULER_VORTEX_H__

#include "tests.h"
#include "dg/dg.h"
#include "physics/euler.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

// Masatsuka2018 section 6.3 & section 7.13.3

/// Spiegel, S. C., Huynh, H. T., and DeBonis, J. R.,
/// “A Survey of the Isentropic Euler Vortex Problem using High-Order Methods,”
/// 22nd AIAA Computational Fluid Dynamics Conference, American Institute of Aeronautics and Astronautics, 2015.
template <int dim, typename real>
class EulerVortexFunction : public dealii::Function<dim,real>
{
public:
    /// Constructor that initializes base_values, amplitudes, frequencies.
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     */
    EulerVortexFunction (
        const Physics::Euler<dim, dim+2, real> euler_physics,
        const dealii::Point<dim> initial_vortex_center,
        const real vortex_strength,
        const real vortex_stddev_decay);

    const Physics::Euler<dim, dim+2, real> euler_physics; ///< Euler physics.
    const real vortex_characteristic_length; ///< R
    const dealii::Point<dim> initial_vortex_center; ///< x_c, y_c
    const real vortex_strength; ///< beta
    const real vortex_stddev_decay; ///< sigma

    /// Manufactured solution exact value
    /** \code
     *  u[s] = A[s]*sin(freq[s][0]*x)*sin(freq[s][1]*y)*sin(freq[s][2]*z);
     *  \endcode
     */
    real value (const dealii::Point<dim> &point, const unsigned int istate = 0) const;

private:
    /// Exact solution using the current \p time provided by the dealii::Function class
    dealii::Point<dim> advected_location(const dealii::Point<dim> old_location) const;

};

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerVortex: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    EulerVortex () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    EulerVortex(const Parameters::AllParameters *const parameters_input);

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
