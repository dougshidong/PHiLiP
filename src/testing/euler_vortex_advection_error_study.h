#ifndef __EULER_VORTEX_ADVECTION_ERROR_STUDY_H__
#define __EULER_VORTEX_ADVECTION_ERROR_STUDY_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nspecies, int nstate>
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

protected:
    //  Compute pressure from conservative solutions using different functions depending on physics model
    double compute_pressure ( const std::array<double,nstate> &conservative_soln ) const;

    //  Compute temperature from conservative solutions using different functions depending on physics model
    double compute_temperature ( const std::array<double,nstate> &conservative_soln ) const;

     //  Compute mass fractions of the #1 species from conservative solutions using different functions depending on physics model
    double compute_mass_fractions_1st ( const std::array<double,nstate> &conservative_soln ) const;  

    // 1/10 cycle exact solutioons
    double compute_exact_at_q ( const dealii::Point<dim,double> &point, const unsigned int istate ) const;

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Returns either the order of convergence or enthalpy, depending on the test type.
    double run_error_study () const;

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

