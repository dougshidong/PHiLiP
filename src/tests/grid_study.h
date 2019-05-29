#ifndef __GRID_STUDY_H__
#define __GRID_STUDY_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class GridStudy: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    GridStudy () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    GridStudy(const Parameters::AllParameters *const parameters_input);

    ~GridStudy() {}; ///< Destructor.

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

protected:
    /// Prints our mesh info and generates eps file if 2D grid.
    void print_mesh_info(const dealii::Triangulation<dim> &triangulation,
                         const std::string &filename) const;
    /// Warps mesh into sinusoidal.
    /** Useful to check non-cartesian linear meshes
     */
    static dealii::Point<dim> warp (const dealii::Point<dim> &p);
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
