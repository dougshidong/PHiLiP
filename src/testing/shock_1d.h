#ifndef __BURGERS_STEADYSHOCK_H__
#define __BURGERS_STEADYSHOCK_H__

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "physics/physics.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class Shock1D: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    Shock1D () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    explicit Shock1D(const Parameters::AllParameters *const parameters_input);

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

    /// Initialize the solution with the exact solution
    /** This is not an issue since the discretized solution will not be the exact solution.
     *  Therefore, the residual does not start at 0.
     */
    void initialize_perturbed_solution(DGBase<dim,double> &dg, const Physics::PhysicsBase<dim,nstate,double> &physics) const;
    /// L2-Integral of the solution over the entire domain.
    /** Used to evaluate error of a functional.
     */
    double integrate_solution_over_domain(DGBase<dim,double> &dg) const;
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
