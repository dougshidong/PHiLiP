#ifndef __DIFFUSION_EXACT_ADJOINT_H__
#define __DIFFUSION_EXACT_ADJOINT_H__

#include <deal.II/base/tensor.h>

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "physics/convection_diffusion.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {
#include "physics/convection_diffusion.h"
/* Test to compare adjoint discrete and continuous adjoints for diffusion in 1D
 *
 * Based on idea from:
 * "Adjoint Recovery of Superconvergent Functionals from PDE Approximations", Pierce and Giles 1998
 * 
 * As the diffusion operator is self adjoint, primal and dual can be be solved in same way then compared.
 * Lu=f, L^*v=g
 * J = <u, g>  = <u, L^*v>
 * J = <v, Lu> = <v, f>
 * 
 * L = d^2/dx^2 = L^* (directly from integration by parts)
 * 
 * To be consistent with the paper, f and g correspond to source/functional weight terms
 *  f(x) = x^3 (1-x)^3,  g(x) = sin(pi*x) and u(0)=u(1)=0 (Dirichlet BC)
 * 
 * Steps:
 *  1. Solve for u and v both directly for primal problems
 *  2. Perform functional evaluation from both cases (direct way) and check they are within tolerance
 *  3. Evaluate the discrete adjoint for both using PHiLiP::Adjoint class
 *  4. Compare (using L2 norm) with the primal solution of the opposing case
 */

// parent class to add the objective function directly to physics as a virtual class
template <int dim, int nstate, typename real>
class diffusion_objective : public Physics::ConvectionDiffusion <dim, nstate, real>
{
public:
    // defnined directly as part of the physics to make passing to the functional simpler
    real objetive_function(
        const dealii::Point<dim,double> &pos) const = 0;
};

template <int dim, int nstate, typename real>
class diffusion_u : public diffusion_objective <dim, nstate, real>
{
public:
    // source term = f
    std::array<real,nstate> source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const override;

    // objective function = g
    real objective_function(
        const dealii::Point<dim,double> &pos) const override;
};

template <int dim, int nstate, typename real>
class diffusion_v : public diffusion_objective <dim, nstate, real>
{
public:
    // source term = g
    std::array<real,nstate> source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const override;

    // objective function = f
    real objective_function(
        const dealii::Point<dim,double> &pos) const override;
};

template <int dim, int nstate>
class DiffusionExactAdjoint : public TestsBase
{
public: 
    // deleting the default constructor
    DiffusionExactAdjoint() = delete;

    // Constructor to call the TestsBase constructor to set parameters = parameters_input
    DiffusionExactAdjoint(const Parameters::AllParameters *const parameters_input);

    // destructor 
    ~DiffusionExactAdjoint(){};

    // perform test described above
    /* Ideally the results from both adjoints will converge to within a sufficient tolerance
     * to be compared over a series of meshes to see if its atleast improving
     */
    int run_test() const;
};

} // Tests namespace
} // PHiLiP namespace

#endif //__DIFFUSION_EXACT_ADJOINT_H__