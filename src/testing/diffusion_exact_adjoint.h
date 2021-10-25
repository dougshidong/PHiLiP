#ifndef __DIFFUSION_EXACT_ADJOINT_H__
#define __DIFFUSION_EXACT_ADJOINT_H__

#include <memory>

#include <deal.II/base/tensor.h>

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "physics/convection_diffusion.h"
#include "physics/manufactured_solution.h"
#include "parameters/all_parameters.h"
#include "functional/functional.h"

namespace PHiLiP {
namespace Tests {

/**Test to compare adjoint discrete and continuous adjoints for diffusion in 1D
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
 * Similar terms chosen to in paper, except using them for the manufactured solution instead of the source
 *  u(x) = x^3 (1-x)^3,  v(x) = sin(pi*x) and hence u(0)=u(1)=0 (Dirichlet BC)
 * 
 * Source term in 1D:
 *  f(x) = -30x^3+60x63-36x^2+6x
 *  g(x) = -pi^2 * sin(pi*x)
 * 
 * In higher dimensions, obtained by taking \f$ \nabla u(x)*u(y)*u(z) \f$
 * 
 * Steps:
 *  1. Solve for u and v both directly for primal problems
 *  2. Perform functional evaluation from both cases (direct way) and check they are within tolerance
 *  3. Evaluate the discrete adjoint for both using PHiLiP::Adjoint class
 *  4. Compare (using L2 norm) with the primal solution of the opposing case
 *
 * Analytic value of the functional was found to be
 *  1D: J =                             [144*(10 - pi^2)/pi^5]
 *  2D: J = 2*[-144*(10 - pi^2)/pi^7]  *[144*(10 - pi^2)/pi^5]
 *  3D: J = 3*[-144*(10 - pi^2)/pi^7]^2*[144*(10 - pi^2)/pi^5]
 */

/// manufactured solution for u
template <int dim, typename real>
class ManufacturedSolutionU : public ManufacturedSolutionFunction <dim, real>
{
protected:
    using dealii::Function<dim,real>::value;
    using dealii::Function<dim,real>::gradient;
    using dealii::Function<dim,real>::hessian;
public:
    /// constructor
    ManufacturedSolutionU(){}

    /// overriding the function for the value and gradient
    real value (const dealii::Point<dim,real> &pos, const unsigned int istate = 0) const override;

    /// Gradient of the manufactured solution
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &pos, const unsigned int istate = 0) const override;

    /// Hessian of manufactured solution is unused but needed to make class concrete
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &/* point */, const unsigned int /* istate = 0 */) const 
    {
        return dealii::SymmetricTensor<2,dim,real>();
    }
};

/// manufactured solution for v
template <int dim, typename real>
class ManufacturedSolutionV : public ManufacturedSolutionFunction <dim, real>
{
protected:
    using dealii::Function<dim,real>::value;
    using dealii::Function<dim,real>::gradient;
    using dealii::Function<dim,real>::hessian;
public:
    /// constructor
    ManufacturedSolutionV(){}

    /// overriding the function for the value and gradient
    real value (const dealii::Point<dim,real> &pos, const unsigned int istate = 0) const override;

    /// Gradient of the manufactured solution
    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim,real> &pos, const unsigned int istate = 0) const override;

    /// Hessian of manufactured solution is unused but needed to make class concrete
    dealii::SymmetricTensor<2,dim,real> hessian (const dealii::Point<dim,real> &/* point */, const unsigned int /* istate = 0 */) const override
    {
        return dealii::SymmetricTensor<2,dim,real>();
    }
};

/// parent class to add the objective function directly to physics as a virtual class
template <int dim, int nstate, typename real>
class diffusion_objective : public Physics::ConvectionDiffusion <dim, nstate, real>
{
public:
    /// constructor
    diffusion_objective(
        const bool                                              convection, 
        const bool                                              diffusion,
        std::shared_ptr<ManufacturedSolutionFunction<dim,real>> manufactured_solution_function): 
            Physics::ConvectionDiffusion<dim,nstate,real>::ConvectionDiffusion(
                convection, 
                diffusion,
                default_diffusion_tensor(),
                Parameters::ManufacturedSolutionParam::get_default_advection_vector(),
                default_diffusion_coefficient(),
                manufactured_solution_function)
    {}

    /// negative one is used for the diffusion coefficient so that the problem becomes \f$\nabla u(x) = f(x)\f$
    static double default_diffusion_coefficient()
    {
        return -1.0;
    }

    /// Default diffusion tensor is set for isotropic diffusion, \f$D=I\f$
    static dealii::Tensor<2,3,double> default_diffusion_tensor()
    {
        dealii::Tensor<2,3,double> diffusion_tensor;
        diffusion_tensor[0][0] = 1;
        if constexpr(dim>=2) {
            diffusion_tensor[0][1] = 0.0;
            diffusion_tensor[1][0] = 0.0;
            diffusion_tensor[1][1] = 1.0;
        }
        if constexpr(dim>=3) {
            diffusion_tensor[0][2] = 0.0;
            diffusion_tensor[1][2] = 0.0;
            diffusion_tensor[2][0] = 0.0;
            diffusion_tensor[2][1] = 0.0;
            diffusion_tensor[2][2] = 1.0;
        }
        return diffusion_tensor;
    }

    /// defnined directly as part of the physics to make passing to the functional simpler
    virtual real objective_function(
        const dealii::Point<dim,real> &pos) const = 0;
};

///physics for the u variable
template <int dim, int nstate, typename real>
class diffusion_u : public diffusion_objective <dim, nstate, real>
{
public:
    /// constructor
    diffusion_u(
        const bool convection, 
        const bool diffusion): 
            diffusion_objective<dim,nstate,real>::diffusion_objective(
                convection, 
                diffusion,
                std::make_shared<ManufacturedSolutionU<dim,real>>())
    {}

    /// source term = f
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &/*solution*/,
        const real /*current_time*/) const override;

    /// objective function = g
    real objective_function(
        const dealii::Point<dim,real> &pos) const override;
};

/// physics for the v variable
template <int dim, int nstate, typename real>
class diffusion_v : public diffusion_objective <dim, nstate, real>
{
public:
    /// constructor
    diffusion_v(
        const bool convection, 
        const bool diffusion): 
            diffusion_objective<dim,nstate,real>::diffusion_objective(
                convection, 
                diffusion,
                std::make_shared<ManufacturedSolutionV<dim,real>>())
    {}

    /// source term = g
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &/*solution*/,
        const real current_time) const override;

    /// objective function = f
    real objective_function(
        const dealii::Point<dim,real> &pos) const override;
};

/// Functional that performs the inner product over the entire domain 
template <int dim, int nstate, typename real>
class DiffusionFunctional : public Functional<dim, nstate, real>
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.
    public:
        /// Constructor
        DiffusionFunctional(
            std::shared_ptr<PHiLiP::DGBase<dim,real>> dg_input,
            std::shared_ptr<PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType>> _physics_fad_fad,
            const bool uses_solution_values = true,
            const bool uses_solution_gradient = false)
        : PHiLiP::Functional<dim,nstate,real>(dg_input,_physics_fad_fad,uses_solution_values,uses_solution_gradient)
        {}
        template <typename real2>
        /// Templated volume integrand
        real2 evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
            const dealii::Point<dim,real2> &phys_coord,
            const std::array<real2,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_grad_at_q) const;

     /// Non-template functions to override the template classes
  real evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
            const dealii::Point<dim,real> &phys_coord,
            const std::array<real,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const override
  {
   return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
  }
     /// Non-template functions to override the template classes
  FadFadType evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &physics,
            const dealii::Point<dim,FadFadType> &phys_coord,
            const std::array<FadFadType,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &soln_grad_at_q) const override
  {
   return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
  }
};

/// test case
template <int dim, int nstate>
class DiffusionExactAdjoint : public TestsBase
{
public: 
    /// deleting the default constructor
    DiffusionExactAdjoint() = delete;

    /// Constructor to call the TestsBase constructor to set parameters = parameters_input
    DiffusionExactAdjoint(const Parameters::AllParameters *const parameters_input);

    /// destructor 
    ~DiffusionExactAdjoint(){};

    /** perform test described above
     *  Ideally the results from both adjoints will converge to within a sufficient tolerance
     *  to be compared over a series of meshes to see if its atleast improving
     */
    int run_test() const;
};

// for evaluating error slopes
double eval_avg_slope(std::vector<double> error, std::vector<double> grid_size, unsigned int n_grids);

} // Tests namespace
} // PHiLiP namespace

#endif //__DIFFUSION_EXACT_ADJOINT_H__
