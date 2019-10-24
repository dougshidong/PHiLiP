#ifndef __VISCOUS_NUMERICAL_FLUX__
#define __VISCOUS_NUMERICAL_FLUX__

#include <deal.II/base/tensor.h>
#include "physics/physics.h"

namespace PHiLiP {
namespace NumericalFlux {

/// Base class of numerical flux associated with dissipation
template<int dim, int nstate, typename real>
class NumericalFluxDissipative
{
public:
virtual ~NumericalFluxDissipative() = 0; ///< Base class destructor required for abstract classes.

/// Solution flux at the interface.
virtual std::array<real, nstate> evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const = 0;

/// Auxiliary flux at the interface.
virtual std::array<real, nstate> evaluate_auxiliary_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary = false) const = 0;

// dealii::Tensor<1,dim, dealii::Tensor<1,nstate,real>> diffusion_matrix_int;
// dealii::Tensor<1,dim, dealii::Tensor<1,nstate,real>> diffusion_matrix_int_transpose;
// dealii::Tensor<1,dim, dealii::Tensor<1,nstate,real>> diffusion_matrix_ext;
// dealii::Tensor<1,dim, dealii::Tensor<1,nstate,real>> diffusion_matrix_ext_transpose;

};

/// Symmetric interior penalty method.
template<int dim, int nstate, typename real>
class SymmetricInternalPenalty: public NumericalFluxDissipative<dim, nstate, real>
{
public:
/// Constructor
SymmetricInternalPenalty(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
:
pde_physics(physics_input)
{};
~SymmetricInternalPenalty() {}; ///< Destructor

/// Evaluate solution flux at the interface
/** \f[\hat{u} = {u_h} \f]
 */
std::array<real, nstate> evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const;

/// Evaluate auxiliary flux at the interface
/** \f[ \hat{A} = {{ A \nabla u_h }} - \mu {{ A }} [[ u_h ]] \f]
 *  
 *  Note that \f$\mu\f$ must be chosen to have a stable scheme.
 *
 *
 */
std::array<real, nstate> evaluate_auxiliary_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary = false) const;
    
protected:
const std::shared_ptr < Physics::PhysicsBase<dim, nstate, real> > pde_physics; ///< Associated physics.

};

//template<int dim, int nstate, typename real>
//class BassiRebay2: public NumericalFluxDissipative<dim, nstate, real>
//{
//public:
///// Constructor
//SymmetricInternalPenalty(Physics::PhysicsBase<dim, nstate, real> *physics_input)
//:
//pde_physics(physics_input)
//{};
///// Destructor
//~SymmetricInternalPenalty() {};

///// Evaluate solution and gradient flux
///*  $\hat{u} = {u_h}$, 
// *  $ \hat{A} = {{ A \nabla u_h }} - \mu {{ A }} [[ u_h ]] $
// */
//std::array<real, nstate> evaluate_solution_flux (
//    const std::array<real, nstate> &soln_int,
//    const std::array<real, nstate> &soln_ext,
//    const dealii::Tensor<1,dim,real> &normal_int) const;

//std::array<real, nstate> evaluate_auxiliary_flux (
//    const std::array<real, nstate> &soln_int,
//    const std::array<real, nstate> &soln_ext,
//    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
//    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
//    const dealii::Tensor<1,dim,real> &normal_int,
//    const real &penalty) const;
//    
//protected:
//const Physics::PhysicsBase<dim, nstate, real> *pde_physics;

//};

} // NumericalFlux namespace
} // PHiLiP namespace

#endif
