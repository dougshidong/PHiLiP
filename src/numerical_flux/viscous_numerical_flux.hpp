#ifndef __VISCOUS_NUMERICAL_FLUX__
#define __VISCOUS_NUMERICAL_FLUX__

#include <deal.II/base/tensor.h>
#include "physics/physics.h"
#include "dg/artificial_dissipation.h"

namespace PHiLiP {
namespace NumericalFlux {

/// Base class of numerical flux associated with dissipation
template<int dim, int nstate, typename real>
class NumericalFluxDissipative
{
public:
/// Constructor
NumericalFluxDissipative(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input, std::shared_ptr<ArtificialDissipationBase<dim, nstate>> artificial_dissipation_input)
: pde_physics(physics_input), artificial_dissip(artificial_dissipation_input)
{};

/// Solution flux at the interface.
virtual std::array<real, nstate> evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const = 0;

/// Auxiliary flux at the interface.
virtual std::array<real, nstate> evaluate_auxiliary_flux (
    const real artificial_diss_coeff_int,
    const real artificial_diss_coeff_ext,
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary = false) const = 0;

protected:
const std::shared_ptr < Physics::PhysicsBase<dim, nstate, real> > pde_physics; ///< Associated physics.
const std::shared_ptr < ArtificialDissipationBase<dim, nstate> > artificial_dissip;  ///< Link to artificial dissipation
};

/// Symmetric interior penalty method.
template<int dim, int nstate, typename real>
class SymmetricInternalPenalty: public NumericalFluxDissipative<dim, nstate, real>
{
using NumericalFluxDissipative<dim,nstate,real>::pde_physics;
using NumericalFluxDissipative<dim,nstate,real>::artificial_dissip;
public:
/// Constructor
SymmetricInternalPenalty(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input, std::shared_ptr < ArtificialDissipationBase<dim, nstate>> artificial_dissipation_input)
: NumericalFluxDissipative<dim,nstate,real>(physics_input,artificial_dissipation_input)
{};

/// Evaluate solution flux at the interface
/** \f[\hat{u} = {u_h} \f]
 */
std::array<real, nstate> evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const override;

/// Evaluate auxiliary flux at the interface
/** \f[ \hat{A} = {{ A \nabla u_h }} - \mu {{ A }} [[ u_h ]] \f]
 *  
 *  Note that \f$\mu\f$ must be chosen to have a stable scheme.
 *
 *
 */
std::array<real, nstate> evaluate_auxiliary_flux (
    const real artificial_diss_coeff_int,
    const real artificial_diss_coeff_ext,
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary = false) const override;
    
};

template<int dim, int nstate, typename real>
class BassiRebay2: public NumericalFluxDissipative<dim, nstate, real>
{
using NumericalFluxDissipative<dim,nstate,real>::pde_physics;
using NumericalFluxDissipative<dim,nstate,real>::artificial_dissip;
public:
/// Constructor
BassiRebay2(std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input, std::shared_ptr<ArtificialDissipationBase<dim, nstate>> artificial_dissipation_input)
: NumericalFluxDissipative<dim,nstate,real>(physics_input,artificial_dissipation_input)
{};

/// Evaluate solution flux at the interface
/** \f[\hat{u} = {u_h} \f]
 */
std::array<real, nstate> evaluate_solution_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const override;

/// Evaluate auxiliary flux at the interface
/** \f[ \hat{A} = {{ A \nabla u_h }} + \Chi {{ A r_e( [[u_h]]_2 ) }}
 *  
 *  Note that \f$ \Chi > N_{faces} \f$ to have a stable scheme.
 *
 *
 */
std::array<real, nstate> evaluate_auxiliary_flux (
    const real artificial_diss_coeff_int,
    const real artificial_diss_coeff_ext,
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_int,
    const std::array<dealii::Tensor<1,dim,real>, nstate> &soln_grad_ext,
    const dealii::Tensor<1,dim,real> &normal_int,
    const real &penalty,
    const bool on_boundary = false) const override;

};

} // NumericalFlux namespace
} // PHiLiP namespace

#endif
