#ifndef __PHILIP_TARGET_WALL_PRESSURE_H__
#define __PHILIP_TARGET_WALL_PRESSURE_H__

/* includes */
#include <vector>
#include <iostream>

#include <Sacado.hpp>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include "dg/dg.h"
#include "physics/physics.h"
#include "target_functional.h"

namespace PHiLiP {

/** Target boundary values.
 *  Simply zero out the default volume contribution.
 */
template <int dim, int nstate, typename real>
class TargetWallPressure : public TargetFunctional<dim, nstate, real>
{
private:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using TargetFunctional<dim,nstate,real>::evaluate_volume_integrand;
    using TargetFunctional<dim,nstate,real>::evaluate_boundary_integrand;
    using TargetFunctional<dim,nstate,real>::TargetFunctional;

public:

    real evaluate_functional( const bool compute_dIdW = false, const bool compute_dIdX = false, const bool compute_d2I = false) override
    {
        double value = TargetFunctional<dim,nstate,real>::evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I);

        this->pcout << "Target pressure error l2_norm: " << value << "\n";

        return value;
    }


    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    template<typename real2>
    real2 evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,real2> &/*phys_coord*/,
        const dealii::Tensor<1,dim,real2> &/*normal*/,
        const std::array<real2,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*target_soln_grad_at_q*/) const
    {
        if (boundary_id == 1001) {
            assert(soln_at_q.size() == dim+2);
            const Physics::Euler<dim,dim+2,real2> &euler = dynamic_cast< const Physics::Euler<dim,dim+2,real2> &> (physics);

            real2 pressure = euler.compute_pressure (soln_at_q);

            std::array<real2,nstate> target_soln_at_q_real2;
            for (int s=0; s<nstate; ++s) {
                target_soln_at_q_real2[s] = target_soln_at_q[s];
            }
            real2 target_pressure = euler.compute_pressure (target_soln_at_q_real2);
            real2 diff = pressure - target_pressure;
            real2 diff2 = diff*diff;

            return diff2;
        } 
        return (real2) 0.0;
    }

    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,real> &phys_coord,
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &target_soln_grad_at_q) const override
    {
        return evaluate_boundary_integrand<real>(
            physics,
            boundary_id,
            phys_coord,
            normal,
            target_soln_at_q,
            soln_at_q,
            soln_grad_at_q,
            target_soln_grad_at_q);
    }
    /// Virtual function for Sacado computation of cell boundary functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual FadFadType evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,FadFadType> &phys_coord,
        const dealii::Tensor<1,dim,FadFadType> &normal,
        const std::array<FadFadType,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &target_soln_grad_at_q) const override
    {
        return evaluate_boundary_integrand<FadFadType>(
            physics,
            boundary_id,
            phys_coord,
            normal,
            soln_at_q,
            target_soln_at_q,
            soln_grad_at_q,
            target_soln_grad_at_q);
    }

    /// Virtual function for computation of cell volume functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const dealii::Point<dim,real> &/*phys_coord*/,
        const std::array<real,nstate> &/*soln_at_q*/,
        const std::array<real,nstate> &/*target_soln_at_q*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_at_q*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*target_soln_grad_at_q*/) const override
    { return (real) 0.0; }
    /// Virtual function for Sacado computation of cell volume functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual FadFadType evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &/*physics*/,
        const dealii::Point<dim,FadFadType> &/*phys_coord*/,
        const std::array<FadFadType,nstate> &/*soln_at_q*/,
        const std::array<real,nstate> &/*target_soln_at_q*/,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &/*soln_grad_at_q*/,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &/*target_soln_grad_at_q*/) const override
    { return (FadFadType) 0.0; }


}; // TargetWallPressure class

} // PHiLiP namespace
#endif

