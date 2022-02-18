#ifndef __BURGERS_REWIENSKI_ROM_H__
#define __BURGERS_REWIENSKI_ROM_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"
#include "functional/functional.h"

namespace PHiLiP {
namespace Tests {

/// Burgers Rewienski snapshot
template <int dim, int nstate>
class ReducedOrderPODAdaptation: public TestsBase
{
public:
    /// Constructor.
    ReducedOrderPODAdaptation(const Parameters::AllParameters *const parameters_input);

    /// Run test
    int run_test () const override;
};

///Functional to take the integral of the solution
template <int dim, int nstate, typename real>
class BurgersRewienskiFunctional : public Functional<dim, nstate, real>
{
public:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.
public:
    /// Constructor
    BurgersRewienskiFunctional(
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

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
