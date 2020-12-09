#ifndef __TARGET_VALUE_FUNCTIONAL_H__
#define __TARGET_VALUE_FUNCTIONAL_H__

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

namespace PHiLiP {

/// Target value functional
/**
  * Given a functional I and a target value T, simply computes (I - T)
  * 
  */
template <int dim, int nstate, typename real>
class TargetValueFunctional : public Functional<dim,nstate,real>
{
private:
    Functional<dim,nstate,real> &functional;

    real target_value;
    

public:
    /** Constructor.
     *  Since we don't have access to the Physics through DGBase, we recreate a Physics
     *  based on the parameter file of DGBase. However, this will not work if the
     *  physics have been overriden through DGWeak::set_physics() as seen in the
     *  diffusion_exact_adjoint test case.
     */
    Functional(
        Functional<dim,nstate,real> &_functional,
        const real _target_value)
        : Functional(_functional.dg, _functional.physics_fad_fad, functional.uses_solution_values, functional.uses_solution_gradient)
        , functional(_functional)
        , target_value(_target_value)
    { }

    /** Constructor.
     *  Uses provided physics instead of creating a new one base on DGBase */
    Functional(
        std::shared_ptr<PHiLiP::DGBase<dim,real>> _dg,
        Functional<dim,nstate,real> &functional,
        std::shared_ptr<PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>> >> _physics_fad_fad,
        const bool _uses_solution_values = true,
        const bool _uses_solution_gradient = true);
    std::shared_ptr<PHiLiP::DGBase<dim,real>> _dg,
    std::shared_ptr<PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>> >> _physics_fad_fad,
    const bool _uses_solution_values,
    const bool _uses_solution_gradient)
    : Functional(_dg, _uses_solution_values, _uses_solution_gradient)
{
    physics_fad_fad = _physics_fad_fad;
}

    /// Destructor.
    ~Functional(){}

public:
    virtual real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

}; // TargetFunctional class

} // PHiLiP namespace

#endif // __FUNCTIONAL_H__

