#ifndef __TARGET_FUNCTIONAL_H__
#define __TARGET_FUNCTIONAL_H__

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
#include "functional.h"

namespace PHiLiP {

/// TargetFunctional base class
/**
  * This base class is used to compute an inverse functional involving integration
  * over the discretized volume and boundary of the domain.
  * It differs from the Functional class by storing a target solution and having default
  * volume and surface integrals corresponding to the L2-norm of the solution difference.
  *
  * Often this is written in the form
  * 
  * \f[
  *      \mathcal{J}\left( \mathbf{u} \right)
  *      = \int_{\Omega} \left( p_{\Omega} \left( \mathbf{u} \right) - g_{\Omega} \left( \mathbf{u}_t \right) \right)^2 \mathrm{d} \Omega
  *      + \int_{\Gamma} \left( p_{\Gamma} \left( \mathbf{u} \right) - g_{\Gamma} \left( \mathbf{u}_t \right) \right)^2 \mathrm{d} \Gamma
  * \f]
  * 
  * where the cellwise or boundary edgewise functions 
  * \f$ g_{\Omega} \left( \mathbf{u} \right) \f$ and \f$ g_{\Gamma} \left( \mathbf{u} \right) \f$
  * are to be overridden in the derived class. Also computes the functional derivatives which 
  * are involved in the computation of the adjoint. If derivatives are needed, the Sacado
  * versions of these functions must also be defined.
  */
template <int dim, int nstate, typename real>
class TargetFunctional : public Functional<dim,nstate,real>
{
public:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Vector for storing the derivatives with respect to each solution DoF
    using Functional<dim,nstate,real>::dIdw;
    /// Vector for storing the derivatives with respect to each grid DoF
    using Functional<dim,nstate,real>::dIdX;
 /// Store the functional value from the last time evaluate_functional() was called.
    using Functional<dim,nstate,real>::current_functional_value;
    /// Sparse matrix for storing the functional partial second derivatives.
    using Functional<dim,nstate,real>::d2IdWdW;
    /// Sparse matrix for storing the functional partial second derivatives.
    using Functional<dim,nstate,real>::d2IdWdX;
    /// Sparse matrix for storing the functional partial second derivatives.
    using Functional<dim,nstate,real>::d2IdXdX;

protected:
    /// Smart pointer to DGBase
    using Functional<dim,nstate,real>::dg;
    /// Physics that should correspond to the one in DGBase
    using Functional<dim,nstate,real>::physics_fad_fad;

    using Functional<dim,nstate,real>::volume_update_flags; ///< Update flags needed at volume points.
    using Functional<dim,nstate,real>::face_update_flags; ///< Update flags needed at face points.
    using Functional<dim,nstate,real>::uses_solution_values; ///< Will evaluate solution values at quadrature points
    using Functional<dim,nstate,real>::uses_solution_gradient; ///< Will evaluate solution gradient at quadrature points

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real>::evaluate_volume_integrand;

    ///// Avoid warning that the function was hidden [-Woverloaded-virtual].
    ///** The compiler would otherwise hide Functional::evaluate_cell_boundary, which is fine for 
    // *  us, but is a typical bug that other people have. This 'using' imports the base class function
    // *  to our derived class even though we don't need it.
    // */
    //using Functional<dim,nstate,real>::evaluate_cell_boundary;

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_cell_functional, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real>::evaluate_volume_cell_functional;


    using Functional<dim,nstate,real>::evaluate_boundary_cell_functional;
    using Functional<dim,nstate,real>::evaluate_boundary_integrand;

protected:
 /// Solution used to evaluate target functional
    const dealii::LinearAlgebra::distributed::Vector<real> target_solution;

public:
    /// Constructor
    /** The target solution is initialized with the one currently within DGBase.
     *  Since we don't have access to the Physics through DGBase, we recreate a Physics
     *  based on the parameter file of DGBase. However, this will not work if the
     *  physics have been overriden through DGWeak::set_physics() as seen in the
     *  diffusion_exact_adjoint test case.
     */
    TargetFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);
    /// Constructor
    /** The target solution is provided instead of using the current solution in the DG object.
     *  Since we don't have access to the Physics through DGBase, we recreate a Physics
     *  based on the parameter file of DGBase. However, this will not work if the
     *  physics have been overriden through DGWeak::set_physics() as seen in the
     *  diffusion_exact_adjoint test case.
     */
    TargetFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);

    /// Constructor
    /** Uses provided physics instead of creating a new one base on DGBase.
      * The target solution is initialized with the one currently within DGBase.
      */
    TargetFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
        std::shared_ptr< Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> > _physics_fad_fad,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);

    /// Constructor
    /** Uses provided physics instead of creating a new one base on DGBase 
     *  The target solution is provided instead of using the current solution in the DG object.
  */
    TargetFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
  const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
        std::shared_ptr< Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> > _physics_fad_fad,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);

    /// Destructor
    ~TargetFunctional(){}

public:
    /// Evaluates the functional derivative with respect to the solution variable
    /** Loops over the discretized domain and determines the sensitivity of the functional value to each 
     *  solution node. Computed from
     * 
     *  \f[
     *      \left. \frac{\partial \mathcal{J}_h}{\partial \mathbf{u}} \right|_{\mathbf{u}_h}
     *      = \sum_{k=1}^{N_e} \left. \frac{\partial}{\partial \mathbf{u}} 
     *          \int_{\Omega_h^k} g_{\Omega} \left( \mathbf{u}_h^k \right) \mathrm{d} \Omega \right|_{\mathbf{u}_h}
     *      + \sum_{k=1}^{N_b} \left. \frac{\partial}{\partial \mathbf{u}} 
     *          \int_{\Gamma_h^k} g_{\Gamma} \left( \mathbf{u}_h^k \right) \mathrm{d} \Gamma \right|_{\mathbf{u}_h}
     *  \f]
     * 
     *  Calls the functions evaluate_volume_integrand() and evaluate_cell_boundary() to be overridden
     */
    virtual real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

    /** Finite difference evaluation of dIdW.
     */
    dealii::LinearAlgebra::distributed::Vector<real> evaluate_dIdw_finiteDifferences(
        DGBase<dim,real> &dg, 
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const double stepsize);

    /** Finite difference evaluation of dIdX.
     */
    dealii::LinearAlgebra::distributed::Vector<real> evaluate_dIdX_finiteDifferences(
        DGBase<dim,real> &dg, 
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const double stepsize);

private:
    /// Templated function to evaluate a cell's volume functional.
    template <typename real2>
    real2 evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real2> &physics,
        const std::vector< real2 > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const;

protected:
    /// Corresponding real function to evaluate a cell's volume functional.
    virtual real evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real> &physics,
        const std::vector< real > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const;
    /// Corresponding FadFadType function to evaluate a cell's volume functional.
    virtual Sacado::Fad::DFad<Sacado::Fad::DFad<real>> evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> &physics_fad_fad,
        const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const;

private:
    /// Templated function to evaluate a cell's face functional.
    template <typename real2>
    real2 evaluate_boundary_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int boundary_id,
        const std::vector< real2 > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim-1> &face_quadrature,
        const unsigned int face_number) const;
protected:
    /// Corresponding real function to evaluate a cell's face functional.
    virtual real evaluate_boundary_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real> &physics,
        const unsigned int boundary_id,
        const std::vector< real > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim-1> &face_quadrature,
        const unsigned int face_number) const;

    /// Corresponding FadFadType function to evaluate a cell's face functional.
    virtual Sacado::Fad::DFad<Sacado::Fad::DFad<real>> evaluate_boundary_cell_functional(
        const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> &physics_fad_fad,
        const unsigned int boundary_id,
        const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim-1> &face_quadrature,
        const unsigned int face_number) const;

    /// Virtual function for computation of cell volume functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const dealii::Point<dim,real> &/*phys_coord*/,
        const std::array<real,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_at_q*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*target_soln_grad_at_q*/) const
    {
        real l2error = 0;

        for (int istate=0; istate<nstate; ++istate) {
            l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
        }

        return l2error;
    }
    /// Virtual function for Sacado computation of cell volume functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual FadFadType evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &/*physics*/,
        const dealii::Point<dim,FadFadType> &/*phys_coord*/,
        const std::array<FadFadType,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &/*soln_grad_at_q*/,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &/*target_soln_grad_at_q*/) const
    {
        FadFadType l2error = 0;
        for (int istate=0; istate<nstate; ++istate) {
            l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
        }
        return l2error;
    }

    /// Virtual function for computation of cell face functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::Point<dim,real> &,//phys_coord,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &,//soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &)//target_soln_grad_at_q)
        const
    {
        real l2error = 0;
        for (int istate=0; istate<nstate; ++istate) {
            l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
        }
        return l2error;
    }
    /// Virtual function for Sacado computation of cell face functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual FadFadType evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::Point<dim,FadFadType> &,//phys_coord,
        const dealii::Tensor<1,dim,FadFadType> &/*normal*/,
        const std::array<FadFadType,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &,//soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &)//target_soln_grad_at_q)
        const
    {
        FadFadType l2error = 0;
        for (int istate=0; istate<nstate; ++istate) {
            l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
        }
        return l2error;
    }

    // /// Virtual function for computation of cell boundary functional term
    // /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    // virtual real evaluate_cell_boundary(
    //     const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
    //     const unsigned int /*boundary_id*/,
    //     const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
    //     std::vector<real> /*soln_coeff*/,
    //     std::vector<real> /*target_soln_coeff*/)
    // {
    //     return (real) 0.0;
    // }

    // /// Virtual function for Sacado computation of cell boundary functional term and derivatives
    // /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    // virtual FadFadType evaluate_cell_boundary(
    //     const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &/*physics*/,
    //     const unsigned int /*boundary_id*/,
    //     const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
    //     std::vector<FadFadType> /*soln_coeff*/,
    //     std::vector<real> /*target_soln_coeff*/)
    // {
    //     return (FadFadType) 0.0;
    // }

}; // TargetFunctional class
} // PHiLiP namespace

#endif // __TARGET_FUNCTIONAL_H__
