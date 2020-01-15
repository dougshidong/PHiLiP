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
#include "dg/high_order_grid.h"
#include "physics/physics.h"

namespace PHiLiP {

/// TargetFunctional base class
/**
  * This base class is used to compute a functional of interest (for example, lift or drag) involving integration
  * over the discretized volume and boundary of the domain. Often this is written in the form
  * 
  * \f[
  *      \mathcal{J}\left( \mathbf{u} \right)
  *      = \int_{\Omega} g_{\Omega} \left( \mathbf{u} \right) \mathrm{d} \Omega
  *      + \int_{\Gamma} g_{\Gamma} \left( \mathbf{u} \right) \mathrm{d} \Gamma
  * \f]
  * 
  * where the cellwise or boundary edgewise functions 
  * \f$ g_{\Omega} \left( \mathbf{u} \right) \f$ and \f$ g_{\Gamma} \left( \mathbf{u} \right) \f$
  * are to be overridden in the derived class. Also computes the functional derivatives which 
  * are involved in the computation of the adjoint. If derivatives are needed, the Sacado
  * versions of these functions must also be defined.
  */
template <int dim, int nstate, typename real>
class TargetFunctional 
{
    using ADType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using ADADType = Sacado::Fad::DFad<ADType>; ///< Sacado AD type that allows 2nd derivatives.
private:
    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,real>> dg;

	/// Solution used to evaluate target functional
    const dealii::LinearAlgebra::distributed::Vector<real> target_solution;

    /// Physics that should correspond to the one in DGBase
    std::shared_ptr<Physics::PhysicsBase<dim,nstate,ADADType>> physics_fad_fad;
public:
    /// Constructor
    /** Since we don't have access to the Physics through DGBase, we recreate a Physics
     *  based on the parameter file of DGBase. However, this will not work if the
     *  physics have been overriden through DGWeak::set_physics() as seen in the
     *  diffusion_exact_adjoint test case.
     */
    TargetFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);

    /// Constructor
    /** Uses provided physics instead of creating a new one base on DGBase */
    TargetFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
        std::shared_ptr<Physics::PhysicsBase<dim,nstate,ADADType>> _physics_fad_fad,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);

    /// Constructor
    /** The target solution is provided instead of using the current solution in the DG object.
     */
    TargetFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
		const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);

    /// Constructor
    /** Uses provided physics instead of creating a new one base on DGBase 
     *  The target solution is provided instead of using the current solution in the DG object.
	 */
    TargetFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
		const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
        std::shared_ptr<Physics::PhysicsBase<dim,nstate,ADADType>> _physics_fad_fad,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);

    /// Destructor
    ~TargetFunctional(){}

    // /// Evaluates the functional of interest
    // /** Loops over the discretized domain and assembles contributions from
    //  *  
    //  * \f[
    //  *      \mathcal{J}_h \left( \mathbf{u} \right)
    //  *      = \sum_{k=1}^{N_e} \int_{\Omega_h^k} g_{\Omega} \left( \mathbf{u}_h^k \right) \mathrm{d} \Omega
    //  *      + \sum_{k=1}^{N_b} \int_{\Gamma_h^k} g_{\Gamma} \left( \mathbf{u}_h^k \right) \mathrm{d} \Gamma
    //  * \f]
    //  * 
    //  *  with terms defined from the functions evaluate_volume_integrand() and evaluate_cell_boundary() to be overridden.
    //  */
    // real evaluate_function(const Physics::PhysicsBase<dim,nstate,real> &physics);
    
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
    real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false);

    /// Templated function to evaluate a cell's volume functional.
    template <typename real2>
    real2 evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real2> &physics,
        const std::vector< real2 > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);
    /// Corresponding real function to evaluate a cell's volume functional.
    real evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real> &physics,
        const std::vector< real > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);
    /// Corresponding ADADType function to evaluate a cell's volume functional.
    ADADType evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,ADADType> &physics,
        const std::vector< ADADType > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< ADADType > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);

    /// Templated function to evaluate a cell's face functional.
    template <typename real2>
    real2 evaluate_face_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real2> &physics,
        const std::vector< real2 > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);
    /// Corresponding real function to evaluate a cell's face functional.
    real evaluate_face_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real> &physics,
        const std::vector< real > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);
    /// Corresponding ADADType function to evaluate a cell's face functional.
    ADADType evaluate_face_cell_functional(
        const Physics::PhysicsBase<dim,nstate,ADADType> &physics,
        const std::vector< ADADType > &soln_coeff,
        const std::vector< real > &target_soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< ADADType > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);

    /// Vector for storing the derivatives with respect to each solution DoF
    dealii::LinearAlgebra::distributed::Vector<real> dIdw;
    /// Vector for storing the derivatives with respect to each grid DoF
    dealii::LinearAlgebra::distributed::Vector<real> dIdX;
	// Store the functional value from the last time evaluate_functional() was called.
	real current_functional_value;

    /// Sparse matrix for storing the functional partial second derivatives.
    dealii::TrilinosWrappers::SparseMatrix d2IdWdW;
    /// Sparse matrix for storing the functional partial second derivatives.
    dealii::TrilinosWrappers::SparseMatrix d2IdWdX;
    /// Sparse matrix for storing the functional partial second derivatives.
    dealii::TrilinosWrappers::SparseMatrix d2IdXdX;

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
    
    /// Virtual function for computation of cell volume functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const dealii::Point<dim,real> &,//phys_coord,
        const std::array<real,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &,//soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &)//target_soln_grad_at_q)
    {
		real l2error = 0;
		
		for (int istate=0; istate<nstate; ++istate) {
			l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
		}

		return l2error;
    }
    /// Virtual function for Sacado computation of cell volume functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual ADADType evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADADType> &/*physics*/,
        const dealii::Point<dim,ADADType> &,//phys_coord,
        const std::array<ADADType,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,ADADType>,nstate> &,//soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,ADADType>,nstate> &)//target_soln_grad_at_q)
    {
		ADADType l2error = 0;
		
		for (int istate=0; istate<nstate; ++istate) {
			l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
		}

		return l2error;
    }

    /// Virtual function for computation of cell face functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_face_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const dealii::Point<dim,real> &,//phys_coord,
        const std::array<real,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &,//soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &)//target_soln_grad_at_q)
    {
		real l2error = 0;
		
		for (int istate=0; istate<nstate; ++istate) {
			l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
		}

		return l2error;
    }
    /// Virtual function for Sacado computation of cell face functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual ADADType evaluate_face_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADADType> &/*physics*/,
        const dealii::Point<dim,ADADType> &,//phys_coord,
        const std::array<ADADType,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
        const std::array<dealii::Tensor<1,dim,ADADType>,nstate> &,//soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,ADADType>,nstate> &)//target_soln_grad_at_q)
    {
		ADADType l2error = 0;
		
		for (int istate=0; istate<nstate; ++istate) {
			l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
		}

		return l2error;
    }

    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
        std::vector<real> /*soln_coeff*/,
        std::vector<real> /*target_soln_coeff*/)
	{return (real) 0.0;}

    /// Virtual function for Sacado computation of cell boundary functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual ADADType evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADADType> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
        std::vector<ADADType> /*soln_coeff*/,
        std::vector<real> /*target_soln_coeff*/)
	{return (ADADType) 0.0;}

protected:
    /// Update flags needed at volume points.
    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;
    /// Update flags needed at face points.
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors;

    const bool uses_solution_values; ///< Will evaluate solution values at quadrature points
    const bool uses_solution_gradient; ///< Will evaluate solution gradient at quadrature points

}; // TargetFunctional class
} // PHiLiP namespace

#endif // __TARGET_FUNCTIONAL_H__
