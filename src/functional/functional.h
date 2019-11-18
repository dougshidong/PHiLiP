#ifndef __FUNCTIONAL_H__
#define __FUNCTIONAL_H__

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

/// Functional base class
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
class Functional 
{
public:
    /// Constructor
    /** Unused in the base class as the functional has no internal memory.
     *  Can be used as a means of specifying additional parameters 
     */
    Functional(){}

    /// destructor
    ~Functional(){}

    /// Evaluates the functional of interest
    /** Loops over the discretized domain and assembles contributions from
     *  
     * \f[
     *      \mathcal{J}_h \left( \mathbf{u} \right)
     *      = \sum_{k=1}^{N_e} \int_{\Omega_h^k} g_{\Omega} \left( \mathbf{u}_h^k \right) \mathrm{d} \Omega
     *      + \sum_{k=1}^{N_b} \int_{\Gamma_h^k} g_{\Gamma} \left( \mathbf{u}_h^k \right) \mathrm{d} \Gamma
     * \f]
     * 
     *  with terms defined from the functions evaluate_cell_volume() and evaluate_cell_boundary() to be overridden.
     */
    real evaluate_function(
        DGBase<dim,real> &dg, 
        const Physics::PhysicsBase<dim,nstate,real> &physics);
    
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
     *  Calls the functions evaluate_cell_volume() and evaluate_cell_boundary() to be overridden
     */
    dealii::LinearAlgebra::distributed::Vector<real> evaluate_dIdw(
        DGBase<dim,real> &dg, 
        const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics);
    
    /// Virtual function for computation of cell volume functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_cell_volume(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const dealii::FEValues<dim,dim> &/*fe_values_volume*/,
        std::vector<real> /*local_solution*/){return (real) 0.0;}
    
    /// Virtual function for Sacado computation of cell volume functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual Sacado::Fad::DFad<real> evaluate_cell_volume(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &/*physics*/,
        const dealii::FEValues<dim,dim> &/*fe_values_volume*/,
        std::vector<Sacado::Fad::DFad<real>> /*local_solution*/){return (Sacado::Fad::DFad<real>) 0.0;}

    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
        std::vector<real> /*local_solution*/){return (real) 0.0;}

    /// Virtual function for Sacado computation of cell boundary functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual Sacado::Fad::DFad<real> evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
        std::vector<Sacado::Fad::DFad<real>> /*local_solution*/){return (Sacado::Fad::DFad<real>) 0.0;}

protected:
    // Update flags needed at volume points.
    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;
    // Update flags needed at face points.
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors;

}; // Functional class

} // PHiLiP namespace

#endif // __FUNCTIONAL_H__