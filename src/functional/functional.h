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

// functional class
template <int dim, int nstate, typename real>
class Functional 
{
public:
    // constructor
    Functional(){}
    // destructor
    ~Functional(){}

    real evaluate_function(
        DGBase<dim,real> &dg, 
        const Physics::PhysicsBase<dim,nstate,real> &physics);

    dealii::LinearAlgebra::distributed::Vector<real> evaluate_dIdw(
        DGBase<dim,real> &dg, 
        const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics);

    virtual real evaluate_cell_volume(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const dealii::FEValues<dim,dim> &/*fe_values_volume*/,
        std::vector<real> /*local_solution*/){return (real) 0.0;}
    
    virtual Sacado::Fad::DFad<real> evaluate_cell_volume(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &/*physics*/,
        const dealii::FEValues<dim,dim> &/*fe_values_volume*/,
        std::vector<Sacado::Fad::DFad<real>> /*local_solution*/){return (Sacado::Fad::DFad<real>) 0.0;}

    virtual real evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
        std::vector<real> /*local_solution*/){return (real) 0.0;}

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