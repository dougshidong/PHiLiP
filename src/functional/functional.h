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
#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class Functional 
{
private:
    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;
public:
    /// Constructor
    Functional(
        std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true);

    /// destructor
    ~Functional(){}

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
    
    using ADType = Sacado::Fad::DFad<real>;
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
        const Physics::PhysicsBase<dim,nstate,ADType> &physics,
        const bool compute_dIdW = false,
        const bool compute_dIdX = false);

    template <typename real2>
    real2 evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real2> &physics,
        const std::vector< real2 > &soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);
    real evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real> &physics,
        const std::vector< real > &soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);
    ADType evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,ADType> &physics,
        const std::vector< ADType > &soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< ADType > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature);

    /// Vector for storing the derivatives with respect to each solution DoF
    dealii::LinearAlgebra::distributed::Vector<real> dIdw;
    /// Vector for storing the derivatives with respect to each grid DoF
    dealii::LinearAlgebra::distributed::Vector<real> dIdX;

    dealii::LinearAlgebra::distributed::Vector<real> evaluate_dIdw_finiteDifferences(
        DGBase<dim,real,MeshType> &dg, 
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const double stepsize);

    dealii::LinearAlgebra::distributed::Vector<real> evaluate_dIdX_finiteDifferences(
        DGBase<dim,real,MeshType> &dg, 
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const double stepsize);
    
    // /// Virtual function for computation of cell volume functional term
    // /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    // virtual real evaluate_volume_integrand(
    //     const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
    //     const dealii::FEValues<dim,dim> &/*fe_values_volume*/,
    //     std::vector<real> /*local_solution*/){return (real) 0.0;}

    // 
    // /// Virtual function for Sacado computation of cell volume functional term and derivatives
    // /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    // virtual ADType evaluate_volume_integrand(
    //     const PHiLiP::Physics::PhysicsBase<dim,nstate,ADType> &/*physics*/,
    //     const dealii::FEValues<dim,dim> &/*fe_values_volume*/,
    //     std::vector<ADType> /*local_solution*/){return (ADType) 0.0;}

    /// Virtual function for computation of cell volume functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const dealii::Point<dim,real> &,//phys_coord,
        const std::array<real,nstate> &,//soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &)//soln_grad_at_q)
    {
        return (real) 0.0;//*phys_coord[0]+0.0*soln_at_q[0]; // Hopefully, multiplying by 0.0 will resize the return value by the correct number of derivatives.
    }
    /// Virtual function for Sacado computation of cell volume functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual ADType evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADType> &/*physics*/,
        const dealii::Point<dim,ADType> &,//phys_coord,
        const std::array<ADType,nstate> &,//soln_at_q,
        const std::array<dealii::Tensor<1,dim,ADType>,nstate> &)//soln_grad_at_q)
    {
        return (real) 0.0;//*phys_coord[0]+0.0*soln_at_q[0]; // Hopefully, multiplying by 0.0 will resize the return value by the correct number of derivatives.
    }

    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
        std::vector<real> /*local_solution*/){return (real) 0.0;}

    /// Virtual function for Sacado computation of cell boundary functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual ADType evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADType> &/*physics*/,
        const unsigned int /*boundary_id*/,
        const dealii::FEFaceValues<dim,dim> &/*fe_values_boundary*/,
        std::vector<ADType> /*local_solution*/){return (ADType) 0.0;}

protected:
    // Update flags needed at volume points.
    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;
    // Update flags needed at face points.
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors;

    const bool uses_solution_values; ///< Will evaluate solution values at quadrature points
    const bool uses_solution_gradient; ///< Will evaluate solution gradient at quadrature points

}; // Functional class

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class FunctionalNormLpVolume : public Functional<dim,nstate,real,MeshType>
{
    using ADtype = Sacado::Fad::DFad<real>;
public:
    FunctionalNormLpVolume(
        const double                               _normLp,
        std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
        const bool                                 _uses_solution_values   = true,
        const bool                                 _uses_solution_gradient = false);

    template <typename real2>
    real2 evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const dealii::Point<dim,real2> &                      phys_coord,
        const std::array<real2,nstate> &                      soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_grad_at_q);

    real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const dealii::Point<dim,real> &                      phys_coord,
        const std::array<real,nstate> &                      soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
    }

    ADtype evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADtype> &physics,
        const dealii::Point<dim,ADtype> &                      phys_coord,
        const std::array<ADtype,nstate> &                      soln_at_q,
        const std::array<dealii::Tensor<1,dim,ADtype>,nstate> &soln_grad_at_q) override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
    }

protected:
    const double normLp;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class FunctionalNormLpBoundary : public Functional<dim,nstate,real,MeshType>
{
    using ADtype = Sacado::Fad::DFad<real>;
public:
    FunctionalNormLpBoundary(
        const double                               _normLp,
        std::vector<unsigned int>                  _boundary_vector,
        const bool                                 _use_all_boundaries,
        std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
        const bool                                 _uses_solution_values   = true,
        const bool                                 _uses_solution_gradient = false);

    template <typename real2>
    real2 evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int                                    boundary_id,
        const dealii::FEFaceValues<dim,dim> &                 fe_values_boundary,
        std::vector<real2>                                    local_solution);
        
    real evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const unsigned int                                   boundary_id,
        const dealii::FEFaceValues<dim,dim> &                fe_values_boundary,
        std::vector<real>                                    local_solution) override
    {
        return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);
    }

    ADtype evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADtype> &physics,
        const unsigned int                                     boundary_id,
        const dealii::FEFaceValues<dim,dim> &                  fe_values_boundary,
        std::vector<ADtype>                                    local_solution) override
    {
        return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);
    }

protected:
    const double              normLp;
    std::vector<unsigned int> boundary_vector;
    const bool                use_all_boundaries;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class FunctionalWeightedIntegralVolume : public Functional<dim,nstate,real,MeshType>
{
    using ADtype = Sacado::Fad::DFad<real>;
public:
    FunctionalWeightedIntegralVolume(
        std::shared_ptr<ManufacturedSolutionFunction<dim,real>>   _weight_function_double,
        std::shared_ptr<ManufacturedSolutionFunction<dim,ADtype>> _weight_function_adtype,
        const bool                                                _use_weight_function_laplacian,
        std::shared_ptr<DGBase<dim,real,MeshType>>                _dg,
        const bool                                                _uses_solution_values   = true,
        const bool                                                _uses_solution_gradient = false);

    template <typename real2>
    real2 evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &  physics,
        const dealii::Point<dim,real2> &                        phys_coord,
        const std::array<real2,nstate> &                        soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &  soln_grad_at_q,
        std::shared_ptr<ManufacturedSolutionFunction<dim,real2>> weight_function);

    real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const dealii::Point<dim,real> &                      phys_coord,
        const std::array<real,nstate> &                      soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q, this->weight_function_double);
    }

    ADtype evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADtype> &physics,
        const dealii::Point<dim,ADtype> &                      phys_coord,
        const std::array<ADtype,nstate> &                      soln_at_q,
        const std::array<dealii::Tensor<1,dim,ADtype>,nstate> &soln_grad_at_q) override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q, this->weight_function_adtype);
    }

protected:
    std::shared_ptr<ManufacturedSolutionFunction<dim,real>>   weight_function_double;
    std::shared_ptr<ManufacturedSolutionFunction<dim,ADtype>> weight_function_adtype;
    const bool                                                use_weight_function_laplacian;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class FunctionalWeightedIntegralBoundary : public Functional<dim,nstate,real,MeshType>
{
    using ADtype = Sacado::Fad::DFad<real>;
public:
    FunctionalWeightedIntegralBoundary(
        std::shared_ptr<ManufacturedSolutionFunction<dim,real>>   _weight_function_double,
        std::shared_ptr<ManufacturedSolutionFunction<dim,ADtype>> _weight_function_adtype,
        const bool                                                _use_weight_function_laplacian,
        std::vector<unsigned int>                                 _boundary_vector,
        const bool                                                _use_all_boundaries,
        std::shared_ptr<DGBase<dim,real,MeshType>>                _dg,
        const bool                                                _uses_solution_values   = true,
        const bool                                                _uses_solution_gradient = false);

    template <typename real2>
    real2 evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &   physics,
        const unsigned int                                       boundary_id,
        const dealii::FEFaceValues<dim,dim> &                    fe_values_boundary,
        std::vector<real2>                                       local_solution,
        std::shared_ptr<ManufacturedSolutionFunction<dim,real2>> weight_function);
        
    real evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const unsigned int                                   boundary_id,
        const dealii::FEFaceValues<dim,dim> &                fe_values_boundary,
        std::vector<real>                                    local_solution) override
    {
        return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution, this->weight_function_double);
    }

    ADtype evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADtype> &physics,
        const unsigned int                                     boundary_id,
        const dealii::FEFaceValues<dim,dim> &                  fe_values_boundary,
        std::vector<ADtype>                                    local_solution) override
    {
        return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution, this->weight_function_adtype);
    }

protected:
    std::shared_ptr<ManufacturedSolutionFunction<dim,real>>   weight_function_double;
    std::shared_ptr<ManufacturedSolutionFunction<dim,ADtype>> weight_function_adtype;
    const bool                                                use_weight_function_laplacian;
    std::vector<unsigned int>                                 boundary_vector;
    const bool                                                use_all_boundaries;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class FunctionalErrorNormLpVolume : public Functional<dim,nstate,real,MeshType>
{
    using ADtype = Sacado::Fad::DFad<real>;
public:
    FunctionalErrorNormLpVolume(
        const double                               _normLp,
        std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
        const bool                                 _uses_solution_values   = true,
        const bool                                 _uses_solution_gradient = false);

    template <typename real2>
    real2 evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const dealii::Point<dim,real2> &                      phys_coord,
        const std::array<real2,nstate> &                      soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_grad_at_q);

    real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const dealii::Point<dim,real> &                      phys_coord,
        const std::array<real,nstate> &                      soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
    }

    ADtype evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADtype> &physics,
        const dealii::Point<dim,ADtype> &                      phys_coord,
        const std::array<ADtype,nstate> &                      soln_at_q,
        const std::array<dealii::Tensor<1,dim,ADtype>,nstate> &soln_grad_at_q) override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
    }

protected:
    const double normLp;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class FunctionalErrorNormLpBoundary : public Functional<dim,nstate,real,MeshType>
{
    using ADtype = Sacado::Fad::DFad<real>;
public:
    FunctionalErrorNormLpBoundary(
        const double                               _normLp,
        std::vector<unsigned int>                  _boundary_vector,
        const bool                                 _use_all_boundaries,
        std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
        const bool                                 _uses_solution_values   = true,
        const bool                                 _uses_solution_gradient = false);

    template <typename real2>
    real2 evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int                                    boundary_id,
        const dealii::FEFaceValues<dim,dim> &                 fe_values_boundary,
        std::vector<real2>                                    local_solution);
        
    real evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const unsigned int                                   boundary_id,
        const dealii::FEFaceValues<dim,dim> &                fe_values_boundary,
        std::vector<real>                                    local_solution) override
    {
        return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);
    }

    ADtype evaluate_cell_boundary(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,ADtype> &physics,
        const unsigned int                                     boundary_id,
        const dealii::FEFaceValues<dim,dim> &                  fe_values_boundary,
        std::vector<ADtype>                                    local_solution) override
    {
        return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);
    }

protected:
    const double              normLp;
    std::vector<unsigned int> boundary_vector;
    const bool                use_all_boundaries;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class FunctionalFactory
{
public:
    static std::shared_ptr< Functional<dim,nstate,real,MeshType> >
    create_Functional(
        PHiLiP::Parameters::AllParameters const *const       param,
        std::shared_ptr< PHiLiP::DGBase<dim,real,MeshType> > dg);

    static std::shared_ptr< Functional<dim,nstate,real,MeshType> >
    create_Functional(
        PHiLiP::Parameters::FunctionalParam                  param,
        std::shared_ptr< PHiLiP::DGBase<dim,real,MeshType> > dg);
};

} // PHiLiP namespace

#endif // __FUNCTIONAL_H__
