#ifndef __GRID_REFINEMENT_FIXED_FRACTION_H__
#define __GRID_REFINEMENT_FIXED_FRACTION_H__

#include "grid_refinement/grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_FixedFraction : public GridRefinementBase<dim,nstate,real,MeshType>
{
public:
    using GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase;
    void refine_grid()    override;
protected:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;   
    std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() override;

    // performs flagging of the domain boundary (for testing)
    void refine_boundary_h();

    virtual void error_indicator() = 0;
    void smoothness_indicator();
    void anisotropic_h();
    void anisotropic_h_jump_based();
    void anisotropic_h_reconstruction_based();
protected:
    dealii::Vector<real> indicator;
    dealii::Vector<real> smoothness;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_FixedFraction_Error : public GridRefinement_FixedFraction<dim,nstate,real,MeshType>
{
public:
    using GridRefinement_FixedFraction<dim,nstate,real,MeshType>::GridRefinement_FixedFraction;
    void error_indicator() override;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_FixedFraction_Hessian : public GridRefinement_FixedFraction<dim,nstate,real,MeshType>
{
public:
    using GridRefinement_FixedFraction<dim,nstate,real,MeshType>::GridRefinement_FixedFraction;
    void error_indicator() override;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_FixedFraction_Residual : public GridRefinement_FixedFraction<dim,nstate,real,MeshType>
{
public:
    using GridRefinement_FixedFraction<dim,nstate,real,MeshType>::GridRefinement_FixedFraction;
    void error_indicator() override;
};

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_FixedFraction_Adjoint : public GridRefinement_FixedFraction<dim,nstate,real,MeshType>
{
public:
    using GridRefinement_FixedFraction<dim,nstate,real,MeshType>::GridRefinement_FixedFraction;
    void error_indicator() override;
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_H__