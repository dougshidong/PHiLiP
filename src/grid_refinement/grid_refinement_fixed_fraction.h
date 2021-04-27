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

    // virtual refinement method from base class
    void refine_grid() override;

protected:

    // specified refinement functions for different cases
    void refine_grid_h();
    void refine_grid_p();
    void refine_grid_hp();   

    // performs flagging of the domain boundary (for testing)
    void refine_boundary_h();

    // additional functions for anisotropic h and 
    void smoothness_indicator();
    void anisotropic_h();
    void anisotropic_h_jump_based();
    void anisotropic_h_reconstruction_based();

    // calls error indicator computation function for error_indicator type
    void error_indicator();

    // error distribution for each indicator type
    void error_indicator_error();
    void error_indicator_hessian();
    void error_indicator_residual();
    void error_indicator_adjoint();

    // vtk output function
    std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() override;

protected:
    dealii::Vector<real> indicator;
    dealii::Vector<real> smoothness;
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_H__