
#ifndef __GRID_REFINEMENT_UNIFORM_H__
#define __GRID_REFINEMENT_UNIFORM_H__

#include <deal.II/grid/tria.h>

#include "grid_refinement/grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_Uniform : public GridRefinementBase<dim,nstate,real,MeshType>
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

    // vtk output function
    std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() override;
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_UNIFORM_H__
