
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
    using GridRefinementBase<dim,nstate,real,MeshType>::MAX_METHOD_VEC;
    void refine_grid()    override;
protected:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
    void output_results_vtk_method(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        std::array<dealii::Vector<real>,MAX_METHOD_VEC>   &dat_vec_vec) override;
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_UNIFORM_H__
