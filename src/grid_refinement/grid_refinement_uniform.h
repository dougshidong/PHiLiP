
#ifndef __GRID_REFINEMENT_UNIFORM_H__
#define __GRID_REFINEMENT_UNIFORM_H__

#include <deal.II/grid/tria.h>

#include "grid_refinement/grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

/// Uniform Grid Refinement Class
/** Simplest form of grid refinement, included primarily as a benchmark for other
  * strategies. This simple class always applies changes in a consistent way throughout 
  * the entire mesh, either by subdividing the mesh cells to form smaller cells ($h$-refinement)
  * incrementing the polynomial orders ($p$-refinement) or a combination of the two 
  * ($hp$-refinement).
  */ 
#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_Uniform : public GridRefinementBase<dim,nstate,real,MeshType>
{
public:
    using GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase;

    /// Perform call to the grid refinement object of choice
    /** This will automatically select the proper subclass, error indicator
      * and various refinement types based on the grid refinement parameters
      * passed at setup to the grid refinement factor class.
      * 
      * See subclass functions for details of refinement types.
      */
    void refine_grid() override;

protected:

    // specified refinement functions for different cases

    /// Uniform \f$h\f$ grid refinement
    /** Uniformly refines the mesh with \f$h\f$-refinement by subdividing all cells.
      */ 
    void refine_grid_h();
    
    /// Uniform \f$p\f$ grid refinement
    /** Increments the polynomial order of each cell by 1.
      */ 
    void refine_grid_p();

    /// Uniform \f$hp\f$ grid refinement
    /** Performs call to uniform mesh \f$h\f$-refinement by splitting all cells followed
      * by uniform \f$p\f$-refinement where the polynomial order of each cell is
      * incremented by 1.
      */ 
    void refine_grid_hp();

    /// Output refinement method dependent results
    /** No additional results are included for uniform grid refinements.
      */ 
    std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() override;
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_UNIFORM_H__
