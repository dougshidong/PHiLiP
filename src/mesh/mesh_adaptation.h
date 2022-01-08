#ifndef __MESHADAPTATION_H__
#define __MESHADAPTATION_H__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

namespace PHiLiP {


#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

/// Contains functions for mesh adaptation. Currently, it supports residual based mesh adaptation with fixed fraction coarsening and refinement.
class MeshAdaptation
{
public:

    /// Constructor to initialize the class with a pointer to DG.
    MeshAdaptation(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    /// Function to adapt the mesh based on input parameters.
    int adapt_mesh();

    /// Computes the vector containing errors in each cell.
    int compute_cellwise_errors();

    /// Computes maximum residual in each cell.
    int compute_max_cellwise_residuals();

    /// Performs fixed fraction refinement based on refinement and coarsening fractions.
    int fixed_fraction_isotropic_refinement_and_coarsening();

protected:
    
    /// Total/maximum refinement steps to be performed while solving a problem.
    int total_refinement_cycles;

    /// Stores the current refinement cycle.
    int current_refinement_cycle = 0;
    
    /// Shared pointer to DGBase.
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    /// Stores errors in each cell
    dealii::Vector<double> cellwise_errors;

};
} // namespace PHiLiP

#endif
