#ifndef __MESHADAPTATION_H__
#define __MESHADAPTATION_H__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "mesh_error_estimate.h"
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
    MeshAdaptation(double critical_res_input, int total_ref_cycle, double refine_frac, double coarsen_frac);

    /// Function to adapt the mesh based on input parameters.
    void adapt_mesh(std::shared_ptr< DGBase<dim, real, MeshType> > dg);

    /// Residual below which mesh adaptation begins.
    double critical_residual;

    /// Total/maximum refinement steps to be performed while solving a problem.
    int total_refinement_cycles;

    /// Stores the current refinement cycle.
    int current_refinement_cycle;

protected:
    
    /// Computes the vector containing errors in each cell.
    void compute_cellwise_errors(std::shared_ptr< DGBase<dim, real, MeshType> > dg);

    /// Computes maximum residual in each cell.
    void compute_max_cellwise_residuals(std::shared_ptr< DGBase<dim, real, MeshType> > dg);

    /// Performs fixed fraction refinement based on refinement and coarsening fractions.
    void fixed_fraction_isotropic_refinement_and_coarsening(std::shared_ptr< DGBase<dim, real, MeshType> > dg);
    
    /// Fraction of cells to be refined in fixed-fraction refinement
    double refinement_fraction;

    /// Fraction of cells to be coarsened in fixed-fraction refinement
    double coarsening_fraction;
    
    /// Stores errors in each cell
    dealii::Vector<real> cellwise_errors;

    /// Parallel std::cout
    dealii::ConditionalOStream pcout;

    /// Pointer to the error estimator class
    std::unique_ptr<MeshErrorEstimateBase<dim, real, MeshType>> mesh_error;

};

} // namespace PHiLiP

#endif
