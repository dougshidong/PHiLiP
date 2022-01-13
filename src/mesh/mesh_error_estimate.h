#ifndef __MESHERRORESTIMATE_H__
#define __MESHERRORESTIMATE_H__

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

/// Abstract class to estimate error for mesh adaptation
class MeshErrorEstimateBase
{

public:

    /// Computes the vector containing errors in each cell.
    virtual dealii::Vector<real> compute_cellwise_errors (std::shared_ptr< DGBase<dim, real, MeshType> > dg) = 0;

};


#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
/// Class to compute residual based error
class ResidualErrorEstimate : public MeshErrorEstimateBase <dim, real, MeshType>
{

public:
    /// Computes maximum residual error in each cell.
    dealii::Vector<real> compute_cellwise_errors (std::shared_ptr< DGBase<dim, real, MeshType> > dg);

};

} // namespace PHiLiP

#endif

