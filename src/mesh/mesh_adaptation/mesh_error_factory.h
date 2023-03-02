#ifndef __MESH_ERROR_FACTORY_H__
#define __MESH_ERROR_FACTORY_H__

#include "mesh_error_estimate.h"

namespace PHiLiP {

/// Returns pointer to appropriate mesh error class depending on the input parameters. 
#if PHILIP_DIM==1 
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class MeshErrorFactory
{
public:
    /// Returns pointer of the mesh error's abstract class.
    static std::unique_ptr<MeshErrorEstimateBase<dim, real, MeshType>> create_mesh_error(std::shared_ptr< DGBase<dim,real,MeshType> > dg);
};

} // namespace PHiLiP

#endif
