#include "mesh_error_factory.h"

namespace PHiLiP {

template <int dim, int nspecies, int nstate, typename real, typename MeshType>
std::unique_ptr <MeshErrorEstimateBase <dim, nspecies, real, MeshType>> MeshErrorFactory<dim, nspecies, nstate, real, MeshType>::create_mesh_error(std::shared_ptr< DGBase<dim,nspecies,real,MeshType>> dg)
{
    if (!(dg->all_parameters->mesh_adaptation_param.use_goal_oriented_mesh_adaptation))
    {
        return std::make_unique<ResidualErrorEstimate<dim, nspecies, real, MeshType>>(dg);
    }

    // Recursive templating required because template parameters must be compile time constants
    // As a results, this recursive template initializes all possible dimensions with all possible nstate
    // without having 15 different if-else statements
    if(dim == dg->all_parameters->dimension)
    {
        // This template parameters dim and nstate match the runtime parameters
        // then create the selected dual-weighted residual type with template parameters dim and nstate
        // Otherwise, keep decreasing nstate and dim until it matches
        if(nstate == dg->all_parameters->nstate) 
        {
            return std::make_unique<DualWeightedResidualError<dim, nspecies, nstate , real, MeshType>>(dg);
        }
        else if constexpr (nstate > 1)
            return MeshErrorFactory<dim, nspecies, nstate-1, real, MeshType>::create_mesh_error(dg);
        else
            return nullptr;
    }
    else
    {
        std::cout<<"Cannot create MeshErrorEstimate. Invalid input"<<std::endl;
        return nullptr;
    }
}

template class MeshErrorFactory<PHILIP_DIM, PHILIP_SPECIES, 5, double,  dealii::Triangulation<PHILIP_DIM> >;
template class MeshErrorFactory<PHILIP_DIM, PHILIP_SPECIES, 5, double,  dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM!=1
template class MeshErrorFactory<PHILIP_DIM, PHILIP_SPECIES, 5, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif
} // namespace PHiLiP
