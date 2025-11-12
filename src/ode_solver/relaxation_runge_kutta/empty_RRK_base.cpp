#include "empty_RRK_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, int nspecies, typename real, typename MeshType> 
EmptyRRKBase<dim,nspecies,real, MeshType>::EmptyRRKBase(std::shared_ptr<RKTableauButcherBase<dim,real,MeshType>> /*rk_tableau*/)
{}

#if PHILIP_SPECIES==1
    template class EmptyRRKBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::Triangulation<PHILIP_DIM>>;
    template class EmptyRRKBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
    #if PHILIP_DIM != 1
        template class EmptyRRKBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
    #endif
#endif
}
}
