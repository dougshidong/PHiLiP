#include "empty_RRK_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType> 
EmptyRRKBase<dim,real, MeshType>::EmptyRRKBase(std::shared_ptr<RKTableauBase<dim,real,MeshType>> /*rk_tableau*/)
{}

template class EmptyRRKBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class EmptyRRKBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
    template class EmptyRRKBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
}
}
