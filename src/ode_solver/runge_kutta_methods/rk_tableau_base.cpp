#include "rk_tableau_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType> 
RKTableauBase<dim,real, MeshType> :: RKTableauBase ()
    : pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{}

template <int dim, typename real, typename MeshType> 
double RKTableauBase<dim,real, MeshType> :: a (int i, int j)
{
    return butcher_tableau_a[i][j];
}

template <int dim, typename real, typename MeshType> 
double RKTableauBase<dim,real, MeshType> :: b (int i)
{
    return butcher_tableau_b[i];
}

template class RKTableauBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class RKTableauBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class RKTableauBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
