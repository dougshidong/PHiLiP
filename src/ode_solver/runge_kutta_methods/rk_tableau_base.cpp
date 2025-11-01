#include "rk_tableau_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, int nspecies, typename real, typename MeshType> 
RKTableauBase<dim,nspecies,real, MeshType> :: RKTableauBase (const int n_rk_stages_input, 
        const std::string rk_method_string_input)
    : n_rk_stages(n_rk_stages_input)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    , rk_method_string(rk_method_string_input)
{
}

template class RKTableauBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::Triangulation<PHILIP_DIM>>;
template class RKTableauBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class RKTableauBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
