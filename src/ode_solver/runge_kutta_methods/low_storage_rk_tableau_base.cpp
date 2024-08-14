#include "low_storage_rk_tableau_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType> 
LowStorageRKTableauBase<dim,real, MeshType> :: LowStorageRKTableauBase (const int n_rk_stages, const int num_delta,
        const std::string rk_method_string_input)
    : rk_method_string(rk_method_string_input)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    this->butcher_tableau_gamma.reinit(n_rk_stages+1,3);
    this->butcher_tableau_beta.reinit(n_rk_stages+1);
    this->butcher_tableau_delta.reinit(num_delta);
    this->butcher_tableau_b_hat.reinit(n_rk_stages+1);
}

template <int dim, typename real, typename MeshType> 
void LowStorageRKTableauBase<dim,real, MeshType> :: set_tableau ()
{
    set_gamma();
    set_beta();
    set_delta();
    set_b_hat();
    pcout << "Assigned RK method: " << rk_method_string << std::endl;
}

template <int dim, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,real, MeshType> :: get_gamma (const int i, const int j) const
{
    return butcher_tableau_gamma[i][j];
}

template <int dim, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,real, MeshType> :: get_beta (const int i) const
{
    return butcher_tableau_beta[i];
}

template <int dim, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,real, MeshType> :: get_delta (const int i) const
{
    return butcher_tableau_delta[i];
}

template <int dim, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,real, MeshType> :: get_b_hat (const int i) const
{
    return butcher_tableau_b_hat[i];
}

template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
