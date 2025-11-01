#include "low_storage_rk_tableau_base.h"


namespace PHiLiP {
namespace ODE {
    
template <int dim, int nspecies, typename real, typename MeshType> 
LowStorageRKTableauBase<dim,nspecies,real, MeshType> :: LowStorageRKTableauBase (const int n_rk_stages_input, const int num_delta_input,
        const std::string rk_method_string_input)
    : RKTableauBase<dim,nspecies,real,MeshType>(n_rk_stages_input,rk_method_string_input)
      , num_delta(num_delta_input)
{
    this->butcher_tableau_gamma.reinit(this->n_rk_stages+1,3);
    this->butcher_tableau_beta.reinit(this->n_rk_stages+1);
    this->butcher_tableau_delta.reinit(this->num_delta);
    this->butcher_tableau_b_hat.reinit(this->n_rk_stages+1);
}

template <int dim, int nspecies, typename real, typename MeshType> 
void LowStorageRKTableauBase<dim,nspecies,real, MeshType> :: set_tableau ()
{
    set_gamma();
    set_beta();
    set_delta();
    set_b_hat();
    this->pcout << "Assigned RK method: " << this->rk_method_string << std::endl;
}

template <int dim, int nspecies, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,nspecies,real, MeshType> :: get_gamma (const int i, const int j) const
{
    return butcher_tableau_gamma[i][j];
}

template <int dim, int nspecies, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,nspecies,real, MeshType> :: get_beta (const int i) const
{
    return butcher_tableau_beta[i];
}

template <int dim, int nspecies, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,nspecies,real, MeshType> :: get_delta (const int i) const
{
    return butcher_tableau_delta[i];
}

template <int dim, int nspecies, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,nspecies,real, MeshType> :: get_b_hat (const int i) const
{
    return butcher_tableau_b_hat[i];
}

template class LowStorageRKTableauBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::Triangulation<PHILIP_DIM>>;
template class LowStorageRKTableauBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class LowStorageRKTableauBase<PHILIP_DIM, PHILIP_SPECIES, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
