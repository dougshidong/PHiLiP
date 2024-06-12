#include "low_storage_rk_tableau_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType> 
LowStorageRKTableauBase<dim,real, MeshType> :: LowStorageRKTableauBase ( 
        const std::string rk_method_string_input)
    : rk_method_string(rk_method_string_input)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    //this->butcher_tableau_a.reinit(n_rk_stages,n_rk_stages);
    //this->butcher_tableau_b.reinit(n_rk_stages);
    //this->butcher_tableau_c.reinit(n_rk_stages);
    this->butcher_tableau_gamma.reinit(6,3);
    this->butcher_tableau_beta.reinit(6);
    this->butcher_tableau_delta.reinit(7);
}

template <int dim, typename real, typename MeshType> 
void LowStorageRKTableauBase<dim,real, MeshType> :: set_tableau ()
{
    set_gamma();
    set_beta();
    set_delta();
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
void LowStorageRKTableauBase<dim,real,MeshType> :: set_gamma()
{
    const double gamma[6][3] = {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-0.497531095840104, 1.384996869124138, 0.0}, {1.010070514199942, 3.878155713328178, 0.0}, {-3.196559004608766,-2.324512951813145, 1.642598936063715}, {1.717835630267259, -0.514633322274467, 0.188295940828347}};
    for (int i = 0; i<6; i++){
        for (int j = 0; j<3; j++){
            this->butcher_tableau_gamma(i,j) = gamma[i][j];
        }
    }
}

template <int dim, typename real, typename MeshType>
void LowStorageRKTableauBase<dim,real,MeshType> :: set_beta()
{
    double beta[6] = {0.0, 0.075152045700771, 0.211361016946069, 1.100713347634329, 0.728537814675568, 0.393172889823198};
    this->butcher_tableau_beta.fill(beta);
}

template <int dim, typename real, typename MeshType>
void LowStorageRKTableauBase<dim,real,MeshType> :: set_delta()
{
    double delta[7] = {1.0, 0.081252332929194, -1.083849060586449, -1.096110881845602, 2.859440022030827, -0.655568367959557, -0.194421504490852};
    this->butcher_tableau_delta.fill(delta);
    
}


template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
