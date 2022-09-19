#include "rk_tableau_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType> 
RKTableauBase<dim,real, MeshType> :: RKTableauBase (const int n_rk_stages, 
        const std::string rk_method_string_input)
    : pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    , rk_method_string(rk_method_string_input)
{
    this->butcher_tableau_a.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_b.reinit(n_rk_stages);
    this->butcher_tableau_c.reinit(n_rk_stages);
}

template <int dim, typename real, typename MeshType> 
void RKTableauBase<dim,real, MeshType> :: set_tableau ()
{
    set_a();
    set_b();
    set_c();
    pcout << "Assigned RK method: " << rk_method_string << std::endl;
}

template <int dim, typename real, typename MeshType> 
double RKTableauBase<dim,real, MeshType> :: get_a (const int i, const int j) const
{
    return butcher_tableau_a[i][j];
}

template <int dim, typename real, typename MeshType> 
double RKTableauBase<dim,real, MeshType> :: get_b (const int i) const
{
    return butcher_tableau_b[i];
}

template <int dim, typename real, typename MeshType> 
double RKTableauBase<dim,real, MeshType> :: get_c (const int i) const
{
    return butcher_tableau_c[i];
}

template class RKTableauBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class RKTableauBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class RKTableauBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
