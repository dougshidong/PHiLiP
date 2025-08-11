#include "PERK_tableau_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType> 
PERKTableauBase<dim,real, MeshType> :: PERKTableauBase (const int n_rk_stages_input, 
        const std::string rk_method_string_input)
    : n_rk_stages(n_rk_stages_input)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    , rk_method_string(rk_method_string_input)
{
    this->butcher_tableau_a1.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_a2.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_a3.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_a4.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_a5.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_a6.reinit(n_rk_stages,n_rk_stages);
    this->butcher_tableau_b.reinit(n_rk_stages);
    this->butcher_tableau_c.reinit(n_rk_stages);
}

template <int dim, typename real, typename MeshType> 
void PERKTableauBase<dim,real, MeshType> :: set_tableau ()
{
    set_a1();
    set_a2();
    set_a3();
    set_a4();
    set_a5();
    set_a6();
    //set_a10();
    set_b();
    set_c();
    pcout << "Assigned RK method: " << rk_method_string << std::endl;
}

template <int dim, typename real, typename MeshType> 
double PERKTableauBase<dim,real, MeshType> :: get_a (const int i, const int j, const int a) const
{
    if (a == 1){
        return butcher_tableau_a1[i][j];
    }
    else if (a ==2) {
        return butcher_tableau_a2[i][j];
    } 
    else if (a ==3) {
        return butcher_tableau_a3[i][j];
    } 
    else if (a ==4) {
        return butcher_tableau_a4[i][j];
    } 
    else if (a ==6) {
        return butcher_tableau_a4[i][j];
    } 
    else if (a ==5) {
        return butcher_tableau_a4[i][j];
    } 
    // else if (a ==10) {
    //     return butcher_tableau_a10[i][j];
    // }
    else {
        pcout << "Butcher tableau 'a' value not implemented " << std::endl;
        std::abort();
    }
}

template <int dim, typename real, typename MeshType> 
double PERKTableauBase<dim,real, MeshType> :: get_b (const int i) const
{
    return butcher_tableau_b[i];
}

template <int dim, typename real, typename MeshType> 
double PERKTableauBase<dim,real, MeshType> :: get_c (const int i) const
{
    return butcher_tableau_c[i];
}

template class PERKTableauBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class PERKTableauBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class PERKTableauBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace