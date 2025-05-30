#include "rk_tableau_butcher_base.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType> 
RKTableauButcherBase<dim,real, MeshType> :: RKTableauButcherBase (const int n_rk_stages_input, 
        const std::string rk_method_string_input)
    : RKTableauBase<dim,real,MeshType>(n_rk_stages_input,rk_method_string_input)
{
    this->butcher_tableau_a.reinit(this->n_rk_stages,this->n_rk_stages);
    this->butcher_tableau_b.reinit(this->n_rk_stages);
    this->butcher_tableau_c.reinit(this->n_rk_stages);
}

template <int dim, typename real, typename MeshType> 
void RKTableauButcherBase<dim,real, MeshType> :: set_tableau ()
{
    set_a();
    set_b();
    set_c();
    this->pcout << "Assigned RK method: " << this->rk_method_string << std::endl;
}


template <int dim, typename real, typename MeshType> 
double RKTableauButcherBase<dim,real, MeshType> :: get_c (const int i) const
{
    return butcher_tableau_c[i];
}

template class RKTableauButcherBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class RKTableauButcherBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class RKTableauButcherBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
