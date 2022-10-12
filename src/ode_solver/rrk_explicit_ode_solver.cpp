#include "rrk_explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType>
RRKExplicitODESolver<dim,real,n_rk_stages,MeshType>::RRKExplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>(dg_input,rk_tableau_input)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void RRKExplicitODESolver<dim,real,n_rk_stages,MeshType>::modify_time_step(real &dt)
{
    real relaxation_parameter = compute_relaxation_parameter_explicit();
    dt *= relaxation_parameter;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
real RRKExplicitODESolver<dim,real,n_rk_stages,MeshType>::compute_relaxation_parameter_explicit() const
{
    double gamma = 1;
    double denominator = 0;
    double numerator = 0;
    for (int i = 0; i < n_rk_stages; ++i){
        const double b_i = this->butcher_tableau->get_b(i);
        for (int j = 0; j < n_rk_stages; ++j){
            real inner_product = compute_inner_product(this->rk_stage[i],this->rk_stage[j]);
            numerator += b_i * this-> butcher_tableau->get_a(i,j) * inner_product; 
            denominator += b_i * this->butcher_tableau->get_b(j) * inner_product;
        }
    }
    numerator *= 2;
    gamma = (denominator < 1E-8) ? 1 : numerator/denominator;
    return gamma;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
real RRKExplicitODESolver<dim,real,n_rk_stages,MeshType>::compute_inner_product (
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_i,
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_j
        ) const
{
    // Intention is to point to physics (mimic structure in flow_solver_cases/periodic_turbulence.cpp for converting to solution for general nodes) 
    // For now, only energy on collocated nodes is implemented.
    
    real inner_product = 0;
    for (unsigned int i = 0; i < this->dg->solution.size(); ++i) {
        inner_product += 1./(this->dg->global_inverse_mass_matrix.diag_element(i))
                         * stage_i[i] * stage_j[i];
    }
    return inner_product;
}

template class RRKExplicitODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RRKExplicitODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RRKExplicitODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RRKExplicitODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RRKExplicitODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RRKExplicitODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RRKExplicitODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RRKExplicitODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    // currently only tested in 1D - commenting out higher dimensions
    /*
    template class RRKExplicitODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RRKExplicitODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RRKExplicitODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RRKExplicitODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    */
#endif

} // ODESolver namespace
} // PHiLiP namespace
