#include "energy_rrk_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType>
EnergyRRKODESolver<dim,real,n_rk_stages,MeshType>::EnergyRRKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RRKODESolverBase<dim,real,n_rk_stages,MeshType>(dg_input,rk_tableau_input)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType>
real EnergyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_relaxation_parameter(real & /*dt*/) const
{
    //See Ketcheson 2019, Eq. 2.4
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
real EnergyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_inner_product (
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_i,
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_j
        ) const
{
    // Calculate by matrix-vector product u_i^T M u_j
    dealii::LinearAlgebra::distributed::Vector<double> temp;
    temp.reinit(stage_j);

    if(this->all_parameters->use_inverse_mass_on_the_fly){
        this->dg->apply_global_mass_matrix(stage_j, temp);
    } else{
        this->dg->global_mass_matrix.vmult(temp,stage_j);
    } //replace stage_j with M*stage_j

    const double result = temp * stage_i;
    return result;
}

template class EnergyRRKODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class EnergyRRKODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class EnergyRRKODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class EnergyRRKODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class EnergyRRKODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class EnergyRRKODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class EnergyRRKODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class EnergyRRKODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class EnergyRRKODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class EnergyRRKODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class EnergyRRKODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class EnergyRRKODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
