#include "energy_rrk_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
EnergyRRKODESolver<dim,real,MeshType>::EnergyRRKODESolver(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : EmptyRRKBase<dim,real,MeshType>(rk_tableau_input)
        , n_rk_stages(rk_tableau_input->n_rk_stages)
{

    relaxation_parameter=1.0;
}

template <int dim, typename real, typename MeshType>
double EnergyRRKODESolver<dim,real,MeshType>::modify_time_step(const double dt)
{
    // Update solution such that dg is holding u^n (not last stage of RK)
    this->dg->solution = this->solution_update;
    this->dg->assemble_residual();

    relaxation_parameter = compute_relaxation_parameter(dt);

    if (relaxation_parameter < 0.5 ){
        this->pcout << "RRK failed to find a reasonable relaxation factor. Aborting..." << std::endl;
        relaxation_parameter=1.0;
        std::abort();
    }
    dt *= relaxation_parameter;

    return dt;
}

template <int dim, typename real, typename MeshType>
real EnergyRRKODESolver<dim,real,MeshType>::compute_relaxation_parameter(real & /*dt*/)
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

template <int dim, typename real, typename MeshType>
real EnergyRRKODESolver<dim,real,MeshType>::compute_inner_product (
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

template class EnergyRRKODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class EnergyRRKODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class EnergyRRKODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace