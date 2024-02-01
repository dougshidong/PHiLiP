#include "energy_rrk_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
AlgebraicRRKODESolver<dim,real,MeshType>::AlgebraicRRKODESolver(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RRKODESolverBase<dim,real,MeshType>(rk_tableau_input)
{
    // Do nothing
}

template <int dim, typename real, typename MeshType>
real AlgebraicRRKODESolver<dim,real,MeshType>::compute_relaxation_parameter(const real /*dt*/,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg,
            const std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
            const dealii::LinearAlgebra::distributed::Vector<double> &/*solution_update*/
        )
{
    //See Ketcheson 2019, Eq. 2.4
    double gamma = 1;
    double denominator = 0;
    double numerator = 0;
    for (int i = 0; i < this->n_rk_stages; ++i){
        const double b_i = this->butcher_tableau->get_b(i);
        for (int j = 0; j < this->n_rk_stages; ++j){
            real inner_product = compute_inner_product(rk_stage[i],rk_stage[j], dg);
            numerator += b_i * this-> butcher_tableau->get_a(i,j) * inner_product; 
            denominator += b_i * this->butcher_tableau->get_b(j) * inner_product;
        }
    }
    numerator *= 2;
    gamma = (denominator < 1E-8) ? 1 : numerator/denominator;
    return gamma;
}

template <int dim, typename real, typename MeshType>
real AlgebraicRRKODESolver<dim,real,MeshType>::compute_inner_product (
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_i,
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_j,
        std::shared_ptr<DGBase<dim,real,MeshType>> dg
        ) const
{
    // Calculate by matrix-vector product u_i^T M u_j
    dealii::LinearAlgebra::distributed::Vector<double> temp;
    temp.reinit(stage_j);

    if(dg->all_parameters->use_inverse_mass_on_the_fly){
        dg->apply_global_mass_matrix(stage_j, temp);
    } else{
        dg->global_mass_matrix.vmult(temp,stage_j);
    } //replace stage_j with M*stage_j

    const double result = temp * stage_i;
    return result;
}

template class AlgebraicRRKODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class AlgebraicRRKODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class AlgebraicRRKODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
