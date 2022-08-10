#include "rrk_explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
RRKExplicitODESolver<dim,real,MeshType>::RRKExplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : ExplicitODESolver<dim,real,MeshType>(dg_input)
        {}

template <int dim, typename real, typename MeshType>
void RRKExplicitODESolver<dim,real,MeshType>::modify_time_step(real &dt)
{
    real relaxation_parameter = compute_relaxation_parameter_explicit();
    dt *= relaxation_parameter;
}

template <int dim, typename real, typename MeshType>
real RRKExplicitODESolver<dim,real,MeshType>::compute_relaxation_parameter_explicit() const
{
    double gamma = 1;
    double denominator = 0;
    double numerator = 0;
    for (int i = 0; i < this->rk_order; ++i){
        for (int j = 0; j < this->rk_order; ++j){
            real inner_product = compute_inner_product(this->rk_stage[i],this->rk_stage[j]);
            numerator += this->butcher_tableau_b[i] *this-> butcher_tableau_a[i][j] * inner_product; 
            denominator += this->butcher_tableau_b[i]*this->butcher_tableau_b[j] * inner_product;
        }
    }
    numerator *= 2;
    gamma = (denominator < 1E-8) ? 1 : numerator/denominator;
    return gamma;
}

template <int dim, typename real, typename MeshType>
real RRKExplicitODESolver<dim,real,MeshType>::compute_inner_product (
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

template class RRKExplicitODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class RRKExplicitODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
//currently only tested in 1D - commenting out higher dimensions
//#if PHILIP_DIM != 1
//template class RRKExplicitODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
//#endif

} // ODESolver namespace
} // PHiLiP namespace
