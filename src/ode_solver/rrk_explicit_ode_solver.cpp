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
    //real relaxation_parameter_implicit = compute_relaxation_parameter_implicit(dt);
    //this -> pcout << "______________________________________________________" << std::endl;
    //this->pcout << "gamma explicit = " << std::setprecision(16) << relaxation_parameter << " gamma implicit = " << relaxation_parameter_implicit << std::endl;
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
real RRKExplicitODESolver<dim,real,MeshType>::compute_relaxation_parameter_implicit(real &dt) const
{
    //For now, assume that the entropy variable is energy and discretization is energy-conservative
    // TEMP : Using variable names per Ranocha paper

    dealii::LinearAlgebra::distributed::Vector<double> d;
    d.reinit(this->rk_stage[0]);
    for (int i = 0; i < this->rk_order; ++i){
        //d += this->butcher_tableau_b[i]*this->rk_stage[i];
        d.add(this->butcher_tableau_b[i], this->rk_stage[i]);
    }
    d *= dt;
    
    //double e = 0; //conservative

    double initial_guess_0 = 1.0;
    double initial_guess_1 = 1.0 + pow(dt, this->rk_order - 1);
    double residual = 1.0;
    dealii::LinearAlgebra::distributed::Vector<double> u_n = this->solution_update;
    double eta_n = compute_numerical_entropy(u_n); //compute_inner_product(u_n, u_n);
    double gamma_k = initial_guess_1;
    double gamma_km1 = initial_guess_0;
    double gamma_kp1; 
    double r_gamma_k = compute_root_function(gamma_k,u_n,d,eta_n);
    double r_gamma_km1 = compute_root_function(gamma_km1, u_n, d, eta_n);

    int iter_limit = 1000;
    double conv_tol = 1E-15;
    int iter_counter = 0;
        //output
        this->pcout << "Iter: " << iter_counter << " (initialization)"
                    << " gamma_0: " << gamma_km1
                    << " gamma_1: " << gamma_k
                    << " residual: " << residual << std::endl;
    while ((residual > conv_tol) && (iter_counter < iter_limit)){
        // For now, secant method
        // TEMP : will replace with Newton's method when eta' has been defined
        gamma_kp1 = gamma_k - r_gamma_k * (gamma_k - gamma_km1)/(r_gamma_k-r_gamma_km1);
        residual = abs(gamma_kp1 - gamma_k);
        iter_counter ++;

        //update values
        gamma_km1 = gamma_k;
        gamma_k = gamma_kp1;
        r_gamma_km1 = r_gamma_k;
        r_gamma_k = compute_root_function(gamma_k, u_n, d, eta_n);

        //output
        this->pcout << "Iter: " << iter_counter
                    << " gamma_k: " << gamma_k
                    << " residual: " << residual << std::endl;
    }

    if (iter_limit == iter_counter) {
        this->pcout << "Error: Iteration limit reached and secant method has not converged" << std::endl;
        return -1;
        std::abort();
    } else {
        this->pcout << "Convergence reached!" << std::endl;
        return gamma_kp1;
    }
}


template <int dim, typename real, int n_rk_stages, typename MeshType>
real RRKExplicitODESolver<dim,real,MeshType>::compute_root_function(
        const double gamma,
        const dealii::LinearAlgebra::distributed::Vector<double> &u_n,
        const dealii::LinearAlgebra::distributed::Vector<double> &d,
        const double eta_n) const
{
    dealii::LinearAlgebra::distributed::Vector<double> temp = u_n;
    temp.add(gamma, d);
    double eta_np1 = compute_numerical_entropy(temp);
    return eta_np1 - eta_n;
}


template <int dim, typename real, typename MeshType>
real RRKExplicitODESolver<dim,real,MeshType>::compute_numerical_entropy(
        const dealii::LinearAlgebra::distributed::Vector<double> &u) const
{
    //For now, return energy (burgers)
    return compute_inner_product(u,u);
}
        

template <int dim, typename real, int n_rk_stages, typename MeshType>
real RRKExplicitODESolver<dim,real,MeshType>::compute_inner_product (
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_i,
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_j
        ) const
{
    // Intention is to point to physics (mimic structure in flow_solver_cases/periodic_turbulence.cpp for converting to solution for general nodes) 
    // For now, only energy on collocated nodes is implemented.
/*    
    real inner_product = 0;
    for (unsigned int i = 0; i < this->dg->solution.size(); ++i) {
        inner_product += 1./(this->dg->global_inverse_mass_matrix.diag_element(i))
                         * stage_i[i] * stage_j[i];
    }
    return inner_product;
*/
    dealii::LinearAlgebra::distributed::Vector<double> temp;
    temp.reinit(stage_j);
    this->dg->global_mass_matrix.vmult(temp, stage_j); //replace stage_j with M*stage_j
    const double result = temp * stage_i;
    return result;
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
