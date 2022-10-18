#include "entropy_rrk_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType>
EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::EntropyRRKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RRKODESolverBase<dim,real,n_rk_stages,MeshType>(dg_input,rk_tableau_input)
{
    this->rk_stage_solution.resize(n_rk_stages);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_stored_quantities(const int istage)
{
    //Store the solution value
    //This function is called before rk_stage is modified to hold the time-derivative
    //this->rk_stage_solution[istage].reinit(this->rk_stage[istage], true); // omit_zeroing_entries=true
    this->rk_stage_solution[istage]=this->rk_stage[istage]; 
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
real EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_relaxation_parameter(real &dt) const
{
    // TEMP : Using variable names per Ranocha paper

    dealii::LinearAlgebra::distributed::Vector<double> d;
    d.reinit(this->rk_stage[0]);
    for (int i = 0; i < n_rk_stages; ++i){
        d.add(this->butcher_tableau->get_b(i), this->rk_stage[i]);
    }
    d *= dt;
    
    const double e = compute_entropy_change_estimate(dt); //conservative
    this->pcout <<"Entropy change estimate: " << e << std::endl;
    
    const dealii::LinearAlgebra::distributed::Vector<double> u_n = this->solution_update;
    const double eta_n = compute_numerical_entropy(u_n); //compute_inner_product(u_n, u_n);
    
    const bool use_secant = true;

    double gamma_kp1; 
    const double conv_tol = 5E-11;
    int iter_counter = 0;
    const int iter_limit = 1000;
    if (use_secant){

        const double initial_guess_0 = this->relaxation_parameter - 1E-5;
        const double initial_guess_1 = this->relaxation_parameter + 1E-5;
        double residual = 1.0;
        double gamma_k = initial_guess_1;
        double gamma_km1 = initial_guess_0;
        double r_gamma_k = compute_root_function(gamma_k, u_n, d, eta_n, e);
        double r_gamma_km1 = compute_root_function(gamma_km1, u_n, d, eta_n,e);

        //output
        //this->pcout << "Iter: " << iter_counter << " (initialization)"
        //            << " gamma_0: " << gamma_km1
        //            << " gamma_1: " << gamma_k
        //            << " residual: " << residual << std::endl;

        while ((residual > conv_tol) && (iter_counter < iter_limit)){
            while (r_gamma_km1 == r_gamma_k){
                this->pcout << "    Roots are identical. Multiplying gamma_k by 1.001 and recomputing..." << std::endl;
                gamma_k *= 1.001;
                r_gamma_km1 = compute_root_function(gamma_km1, u_n, d, eta_n, e);
                r_gamma_k = compute_root_function(gamma_k, u_n, d, eta_n, e);
                //this->pcout << "Current r_gamma_km1 = " << r_gamma_km1 << " r_gamma_k = " << r_gamma_k << std::endl;
                //this->pcout << "Current gamma_km1 = " << gamma_km1 << " gamma_k = " << gamma_k << std::endl;
            }
            // Secant method, as recommended by Rogowski et al. 2022
            gamma_kp1 = gamma_k - r_gamma_k * (gamma_k - gamma_km1)/(r_gamma_k-r_gamma_km1);
            residual = abs(gamma_kp1 - gamma_k);
            iter_counter ++;

            //update values
            gamma_km1 = gamma_k;
            gamma_k = gamma_kp1;
            r_gamma_km1 = r_gamma_k;
            r_gamma_k = compute_root_function(gamma_k, u_n, d, eta_n, e);

            //output
            this->pcout << "Iter: " << iter_counter
                        << " gamma_k: " << gamma_k
                        << " residual: " << residual << std::endl;
        }
    } else {
        //Bisection method
        double l_limit = this->relaxation_parameter - 0.01;
        double u_limit = this->relaxation_parameter + 0.01;
        double root_l_limit = compute_root_function(l_limit, u_n, d, eta_n, e);
        double root_u_limit = compute_root_function(u_limit, u_n, d, eta_n, e);

        double residual = 1.0;

        while ((residual > conv_tol) && (iter_counter < iter_limit)){
            if (root_l_limit * root_u_limit > 0){
                this->pcout << "No root in the interval. Aborting..." << std::endl;
                std::abort();
            }

            gamma_kp1 = 0.5 * (l_limit + u_limit);
            this->pcout << "Iter: " << iter_counter;
            this->pcout << " Gamma by bisection is " << gamma_kp1;
            double root_at_gamma = compute_root_function(gamma_kp1, u_n, d, eta_n, e);
            if (root_at_gamma < 0) {
                l_limit = gamma_kp1;
                root_l_limit = root_at_gamma;
            } else {
                u_limit = gamma_kp1;
                root_u_limit = root_at_gamma;
            }
            residual = abs(root_at_gamma);
            this->pcout << " With residual " << residual << std::endl;
            iter_counter++;
        }    
    }

    if (iter_limit == iter_counter) {
        this->pcout << "Error: Iteration limit reached and secant method has not converged" << std::endl;
        std::abort();
        return -1;
    } else {
        this->pcout << "Convergence reached!" << std::endl;
        return gamma_kp1;
    }
}


template <int dim, typename real, int n_rk_stages, typename MeshType>
real EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_root_function(
        const double gamma,
        const dealii::LinearAlgebra::distributed::Vector<double> &u_n,
        const dealii::LinearAlgebra::distributed::Vector<double> &d,
        const double eta_n,
        const double e) const
{
    dealii::LinearAlgebra::distributed::Vector<double> temp = u_n;
    temp.add(gamma, d);
    double eta_np1 = compute_numerical_entropy(temp);
    return eta_np1 - eta_n - gamma * e;
}


template <int dim, typename real, int n_rk_stages, typename MeshType>
real EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_numerical_entropy(
        const dealii::LinearAlgebra::distributed::Vector<double> &u) const
{
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(this->dg->right_hand_side);
    if(this->dg->all_parameters->use_inverse_mass_on_the_fly)
        this->dg->apply_global_mass_matrix(u,mass_matrix_times_solution);
    else
        this->dg->global_mass_matrix.vmult( mass_matrix_times_solution, u);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global = compute_entropy_vars(u);

    double entropy = entropy_var_hat_global * mass_matrix_times_solution;
    //double entropy_mpi = (dealii::Utilities::MPI::sum(entropy, this->mpi_communicator));
    return entropy;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
real EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_entropy_change_estimate(real &dt) const
{
    double entropy_change_estimate = 0;
    for (int istage = 0; istage<n_rk_stages; ++istage){

        // Recall rk_stage is IMM * RHS
        // therefore, RHS = M * rk_stage = M * du/dt
        dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_rk_stage(this->dg->solution);
        this->dg->global_mass_matrix.vmult(mass_matrix_times_rk_stage, this->rk_stage[istage]);
        
        //transform solution into entropy variables
        dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global = compute_entropy_vars(this->rk_stage_solution[istage]);
        
        double entropy = entropy_var_hat_global * mass_matrix_times_rk_stage;
        // MPI sum
        //double entropy_mpi = (dealii::Utilities::MPI::sum(entropy, this->mpi_communicator));
        entropy_change_estimate += this->butcher_tableau->get_b(istage) * entropy;
    }

    return dt * entropy_change_estimate;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_entropy_vars(const dealii::LinearAlgebra::distributed::Vector<double> &u) const
{

        // TEMP : hard-code poly_degree and nstate
        const unsigned int nstate = dim + 2;
        const unsigned int poly_degree = 3;  

        //TEMP : Should move following code to physics....
        const unsigned int n_dofs_cell = this->dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_quad_pts = this->dg->volume_quadrature_collection[poly_degree].size();
        const unsigned int n_shape_fns = n_dofs_cell / nstate;
        //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
        OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, this->dg->max_grid_degree);
        vol_projection.build_1D_volume_operator(this->dg->oneD_fe_collection_1state[poly_degree], this->dg->oneD_quadrature_collection[poly_degree]);

        OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, this->dg->max_grid_degree); 
        soln_basis.build_1D_volume_operator(this->dg->oneD_fe_collection_1state[poly_degree], this->dg->oneD_quadrature_collection[poly_degree]);

        dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(this->dg->right_hand_side);
        std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

        //std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(this->dg->all_parameters);

        for (auto cell = this->dg->dof_handler.begin_active(); cell!=this->dg->dof_handler.end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices (dofs_indices);

            std::array<std::vector<double>,nstate> soln_coeff;
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = this->dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                const unsigned int ishape = this->dg->fe_collection[poly_degree].system_to_component_index(idof).second;
                if(ishape == 0)
                    soln_coeff[istate].resize(n_shape_fns);
                soln_coeff[istate][ishape] = u(dofs_indices[idof]);
            }

            std::array<std::vector<double>,nstate> soln_at_q;
            for(unsigned int istate=0; istate<nstate; istate++){
                soln_at_q[istate].resize(n_quad_pts);
                soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                                 soln_basis.oneD_vol_operator);
            }
            std::array<std::vector<double>,nstate> entropy_var_at_q;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                std::array<double,nstate> soln_state;
                for(unsigned int istate=0; istate<nstate; istate++){
                    soln_state[istate] = soln_at_q[istate][iquad];
                }

                std::array<double,nstate> entropy_var;
                const double density = soln_state[0];
                dealii::Tensor<1,dim,double> vel;
                double vel2 = 0.0;
                for(int idim=0; idim<dim; idim++){
                    vel[idim] = soln_state[idim+1]/soln_state[0];
                    vel2 += vel[idim]*vel[idim];
                }
                const double pressure = 0.4*(soln_state[nstate-1] - 0.5*density*vel2);
                const double entropy = log(pressure) - 1.4 * log(density);
                 
                entropy_var[0] = (1.4-entropy)/0.4 - 0.5 * density / pressure * vel2;
                for(int idim=0; idim<dim; idim++){
                    entropy_var[idim+1] = soln_state[idim+1] / pressure;
                }
                entropy_var[nstate-1] = - density / pressure;

                for(unsigned int istate=0; istate<nstate; istate++){
                    if(iquad==0)
                        entropy_var_at_q[istate].resize(n_quad_pts);
                    entropy_var_at_q[istate][iquad] = entropy_var[istate];
                }
            }
            for(unsigned int istate=0; istate<nstate; istate++){
                //Projected vector of entropy variables.
                std::vector<double> entropy_var_hat(n_shape_fns);
                vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                     vol_projection.oneD_vol_operator);
                                                    
                for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                    const unsigned int idof = istate * n_shape_fns + ishape;
                    entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
                }
            }
        }
    return entropy_var_hat_global;
}

template class EntropyRRKODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class EntropyRRKODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class EntropyRRKODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class EntropyRRKODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class EntropyRRKODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class EntropyRRKODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class EntropyRRKODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class EntropyRRKODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class EntropyRRKODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class EntropyRRKODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class EntropyRRKODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class EntropyRRKODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
