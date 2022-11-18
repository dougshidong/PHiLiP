#include "entropy_rrk_ode_solver.h"
#include "physics/euler.h"
#include "physics/physics_factory.h"

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
    this->rk_stage_solution[istage]=this->rk_stage[istage]; 
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
real EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_relaxation_parameter(real &dt) const
{
    // TEMP : Using variable names per Ranocha paper

    // Console output is based on linearsolverparam
    bool do_output = (this->dg->all_parameters->linear_solver_param.linear_solver_output == Parameters::OutputEnum::verbose); 

    dealii::LinearAlgebra::distributed::Vector<double> d;
    d.reinit(this->rk_stage[0]);
    for (int i = 0; i < n_rk_stages; ++i){
        d.add(this->butcher_tableau->get_b(i), this->rk_stage[i]);
    }
    d *= dt;
    
    const double e = compute_entropy_change_estimate(dt); //calculate
    if (do_output) this->pcout <<"Entropy change estimate: " << e << std::endl;
    
    const dealii::LinearAlgebra::distributed::Vector<double> u_n = this->solution_update;
    const double eta_n = compute_numerical_entropy(u_n); //compute_inner_product(u_n, u_n);
    
    const bool use_secant = true;

    double gamma_kp1; 
    const double conv_tol = 1E-8; //2.4E-10;
    int iter_counter = 0;
    const int iter_limit = 100;
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
            if (r_gamma_km1 == r_gamma_k){
                if (do_output) this->pcout << "    Roots are identical. Multiplying gamma_k by 1.001 and recomputing..." << std::endl;
                gamma_k *= 1.001;
                r_gamma_km1 = compute_root_function(gamma_km1, u_n, d, eta_n, e);
                r_gamma_k = compute_root_function(gamma_k, u_n, d, eta_n, e);
                //this->pcout << "Current r_gamma_km1 = " << r_gamma_km1 << " r_gamma_k = " << r_gamma_k << std::endl;
                //this->pcout << "Current gamma_km1 = " << gamma_km1 << " gamma_k = " << gamma_k << std::endl;
            }
            if ((gamma_k < 0.5) || (gamma_k > 1.5)) {
                if (do_output) this->pcout << "    Gamma is far from 1. Setting gamma_k = 1 and contining iterations." << std::endl;
                gamma_k = 1.0;
                r_gamma_k = compute_root_function(gamma_k, u_n, d, eta_n, e);

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
            
            //residual = abs(r_gamma_k); 

            //output
            if (do_output) {
                this->pcout << "Iter: " << iter_counter
                            << " gamma_k: " << gamma_k
                            << " residual: " << residual << std::endl;
            }
            
            if (isnan(gamma_k) || isnan(gamma_km1)) {
                if (do_output) this->pcout << "    NaN detected. Restarting iterations from 1.0." << std::endl;
                gamma_k   = 1.0 - 1E-5;
                r_gamma_k = compute_root_function(gamma_k, u_n, d, eta_n, e);
                gamma_km1 = 1.0 + 1E-5;
                r_gamma_km1 = compute_root_function(gamma_km1, u_n, d, eta_n, e);
                residual = 1.0;
            }
        }
    } else {
        //Bisection method
        double l_limit = this->relaxation_parameter - 0.5;
        double u_limit = this->relaxation_parameter + 0.5;
        double root_l_limit = compute_root_function(l_limit, u_n, d, eta_n, e);
        double root_u_limit = compute_root_function(u_limit, u_n, d, eta_n, e);

        double residual = 1.0;

        while ((residual > conv_tol) && (iter_counter < iter_limit)){
            if (root_l_limit * root_u_limit > 0){
                this->pcout << "No root in the interval. Aborting..." << std::endl;
                std::abort();
            }

            gamma_kp1 = 0.5 * (l_limit + u_limit);
            if (do_output) this->pcout << "Iter: " << iter_counter;
            if (do_output) this->pcout << " Gamma by bisection is " << gamma_kp1;
            double root_at_gamma = compute_root_function(gamma_kp1, u_n, d, eta_n, e);
            if (root_at_gamma < 0) {
                l_limit = gamma_kp1;
                root_l_limit = root_at_gamma;
            } else {
                u_limit = gamma_kp1;
                root_u_limit = root_at_gamma;
            }
            residual = abs(root_at_gamma);
            residual = u_limit-l_limit;
            if (do_output) this->pcout << " With residual " << residual << std::endl;
            iter_counter++;
        }    
    }

    if (iter_limit == iter_counter) {
        this->pcout << "Error: Iteration limit reached and secant method has not converged" << std::endl;
        std::abort();
        return -1;
    } else {
        if (do_output) this->pcout << "Convergence reached!" << std::endl;
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
    real num_entropy = compute_integrated_numerical_entropy(u);
    //this->pcout << num_entropy << " ";
    return num_entropy;
    
    /*dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(this->dg->right_hand_side);
    if(this->dg->all_parameters->use_inverse_mass_on_the_fly)
        this->dg->apply_global_mass_matrix(u,mass_matrix_times_solution);
    else
        this->dg->global_mass_matrix.vmult( mass_matrix_times_solution, u);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global = compute_entropy_vars(u);

    double entropy = entropy_var_hat_global * mass_matrix_times_solution;
    //double entropy_mpi = (dealii::Utilities::MPI::sum(entropy, this->mpi_communicator));
    return entropy;
    */
}


template <int dim, typename real, int n_rk_stages, typename MeshType>
real EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_integrated_numerical_entropy(
        const dealii::LinearAlgebra::distributed::Vector<double> &u) const
{
    const int nstate = dim+2; //factory has already checked that this is Euler or NS
    const unsigned int poly_degree = this->dg->max_degree;

    const unsigned int n_dofs_cell = this->dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = this->dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;

    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, this->dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(this->dg->oneD_fe_collection_1state[poly_degree], this->dg->oneD_quadrature_collection[poly_degree]);

    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, this->dg->max_grid_degree); 
    soln_basis.build_1D_volume_operator(this->dg->oneD_fe_collection_1state[poly_degree], this->dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, poly_degree, this->dg->max_grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(this->dg->high_order_grid->oneD_fe_system, this->dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(this->dg->high_order_grid->oneD_fe_system, this->dg->oneD_quadrature_collection[poly_degree], this->dg->oneD_face_quadrature);

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);
    
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(this->dg->all_parameters));

    double integrand_numerical_entropy_function=0;
    double integral_numerical_entropy_function=0;
    const std::vector<double> &quad_weights = this->dg->volume_quadrature_collection[poly_degree].get_weights();

    auto metric_cell = this->dg->high_order_grid->dof_handler_grid.begin_active();
    // Changed for loop to update metric_cell.
    for (auto cell = this->dg->dof_handler.begin_active(); cell!= this->dg->dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = this->dg->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(this->dg->max_grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (this->dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, this->dg->max_grid_degree, false, false);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            this->dg->all_parameters->use_invariant_curl_form);

        // Fetch the modal soln coefficients
        // We immediately separate them by state as to be able to use sum-factorization
        // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
        // mult would sum the states at the quadrature point.
        // That is why the basis functions are based off the 1state oneD fe_collection.
        std::array<std::vector<double>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
            const unsigned int istate = this->dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = this->dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
            }
            soln_coeff[istate][ishape] = u(dofs_indices[idof]);
        }
        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        std::array<std::vector<double>,nstate> energy_var_at_q;
        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_state;
            // Extract solution and gradient in a way that the physics ca n use them.
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }

            integrand_numerical_entropy_function = euler_physics->compute_numerical_entropy_function(soln_state);
            integral_numerical_entropy_function += integrand_numerical_entropy_function * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
        }
    }
    // update integrated quantities and return
    return dealii::Utilities::MPI::sum(integral_numerical_entropy_function, this->mpi_communicator);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
real EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_entropy_change_estimate(real &dt) const
{
    double entropy_change_estimate = 0;
    for (int istage = 0; istage<n_rk_stages; ++istage){

        // Recall rk_stage is IMM * RHS
        // therefore, RHS = M * rk_stage = M * du/dt
        dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_rk_stage(this->dg->solution);
        if(this->dg->all_parameters->use_inverse_mass_on_the_fly)
            this->dg->apply_global_mass_matrix(this->rk_stage[istage],mass_matrix_times_rk_stage);
        else
            this->dg->global_mass_matrix.vmult( mass_matrix_times_rk_stage, this->rk_stage[istage]);
        
        //transform solution into entropy variables
        dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global = compute_entropy_vars(this->rk_stage_solution[istage]);
        
        double entropy = entropy_var_hat_global * mass_matrix_times_rk_stage;
        
        entropy_change_estimate += this->butcher_tableau->get_b(istage) * entropy;
    }

    return dt * entropy_change_estimate;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> EntropyRRKODESolver<dim,real,n_rk_stages,MeshType>::compute_entropy_vars(const dealii::LinearAlgebra::distributed::Vector<double> &u) const
{
    // hard-code nstate for Euler/NS - ODESolverFactory has already ensured that we use Euler/NS
    const unsigned int nstate = dim + 2;
    // Currently only implemented for constant p
    const unsigned int poly_degree = this->dg->get_max_fe_degree();
    if (poly_degree != this->dg->get_min_fe_degree()){
        this->pcout << "Error: Entropy RRK is only implemented for uniform p. Aborting..." << std::endl;
        std::abort();
    }

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

    std::shared_ptr< Physics::Euler<dim,dim+2,double> > euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(this->dg->all_parameters));

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

            std::array<double,nstate> entropy_var = euler_physics->convert_conservative_to_entropy(soln_state);

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
