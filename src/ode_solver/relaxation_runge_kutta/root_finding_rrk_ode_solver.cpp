#include "root_finding_rrk_ode_solver.h"
#include "physics/euler.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
RootFindingRRKODESolver<dim,real,MeshType>::RootFindingRRKODESolver(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RRKODESolverBase<dim,real,MeshType>(rk_tableau_input)
{
}

template <int dim, typename real, typename MeshType>
real RootFindingRRKODESolver<dim,real,MeshType>::compute_relaxation_parameter(const real dt,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg,
            const std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
            const dealii::LinearAlgebra::distributed::Vector<double> &solution_update
            ) 
{
    // Console output is based on linearsolverparam
    const bool do_output = (dg->all_parameters->ode_solver_param.rrk_root_solver_output == Parameters::OutputEnum::verbose); 

    // Note: there is some overlap in computations here and in runge_kutta_ode_solver.
    // In future optimization, this could be improved.
    dealii::LinearAlgebra::distributed::Vector<double> step_direction;
    step_direction.reinit(rk_stage[0]);
    for (int i = 0; i < this->n_rk_stages; ++i){
        step_direction.add(this->butcher_tableau->get_b(i), rk_stage[i]);
    }
    step_direction *= dt;
    
    // Compute entropy change estimate in M norm , [ v^T (M) du/dt ]
    const double entropy_change_est = compute_entropy_change_estimate(dt, dg, rk_stage);
    if (do_output) this->pcout <<"Entropy change estimate: " << std::setprecision(16) << entropy_change_est << std::endl;

    // n and np1 denote timestep indices
    const dealii::LinearAlgebra::distributed::Vector<double> u_n = solution_update;
    const double num_entropy_n = compute_numerical_entropy(u_n,dg);
    
    // Allow user to manually select bisection or secant solver.
    // As secant method is nearly always preferable, this is hard-coded.
    const bool use_secant = true;
    bool secant_failed = false;

    // k, kp1, km1 denote iteration indices of secant or bisection solvers
    double gamma_kp1 = 0; 
    const double conv_tol = dg->all_parameters->ode_solver_param.relaxation_runge_kutta_root_tolerance;
    int iter_counter = 0;
    const int iter_limit = 100;
    if (use_secant){

        const double initial_guess_0 = this->relaxation_parameter - 1E-5;
        const double initial_guess_1 = this->relaxation_parameter + 1E-5;
        double residual = 1.0;
        double gamma_k = initial_guess_1;
        double gamma_km1 = initial_guess_0;
        double r_gamma_k = compute_root_function(gamma_k, u_n, step_direction, num_entropy_n, entropy_change_est,dg);
        double r_gamma_km1 = compute_root_function(gamma_km1, u_n, step_direction, num_entropy_n,entropy_change_est,dg);

        while ((residual > conv_tol) && (iter_counter < iter_limit)){
            if (r_gamma_km1 == r_gamma_k){
                if (do_output) this->pcout << "    Roots are identical. Multiplying gamma_k by 1.001 and recomputing..." << std::endl;
                gamma_k *= 1.001;
                r_gamma_km1 = compute_root_function(gamma_km1, u_n, step_direction, num_entropy_n, entropy_change_est,dg);
                r_gamma_k = compute_root_function(gamma_k, u_n, step_direction, num_entropy_n, entropy_change_est,dg);
            }
            if ((gamma_k < 0.5) || (gamma_k > 1.5)) {
                if (do_output) this->pcout << "    Gamma is far from 1. Setting gamma_k = 1 and contining iterations." << std::endl;
                gamma_k = 1.0;
                r_gamma_k = compute_root_function(gamma_k, u_n, step_direction, num_entropy_n, entropy_change_est,dg);

            }
            // Secant method, as recommended by Rogowski et al. 2022
            gamma_kp1 = gamma_k - r_gamma_k * (gamma_k - gamma_km1)/(r_gamma_k-r_gamma_km1);
            residual = abs(gamma_kp1 - gamma_k);
            iter_counter ++;

            //update values
            gamma_km1 = gamma_k;
            gamma_k = gamma_kp1;
            r_gamma_km1 = r_gamma_k;
            r_gamma_k = compute_root_function(gamma_k, u_n, step_direction, num_entropy_n, entropy_change_est,dg);
            
            //output
            if (do_output) {
                this->pcout << "Iter: " << iter_counter
                            << " gamma_k: " << gamma_k
                            << " residual: " << residual << std::endl;
            }
            
            if (isnan(gamma_k) || isnan(gamma_km1)) {
                if (do_output) this->pcout << "    NaN detected. Restarting iterations from 1.0." << std::endl;
                gamma_k   = 1.0 - 1E-5;
                r_gamma_k = compute_root_function(gamma_k, u_n, step_direction, num_entropy_n, entropy_change_est,dg);
                gamma_km1 = 1.0 + 1E-5;
                r_gamma_km1 = compute_root_function(gamma_km1, u_n, step_direction, num_entropy_n, entropy_change_est,dg);
                residual = 1.0;
            }
        }

        // If secant method fails to find a root within the specified number of iterations, fall back on bisection method.
        if (iter_limit == iter_counter) {
            this->pcout << "Secant method failed to find a root within the iteration limit. Restarting with bisection method." << std::endl;
            secant_failed = true;
        }
    }
    if (!use_secant || secant_failed) {
        //Bisection method

        iter_counter = 0;

        double l_limit = this->relaxation_parameter - 0.1;
        double u_limit = this->relaxation_parameter + 0.1;
        double root_l_limit = compute_root_function(l_limit, u_n, step_direction, num_entropy_n, entropy_change_est,dg);
        double root_u_limit = compute_root_function(u_limit, u_n, step_direction, num_entropy_n, entropy_change_est,dg);

        double residual = 1.0;

        while ((residual > conv_tol) && (iter_counter < iter_limit)){
            if (root_l_limit * root_u_limit > 0){
                this->pcout << "Bisection solver: No root in the interval. Increasing interval size..." << std::endl;
                l_limit -= 0.1;
                u_limit += 0.1;
            }

            gamma_kp1 = 0.5 * (l_limit + u_limit);
            if (do_output) this->pcout << "Iter: " << iter_counter;
            if (do_output) this->pcout << " Gamma by bisection is " << gamma_kp1;
            double root_at_gamma = compute_root_function(gamma_kp1, u_n, step_direction, num_entropy_n, entropy_change_est,dg);
            if (root_at_gamma < 0) {
                l_limit = gamma_kp1;
                root_l_limit = root_at_gamma;
            } else {
                u_limit = gamma_kp1;
                root_u_limit = root_at_gamma;
            }
            residual = u_limit-l_limit;
            if (do_output) this->pcout << " With residual " << residual << std::endl;
            iter_counter++;
        }    
    }

    if (iter_limit == iter_counter) {
        this->pcout << "Error: Iteration limit reached and root finding was not successful." << std::endl;
        secant_failed = true;
        std::abort();
        return -1;
    } else { // Root-finding was successful

        if (do_output) {
            // Use [ gamma * dt * (v^T (M+K) du/dt - v^T (M) du/dt ) ] as a workaround to calculate [ gamma * (v^T (K) du/dt) ]
            const double FR_entropy_contribution = gamma_kp1 *(
                    compute_entropy_change_estimate(dt, dg, rk_stage, false) 
                    - compute_entropy_change_estimate(dt, dg, rk_stage, true)
                    );

            // TEMP store in dg so that flow solver case can access it
            dealii::LinearAlgebra::distributed::Vector<double> temp = u_n;
            temp.add(gamma_kp1, step_direction);
            const double num_entropy_npgamma = compute_numerical_entropy(temp,dg);
            const double FR_entropy_this_tstep = num_entropy_npgamma - num_entropy_n + FR_entropy_contribution;
            this->FR_entropy_cumulative += FR_entropy_this_tstep;

            this->pcout << "Convergence reached!" << std::endl;
            this->pcout << "  Entropy at prev timestep (DG) :     " << num_entropy_n << std::endl
                        << "  Entropy at current timestep (DG) :  " << num_entropy_npgamma << std::endl;
            this->pcout << "    Estimate entropy change (M norm): " << entropy_change_est << std::endl
                        << "    Actual entropy change (DG):       " << num_entropy_npgamma - num_entropy_n << std::endl
                        << "    FR contribution:                  " << FR_entropy_contribution << std::endl
                        << "    Corrected entropy change (FR):    " << FR_entropy_this_tstep << std::endl
                        << "  Cumulative entropy change (FR):     " << this->FR_entropy_cumulative << std::endl
                        << std::endl;
        }

        return gamma_kp1;
    }
}


template <int dim, typename real, typename MeshType>
real RootFindingRRKODESolver<dim,real,MeshType>::compute_root_function(
        const double gamma,
        const dealii::LinearAlgebra::distributed::Vector<double> &u_n,
        const dealii::LinearAlgebra::distributed::Vector<double> &step_direction,
        const double num_entropy_n,
        const double entropy_change_est,
        std::shared_ptr<DGBase<dim,real,MeshType>> dg) const
{
    dealii::LinearAlgebra::distributed::Vector<double> temp = u_n;
    temp.add(gamma, step_direction);
    double num_entropy_np1 = compute_numerical_entropy(temp,dg);
    return num_entropy_np1 - num_entropy_n - gamma * entropy_change_est;
}


template <int dim, typename real, typename MeshType>
real RootFindingRRKODESolver<dim,real,MeshType>::compute_numerical_entropy(
        const dealii::LinearAlgebra::distributed::Vector<double> &u,
        std::shared_ptr<DGBase<dim,real,MeshType>> dg) const 
{
    real num_entropy = compute_integrated_numerical_entropy(u,dg);

    return num_entropy;
    
}


template <int dim, typename real, typename MeshType>
real RootFindingRRKODESolver<dim,real,MeshType>::compute_integrated_numerical_entropy(
        const dealii::LinearAlgebra::distributed::Vector<double> &u,
        std::shared_ptr<DGBase<dim,real,MeshType>> dg) const
{
    // This function is reproduced from flow_solver_cases/periodic_turbulence
    // Check that poly_degree is uniform everywhere
    if (dg->get_max_fe_degree() != dg->get_min_fe_degree()) {
        // Note: This function may have issues with nonuniform p. Should test in debug mode if developing in the future.
        this->pcout << "ERROR: compute_integrated_quantities() is untested for nonuniform p. Aborting..." << std::endl;
        std::abort();
    }

    PHiLiP::Parameters::AllParameters parameters_euler = *(dg->all_parameters);
    if (parameters_euler.pde_type != Parameters::AllParameters::PartialDifferentialEquation::euler
            &&
            parameters_euler.pde_type != Parameters::AllParameters::PartialDifferentialEquation::navier_stokes){
        this->pcout << "ERROR: Only implemented for Euler or Navier-Stokes. Aborting..." << std::endl;
        std::abort();
    }
    std::shared_ptr < Physics::Euler<dim, dim+2, double > > euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));
    
    const int nstate = dim+2;
    double integrated_quantity = 0.0;

    const double poly_degree = dg->all_parameters->flow_solver_param.poly_degree;

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;

    OPERATOR::vol_projection_operator<dim,2*dim,double> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], 
                                            dg->oneD_quadrature_collection[poly_degree]);

    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, dg->max_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], 
                                        dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, dg->max_grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, 
                                                         dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, 
                                                         dg->oneD_quadrature_collection[poly_degree], 
                                                         dg->oneD_face_quadrature);

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);
    
    const std::vector<double> &quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();

    // If in the future we need the physical quadrature node location, turn these flags to true and the constructor will
    // automatically compute it for you. Currently set to false as to not compute extra unused terms.
    const bool store_vol_flux_nodes = false;//currently doesn't need the volume physical nodal position
    const bool store_surf_flux_nodes = false;//currently doesn't need the surface physical nodal position

    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    
    // Changed for loop to update metric_cell.
    for (auto cell = dg->dof_handler.begin_active(); cell!= dg->dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->fe_system;
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
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(dg->max_grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, dg->max_grid_degree, store_vol_flux_nodes, store_surf_flux_nodes);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg->all_parameters->use_invariant_curl_form);

        // Fetch the modal soln coefficients
        // We immediately separate them by state as to be able to use sum-factorization
        // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
        // mult would sum the states at the quadrature point.
        // That is why the basis functions are based off the 1state oneD fe_collection.
        std::array<std::vector<double>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
            }
         
            soln_coeff[istate][ishape] = u(dofs_indices[idof]);
        }

        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<std::vector<double>,nstate> soln_at_q_vect;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q_vect[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_vect[istate],
                                             soln_basis.oneD_vol_operator);
        }

        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_at_q;
            // Extract solution and gradient in a way that the physics ca n use them.
            for(int istate=0; istate<nstate; istate++){
                soln_at_q[istate] = soln_at_q_vect[istate][iquad];
            }
            
            //#####################################################################
            // Compute integrated quantities here
            //#####################################################################
            const double quadrature_entropy = euler_physics->compute_numerical_entropy_function(soln_at_q);
            integrated_quantity += quadrature_entropy * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            //#####################################################################
        }
    }
    //MPI
    integrated_quantity = dealii::Utilities::MPI::sum(integrated_quantity, this->mpi_communicator);

    return integrated_quantity;
}

template <int dim, typename real, typename MeshType>
real RootFindingRRKODESolver<dim,real,MeshType>::compute_entropy_change_estimate(const real dt,
        std::shared_ptr<DGBase<dim,real,MeshType>> dg,
        const std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
        const bool use_M_norm_for_entropy_change_est) const
{
    double entropy_change_estimate = 0;

    for (int istage = 0; istage<this->n_rk_stages; ++istage){

        // Recall rk_stage is IMM * RHS
        // therefore, RHS = M * rk_stage = M * du/dt
        dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_rk_stage(dg->solution);
        if(dg->all_parameters->use_inverse_mass_on_the_fly)
        {
            if (use_M_norm_for_entropy_change_est)
                dg->apply_global_mass_matrix(rk_stage[istage], mass_matrix_times_rk_stage,
                        false, // use_auxiliary_eq,
                        true // use_M_norm
                        );
            else
                dg->apply_global_mass_matrix(rk_stage[istage],mass_matrix_times_rk_stage);

        }
        else
            dg->global_mass_matrix.vmult( mass_matrix_times_rk_stage, rk_stage[istage]);
        
        //transform solution into entropy variables
        dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global = this->compute_entropy_vars(this->rk_stage_solution[istage],dg);
        
        double entropy = entropy_var_hat_global * mass_matrix_times_rk_stage;
        
        entropy_change_estimate += this->butcher_tableau->get_b(istage) * entropy;
    }

    return dt * entropy_change_estimate;
}

template class RootFindingRRKODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RootFindingRRKODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RootFindingRRKODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
