#include "periodic_entropy_tests.h"
#include "physics/physics_factory.h"

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// TEST FOR ENTROPY CONSERVATION/STABILITY ON PERIODIC DOMAINS (EULER/NS)
//=========================================================

template <int dim, int nstate>
PeriodicEntropyTests<dim, nstate>::PeriodicEntropyTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicCubeFlow<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&this->all_param));
}

template <int dim, int nstate>
double PeriodicEntropyTests<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    // For Euler simulations, use CFL
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
    // U_infinity is initialized as M_infinity
    const double constant_time_step = this->all_param.flow_solver_param.courant_friedrich_lewy_number * approximate_grid_spacing / this->all_param.euler_param.mach_inf;
    return constant_time_step;
}

template <int dim, int nstate>
double PeriodicEntropyTests<dim, nstate>::compute_entropy(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
    const double poly_degree = this->all_param.flow_solver_param.poly_degree;

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;

    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, poly_degree, dg->max_grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, dg->oneD_quadrature_collection[poly_degree], dg->oneD_face_quadrature);

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);
    
    double integrand_numerical_entropy_function=0;
    double integral_numerical_entropy_function=0;
    const std::vector<double> &quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

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
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, dg->max_grid_degree, false, false);
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
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
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

        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_state;
            // Extract solution and gradient in a way that the physics ca n use them.
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            const double density = soln_state[0];                                                                                                                                                              
            const double pressure = euler_physics->compute_pressure(soln_state);
            const double entropy = log(pressure) - 1.4* log(density);
            const double quadrature_entropy = -density*entropy/0.4;
            //integrand_numerical_entropy_function = this->euler_physics->compute_numerical_entropy_function(soln_state); //SOMEHOW NOT REACHING PHYSICS
            integrand_numerical_entropy_function = quadrature_entropy;
            integral_numerical_entropy_function += integrand_numerical_entropy_function * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
        }
    }
    // update integrated quantities and return
    return dealii::Utilities::MPI::sum(integral_numerical_entropy_function, this->mpi_communicator);
}
    /*
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
    if(dg->all_parameters->use_inverse_mass_on_the_fly)
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
    else
        dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    //std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters);

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
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
            
            //pcout << pressure << " " << entropy << std::endl;        
            entropy_var[0] = (1.4-entropy)/0.4 - 0.5 * density / pressure * vel2;
            for(int idim=0; idim<dim; idim++){
                entropy_var[idim+1] = soln_state[idim+1] / pressure;
            }
            entropy_var[nstate-1] = - density / pressure;
            //pcout << entropy_var[0] << " ";

            for(int istate=0; istate<nstate; istate++){
                if(iquad==0)
                    entropy_var_at_q[istate].resize(n_quad_pts);
                entropy_var_at_q[istate][iquad] = entropy_var[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
                                                
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
            }
            //this->pcout << entropy_var_hat_global[0] << " ";
        }
    }

    double entropy = entropy_var_hat_global * mass_matrix_times_solution;
    //double entropy_mpi = (dealii::Utilities::MPI::sum(entropy, this->mpi_communicator));
    return entropy;
}
*/
template <int dim, int nstate>
void PeriodicEntropyTests<dim, nstate>::compute_unsteady_data_and_write_to_table(
       const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg ,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table )
{
    const double dt = this->get_constant_time_step(dg);
    int output_solution_every_n_iterations = round(this->all_param.ode_solver_param.output_solution_every_dt_time_intervals/dt);
    if (this->all_param.ode_solver_param.output_solution_every_x_steps > output_solution_every_n_iterations)
        output_solution_every_n_iterations = this->all_param.ode_solver_param.output_solution_every_x_steps;
    
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    const bool is_rrk = (this->all_param.ode_solver_param.ode_solver_type == ODEEnum::rrk_explicit_solver);
    const double dt_actual = current_time - previous_time;

    const double entropy = this->compute_entropy(dg);
    if (current_iteration == 0)  initial_entropy = entropy;

    double relaxation_parameter = 0;
    if (is_rrk) relaxation_parameter = dt_actual/dt;

    if ((current_iteration % output_solution_every_n_iterations) == 0){

        this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << std::setprecision(16) << current_time
                    << "    Entropy: " << entropy
                    << "    U/Uo: " << entropy/initial_entropy;
        if (is_rrk)
            this->pcout << "    gamma^n: " << relaxation_parameter;
        this->pcout << std::endl;
    }
    unsteady_data_table->add_value("iteration", current_iteration);
    unsteady_data_table->set_scientific("iteration", false);
    this->add_value_to_data_table(current_time,"time",unsteady_data_table);
    unsteady_data_table->set_scientific("time", false);
    this->add_value_to_data_table(entropy,"entropy",unsteady_data_table);
    unsteady_data_table->set_scientific("entropy", false);
    this->add_value_to_data_table(entropy/initial_entropy,"U/Uo",unsteady_data_table);
    unsteady_data_table->set_scientific("U/Uo", false);
    if (is_rrk) {
        this->add_value_to_data_table(relaxation_parameter,"gamma",unsteady_data_table); 
        unsteady_data_table->set_scientific("gamma", false);
    }
    std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
    unsteady_data_table->write_text(unsteady_data_table_file);

    //for next iteration
    previous_time = current_time;

}

//Only template for Euler/NS
template class PeriodicEntropyTests <PHILIP_DIM,PHILIP_DIM+2>;

} // FlowSolver namespace
} // PHiLiP namespace

