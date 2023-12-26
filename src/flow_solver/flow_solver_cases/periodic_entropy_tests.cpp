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
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
            PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&(this->all_param)));
}

template <int dim, int nstate>
double PeriodicEntropyTests<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_case = this->all_param.flow_solver_param.flow_case_type;
    const double CFL = this->all_param.flow_solver_param.courant_friedrichs_lewy_number;
    // For Euler simulations, use CFL
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));

    double constant_time_step=0;
    if (flow_case == FlowCaseEnum::isentropic_vortex){
        // Using dt = CFL * delta_x/U_infinity, consistent with Ranocha's choice (Relaxation Runge Kutta methods... 2020)
        // U_infinity is initialized as M_infinity
        constant_time_step = CFL * approximate_grid_spacing / this->all_param.euler_param.mach_inf;
    } else if (flow_case == FlowCaseEnum::kelvin_helmholtz_instability){
        /*
        const double max_wave_speed = this->compute_integrated_quantities(*dg, IntegratedQuantityEnum::max_wave_speed);
        constant_time_step = CFL * approximate_grid_spacing / max_wave_speed;
        */
        // TEMP using same as is defined in periodic turbulence for consistency with some existing results
        const double constant_time_step = this->all_param.flow_solver_param.courant_friedrichs_lewy_number * approximate_grid_spacing;
        return constant_time_step;
    } else{
        this->pcout << "Timestep size has not been defined in periodic_entropy_tests for this flow_case_type. Aborting..." << std::endl;
        std::abort();
    }

    return constant_time_step;
}

template<int dim, int nstate>
double PeriodicEntropyTests<dim, nstate>::compute_integrated_quantities(DGBase<dim, double> &dg, IntegratedQuantityEnum quantity, const int overintegrate) const
{
    // Check that poly_degree is uniform everywhere
    if (dg.get_max_fe_degree() != dg.get_min_fe_degree()) {
        // Note: This function may have issues with nonuniform p. Should test in debug mode if developing in the future.
        this->pcout << "ERROR: compute_integrated_quantities() is untested for nonuniform p. Aborting..." << std::endl;
        std::abort();
    }

    double integrated_quantity = 0.0;

    // Set the quadrature of size dim and 1D for sum-factorization.
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::QGauss<1> quad_extra_1D(dg.max_degree+1+overintegrate);

    const unsigned int n_quad_pts = quad_extra.size();
    const unsigned int grid_degree = dg.high_order_grid->get_current_fe_system().tensor_degree();
    const unsigned int poly_degree = dg.max_degree;
    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, grid_degree); 
    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, poly_degree, grid_degree);
    // Build basis function volume operator and gradient operator from 1D finite element for 1 state.
    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    soln_basis.build_1D_gradient_operator(dg.oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    // Build mapping shape functions operators using the oneD high_ordeR_grid finite element
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg.high_order_grid->oneD_fe_system, dg.high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg.high_order_grid->oneD_fe_system, quad_extra_1D, dg.oneD_face_quadrature);
    const std::vector<double> &quad_weights = quad_extra.get_weights();
    // If in the future we need the physical quadrature node location, turn these flags to true and the constructor will
    // automatically compute it for you. Currently set to false as to not compute extra unused terms.
    const bool store_vol_flux_nodes = false;//currently doesn't need the volume physical nodal position
    const bool store_surf_flux_nodes = false;//currently doesn't need the surface physical nodal position

    const unsigned int n_dofs = dg.fe_collection[poly_degree].n_dofs_per_cell();
    const unsigned int n_shape_fns = n_dofs / nstate;
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs);
    auto metric_cell = dg.high_order_grid->dof_handler_grid.begin_active();
    // Changed for loop to update metric_cell.
    for (auto cell = dg.dof_handler.begin_active(); cell!= dg.dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg.high_order_grid->get_current_fe_system();
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
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg.high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, grid_degree, store_vol_flux_nodes, store_surf_flux_nodes);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg.all_parameters->use_invariant_curl_form);

        // Fetch the modal soln coefficients
        // We immediately separate them by state as to be able to use sum-factorization
        // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
        // mult would sum the states at the quadrature point.
        // That is why the basis functions are based off the 1state oneD fe_collection.
        std::array<std::vector<double>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs; ++idof) {
            const unsigned int istate = dg.fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg.fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
            }
         
            soln_coeff[istate][ishape] = dg.solution(dofs_indices[idof]);
        }
        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<std::vector<double>,nstate> soln_at_q_vect;
        std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> soln_grad_at_q_vect;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q_vect[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_vect[istate],
                                             soln_basis.oneD_vol_operator);
            // We need to first compute the reference gradient of the solution, then transform that to a physical gradient.
            dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
            for(int idim=0; idim<dim; idim++){
                ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
                soln_grad_at_q_vect[istate][idim].resize(n_quad_pts);
            }
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.
            soln_basis.gradient_matrix_vector_mult_1D(soln_coeff[istate], ref_gradient_basis_fns_times_soln,
                                                      soln_basis.oneD_vol_operator,
                                                      soln_basis.oneD_grad_operator);
            // Transform the reference gradient into a physical gradient operator.
            for(int idim=0; idim<dim; idim++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(int jdim=0; jdim<dim; jdim++){
                        //transform into the physical gradient
                        soln_grad_at_q_vect[istate][idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                                  * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                                  / metric_oper.det_Jac_vol[iquad];
                    }
                }
            }
        }

        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_at_q;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;
            // Extract solution and gradient in a way that the physics ca n use them.
            for(int istate=0; istate<nstate; istate++){
                soln_at_q[istate] = soln_at_q_vect[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    soln_grad_at_q[istate][idim] = soln_grad_at_q_vect[istate][idim][iquad];
                }
            }
            
            //#####################################################################
            // Compute integrated quantities here
            //#####################################################################
            if (quantity == IntegratedQuantityEnum::kinetic_energy) { 
                const double KE_integrand = this->euler_physics->compute_kinetic_energy_from_conservative_solution(soln_at_q);
                integrated_quantity += KE_integrand * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            } else if (quantity == IntegratedQuantityEnum::numerical_entropy) {
                const double quadrature_entropy = this->euler_physics->compute_numerical_entropy_function(soln_at_q);
                //Using std::cout because of cell->is_locally_owned check 
                if (isnan(quadrature_entropy))  std::cout << "WARNING: NaN entropy detected at a node!"  << std::endl;
                integrated_quantity += quadrature_entropy * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            } else if (quantity == IntegratedQuantityEnum::max_wave_speed) {
                const double local_wave_speed = this->euler_physics->max_convective_eigenvalue(soln_at_q);
                if(local_wave_speed > integrated_quantity) integrated_quantity = local_wave_speed;
            } else {
                std::cout << "Integrated quantity is not correctly defined." << std::endl;
            }
            //#####################################################################
        }
    }

    //MPI
    if (quantity == IntegratedQuantityEnum::max_wave_speed) {
        integrated_quantity = dealii::Utilities::MPI::max(integrated_quantity, this->mpi_communicator);
    } else {
        integrated_quantity = dealii::Utilities::MPI::sum(integrated_quantity, this->mpi_communicator);
    }

    return integrated_quantity;
}


template <int dim, int nstate>
double PeriodicEntropyTests<dim, nstate>::compute_entropy(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
     return compute_integrated_quantities(*dg, IntegratedQuantityEnum::numerical_entropy, 0);
}

template <int dim, int nstate>
void PeriodicEntropyTests<dim, nstate>::compute_unsteady_data_and_write_to_table(
       const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg ,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table )
{
    const double dt = this->get_constant_time_step(dg);
    
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    const bool is_rrk = (this->all_param.ode_solver_param.ode_solver_type == ODEEnum::rrk_explicit_solver);
    const double dt_actual = current_time - previous_time;

    // All discrete proofs use solution nodes, therefore it is best to report 
    // entropy on the solution nodes rather than by overintegrating.
    const double entropy = this->compute_integrated_quantities(*dg, IntegratedQuantityEnum::numerical_entropy, 0); //do not overintegrate
    if (std::isnan(entropy)){
        // Note that this throws an exception rather than using abort()
        // such that the test khi_robustness can start another test after
        // an expected crash.
        this->pcout << "Entropy is nan. Ending flow simulation by throwing an exception..." << std::endl << std::flush;
        throw current_time;
    }
    if (current_iteration == 0)  initial_entropy = entropy;

    double relaxation_parameter = 0;
    if (is_rrk) relaxation_parameter = dt_actual/dt;

    const double kinetic_energy = this->compute_integrated_quantities(*dg, IntegratedQuantityEnum::kinetic_energy);
    if (std::isnan(kinetic_energy)){
        this->pcout << "Kinetic energy is nan. Ending flow simulation by throwing an exception..." << std::endl << std::flush;
        throw current_time;
    }

    // Output solution to console according to output_solution_every_n_iterations
    int output_solution_every_n_iterations = round(this->all_param.ode_solver_param.output_solution_every_dt_time_intervals/dt);
    if (this->all_param.ode_solver_param.output_solution_every_x_steps > output_solution_every_n_iterations)
        output_solution_every_n_iterations = this->all_param.ode_solver_param.output_solution_every_x_steps;
    if (output_solution_every_n_iterations > 0){
        //Need to check that output_solution_every_n_iterations is nonzero to avoid
        //floating point exception
        if ((current_iteration % output_solution_every_n_iterations) == 0){

            this->pcout << "    Iter: " << current_iteration
                        << "    Time: " << std::setprecision(16) << current_time
                        << "    Entropy: " << entropy
                        << "    U/Uo: " << entropy/initial_entropy
                        << "    Kinetic energy: " << kinetic_energy;
            if (is_rrk)
                this->pcout << "    gamma^n: " << relaxation_parameter;
            this->pcout << std::endl;
        }
    }

    // Write to file at every iteration
    unsteady_data_table->add_value("iteration", current_iteration);
    unsteady_data_table->set_scientific("iteration", false);
    this->add_value_to_data_table(current_time,"time",unsteady_data_table);
    unsteady_data_table->set_scientific("time", false);
    this->add_value_to_data_table(entropy,"entropy",unsteady_data_table);
    unsteady_data_table->set_scientific("entropy", false);
    this->add_value_to_data_table(entropy/initial_entropy,"U/Uo",unsteady_data_table);
    unsteady_data_table->set_scientific("U/Uo", false);
    this->add_value_to_data_table(kinetic_energy,"kinetic_energy",unsteady_data_table);
    unsteady_data_table->set_scientific("kinetic_energy", false);
    if (is_rrk) {
        this->add_value_to_data_table(relaxation_parameter,"gamma",unsteady_data_table); 
        unsteady_data_table->set_scientific("gamma", false);
    }
    std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
    unsteady_data_table->write_text(unsteady_data_table_file);

    //for next iteration
    previous_time = current_time;

}

#if PHILIP_DIM>1
    template class PeriodicEntropyTests <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

