#include "periodic_turbulence.h"

#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor.h>
#include "math.h"
#include <string>
#include <deal.II/base/quadrature_lib.h>

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// TURBULENCE IN PERIODIC CUBE DOMAIN
//=========================================================
template <int dim, int nstate>
PeriodicTurbulence<dim, nstate>::PeriodicTurbulence(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicCubeFlow<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
        , number_of_times_to_output_velocity_field(this->all_param.flow_solver_param.number_of_times_to_output_velocity_field)
        , output_velocity_field_at_fixed_times(this->all_param.flow_solver_param.output_velocity_field_at_fixed_times)
        , output_vorticity_magnitude_field_in_addition_to_velocity(this->all_param.flow_solver_param.output_vorticity_magnitude_field_in_addition_to_velocity)
        , output_flow_field_files_directory_name(this->all_param.flow_solver_param.output_flow_field_files_directory_name)
        , output_solution_at_exact_fixed_times(this->all_param.ode_solver_param.output_solution_at_exact_fixed_times)
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_param.flow_solver_param.flow_case_type;

    // Flow case identifiers
    this->is_taylor_green_vortex = (flow_type == FlowCaseEnum::taylor_green_vortex);
    this->is_decaying_homogeneous_isotropic_turbulence = (flow_type == FlowCaseEnum::decaying_homogeneous_isotropic_turbulence);
    this->is_viscous_flow = (this->all_param.pde_type != Parameters::AllParameters::PartialDifferentialEquation::euler);
    this->do_calculate_numerical_entropy= this->all_param.flow_solver_param.do_calculate_numerical_entropy;

    // Navier-Stokes object; create using dynamic_pointer_cast and the create_Physics factory
    PHiLiP::Parameters::AllParameters parameters_navier_stokes = this->all_param;
    parameters_navier_stokes.pde_type = Parameters::AllParameters::PartialDifferentialEquation::navier_stokes;
    this->navier_stokes_physics = std::dynamic_pointer_cast<Physics::NavierStokes<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_navier_stokes));

    /* Initialize integrated quantities as NAN; 
       done as a precaution in the case compute_integrated_quantities() is not called
       before a member function of kind get_integrated_quantity() is called
     */
    std::fill(this->integrated_quantities.begin(), this->integrated_quantities.end(), NAN);

    /// For outputting velocity field
    if(output_velocity_field_at_fixed_times && (number_of_times_to_output_velocity_field > 0)) {
        exact_output_times_of_velocity_field_files_table = std::make_shared<dealii::TableHandler>();
        this->output_velocity_field_times.reinit(number_of_times_to_output_velocity_field);
        
        // Get output_velocity_field_times from string
        const std::string output_velocity_field_times_string = this->all_param.flow_solver_param.output_velocity_field_times_string;
        std::string line = output_velocity_field_times_string;
        std::string::size_type sz1;
        output_velocity_field_times[0] = std::stod(line,&sz1);
        for(unsigned int i=1; i<number_of_times_to_output_velocity_field; ++i) {
            line = line.substr(sz1);
            sz1 = 0;
            output_velocity_field_times[i] = std::stod(line,&sz1);
        }

        // Get flow_field_quantity_filename_prefix
        flow_field_quantity_filename_prefix = "velocity";
        if(output_vorticity_magnitude_field_in_addition_to_velocity) {
            flow_field_quantity_filename_prefix += std::string("_vorticity");
        }
    }

    this->index_of_current_desired_time_to_output_velocity_field = 0;
    if(this->all_param.flow_solver_param.restart_computation_from_file) {
        // If restarting, get the index of the current desired time to output velocity field based on the initial time
        const double initial_simulation_time = this->all_param.ode_solver_param.initial_time;
        for(unsigned int i=1; i<number_of_times_to_output_velocity_field; ++i) {
            if((output_velocity_field_times[i-1] < initial_simulation_time) && (initial_simulation_time < output_velocity_field_times[i])) {
                this->index_of_current_desired_time_to_output_velocity_field = i;
            }
        }
    }
}

template <int dim, int nstate>
void PeriodicTurbulence<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    std::string flow_type_string;
    if(this->is_taylor_green_vortex || this->is_decaying_homogeneous_isotropic_turbulence) {
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
        this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    }
    this->display_grid_parameters();
}

template <int dim, int nstate>
double PeriodicTurbulence<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    if(this->all_param.flow_solver_param.constant_time_step > 0.0) {
        const double constant_time_step = this->all_param.flow_solver_param.constant_time_step;
        this->pcout << "- - Using constant time step in FlowSolver parameters: " << constant_time_step << std::endl;
        return constant_time_step;
    } else {
        const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
        const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
        const double constant_time_step = this->all_param.flow_solver_param.courant_friedrichs_lewy_number * approximate_grid_spacing;
        return constant_time_step;
    }
}

template <int dim, int nstate>
double PeriodicTurbulence<dim,nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
    const double cfl_number = this->all_param.flow_solver_param.courant_friedrichs_lewy_number;
    const double time_step = cfl_number * approximate_grid_spacing / this->maximum_local_wave_speed;
    return time_step;
}

template <int dim, int nstate>
double PeriodicTurbulence<dim,nstate>::get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg)
{
    // initialize the maximum local wave speed
    update_maximum_local_wave_speed(*dg);
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const double time_step = get_adaptive_time_step(dg);
    return time_step;
}

std::string get_padded_mpi_rank_string(const int mpi_rank_input) {
    // returns the mpi rank as a string with appropriate padding
    std::string mpi_rank_string = std::to_string(mpi_rank_input);
    const unsigned int length_of_mpi_rank_with_padding = 5;
    const int number_of_zeros = length_of_mpi_rank_with_padding - mpi_rank_string.length();
    mpi_rank_string.insert(0, number_of_zeros, '0');

    return mpi_rank_string;
}

template<int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::output_velocity_field(
    std::shared_ptr<DGBase<dim,double>> dg,
    const unsigned int output_file_index,
    const double current_time) const
{
    this->pcout << "  ... Writting velocity field ... " << std::flush;

    // NOTE: Same loop from read_values_from_file_and_project() in set_initial_condition.cpp
    
    // Get filename prefix based on output file index and the flow field quantity filename prefix
    const std::string filename_prefix = flow_field_quantity_filename_prefix + std::string("-") + std::to_string(output_file_index);

    // (1) Get filename based on MPI rank
    //-------------------------------------------------------------
    // -- Get padded mpi rank string
    const std::string mpi_rank_string = get_padded_mpi_rank_string(this->mpi_rank);
    // -- Assemble filename string
    const std::string filename_without_extension = filename_prefix + std::string("-") + mpi_rank_string;
    const std::string filename = output_flow_field_files_directory_name + std::string("/") + filename_without_extension + std::string(".dat");
    //-------------------------------------------------------------

    // (1.5) Write the exact output time for the file to the table 
    //-------------------------------------------------------------
    if(this->mpi_rank==0) {
        const std::string filename_for_time_table = output_flow_field_files_directory_name + std::string("/") + std::string("exact_output_times_of_velocity_field_files.txt");
        // Add values to data table
        this->add_value_to_data_table(output_file_index,"output_file_index",this->exact_output_times_of_velocity_field_files_table);
        this->add_value_to_data_table(current_time,"time",this->exact_output_times_of_velocity_field_files_table);
        // Write to file
        std::ofstream data_table_file(filename_for_time_table);
        this->exact_output_times_of_velocity_field_files_table->write_text(data_table_file);
    }
    //-------------------------------------------------------------

    // (2) Write file
    //-------------------------------------------------------------
    std::ofstream FILE (filename);
    
    // check that the file is open and write DOFs
    if (!FILE.is_open()) {
        this->pcout << "ERROR: Cannot open file " << filename << std::endl;
        std::abort();
    } else if(this->mpi_rank==0) {
        const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
        FILE << number_of_degrees_of_freedom_per_state << std::string("\n");
    }

    // build a basis oneD on equidistant nodes in 1D
    dealii::Quadrature<1> vol_quad_equidistant_1D = dealii::QIterated<1>(dealii::QTrapez<1>(),dg->max_degree);
    dealii::FE_DGQArbitraryNodes<1,1> equidistant_finite_element(vol_quad_equidistant_1D);

    const unsigned int init_grid_degree = dg->high_order_grid->get_current_fe_system().tensor_degree();
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, dg->max_degree, init_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);
    soln_basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);

    // mapping basis for the equidistant node set because we output the physical coordinates
    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis_at_equidistant(1, dg->max_degree, init_grid_degree);
    mapping_basis_at_equidistant.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis_at_equidistant.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, vol_quad_equidistant_1D, dg->oneD_face_quadrature);

    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
        if (!current_cell->is_locally_owned()) continue;
    
        const int i_fele = current_cell->active_fe_index();
        const unsigned int poly_degree = i_fele;
        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_shape_fns = n_dofs_cell / nstate;
        const unsigned int n_quad_pts = n_shape_fns;

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->get_current_fe_system();
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
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(init_grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first;
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second;
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val;
        }
        // Construct the metric operators
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper_equid(nstate, poly_degree, init_grid_degree, true, false);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix.
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper_equid.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis_at_equidistant,
            dg->all_parameters->use_invariant_curl_form);

        
        current_dofs_indices.resize(n_dofs_cell);
        current_cell->get_dof_indices (current_dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0) {
                soln_coeff[istate].resize(n_shape_fns);
            }
            soln_coeff[istate][ishape] = dg->solution(current_dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> soln_grad_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
            // apply gradient of reference basis functions on the solution at volume cubature nodes
            dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
            for(int idim=0; idim<dim; idim++){
                ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
            }
            soln_basis.gradient_matrix_vector_mult_1D(soln_coeff[istate], ref_gradient_basis_fns_times_soln,
                                                      soln_basis.oneD_vol_operator,
                                                      soln_basis.oneD_grad_operator);
            // transform the gradient into a physical gradient operator scaled by determinant of metric Jacobian
            // then apply the inner product in each direction
            for(int idim=0; idim<dim; idim++){
                soln_grad_at_q[istate][idim].resize(n_quad_pts);
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(int jdim=0; jdim<dim; jdim++){
                        //transform into the physical gradient
                        soln_grad_at_q[istate][idim][iquad] += metric_oper_equid.metric_cofactor_vol[idim][jdim][iquad]
                                                            * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                            / metric_oper_equid.det_Jac_vol[iquad];
                    }
                }
            }
        }
        // compute quantities at quad nodes (equisdistant)
        dealii::Tensor<1,dim,std::vector<double>> velocity_at_q;
        std::vector<double> vorticity_magnitude_at_q(n_quad_pts);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            std::array<double,nstate> soln_state;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    soln_grad_state[istate][idim] = soln_grad_at_q[istate][idim][iquad];
                }
            }
            const dealii::Tensor<1,dim,double> velocity = this->navier_stokes_physics->compute_velocities(soln_state);
            for(int idim=0; idim<dim; idim++){
                if(iquad==0)
                    velocity_at_q[idim].resize(n_quad_pts);
                velocity_at_q[idim][iquad] = velocity[idim];
            }
            
            // write vorticity magnitude field if desired
            if(output_vorticity_magnitude_field_in_addition_to_velocity) {
                vorticity_magnitude_at_q[iquad] = this->navier_stokes_physics->compute_vorticity_magnitude(soln_state, soln_grad_state);
            }
        }
        // write out all values at equidistant nodes
        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
            dealii::Point<dim,double> vol_equid_node;
            // write coordinates
            for(int idim=0; idim<dim; idim++) {
                vol_equid_node[idim] = metric_oper_equid.flux_nodes_vol[idim][ishape];
                FILE << std::setprecision(17) << vol_equid_node[idim] << std::string(" ");
            }
            // write velocity field
            for (int d=0; d<dim; ++d) {
                FILE << std::setprecision(17) << velocity_at_q[d][ishape] << std::string(" ");
            }
            // write vorticity magnitude field if desired
            if(output_vorticity_magnitude_field_in_addition_to_velocity) {
                FILE << std::setprecision(17) << vorticity_magnitude_at_q[ishape] << std::string(" ");
            }
            FILE << std::string("\n"); // next line
        }
    }
    FILE.close();
    this->pcout << "done." << std::endl;
}

template<int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::update_maximum_local_wave_speed(DGBase<dim, double> &dg)
{    
    // Initialize the maximum local wave speed to zero
    this->maximum_local_wave_speed = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                                              dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell = dg.dof_handler.begin_active(); cell!=dg.dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            // Update the maximum local wave speed (i.e. convective eigenvalue)
            const double local_wave_speed = this->navier_stokes_physics->max_convective_eigenvalue(soln_at_q);
            if(local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
        }
    }
    this->maximum_local_wave_speed = dealii::Utilities::MPI::max(this->maximum_local_wave_speed, this->mpi_communicator);
}

template<int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::compute_and_update_integrated_quantities(DGBase<dim, double> &dg)
{
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integral_values;
    std::fill(integral_values.begin(), integral_values.end(), 0.0);
    
    // Initialize the maximum local wave speed to zero; only used for adaptive time step
    if(this->all_param.flow_solver_param.adaptive_time_step == true) this->maximum_local_wave_speed = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;

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
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.}
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

            std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrand_values;
            std::fill(integrand_values.begin(), integrand_values.end(), 0.0);
            integrand_values[IntegratedQuantitiesEnum::kinetic_energy] = this->navier_stokes_physics->compute_kinetic_energy_from_conservative_solution(soln_at_q);
            integrand_values[IntegratedQuantitiesEnum::enstrophy] = this->navier_stokes_physics->compute_enstrophy(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::pressure_dilatation] = this->navier_stokes_physics->compute_pressure_dilatation(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::deviatoric_strain_rate_tensor_magnitude_sqr] = this->navier_stokes_physics->compute_deviatoric_strain_rate_tensor_magnitude_sqr(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::strain_rate_tensor_magnitude_sqr] = this->navier_stokes_physics->compute_strain_rate_tensor_magnitude_sqr(soln_at_q,soln_grad_at_q);

            for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
                integral_values[i_quantity] += integrand_values[i_quantity] * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            }

            // Update the maximum local wave speed (i.e. convective eigenvalue) if using an adaptive time step
            if(this->all_param.flow_solver_param.adaptive_time_step == true) {
                const double local_wave_speed = this->navier_stokes_physics->max_convective_eigenvalue(soln_at_q);
                if(local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
            }
        }
    }
    this->maximum_local_wave_speed = dealii::Utilities::MPI::max(this->maximum_local_wave_speed, this->mpi_communicator);

    // update integrated quantities
    for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
        this->integrated_quantities[i_quantity] = dealii::Utilities::MPI::sum(integral_values[i_quantity], this->mpi_communicator);
        this->integrated_quantities[i_quantity] /= this->domain_size; // divide by total domain volume
    }
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_integrated_kinetic_energy() const
{
    const double integrated_kinetic_energy = this->integrated_quantities[IntegratedQuantitiesEnum::kinetic_energy];
    // // Abort if energy is nan
    // if(std::isnan(integrated_kinetic_energy)) {
    //     this->pcout << " ERROR: Kinetic energy at time " << current_time << " is nan." << std::endl;
    //     this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
    //     std::abort();
    // }
    return integrated_kinetic_energy;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_integrated_enstrophy() const
{
    return this->integrated_quantities[IntegratedQuantitiesEnum::enstrophy];
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_vorticity_based_dissipation_rate() const
{
    const double integrated_enstrophy = this->integrated_quantities[IntegratedQuantitiesEnum::enstrophy];
    double vorticity_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        vorticity_based_dissipation_rate = this->navier_stokes_physics->compute_vorticity_based_dissipation_rate_from_integrated_enstrophy(integrated_enstrophy);
    }
    return vorticity_based_dissipation_rate;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_pressure_dilatation_based_dissipation_rate() const
{
    const double integrated_pressure_dilatation = this->integrated_quantities[IntegratedQuantitiesEnum::pressure_dilatation];
    return (-1.0*integrated_pressure_dilatation); // See reference (listed in header file), equation (57b)
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_deviatoric_strain_rate_tensor_based_dissipation_rate() const
{
    const double integrated_deviatoric_strain_rate_tensor_magnitude_sqr = this->integrated_quantities[IntegratedQuantitiesEnum::deviatoric_strain_rate_tensor_magnitude_sqr];
    double deviatoric_strain_rate_tensor_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        deviatoric_strain_rate_tensor_based_dissipation_rate = 
            this->navier_stokes_physics->compute_deviatoric_strain_rate_tensor_based_dissipation_rate_from_integrated_deviatoric_strain_rate_tensor_magnitude_sqr(integrated_deviatoric_strain_rate_tensor_magnitude_sqr);
    }
    return deviatoric_strain_rate_tensor_based_dissipation_rate;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_strain_rate_tensor_based_dissipation_rate() const
{
    const double integrated_strain_rate_tensor_magnitude_sqr = this->integrated_quantities[IntegratedQuantitiesEnum::strain_rate_tensor_magnitude_sqr];
    double strain_rate_tensor_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        strain_rate_tensor_based_dissipation_rate = 
            this->navier_stokes_physics->compute_strain_rate_tensor_based_dissipation_rate_from_integrated_strain_rate_tensor_magnitude_sqr(integrated_strain_rate_tensor_magnitude_sqr);
    }
    return strain_rate_tensor_based_dissipation_rate;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_numerical_entropy(
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

    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    // Changed for loop to update metric_cell.
    for (auto cell = dg->dof_handler.begin_active(); cell!= dg->dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);
        
        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->get_current_fe_system();
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
            integrand_numerical_entropy_function = this->navier_stokes_physics->compute_numerical_entropy_function(soln_state);
            integral_numerical_entropy_function += integrand_numerical_entropy_function * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
        }
    }
    // update integrated quantities and return
    return dealii::Utilities::MPI::sum(integral_numerical_entropy_function, this->mpi_communicator);
}

template <int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    // Compute and update integrated quantities
    this->compute_and_update_integrated_quantities(*dg);
    // Get computed quantities
    const double integrated_kinetic_energy = this->get_integrated_kinetic_energy();
    const double integrated_enstrophy = this->get_integrated_enstrophy();
    const double vorticity_based_dissipation_rate = this->get_vorticity_based_dissipation_rate();
    const double pressure_dilatation_based_dissipation_rate = this->get_pressure_dilatation_based_dissipation_rate();
    const double deviatoric_strain_rate_tensor_based_dissipation_rate = this->get_deviatoric_strain_rate_tensor_based_dissipation_rate();
    const double strain_rate_tensor_based_dissipation_rate = this->get_strain_rate_tensor_based_dissipation_rate();
    
    double numerical_entropy = 0;
    if (do_calculate_numerical_entropy) numerical_entropy = this->get_numerical_entropy(dg);

    if(this->mpi_rank==0) {
        // Add values to data table
        this->add_value_to_data_table(current_time,"time",unsteady_data_table);
        if(do_calculate_numerical_entropy) this->add_value_to_data_table(numerical_entropy,"numerical_entropy",unsteady_data_table);
        this->add_value_to_data_table(integrated_kinetic_energy,"kinetic_energy",unsteady_data_table);
        this->add_value_to_data_table(integrated_enstrophy,"enstrophy",unsteady_data_table);
        if(is_viscous_flow) this->add_value_to_data_table(vorticity_based_dissipation_rate,"eps_vorticity",unsteady_data_table);
        this->add_value_to_data_table(pressure_dilatation_based_dissipation_rate,"eps_pressure",unsteady_data_table);
        if(is_viscous_flow) this->add_value_to_data_table(strain_rate_tensor_based_dissipation_rate,"eps_strain",unsteady_data_table);
        if(is_viscous_flow) this->add_value_to_data_table(deviatoric_strain_rate_tensor_based_dissipation_rate,"eps_dev_strain",unsteady_data_table);
        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }
    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Energy: " << integrated_kinetic_energy
                << "    Enstrophy: " << integrated_enstrophy;
    if(is_viscous_flow) {
        this->pcout << "    eps_vorticity: " << vorticity_based_dissipation_rate
                    << "    eps_p+eps_strain: " << (pressure_dilatation_based_dissipation_rate + strain_rate_tensor_based_dissipation_rate);
    }
    if(do_calculate_numerical_entropy){
        this->pcout << "    Num. Entropy: " << std::setprecision(16) << numerical_entropy;
    }
    this->pcout << std::endl;

    // Abort if energy is nan
    if(std::isnan(integrated_kinetic_energy)) {
        this->pcout << " ERROR: Kinetic energy at time " << current_time << " is nan." << std::endl;
        this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }

    // Output velocity field for spectra obtaining kinetic energy spectra
    if(output_velocity_field_at_fixed_times) {
        const double time_step = this->get_time_step();
        const double next_time = current_time + time_step;
        const double desired_time = this->output_velocity_field_times[this->index_of_current_desired_time_to_output_velocity_field];
        // Check if current time is an output time
        bool is_output_time = false; // default initialization
        if(this->output_solution_at_exact_fixed_times) {
            is_output_time = current_time == desired_time;
        } else {
            is_output_time = ((current_time<=desired_time) && (next_time>desired_time));
        }
        if(is_output_time) {
            // Output velocity field for current index
            this->output_velocity_field(dg, this->index_of_current_desired_time_to_output_velocity_field, current_time);
            
            // Update index s.t. it never goes out of bounds
            if(this->index_of_current_desired_time_to_output_velocity_field 
                < (this->number_of_times_to_output_velocity_field-1)) {
                this->index_of_current_desired_time_to_output_velocity_field += 1;
            }
        }
    }
}

#if PHILIP_DIM==3
template class PeriodicTurbulence <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

