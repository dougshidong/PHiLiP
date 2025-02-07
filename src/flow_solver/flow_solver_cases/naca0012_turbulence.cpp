#include "naca0012_turbulence.h"
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include <deal.II/base/table_handler.h>
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/gmsh_reader.hpp"
#include "functional/lift_drag.hpp"
#include "functional/extraction_functional.hpp"
#include "functional/amiet_model.hpp"

namespace PHiLiP {
namespace FlowSolver {
//=========================================================
// NACA0012
//=========================================================
template <int dim, int nstate>
NACA0012_LES<dim, nstate>::NACA0012_LES(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
        , number_of_times_to_output_velocity_field(this->all_param.flow_solver_param.number_of_times_to_output_velocity_field)
        , output_velocity_field_at_fixed_times(this->all_param.flow_solver_param.output_velocity_field_at_fixed_times)
        , output_vorticity_magnitude_field_in_addition_to_velocity(this->all_param.flow_solver_param.output_vorticity_magnitude_field_in_addition_to_velocity)
        , output_flow_field_files_directory_name(this->all_param.flow_solver_param.output_flow_field_files_directory_name)
        , output_solution_at_exact_fixed_times(this->all_param.ode_solver_param.output_solution_at_exact_fixed_times)
        , output_density_field_in_addition_to_velocity(this->all_param.flow_solver_param.output_density_field_in_addition_to_velocity)
        , output_viscosity_field_in_addition_to_velocity(this->all_param.flow_solver_param.output_viscosity_field_in_addition_to_velocity)
        , compute_time_averaged_solution(this->all_param.flow_solver_param.compute_time_averaged_solution)
        , time_to_start_averaging(this->all_param.flow_solver_param.time_to_start_averaging)
{
    
    // Navier-Stokes object; create using dynamic_pointer_cast and the create_Physics factory
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PHiLiP::Parameters::AllParameters parameters_navier_stokes = this->all_param;
    parameters_navier_stokes.pde_type = PDE_enum::navier_stokes;
    this->navier_stokes_physics = std::dynamic_pointer_cast<Physics::NavierStokes<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_navier_stokes));

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
void NACA0012_LES<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    if (pde_type == PDE_enum::navier_stokes){
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    }
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    const double pi = atan(1.0) * 4.0;
    this->pcout << "- - Angle of attack [deg]: " << this->all_param.euler_param.angle_of_attack*180/pi << std::endl;
    this->pcout << "- - Side-slip angle [deg]: " << this->all_param.euler_param.side_slip_angle*180/pi << std::endl;
    this->pcout << "- - Farfield conditions: " << std::endl;
    const dealii::Point<dim> dummy_point;
    for (int s=0;s<nstate;s++) {
        this->pcout << "- - - State " << s << "; Value: " << this->initial_condition_function->value(dummy_point, s) << std::endl;
    }
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NACA0012_LES<dim,nstate>::generate_grid() const
{
    //Dummy triangulation
    if constexpr(dim==2) {
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );
        dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
        dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
        grid->refine_global();
        return grid;
    } 
    else if constexpr(dim==3) {
        const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
        const bool use_mesh_smoothing = false;
        std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename, 
                this->all_param.flow_solver_param.use_periodic_BC_in_x, 
                this->all_param.flow_solver_param.use_periodic_BC_in_y, 
                this->all_param.flow_solver_param.use_periodic_BC_in_z, 
                this->all_param.flow_solver_param.x_periodic_id_face_1, 
                this->all_param.flow_solver_param.x_periodic_id_face_2, 
                this->all_param.flow_solver_param.y_periodic_id_face_1, 
                this->all_param.flow_solver_param.y_periodic_id_face_2, 
                this->all_param.flow_solver_param.z_periodic_id_face_1, 
                this->all_param.flow_solver_param.z_periodic_id_face_2,
                this->all_param.flow_solver_param.mesh_reader_verbose_output, 
                this->all_param.do_renumber_dofs, 0, use_mesh_smoothing);
        
        return naca0012_mesh->triangulation;
    }
    
    // TO DO: Avoid reading the mesh twice (here and in set_high_order_grid -- need a default dummy triangulation)
}

template <int dim, int nstate>
void NACA0012_LES<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    const bool use_mesh_smoothing = false;
    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename, this->all_param.do_renumber_dofs, 0, use_mesh_smoothing);
    dg->set_high_order_grid(naca0012_mesh);
    for (int i=0; i<this->all_param.flow_solver_param.number_of_mesh_refinements; ++i) {
        dg->high_order_grid->refine_global();
    }
}

template <int dim, int nstate>
double NACA0012_LES<dim,nstate>::compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> lift_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::lift);
    const double lift = lift_functional.evaluate_functional();
    return lift;
}

template <int dim, int nstate>
double NACA0012_LES<dim,nstate>::compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> drag_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::drag);
    const double drag = drag_functional.evaluate_functional();
    return drag;
}


std::string get_padded_mpi_rank_strings(const int mpi_rank_input) {
    // returns the mpi rank as a string with appropriate padding
    std::string mpi_rank_string = std::to_string(mpi_rank_input);
    const unsigned int length_of_mpi_rank_with_padding = 5;
    const int number_of_zeros = length_of_mpi_rank_with_padding - mpi_rank_string.length();
    mpi_rank_string.insert(0, number_of_zeros, '0');

    return mpi_rank_string;
}

template<int dim, int nstate>
void NACA0012_LES<dim, nstate>::output_velocity_field(
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
    const std::string mpi_rank_string = get_padded_mpi_rank_strings(this->mpi_rank);
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

    const unsigned int higher_poly_degree = /*this->output_velocity_number_of_subvisions*/2*(dg->max_degree+1)-1; // Note: -1 so that n_quad_pts in 1D is n_subdiv*(P+1)

    // check that the file is open and write DOFs
    if (!FILE.is_open()) {
        this->pcout << "ERROR: Cannot open file " << filename << std::endl;
        std::abort();
    } else if(this->mpi_rank==0) {
        //const unsigned int number_of_degrees_of_freedom_per_state = this->get_number_of_degrees_of_freedom_per_state_from_poly_degree(higher_poly_degree);
        //FILE << number_of_degrees_of_freedom_per_state << std::string("\n");
    }

    // build a basis oneD on equidistant nodes in 1D
    dealii::Quadrature<1> vol_quad_equidistant_1D = dealii::QIterated<1>(dealii::QTrapez<1>(),higher_poly_degree);
    const unsigned int n_quad_pts = pow(vol_quad_equidistant_1D.size(),dim);

    const unsigned int init_grid_degree = dg->high_order_grid->fe_system.tensor_degree();
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, dg->max_degree, init_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);
    soln_basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);

    // mapping basis for the equidistant node set because we output the physical coordinates
    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis_at_equidistant(1, dg->max_degree, init_grid_degree);
    mapping_basis_at_equidistant.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis_at_equidistant.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, vol_quad_equidistant_1D, dg->oneD_face_quadrature);

    double kinetic_energy_sum_P1 = 0;
    double kinetic_energy_index_P1 = 0;

    // Loop over all cells
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
        if (!current_cell->is_locally_owned()) continue;

        const int i_fele = current_cell->active_fe_index();
        const unsigned int poly_degree = i_fele;
        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_shape_fns = n_dofs_cell / nstate;

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
        std::array<std::vector<double>,nstate> time_averaged_soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0) {
                soln_coeff[istate].resize(n_shape_fns);
                time_averaged_soln_coeff[istate].resize(n_shape_fns);
            }
            soln_coeff[istate][ishape] = dg->solution(current_dofs_indices[idof]);
            time_averaged_soln_coeff[istate][ishape] = dg->time_averaged_solution(current_dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        std::array<std::vector<double>,nstate> time_averaged_soln_at_q;
        std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> soln_grad_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            time_averaged_soln_at_q[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(time_averaged_soln_coeff[istate], time_averaged_soln_at_q[istate],
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
        dealii::Tensor<1,dim,std::vector<double>> time_averaged_velocity_at_q;
        dealii::Tensor<1,dim,std::vector<double>> velocity_fluctuations_at_q;
        //dealii::Tensor<1,dim,std::vector<double>> velocity_fluctuations_at_q;
        std::vector<double> vorticity_magnitude_at_q(n_quad_pts);
        std::vector<double> density_at_q(n_quad_pts);
        std::vector<double> viscosity_at_q(n_quad_pts);
        std::vector<double> kinetic_energy_at_q(n_quad_pts);
        std::vector<double> enstrophy_at_q(n_quad_pts);
        std::vector<double> pressure_dilatation_at_q(n_quad_pts);      
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            std::array<double,nstate> soln_state;
            std::array<double,nstate> time_averaged_soln_state;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
                time_averaged_soln_state[istate] = time_averaged_soln_at_q[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    soln_grad_state[istate][idim] = soln_grad_at_q[istate][idim][iquad];
                }
            }
            const dealii::Tensor<1,dim,double> velocity = this->navier_stokes_physics->compute_velocities(soln_state);
            const dealii::Tensor<1,dim,double> time_averaged_velocity = this->navier_stokes_physics->compute_velocities(time_averaged_soln_state);
            for(int idim=0; idim<dim; idim++){
                if(iquad==0){
                    velocity_at_q[idim].resize(n_quad_pts);
                    time_averaged_velocity_at_q[idim].resize(n_quad_pts);
                    velocity_fluctuations_at_q[idim].resize(n_quad_pts);
                }
                velocity_at_q[idim][iquad] = velocity[idim];
                time_averaged_velocity_at_q[idim][iquad] = time_averaged_velocity[idim];
                velocity_fluctuations_at_q[idim][iquad] = velocity_at_q[idim][iquad] - time_averaged_velocity_at_q[idim][iquad];
            } 
            // write vorticity magnitude field if desired
            if(output_vorticity_magnitude_field_in_addition_to_velocity) {
                vorticity_magnitude_at_q[iquad] = this->navier_stokes_physics->compute_vorticity_magnitude(soln_state, soln_grad_state);
            }
            // write density field if desired
            if(output_density_field_in_addition_to_velocity) {
                density_at_q[iquad] = soln_state[0];
            }
            // write viscosity field if desired
            if(output_viscosity_field_in_addition_to_velocity) {
                const std::array<double,nstate> primitive_soln = this->navier_stokes_physics->convert_conservative_to_primitive(soln_state);
                viscosity_at_q[iquad] = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln);
            }
            kinetic_energy_at_q[iquad] = this->navier_stokes_physics->compute_kinetic_energy_from_conservative_solution(soln_state);
            enstrophy_at_q[iquad] = this->navier_stokes_physics->compute_enstrophy(soln_state,soln_grad_state);
            pressure_dilatation_at_q[iquad] = this->navier_stokes_physics->compute_pressure_dilatation(soln_state,soln_grad_state);
        }
        // write out all values at equidistant nodes
        for(unsigned int ishape=0; ishape<n_quad_pts; ishape++){
            dealii::Point<dim,double> vol_equid_node;
            // write coordinates
            for(int idim=0; idim<dim; idim++) {
                vol_equid_node[idim] = metric_oper_equid.flux_nodes_vol[idim][ishape];
                FILE << std::setprecision(17) << vol_equid_node[idim] << std::string(" ");
            }
            const double precision = 2E-3;
            // if( ((0.5-precision)<vol_equid_node[0] && vol_equid_node[0]<(0.5+precision)) && ((0.11-precision)<vol_equid_node[1] && vol_equid_node[1]<(0.11+precision)) ){
            //     for(int idim=0; idim<dim; idim++) {
            //         std::cout << "vol_equid_node["<<idim<<"]: " << vol_equid_node[idim] << std::endl;
            //     }
            // } 
            if( ((0.5-precision)<vol_equid_node[0] && vol_equid_node[0]<(0.5+precision)) && ((0.05-precision)<vol_equid_node[1] && vol_equid_node[1]<(0.05+precision)) ){
                for(int idim=0; idim<dim; idim++) {
                    std::cout << "vol_equid_node["<<idim<<"]: " << vol_equid_node[idim] << std::endl;
                }
                std::cout << "kinetic energy at q: " << kinetic_energy_at_q[ishape] << std::endl;
                kinetic_energy_sum_P1 +=  kinetic_energy_at_q[ishape];
                kinetic_energy_index_P1 += 1;
            }
            // // write velocity field
            // for (int d=0; d<dim; ++d) {
            //     FILE << std::setprecision(17) << velocity_at_q[d][ishape] << std::string(" ");
            // }
            // // write velocity fluctuations field
            // if(compute_time_averaged_solution && current_time >= time_to_start_averaging) {
            //     for (int d=0; d<dim; ++d) {
            //         FILE << std::setprecision(17) << velocity_fluctuations_at_q[d][ishape] << std::string(" ");
            //     }
            // }
            // // write vorticity magnitude field if desired
            // if(output_vorticity_magnitude_field_in_addition_to_velocity) {
            //     FILE << std::setprecision(17) << vorticity_magnitude_at_q[ishape] << std::string(" ");
            // }
            // // write density field if desired
            // if(output_density_field_in_addition_to_velocity) {
            //     FILE << std::setprecision(17) << density_at_q[ishape] << std::string(" ");
            // }
            // // write viscosity field if desired
            // if(output_viscosity_field_in_addition_to_velocity) {
            //     FILE << std::setprecision(17) << viscosity_at_q[ishape] << std::string(" ");
            // }
            FILE << std::string("\n"); // next line
        }
    }//End of cell loop
    // write vkinetic energy average at point
    double kinetic_energy_mpi_sum_P1 = dealii::Utilities::MPI::sum(kinetic_energy_sum_P1, this->mpi_communicator);
    double kinetic_energy_mpi_index_P1 = dealii::Utilities::MPI::sum(kinetic_energy_index_P1, this->mpi_communicator);
    FILE << std::setprecision(17) << kinetic_energy_mpi_sum_P1/kinetic_energy_mpi_index_P1 << std::string(" ");
    FILE.close();
    this->pcout << "done." << std::endl;
}

template<int dim, int nstate>
void NACA0012_LES<dim, nstate>::output_kinetic_energy_at_points(
    std::shared_ptr<DGBase<dim,double>> dg,
    const double current_time,
    const dealii::Point<dim,double> P1,
    const dealii::Point<dim,double> P2,
    const dealii::Point<dim,double> P3,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const
{
    const unsigned int higher_poly_degree = /*this->output_velocity_number_of_subvisions*/2*(dg->max_degree+1)-1; // Note: -1 so that n_quad_pts in 1D is n_subdiv*(P+1)
       // build a basis oneD on equidistant nodes in 1D
    dealii::Quadrature<1> vol_quad_equidistant_1D = dealii::QIterated<1>(dealii::QTrapez<1>(),higher_poly_degree);
    const unsigned int n_quad_pts = pow(vol_quad_equidistant_1D.size(),dim);

    const unsigned int init_grid_degree = dg->high_order_grid->fe_system.tensor_degree();
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, dg->max_degree, init_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);
    soln_basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);

    // mapping basis for the equidistant node set because we output the physical coordinates
    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis_at_equidistant(1, dg->max_degree, init_grid_degree);
    mapping_basis_at_equidistant.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis_at_equidistant.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, vol_quad_equidistant_1D, dg->oneD_face_quadrature);

    //Instantaneous kinetic energy at three points
    double kinetic_energy_sum_P1 = 0;
    double kinetic_energy_index_P1 = 0;
    double kinetic_energy_sum_P2 = 0;
    double kinetic_energy_index_P2 = 0;
    double kinetic_energy_sum_P3 = 0;
    double kinetic_energy_index_P3 = 0;

    //Time-averaged kinetic energy and velocity magnitude at Point 1
    double time_averaged_kinetic_energy_sum_P1 = 0;
    double time_averaged_kinetic_energy_index_P1 = 0;
    double time_averaged_velocity_magnitude_sum_P1 = 0;
    double time_averaged_velocity_magnitude_index_P1 = 0;

    //Time-averaged kinetic energy and velocity magnitude at Point 2
    double time_averaged_kinetic_energy_sum_P2 = 0;
    double time_averaged_kinetic_energy_index_P2 = 0;
    double time_averaged_velocity_magnitude_sum_P2 = 0;
    double time_averaged_velocity_magnitude_index_P2 = 0;

    //Time-averaged kinetic energy and velocity magnitude at Point 3
    double time_averaged_kinetic_energy_sum_P3 = 0;
    double time_averaged_kinetic_energy_index_P3 = 0;
    double time_averaged_velocity_magnitude_sum_P3 = 0;
    double time_averaged_velocity_magnitude_index_P3 = 0;
    // Loop over all cells
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
        if (!current_cell->is_locally_owned()) continue;

        const int i_fele = current_cell->active_fe_index();
        const unsigned int poly_degree = i_fele;
        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_shape_fns = n_dofs_cell / nstate;

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
        std::array<std::vector<double>,nstate> time_averaged_soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0) {
                soln_coeff[istate].resize(n_shape_fns);
                time_averaged_soln_coeff[istate].resize(n_shape_fns);
            }
            soln_coeff[istate][ishape] = dg->solution(current_dofs_indices[idof]);
            time_averaged_soln_coeff[istate][ishape] = dg->time_averaged_solution(current_dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        std::array<std::vector<double>,nstate> time_averaged_soln_at_q;
        std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> soln_grad_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            time_averaged_soln_at_q[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(time_averaged_soln_coeff[istate], time_averaged_soln_at_q[istate],
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
        //dealii::Tensor<1,dim,std::vector<double>> velocity_fluctuations_at_q;
        std::vector<double> kinetic_energy_at_q(n_quad_pts);
        std::vector<double> time_averaged_kinetic_energy_at_q(n_quad_pts);
        std::vector<double> velocity_magnitude_at_q(n_quad_pts); 
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            std::array<double,nstate> soln_state;
            std::array<double,nstate> time_averaged_soln_state;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
                time_averaged_soln_state[istate] = time_averaged_soln_at_q[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    soln_grad_state[istate][idim] = soln_grad_at_q[istate][idim][iquad];
                }
            }
            kinetic_energy_at_q[iquad] = this->navier_stokes_physics->compute_kinetic_energy_from_conservative_solution(soln_state);
            time_averaged_kinetic_energy_at_q[iquad] = this->navier_stokes_physics->compute_kinetic_energy_from_conservative_solution(time_averaged_soln_state);
            const dealii::Tensor<1,dim,double> velocity = this->navier_stokes_physics->compute_velocities(time_averaged_soln_state);
            
            velocity_magnitude_at_q[iquad] = sqrt(this->navier_stokes_physics->compute_velocity_squared(velocity));
        }
        // write out all values at equidistant nodes
        for(unsigned int ishape=0; ishape<n_quad_pts; ishape++){
            dealii::Point<dim,double> vol_equid_node;
            // write coordinates
            for(int idim=0; idim<dim; idim++) {
                vol_equid_node[idim] = metric_oper_equid.flux_nodes_vol[idim][ishape];
            }
            const double x_precision_P1 = 6E-4;
            const double y_precision_P1 = 6E-4;
            if( ((P1[0]-x_precision_P1)<vol_equid_node[0] && vol_equid_node[0]<(P1[0]+x_precision_P1)) && ((P1[1]-y_precision_P1)<vol_equid_node[1] && vol_equid_node[1]<(P1[1]+y_precision_P1)) ){
                // for(int idim=0; idim<dim; idim++) {
                //     std::cout << "P1 vol_equid_node["<<idim<<"]: " << vol_equid_node[idim] << std::endl;
                // }
                // std::cout << "kinetic energy at q: " << kinetic_energy_at_q[ishape] << std::endl;
                kinetic_energy_sum_P1 +=  kinetic_energy_at_q[ishape];
                kinetic_energy_index_P1 += 1;
                time_averaged_kinetic_energy_sum_P1 +=  time_averaged_kinetic_energy_at_q[ishape];
                time_averaged_kinetic_energy_index_P1 += 1;
                time_averaged_velocity_magnitude_sum_P1 += velocity_magnitude_at_q[ishape];
                time_averaged_velocity_magnitude_index_P1 += 1;
            }
            const double x_precision_P2 = 1E-5;
            const double y_precision_P2 = 1E-5;
            if( ((P2[0]-x_precision_P2)<vol_equid_node[0] && vol_equid_node[0]<(P2[0]+x_precision_P2)) && ((P2[1]-y_precision_P2)<vol_equid_node[1] && vol_equid_node[1]<(P2[1]+y_precision_P2)) ){
                // for(int idim=0; idim<dim; idim++) {
                //     std::cout << "P2 vol_equid_node["<<idim<<"]: " << vol_equid_node[idim] << std::endl;
                // }
                // std::cout << "kinetic energy at q: " << kinetic_energy_at_q[ishape] << std::endl;
                kinetic_energy_sum_P2 +=  kinetic_energy_at_q[ishape];
                kinetic_energy_index_P2 += 1;
                time_averaged_kinetic_energy_sum_P2 +=  time_averaged_kinetic_energy_at_q[ishape];
                time_averaged_kinetic_energy_index_P2 += 1;
                time_averaged_velocity_magnitude_sum_P2 += velocity_magnitude_at_q[ishape];
                time_averaged_velocity_magnitude_index_P2 += 1;
            }
            const double x_precision_P3 = 1E-7;
            const double y_precision_P3 = 1E-7;
            if( ((P3[0]-x_precision_P3)<vol_equid_node[0] && vol_equid_node[0]<(P3[0]+x_precision_P3)) && ((P3[1]-y_precision_P3)<vol_equid_node[1] && vol_equid_node[1]<(P3[1]+y_precision_P3)) ){
                // for(int idim=0; idim<dim; idim++) {
                //     std::cout << "P3 vol_equid_node["<<idim<<"]: " << vol_equid_node[idim] << std::endl;
                // }
                // std::cout << "kinetic energy at q: " << kinetic_energy_at_q[ishape] << std::endl;
                kinetic_energy_sum_P3 +=  kinetic_energy_at_q[ishape];
                kinetic_energy_index_P3 += 1;
                time_averaged_kinetic_energy_sum_P3 +=  time_averaged_kinetic_energy_at_q[ishape];
                time_averaged_kinetic_energy_index_P3 += 1;
                time_averaged_velocity_magnitude_sum_P3 += velocity_magnitude_at_q[ishape];
                time_averaged_velocity_magnitude_index_P3 += 1;
            }
        }
    }//End of cell loop
    // write vkinetic energy average at point 1
    double kinetic_energy_mpi_sum_P1 = dealii::Utilities::MPI::sum(kinetic_energy_sum_P1, this->mpi_communicator);
    double kinetic_energy_mpi_index_P1 = dealii::Utilities::MPI::sum(kinetic_energy_index_P1, this->mpi_communicator);
    double spanwise_average_kinetic_energy_P1 = kinetic_energy_mpi_sum_P1/kinetic_energy_mpi_index_P1;
    this->add_value_to_data_table(spanwise_average_kinetic_energy_P1,"k, P1",unsteady_data_table);
    // write vkinetic energy average at point 2
    double kinetic_energy_mpi_sum_P2 = dealii::Utilities::MPI::sum(kinetic_energy_sum_P2, this->mpi_communicator);
    double kinetic_energy_mpi_index_P2 = dealii::Utilities::MPI::sum(kinetic_energy_index_P2, this->mpi_communicator);
    double spanwise_average_kinetic_energy_P2 = kinetic_energy_mpi_sum_P2/kinetic_energy_mpi_index_P2;
    this->add_value_to_data_table(spanwise_average_kinetic_energy_P2,"k, P2",unsteady_data_table);
    // write vkinetic energy average at point 3
    double kinetic_energy_mpi_sum_P3 = dealii::Utilities::MPI::sum(kinetic_energy_sum_P3, this->mpi_communicator);
    double kinetic_energy_mpi_index_P3 = dealii::Utilities::MPI::sum(kinetic_energy_index_P3, this->mpi_communicator);
    double spanwise_average_kinetic_energy_P3 = kinetic_energy_mpi_sum_P3/kinetic_energy_mpi_index_P3;
    this->add_value_to_data_table(spanwise_average_kinetic_energy_P3,"k, P3",unsteady_data_table);
    if(this->all_param.flow_solver_param.compute_time_averaged_solution && (current_time >= this->all_param.flow_solver_param.time_to_start_averaging)) {
        double time_averaged_kinetic_energy_mpi_sum_P1 = dealii::Utilities::MPI::sum(time_averaged_kinetic_energy_sum_P1, this->mpi_communicator);
        double time_averaged_kinetic_energy_mpi_index_P1 = dealii::Utilities::MPI::sum(time_averaged_kinetic_energy_index_P1, this->mpi_communicator);
        double time_averaged_spanwise_average_kinetic_energy_P1 = time_averaged_kinetic_energy_mpi_sum_P1/time_averaged_kinetic_energy_mpi_index_P1;
        this->add_value_to_data_table(time_averaged_spanwise_average_kinetic_energy_P1,"time_averaged_k, P1",unsteady_data_table);
        // write vkinetic energy average at point 2
        double time_averaged_kinetic_energy_mpi_sum_P2 = dealii::Utilities::MPI::sum(time_averaged_kinetic_energy_sum_P2, this->mpi_communicator);
        double time_averaged_kinetic_energy_mpi_index_P2 = dealii::Utilities::MPI::sum(time_averaged_kinetic_energy_index_P2, this->mpi_communicator);
        double time_averaged_spanwise_average_kinetic_energy_P2 = time_averaged_kinetic_energy_mpi_sum_P2/time_averaged_kinetic_energy_mpi_index_P2;
        this->add_value_to_data_table(time_averaged_spanwise_average_kinetic_energy_P2,"time_averaged_k, P2",unsteady_data_table);
        // write vkinetic energy average at point 3
        double time_averaged_kinetic_energy_mpi_sum_P3 = dealii::Utilities::MPI::sum(time_averaged_kinetic_energy_sum_P3, this->mpi_communicator);
        double time_averaged_kinetic_energy_mpi_index_P3 = dealii::Utilities::MPI::sum(time_averaged_kinetic_energy_index_P3, this->mpi_communicator);
        double time_averaged_spanwise_average_kinetic_energy_P3 = time_averaged_kinetic_energy_mpi_sum_P3/time_averaged_kinetic_energy_mpi_index_P3;
        this->add_value_to_data_table(time_averaged_spanwise_average_kinetic_energy_P3,"time_averaged_k, P3",unsteady_data_table);
    } else{
        this->add_value_to_data_table(0,"time_averaged_k, P1",unsteady_data_table);
        this->add_value_to_data_table(0,"time_averaged_k, P2",unsteady_data_table);
        this->add_value_to_data_table(0,"time_averaged_k, P3",unsteady_data_table);
    }
    // write velocity magnitude at point 1
    double velocity_magnitude_mpi_sum_P1 = dealii::Utilities::MPI::sum(time_averaged_velocity_magnitude_sum_P1, this->mpi_communicator);
    double velocity_magnitude_mpi_index_P1 = dealii::Utilities::MPI::sum(time_averaged_velocity_magnitude_index_P1, this->mpi_communicator);
    double spanwise_average_velocity_magnitude_P1 = velocity_magnitude_mpi_sum_P1/velocity_magnitude_mpi_index_P1;
    this->add_value_to_data_table(spanwise_average_velocity_magnitude_P1,"t_ave_vel_mag, P1",unsteady_data_table);
    // write velocity magnitude at point 2
    double velocity_magnitude_mpi_sum_P2 = dealii::Utilities::MPI::sum(time_averaged_velocity_magnitude_sum_P2, this->mpi_communicator);
    double velocity_magnitude_mpi_index_P2 = dealii::Utilities::MPI::sum(time_averaged_velocity_magnitude_index_P2, this->mpi_communicator);
    double spanwise_average_velocity_magnitude_P2 = velocity_magnitude_mpi_sum_P2/velocity_magnitude_mpi_index_P2;
    this->add_value_to_data_table(spanwise_average_velocity_magnitude_P2,"t_ave_vel_mag, P2",unsteady_data_table);
    // write velocity magnitude at point 3
    double velocity_magnitude_mpi_sum_P3 = dealii::Utilities::MPI::sum(time_averaged_velocity_magnitude_sum_P3, this->mpi_communicator);
    double velocity_magnitude_mpi_index_P3 = dealii::Utilities::MPI::sum(time_averaged_velocity_magnitude_index_P3, this->mpi_communicator);
    double spanwise_average_velocity_magnitude_P3 = velocity_magnitude_mpi_sum_P3/velocity_magnitude_mpi_index_P3;
    this->add_value_to_data_table(spanwise_average_velocity_magnitude_P3,"t_ave_vel_mag, P3",unsteady_data_table);
    // Add lift to data table
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg);
    this->add_value_to_data_table(lift,"lift",unsteady_data_table);
    this->add_value_to_data_table(drag,"drag",unsteady_data_table);
}

template <int dim, int nstate>
void NACA0012_LES<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg);

    this->pcout << " Resulting lift : " << lift << std::endl;
    this->pcout << " Resulting drag : " << drag << std::endl;
}

template <int dim, int nstate>
void NACA0012_LES<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table,
            const bool do_write_unsteady_data_table_file)
{
    // Compute aerodynamic values
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg);
    
    if(output_counter == 4){
        if(this->all_param.flow_solver_param.compute_time_averaged_solution && (current_time >= this->all_param.flow_solver_param.time_to_start_averaging)){
            if(this->mpi_rank==0) {
                // Add values to data table
                this->add_value_to_data_table(current_time,"time",unsteady_data_table);
                // this->add_value_to_data_table(lift,"lift",unsteady_data_table);
                // this->add_value_to_data_table(drag,"drag",unsteady_data_table);
            }
            dealii::Point<dim,double> P1;
            dealii::Point<dim,double> P2;
            dealii::Point<dim,double> P3;
            if(dim == 2){
                P1[0] = 0.5;
                P1[1] = 0.11;
                P2[0] = 0.5;
                P2[1] = 0.05;
                P3[0] = 1.3;
                P3[1] = 0;
            }
            if(dim == 3){
                P1[0] = 0.5;
                P1[1] = 0.11;
                P1[2] = 0;
                P2[0] = 0.499696;
                P2[1] = 0.0501122;
                P2[2] = 0;
                P3[0] = 1.2997177;
                P3[1] = 0.00014232968;
                P3[2] = 0;
            }
            this->output_kinetic_energy_at_points(dg, current_time, P1, P2, P3, unsteady_data_table);
        }

        // Reset counter
        output_counter = 0;
    }else{
        // Add to counter
        output_counter += 1;
    }
    
    if(terminal_counter == 19999){
        // Print to console
        this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << current_time
                    << "    Lift: " << lift
                    << "    Drag: " << drag;
        this->pcout << std::endl;
        
        terminal_counter = 0;
    }else{
        // Add to counter
        terminal_counter += 1;
    }


    if(this->mpi_rank==0) {
        // Write to file
        if(do_write_unsteady_data_table_file) {
            std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
            unsteady_data_table->write_text(unsteady_data_table_file);
        }
    }
    // Abort if energy is nan
    // if(std::isnan(lift) || std::isnan(drag)) {
    //     this->pcout << " ERROR: Lift or drag at time " << current_time << " is nan." << std::endl;
    //     this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
    //     std::abort();
    // }
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

    bool output_velocity_field_at_fixed_location = false;
    // Output velocity field at fixed location along airfoil surface
    if(output_velocity_field_at_fixed_location) {
        if constexpr(nstate!=1){
            dealii::Point<dim,double> extraction_point;
            if constexpr(dim==2){
                extraction_point[0] = this->all_param.boundary_layer_extraction_param.extraction_point_x;
                extraction_point[1] = this->all_param.boundary_layer_extraction_param.extraction_point_y;
            } else if constexpr(dim==3){
                extraction_point[0] = this->all_param.boundary_layer_extraction_param.extraction_point_x;
                extraction_point[1] = this->all_param.boundary_layer_extraction_param.extraction_point_y;
                extraction_point[2] = this->all_param.boundary_layer_extraction_param.extraction_point_z;
            }
            int number_of_sampling = this->all_param.boundary_layer_extraction_param.number_of_sampling;

            ExtractionFunctional<dim,nstate,double,Triangulation> boundary_layer_extraction(dg, extraction_point, number_of_sampling);

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
                const double displacement_thickness = boundary_layer_extraction.evaluate_displacement_thickness();

                const double momentum_thickness = boundary_layer_extraction.evaluate_momentum_thickness();

                const double edge_velocity = boundary_layer_extraction.evaluate_edge_velocity();

                const double wall_shear_stress = boundary_layer_extraction.evaluate_wall_shear_stress();

                const double maximum_shear_stress = boundary_layer_extraction.evaluate_maximum_shear_stress();

                const double friction_velocity = boundary_layer_extraction.evaluate_friction_velocity();

                const double boundary_layer_thickness = boundary_layer_extraction.evaluate_boundary_layer_thickness();

                this->pcout << " Extracted displacement_thickness : "   << displacement_thickness   << std::endl;
                this->pcout << " Extracted momentum_thickness : "       << momentum_thickness       << std::endl;
                this->pcout << " Extracted edge_velocity : "            << edge_velocity            << std::endl;
                this->pcout << " Extracted wall_shear_stress : "        << wall_shear_stress        << std::endl;
                this->pcout << " Extracted maximum_shear_stress : "     << maximum_shear_stress     << std::endl;
                this->pcout << " Extracted friction_velocity : "        << friction_velocity        << std::endl;
                this->pcout << " Extracted boundary_layer_thickness : " << boundary_layer_thickness << std::endl;
                
                // Update index s.t. it never goes out of bounds
                if(this->index_of_current_desired_time_to_output_velocity_field 
                    < (this->number_of_times_to_output_velocity_field-1)) {
                    this->index_of_current_desired_time_to_output_velocity_field += 1;
                }
            }
        }
    }
}

#if PHILIP_DIM!=1
    template class NACA0012_LES<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

