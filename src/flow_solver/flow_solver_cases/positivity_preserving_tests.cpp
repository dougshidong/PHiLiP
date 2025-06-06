#include "positivity_preserving_tests.h"
#include "mesh/grids/positivity_preserving_tests_grid.h"
#include "mesh/grids/straight_periodic_cube.hpp"
#include <deal.II/grid/grid_generator.h>
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
PositivityPreservingTests<dim, nstate>::PositivityPreservingTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : CubeFlow_UniformGrid<dim, nstate>(parameters_input)
    , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
            PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&(this->all_param)));
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> PositivityPreservingTests<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );
    
    if(dim >= 1) {
        if(this->all_param.flow_solver_param.grid_xmax == this->all_param.flow_solver_param.grid_xmin) {
            std::cout << "Error: xmax and xmin need to be provided as parameters - Aborting... " << std::endl << std::flush;
            std::abort();
        }
    }

    if(dim >= 2) {
        if(this->all_param.flow_solver_param.grid_ymax == this->all_param.flow_solver_param.grid_ymin) {
            std::cout << "Error: ymax and ymin need to be provided as parameters - Aborting... " << std::endl << std::flush;
            std::abort();
        }
    }

    if(dim == 3) {
        if(this->all_param.flow_solver_param.grid_zmax == this->all_param.flow_solver_param.grid_zmin) {
            std::cout << "Error: zmax and zmin need to be provided as parameters - Aborting... " << std::endl << std::flush;
            std::abort();
        }
    }
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    if (dim==1 && (flow_case_type == flow_case_enum::sod_shock_tube
        || flow_case_type == flow_case_enum::leblanc_shock_tube
        || flow_case_type == flow_case_enum::shu_osher_problem)) {
        Grids::shock_tube_1D_grid<dim>(*grid, &this->all_param.flow_solver_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::shock_diffraction) {
        Grids::shock_diffraction_grid<dim>(*grid, &this->all_param.flow_solver_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::astrophysical_jet) {
        Grids::astrophysical_jet_grid<dim>(*grid, &this->all_param.flow_solver_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::double_mach_reflection) {
            Grids::double_mach_reflection_grid<dim>(*grid, &this->all_param.flow_solver_param);
    }
    else if (dim==2 && flow_case_type == flow_case_enum::strong_vortex_shock_wave) {
            Grids::svsw_grid<dim>(*grid, &this->all_param.flow_solver_param);
    }
    return grid;
}

template <int dim, int nstate>
void PositivityPreservingTests<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
}

template<int dim, int nstate>
void PositivityPreservingTests<dim, nstate>::check_positivity_density(DGBase<dim, double>& dg)
{
    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = dg.max_grid_degree;
    const unsigned int poly_degree = this->all_param.flow_solver_param.poly_degree;
    //Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, double> soln_basis(1, poly_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, double> soln_basis_projection_oper(1, dg.max_degree, init_grid_degree);


    // Build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], dg.oneD_quadrature_collection[poly_degree]);
    soln_basis_projection_oper.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], dg.oneD_quadrature_collection[poly_degree]);

    for (auto soln_cell = dg.dof_handler.begin_active(); soln_cell != dg.dof_handler.end(); ++soln_cell) {
        if (!soln_cell->is_locally_owned()) continue;


        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = dg.fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<double>, nstate> soln_coeff;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        // Allocate solution dofs and set local max and min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = dg.fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg.fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = dg.solution[current_dofs_indices[idof]];
        }

        const unsigned int n_quad_pts = dg.volume_quadrature_collection[poly_degree].size();

        std::array<std::vector<double>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate], soln_basis.oneD_vol_operator);
        }

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            // Verify that positivity of density is preserved
            if (soln_at_q[0][iquad] < 0) {
                std::cout << "Flow Solver Error: Density is negative - Aborting... " << std::endl << std::flush;
                std::abort();
            }

            if (soln_at_q[nstate - 1][iquad] < 0) {
                std::cout << "Flow Solver Error: Total Energy is negative - Aborting... " << std::endl << std::flush;
                std::abort();
            }

            if ((isnan(soln_at_q[0][iquad]))) {
                std::cout << "Flow Solver Error: Density is NaN - Aborting... " << std::endl << std::flush;
                std::abort();
            }
        }
    }
}

template<int dim, int nstate>
double PositivityPreservingTests<dim, nstate>::compute_integrated_entropy(DGBase<dim, double> &dg) const
{
    // Check that poly_degree is uniform everywhere
    if (dg.get_max_fe_degree() != dg.get_min_fe_degree()) {
        // Note: This function may have issues with nonuniform p. Should test in debug mode if developing in the future.
        this->pcout << "ERROR: compute_integrated_quantities() is untested for nonuniform p. Aborting..." << std::endl;
        std::abort();
    }

    double integrated_quantity = 0.0;

    const unsigned int grid_degree = dg.high_order_grid->fe_system.tensor_degree();
    const unsigned int poly_degree = dg.max_degree;

    // Set the quadrature of size dim and 1D for sum-factorization.
    dealii::Quadrature<1> quad_1D = dg.oneD_quadrature_collection[poly_degree];
    std::vector<double> quad_weights = dg.volume_quadrature_collection[poly_degree].get_weights();
    unsigned int n_quad_pts = dg.volume_quadrature_collection[poly_degree].size();

    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, grid_degree); 
    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, grid_degree);
    // Build basis function volume operator from 1D finite element for 1 state.
    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], quad_1D);
    // Build mapping shape functions operators using the oneD high_ordeR_grid finite element
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg.high_order_grid->oneD_fe_system, dg.high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg.high_order_grid->oneD_fe_system, quad_1D, dg.oneD_face_quadrature);
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
        const dealii::FESystem<dim> &fe_metric = dg.high_order_grid->fe_system;
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
        for(int istate=0; istate<nstate; istate++){
            soln_at_q_vect[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_vect[istate],
                                             soln_basis.oneD_vol_operator);
        }

        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_at_q;
            // Extract solution in a way that the physics ca n use them.
            for(int istate=0; istate<nstate; istate++){
                soln_at_q[istate] = soln_at_q_vect[istate][iquad];
            }
            
            //#####################################################################
            // Compute integrated quantities here
            //#####################################################################
            const double quadrature_entropy = this->euler_physics->compute_numerical_entropy_function(soln_at_q);
            //Using std::cout because of cell->is_locally_owned check 
            if (isnan(quadrature_entropy))  std::cout << "WARNING: NaN entropy detected at a node!"  << std::endl;
            integrated_quantity += quadrature_entropy * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            //#####################################################################
        }
    }

    //MPI
    integrated_quantity = dealii::Utilities::MPI::sum(integrated_quantity, this->mpi_communicator);
    
    return integrated_quantity;
}

template <int dim, int nstate>
void PositivityPreservingTests<dim, nstate>::compute_unsteady_data_and_write_to_table(
    const std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver,
    const std::shared_ptr <DGBase<dim, double>> dg,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    //unpack current iteration and current time from ode solver
    const unsigned int current_iteration = ode_solver->current_iteration;
    const double current_time = ode_solver->current_time;

    // All discrete proofs use solution nodes, therefore it is best to report 
    // entropy on the solution nodes rather than by overintegrating.
    const double current_numerical_entropy = this->compute_integrated_entropy(*dg); // no overintegration
    if (current_iteration==0) this->previous_numerical_entropy = current_numerical_entropy;
    const double entropy = current_numerical_entropy - previous_numerical_entropy + ode_solver->FR_entropy_contribution_RRK_solver;
    this->previous_numerical_entropy = current_numerical_entropy;

    if (std::isnan(entropy)){
        this->pcout << "Entropy is nan. Aborting flow simulation..." << std::endl << std::flush;
        std::abort();
    }
    if (current_iteration == 0)  initial_entropy = current_numerical_entropy;

    if(nstate == dim + 2)
        this->check_positivity_density(*dg);

    if (this->mpi_rank == 0) {

        unsteady_data_table->add_value("iteration", current_iteration);
        // Add values to data table
        this->add_value_to_data_table(current_time, "time", unsteady_data_table);
        this->add_value_to_data_table(entropy,"entropy",unsteady_data_table);
        unsteady_data_table->set_scientific("entropy", false);
        this->add_value_to_data_table(current_numerical_entropy,"current_numerical_entropy",unsteady_data_table);
        unsteady_data_table->set_scientific("current_numerical_entropy", false);
        this->add_value_to_data_table(entropy/initial_entropy,"U/Uo",unsteady_data_table);
        unsteady_data_table->set_scientific("U/Uo", false);


        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

    if (current_iteration % this->all_param.ode_solver_param.print_iteration_modulo == 0) {
        // Print to console
        this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << std::setprecision(16) << current_time
                    << "    Current Numerical Entropy:  " << current_numerical_entropy
                    << "    Entropy: " << entropy
                    << "    (U-Uo)/Uo: " << entropy/initial_entropy;

        this->pcout << std::endl;
    }

    // Update local maximum wave speed before calculating next time step
    update_maximum_local_wave_speed(*dg);
}

template class PositivityPreservingTests<PHILIP_DIM, PHILIP_DIM+2>;

} // FlowSolver namespace
} // PHiLiP namespace