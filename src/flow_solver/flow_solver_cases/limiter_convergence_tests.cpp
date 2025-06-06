#include "limiter_convergence_tests.h"
#include <iostream>
#include <stdlib.h>
#include "mesh/gmsh_reader.hpp"
#include "physics/physics_factory.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include "mesh/grids/straight_periodic_cube.hpp"


namespace PHiLiP{
namespace FlowSolver{

template <int dim, int nstate>
LimiterConvergenceTests<dim, nstate>::LimiterConvergenceTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
    , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
    //create the Physics object
    this->pde_physics = std::dynamic_pointer_cast<Physics::PhysicsBase<dim,nstate,double>>(
                Physics::PhysicsFactory<dim,nstate,double>::create_Physics(parameters_input));
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> LimiterConvergenceTests<dim,nstate>::generate_grid() const
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

    double left = this->all_param.flow_solver_param.grid_left_bound;
    double right = this->all_param.flow_solver_param.grid_right_bound;

    const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;

    PHiLiP::Grids::straight_periodic_cube<dim, Triangulation>(grid, left, right, pow(2.0, number_of_refinements));

    std::cout << "Grid generated and refined" << std::endl;

    return grid;
}

template <int dim, int nstate>
double LimiterConvergenceTests<dim, nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim, double>> dg) const
{
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case = this->all_param.flow_solver_param.flow_case_type;

    double left = this->all_param.flow_solver_param.grid_left_bound;
    double right = this->all_param.flow_solver_param.grid_right_bound;
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;

    const unsigned int n_global_active_cells = dg->triangulation->n_global_active_cells();
    const unsigned int n_dofs_cfl = dg->dof_handler.n_dofs() / nstate;
    double delta_x = (PHILIP_DIM == 2) ? (right - left) / pow(n_global_active_cells, (1.0 / dim)) : (right - left) / pow(n_dofs_cfl, (1.0 / dim));
    double time_step = 1e-5;

    /**********************************
    * These values for the time step are chosen to show dominant spatial accuracy in the OOA results for P2
    * For >=P3 timestep values  refer to: 
    * Zhang, Xiangxiong, and Chi-Wang Shu. 
    * "On maximum-principle-satisfying high order schemes for scalar conservation laws." 
    * Journal of Computational Physics 229.9 (2010): 3091-3120.
    **********************************/
   
    if(flow_case == flow_case_enum::advection_limiter)
        time_step = (PHILIP_DIM == 2) ? (1.0 / 14.0) * delta_x : (1.0 / 3.0) * pow(delta_x, 2.0);
    
    if(flow_case == flow_case_enum::burgers_limiter)
        time_step = (PHILIP_DIM == 2) ? (1.0 / 14.0) * delta_x : (1.0 / 24.0) * delta_x;

    if (flow_case == flow_case_enum::low_density){
        const double approximate_grid_spacing = (this->all_param.flow_solver_param.grid_xmax-this->all_param.flow_solver_param.grid_xmin)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
        const double cfl_number = this->all_param.flow_solver_param.courant_friedrichs_lewy_number;
        time_step = cfl_number * approximate_grid_spacing / this->maximum_local_wave_speed;
    }

    return time_step;
}

template <int dim, int nstate>
double LimiterConvergenceTests<dim, nstate>::get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim, double>> dg)
{
    if(nstate == dim + 2){
        // initialize the maximum local wave speed
        update_maximum_local_wave_speed(*dg);
    }
    // compute time step for each case such that results show dominant spatial accuracy
    const double time_step = get_adaptive_time_step(dg);

    return time_step;
}

template<int dim, int nstate>
void LimiterConvergenceTests<dim, nstate>::update_maximum_local_wave_speed(DGBase<dim, double> &dg)
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
            const double local_wave_speed = this->pde_physics->max_convective_eigenvalue(soln_at_q);
            if(local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
        }
    }
    this->maximum_local_wave_speed = dealii::Utilities::MPI::max(this->maximum_local_wave_speed, this->mpi_communicator);
}


template <int dim, int nstate>
void LimiterConvergenceTests<dim, nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
}

template<int dim, int nstate>
void LimiterConvergenceTests<dim, nstate>::check_limiter_principle(DGBase<dim, double>& dg)
{
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case = this->all_param.flow_solver_param.flow_case_type;

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
        if(flow_case == flow_case_enum::advection_limiter || flow_case == flow_case_enum::burgers_limiter){
            for (unsigned int istate = 0; istate < nstate; ++istate) {
                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    // Verify that strict maximum principle is satisfied
                    if (soln_at_q[istate][iquad] > 1.00 + 1e-13 || soln_at_q[istate][iquad] < -1.00 - 1e-13|| (isnan(soln_at_q[istate][iquad])))  {
                            std::cout << "Flow Solver Error: Strict Maximum Principle is violated - Aborting... " << soln_at_q[istate][iquad] << std::endl << std::flush;
                            std::abort();
                    } 
                }
            }
        } else if (flow_case == flow_case_enum::low_density) {
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                    // Verify that positivity of density is preserved
                    if (soln_at_q[0][iquad] < 0 || (isnan(soln_at_q[0][iquad])) ) {
                        std::cout << "Error: Density is negative or NaN - Aborting... " << std::endl << std::flush;
                        std::abort();
                    }
            }
        }
    }
}

template <int dim, int nstate>
void LimiterConvergenceTests<dim, nstate>::compute_unsteady_data_and_write_to_table(
    const unsigned int current_iteration,
    const double current_time,
    const std::shared_ptr <DGBase<dim, double>> dg,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    this->check_limiter_principle(*dg);
    if (this->mpi_rank == 0) {

        unsteady_data_table->add_value("iteration", current_iteration);
        // Add values to data table
        this->add_value_to_data_table(current_time, "time", unsteady_data_table);


        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

    if (current_iteration % this->all_param.ode_solver_param.print_iteration_modulo == 0) {
        // Print to console
        this->pcout << "    Iter: " << current_iteration
            << "    Time: " << current_time;

        this->pcout << std::endl;
    }
}

#if PHILIP_DIM==1
    template class LimiterConvergenceTests<PHILIP_DIM, PHILIP_DIM>;
    template class LimiterConvergenceTests<PHILIP_DIM, PHILIP_DIM+2>;
#elif PHILIP_DIM==2
    template class LimiterConvergenceTests<PHILIP_DIM, PHILIP_DIM>;
    template class LimiterConvergenceTests<PHILIP_DIM, PHILIP_DIM+2>;
    template class LimiterConvergenceTests<PHILIP_DIM, 1>;
#endif

}
}