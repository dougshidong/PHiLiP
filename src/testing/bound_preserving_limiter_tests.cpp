#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

#include "bound_preserving_limiter_tests.h"

#include "physics/initial_conditions/initial_condition_function.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BoundPreservingLimiterTests<dim, nstate>::BoundPreservingLimiterTests(
    const PHiLiP::Parameters::AllParameters* const parameters_input,
    const dealii::ParameterHandler& parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
double BoundPreservingLimiterTests<dim, nstate>::get_time_step(std::shared_ptr<DGBase<dim, double>> dg) const
{
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case = all_parameters_new.flow_solver_param.flow_case_type;

    double left = all_parameters_new.flow_solver_param.grid_left_bound;
    double right = all_parameters_new.flow_solver_param.grid_right_bound;
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
        // Initialize the maximum local wave speed to zero
        double maximum_local_wave_speed = 0.0;

        // Overintegrate the error to make sure there is not integration error in the error estimate
        int overintegrate = 10;
        dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
        dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[dg->max_degree], quad_extra,
                                                dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

        const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
        std::array<double,nstate> soln_at_q;

        std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
        for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            fe_values_extra.reinit (cell);
            cell->get_dof_indices (dofs_indices);

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                    const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                    soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                }
                double local_wave_speed = 0.0;
                if(nstate == dim + 2) {
                    // Update the maximum local wave speed (i.e. convective eigenvalue)
                    Physics::Euler<dim,nstate,double> euler_physics_double
                        = Physics::Euler<dim, nstate, double>(
                                all_parameters,
                                all_parameters_new.euler_param.ref_length,
                                all_parameters_new.euler_param.gamma_gas,
                                all_parameters_new.euler_param.mach_inf,
                                all_parameters_new.euler_param.angle_of_attack,
                                all_parameters_new.euler_param.side_slip_angle);

                    local_wave_speed = euler_physics_double.max_convective_eigenvalue(soln_at_q);
                }
                if(local_wave_speed > maximum_local_wave_speed) maximum_local_wave_speed = local_wave_speed;
            }
        }
        maximum_local_wave_speed = dealii::Utilities::MPI::max(maximum_local_wave_speed, this->mpi_communicator);

        const double approximate_grid_spacing = (all_parameters_new.flow_solver_param.grid_right_bound-all_parameters_new.flow_solver_param.grid_left_bound)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
        const double cfl_number = all_parameters_new.flow_solver_param.courant_friedrichs_lewy_number;
        time_step = cfl_number * approximate_grid_spacing / maximum_local_wave_speed;
    }

    return time_step;
}

template <int dim, int nstate>
double BoundPreservingLimiterTests<dim, nstate>::calculate_uexact(const dealii::Point<dim> qpoint, const dealii::Tensor<1, 3, double> adv_speeds, double final_time) const
{
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case = all_parameters_new.flow_solver_param.flow_case_type;
    const double pi = atan(1) * 4.0;

    double uexact = 1.0;
    if (flow_case == Parameters::FlowSolverParam::FlowCaseType::low_density && dim == 2) {
        uexact = 0.01 + exp(-500.0*(pow(qpoint[0], 2.0)+pow(qpoint[1], 2.0)));
    }
    if (flow_case == Parameters::FlowSolverParam::FlowCaseType::low_density && dim == 1) {
        uexact = 0.01 + exp(-500.0*pow(qpoint[0], 2.0));
    }
    else {
        for (int idim = 0; idim < dim; idim++) {
            if (flow_case == Parameters::FlowSolverParam::FlowCaseType::burgers_limiter)
                uexact *= cos(pi * (qpoint[idim] - final_time));//for grid 1-3
            if (flow_case == Parameters::FlowSolverParam::FlowCaseType::advection_limiter)
                uexact *= sin(2.0 * pi * (qpoint[idim] - adv_speeds[idim] * final_time));//for grid 1-3
        }
    }

    return uexact;
}

template <int dim, int nstate>
std::array<double,3> BoundPreservingLimiterTests<dim, nstate>::calculate_l_n_error(
    std::shared_ptr<DGBase<dim, double>> dg,
    const int poly_degree,
    const double final_time) const
{
    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(poly_degree + 1 + overintegrate);
    dealii::FEValues<dim, dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra,
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double, nstate> soln_at_q;

    double l1error = 0.0;
    double l2error = 0.0;
    double linferror = 0.0;

    // Integrate every cell and compute L2
    std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
    const dealii::Tensor<1, 3, double> adv_speeds = Parameters::ManufacturedSolutionParam::get_default_advection_vector();
    for (auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit(cell);
        cell->get_dof_indices(dofs_indices);

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
            double uexact = calculate_uexact(qpoint, adv_speeds, final_time);   

            //std::cout << "u:   " << soln_at_q[0] << "   uexact:   " << uexact << std::endl;       
            l1error += pow(abs(soln_at_q[0] - uexact), 1.0) * fe_values_extra.JxW(iquad);
            l2error += pow(abs(soln_at_q[0] - uexact), 2.0) * fe_values_extra.JxW(iquad);
            //L-infinity norm
            linferror = std::max(abs(soln_at_q[0]-uexact), linferror);
        }
    }
    //MPI sum
    double l1error_mpi = dealii::Utilities::MPI::sum(l1error, this->mpi_communicator);

    double l2error_mpi = dealii::Utilities::MPI::sum(l2error, this->mpi_communicator);
    l2error_mpi = pow(l2error_mpi, 1.0/2.0);

    double linferror_mpi = dealii::Utilities::MPI::max(linferror, this->mpi_communicator);

    std::array<double,3> lerror_mpi;
    lerror_mpi[0] = l1error_mpi;
    lerror_mpi[1] = l2error_mpi;
    lerror_mpi[2] = linferror_mpi;
    return lerror_mpi;
}

template <int dim, int nstate>
int BoundPreservingLimiterTests<dim, nstate>::run_test() const
{
    pcout << " Running Bound Preserving Limiter test. " << std::endl;
    pcout << dim << "    " << nstate << std::endl;
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;

    int test_result = 1;

    if (!all_parameters_new.limiter_param.use_OOA) {
        test_result = run_full_limiter_test();
    }
    else {
        test_result = run_convergence_test();
    }
    return test_result; //if got to here means passed the test, otherwise would've failed earlier
}

template <int dim, int nstate>
int BoundPreservingLimiterTests<dim, nstate>::run_full_limiter_test() const
{
    pcout << "\n" << "Creating FlowSolver" << std::endl;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case = all_parameters_new.flow_solver_param.flow_case_type;

    if (flow_case == Parameters::FlowSolverParam::FlowCaseType::low_density) {
        param.flow_solver_param.number_of_grid_elements_x = pow(2.0,param.flow_solver_param.number_of_mesh_refinements);
        if(dim == 2)
            param.flow_solver_param.number_of_grid_elements_y = pow(2.0,param.flow_solver_param.number_of_mesh_refinements);
    }

    // Create flow solver to access DG object which is needed to calculate time step
    std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);
    double time_step = get_time_step(flow_solver->dg);
    param.flow_solver_param.constant_time_step = time_step;
    // Delete flow solver object (the time_step param cannot be changed directly because flow_solver_param is protected in flow_solver)
    flow_solver.reset();
    // Reinitialize flow solver with new time step parameter
    flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();

    return 0;
}

template <int dim, int nstate>
int BoundPreservingLimiterTests<dim, nstate>::run_convergence_test() const
{
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    PHiLiP::Parameters::ManufacturedConvergenceStudyParam manu_grid_conv_param = all_parameters_new.manufactured_convergence_study_param;

    const unsigned int n_grids = manu_grid_conv_param.number_of_grids;
    dealii::ConvergenceTable convergence_table;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error_l2(n_grids);
    double final_order = 0.0;
    const double expected_order = all_parameters_new.flow_solver_param.expected_order_at_final_time;

    for (unsigned int igrid = 3; igrid < n_grids; igrid++) {

        pcout << "\n" << "Creating FlowSolver" << std::endl;

        Parameters::AllParameters param = *(TestsBase::all_parameters);
        param.flow_solver_param.number_of_grid_elements_per_dimension = pow(2.0,igrid);
        int grid_elem = param.flow_solver_param.number_of_grid_elements_per_dimension;

        using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
        flow_case_enum flow_case = all_parameters_new.flow_solver_param.flow_case_type;
        //const double pi = atan(1) * 4.0;

        if (flow_case == Parameters::FlowSolverParam::FlowCaseType::low_density) {    
            param.flow_solver_param.number_of_grid_elements_x = grid_elem;
            
            if (dim == 2) {
                param.flow_solver_param.number_of_grid_elements_y = grid_elem;
            }
        }
        
        // Create flow solver to access DG object which is needed to calculate time step
        std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);
        const unsigned int n_global_active_cells = flow_solver->dg->triangulation->n_global_active_cells();
        const int poly_degree = all_parameters_new.flow_solver_param.poly_degree;
        double time_step = get_time_step(flow_solver->dg);
        param.flow_solver_param.constant_time_step = time_step;
        // Delete flow solver object (the time_step param cannot be changed directly because flow_solver_param is protected in flow_solver)
        flow_solver.reset();
        // Reinitialize flow solver with new time step parameter
        flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);
        flow_solver->run();
        const double final_time_actual = flow_solver->ode_solver->current_time;

        // output results
        const unsigned int n_dofs = flow_solver->dg->dof_handler.n_dofs();
        this->pcout << "Dimension: " << dim
        << "\t Polynomial degree p: " << poly_degree
        << std::endl
        << "Grid number: " << igrid + 1 << "/" << n_grids
        << ". Number of active cells: " << n_global_active_cells
        << ". Number of degrees of freedom: " << n_dofs
        << std::endl;

        const std::array<double,3> lerror_mpi_sum = calculate_l_n_error(flow_solver->dg, poly_degree, final_time_actual);

        // Convergence table
        const double dx = 1.0 / pow(n_dofs, (1.0 / dim));
        grid_size[igrid] = dx;
        soln_error_l2[igrid] = lerror_mpi_sum[1];

        convergence_table.add_value("p", poly_degree);
        convergence_table.add_value("cells", n_global_active_cells);
        convergence_table.add_value("DoFs", n_dofs);
        convergence_table.add_value("dx", dx);
        convergence_table.add_value("soln_L1_error", lerror_mpi_sum[0]);
        convergence_table.add_value("soln_L2_error", lerror_mpi_sum[1]);
        convergence_table.add_value("soln_Linf_error", lerror_mpi_sum[2]);

        this->pcout << " Grid size h: " << dx
            << " L1-soln_error: " << lerror_mpi_sum[0]
            << " L2-soln_error: " << lerror_mpi_sum[1]
            << " Linf-soln_error: " << lerror_mpi_sum[2]
            << " Residual: " << flow_solver->ode_solver->residual_norm
            << std::endl;

        if (igrid > 0) {
            const double slope_soln_err = log(soln_error_l2[igrid] / soln_error_l2[igrid - 1])
                / log(grid_size[igrid] / grid_size[igrid - 1]);

            if (igrid == n_grids - 1)
                final_order = slope_soln_err;

            this->pcout << "From grid " << igrid - 1
                << "  to grid " << igrid
                << "  dimension: " << dim
                << "  polynomial degree p: " << poly_degree
                << std::endl
                << "  solution_error1 " << soln_error_l2[igrid - 1]
                << "  solution_error2 " << soln_error_l2[igrid]
                << "  slope " << slope_soln_err
                << std::endl;
        }

        this->pcout << " ********************************************"
            << std::endl
            << " Convergence rates for p = " << poly_degree
            << std::endl
            << " ********************************************"
            << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L1_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_Linf_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L1_error", true);
        convergence_table.set_scientific("soln_L2_error", true);
        convergence_table.set_scientific("soln_Linf_error", true);
        if (this->pcout.is_active()) convergence_table.write_text(this->pcout.get_stream());

        std::ofstream table_file("convergence_rates.txt");
        convergence_table.write_text(table_file);

        
    }//end of grid loop

    if(abs(final_order - expected_order) < 1e-4)
        return 0;
    else
        return 1;
}

#if PHILIP_DIM==1
template class BoundPreservingLimiterTests<PHILIP_DIM, PHILIP_DIM>;
template class BoundPreservingLimiterTests<PHILIP_DIM, PHILIP_DIM + 2>;
#elif PHILIP_DIM==2
template class BoundPreservingLimiterTests<PHILIP_DIM, PHILIP_DIM>;
template class BoundPreservingLimiterTests<PHILIP_DIM, PHILIP_DIM + 2>;
template class BoundPreservingLimiterTests<PHILIP_DIM, 1>;
#endif

} // Tests namespace
} // PHiLiP namespace
