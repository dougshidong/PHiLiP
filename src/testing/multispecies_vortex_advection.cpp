#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

#include "multispecies_vortex_advection.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nspecies, int nstate>
MultispeciesVortexAdvection<dim, nspecies, nstate>::MultispeciesVortexAdvection(
    const PHiLiP::Parameters::AllParameters* const parameters_input,
    const dealii::ParameterHandler& parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{
    //create the Physics object
    this->real_gas_physics = std::dynamic_pointer_cast<Physics::RealGas<dim,nspecies,dim+nspecies+1,double>>(
            PHiLiP::Physics::PhysicsFactory<dim,nspecies,nstate,double>::create_Physics(parameters_input));

    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case = parameters_input->flow_solver_param.flow_case_type;

    if (flow_case == Parameters::FlowSolverParam::FlowCaseType::multi_species_vortex_advection) {    
        this->high_temp = false;
    } else if (flow_case == Parameters::FlowSolverParam::FlowCaseType::multi_species_vortex_advection_high_temp) {    
        this->high_temp = true;
    }
}

template <int dim, int nspecies, int nstate>
double MultispeciesVortexAdvection<dim, nspecies, nstate>::get_time_step(std::shared_ptr<DGBase<dim, nspecies, double>> dg) const
{
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    double time_step = 1e-5;

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
            double local_wave_speed = this->real_gas_physics->max_convective_eigenvalue(soln_at_q);
            if(local_wave_speed > maximum_local_wave_speed) maximum_local_wave_speed = local_wave_speed;
        }
    }
    maximum_local_wave_speed = dealii::Utilities::MPI::max(maximum_local_wave_speed, this->mpi_communicator);

    const double approximate_grid_spacing = (all_parameters_new.flow_solver_param.grid_right_bound-all_parameters_new.flow_solver_param.grid_left_bound)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
    const double cfl_number = all_parameters_new.flow_solver_param.courant_friedrichs_lewy_number;
    time_step = cfl_number * approximate_grid_spacing / maximum_local_wave_speed;

    return time_step;
}

template <int dim, int nspecies, int nstate>
std::array<std::array<double,3>,nstate+1> MultispeciesVortexAdvection<dim, nspecies, nstate>::calculate_l_n_error(
    std::shared_ptr<DGBase<dim, nspecies, double>> dg,
    const int poly_degree,
    const double /*final_time*/,
    std::shared_ptr<FlowSolver::FlowSolver<dim, nspecies, nstate>> flow_solver) const
{
    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(poly_degree + 1 + overintegrate);
    dealii::FEValues<dim, dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra,
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double, nstate> soln_at_q, soln_exact_primitive;

    std::array<std::array<double,3>,nstate+1> lerror_primitive;
    for (int istate = 0; istate < nstate+1; ++istate) {
        lerror_primitive[istate][0] = 0.0;
        lerror_primitive[istate][1] = 0.0;
        lerror_primitive[istate][2] = 0.0;
    }
    // Integrate every cell and compute L2
    std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
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
            double temperature_at_q = this->real_gas_physics->compute_temperature(soln_at_q);

            const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

            std::array<double, nstate> soln_exact;
            for(int istate = 0; istate < nstate; istate++)
                soln_exact[istate] = flow_solver->flow_solver_case->initial_condition_function->value(qpoint,istate);
            soln_exact_primitive = this->real_gas_physics->convert_conservative_to_primitive(soln_exact);
            double temperature_exact = this->real_gas_physics->compute_temperature(soln_exact);
            
            for(int istate = 0; istate < nstate; ++istate) {
                std::array<double, nstate> soln_at_q_primitive = this->real_gas_physics->convert_conservative_to_primitive(soln_at_q);
                // if(istate==0)
                //     std::cout << soln_at_q_primitive[istate] << " " << soln_exact_primitive[istate] << std::endl;
                lerror_primitive[istate][0] += pow(abs(soln_at_q_primitive[istate] - soln_exact_primitive[istate]), 1.0) * fe_values_extra.JxW(iquad);
                lerror_primitive[istate][1] += pow(abs(soln_at_q_primitive[istate] - soln_exact_primitive[istate]), 2.0) * fe_values_extra.JxW(iquad);
                //L-infinity norm
                lerror_primitive[istate][2] = std::max(abs(soln_at_q_primitive[istate]-soln_exact_primitive[istate]), lerror_primitive[istate][2]);
            }
            lerror_primitive[nstate][0] += pow(abs(temperature_at_q - temperature_exact), 1.0) * fe_values_extra.JxW(iquad);
            lerror_primitive[nstate][1] += pow(abs(temperature_at_q - temperature_exact), 2.0) * fe_values_extra.JxW(iquad);
            lerror_primitive[nstate][2] = std::max(abs(temperature_at_q-temperature_exact), lerror_primitive[nstate][2]);
        }
    }
    //MPI sum
    std::array<std::array<double,3>,nstate+1> lerror_mpi;
    for(int istate = 0; istate < nstate+1; ++istate) {
    
        lerror_mpi[istate][0] = dealii::Utilities::MPI::sum(lerror_primitive[istate][0], this->mpi_communicator);
        lerror_mpi[istate][1] = dealii::Utilities::MPI::sum(lerror_primitive[istate][1], this->mpi_communicator);

        lerror_mpi[istate][1] = pow(lerror_mpi[istate][1], 1.0/2.0);

        lerror_mpi[istate][2] = dealii::Utilities::MPI::max(lerror_primitive[istate][2], this->mpi_communicator);
    }

    return lerror_mpi;
}

template <int dim, int nspecies, int nstate>
int MultispeciesVortexAdvection<dim, nspecies, nstate>::run_test() const
{
    pcout << " Running Multispecies Vortex Advection test. " << std::endl;
    pcout << dim << "    " << nstate << std::endl;
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;

    PHiLiP::Parameters::ManufacturedConvergenceStudyParam manu_grid_conv_param = all_parameters_new.manufactured_convergence_study_param;

    const unsigned int n_grids = manu_grid_conv_param.number_of_grids;
    dealii::ConvergenceTable convergence_table;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error_l2(n_grids);
    double final_order = 0.0;
    double expected_order = all_parameters_new.flow_solver_param.expected_order_at_final_time;
    if(expected_order==0.0)
        expected_order = all_parameters_new.flow_solver_param.poly_degree + 1.0;

    for (unsigned int igrid = 1; igrid < n_grids; igrid++) {

        pcout << "\n" << "Creating FlowSolver" << std::endl;

        Parameters::AllParameters param = *(TestsBase::all_parameters);
        param.flow_solver_param.grid_degree = param.flow_solver_param.poly_degree + 1;
        param.flow_solver_param.number_of_grid_elements_per_dimension = pow(2.0,igrid);
        int grid_elem = param.flow_solver_param.number_of_grid_elements_per_dimension;
        param.flow_solver_param.number_of_grid_elements_x = grid_elem;
        
        // Create flow solver to access DG object which is needed to calculate time step
        std::shared_ptr<FlowSolver::FlowSolver<dim, nspecies, nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim, nspecies, nstate>::select_flow_case(&param, parameter_handler);

        const unsigned int n_global_active_cells = flow_solver->dg->triangulation->n_global_active_cells();
        const int poly_degree = all_parameters_new.flow_solver_param.poly_degree;
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

        const std::array<std::array<double,3>,nstate+1> lerror_mpi_sum = calculate_l_n_error(flow_solver->dg, poly_degree, final_time_actual, flow_solver);

        // Convergence table
        const double dx = 10.0 / pow(n_dofs, (1.0 / dim));
        grid_size[igrid] = dx;
        soln_error_l2[igrid] = lerror_mpi_sum[0][1];

        convergence_table.add_value("p", poly_degree);
        convergence_table.add_value("cells", n_global_active_cells);
        convergence_table.add_value("DoFs", n_dofs);
        convergence_table.add_value("dx", dx);
        convergence_table.add_value("density_L1", lerror_mpi_sum[0][0]);
        convergence_table.add_value("density_L2", lerror_mpi_sum[0][1]);
        convergence_table.add_value("density_Linf", lerror_mpi_sum[0][2]);
        convergence_table.add_value("pressure_L1", lerror_mpi_sum[dim+1][0]);
        convergence_table.add_value("pressure_L2", lerror_mpi_sum[dim+1][1]);
        convergence_table.add_value("pressure_Linf", lerror_mpi_sum[dim+1][2]);
        // convergence_table.add_value("temp_L1", lerror_mpi_sum[nstate][0]);
        // convergence_table.add_value("temp_L2", lerror_mpi_sum[nstate][1]);
        // convergence_table.add_value("temp_Linf", lerror_mpi_sum[nstate][2]);
        convergence_table.add_value("Y_H2_L1", lerror_mpi_sum[dim+2][0]);
        convergence_table.add_value("Y_H2_L2", lerror_mpi_sum[dim+2][1]);
        convergence_table.add_value("Y_H2_Linf", lerror_mpi_sum[dim+2][2]);

        this->pcout << " Grid size h: " << dx
            << " Density L1-soln_error: " << lerror_mpi_sum[0][0]
            << " Density L2-soln_error: " << lerror_mpi_sum[0][1]
            << " Density Linf-soln_error: " << lerror_mpi_sum[0][2]
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
        convergence_table.evaluate_convergence_rates("density_L1", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("density_L2", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("density_Linf", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("pressure_L1", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("pressure_L2", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("pressure_Linf", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        // convergence_table.evaluate_convergence_rates("temp_L1", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        // convergence_table.evaluate_convergence_rates("temp_L2", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        // convergence_table.evaluate_convergence_rates("temp_Linf", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("Y_H2_L1", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("Y_H2_L2", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("Y_H2_Linf", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("density_L1", true);
        convergence_table.set_scientific("density_L2", true);
        convergence_table.set_scientific("density_Linf", true);
        convergence_table.set_scientific("pressure_L1", true);
        convergence_table.set_scientific("pressure_L2", true);
        convergence_table.set_scientific("pressure_Linf", true);
        // convergence_table.set_scientific("temp_L1", true);
        // convergence_table.set_scientific("temp_L2", true);
        // convergence_table.set_scientific("temp_Linf", true);
        convergence_table.set_scientific("Y_H2_L1", true);
        convergence_table.set_scientific("Y_H2_L2", true);
        convergence_table.set_scientific("Y_H2_Linf", true);
        if (this->pcout.is_active()) convergence_table.write_text(this->pcout.get_stream());

        std::ofstream table_file("convergence_rates.txt");
        convergence_table.write_text(table_file);

        
    }//end of grid loop

    if(final_order > expected_order - 0.1)
        return 0;
    else
        return 1;
}

#if PHILIP_SPECIES>1
template class MultispeciesVortexAdvection<PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1>;
#endif
} // Tests namespace
} // PHiLiP namespace
