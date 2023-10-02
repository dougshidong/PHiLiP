#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

#include "burgers_limiter.h"

#include "physics/initial_conditions/initial_condition_function.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersLimiter<dim, nstate>::BurgersLimiter(
    const PHiLiP::Parameters::AllParameters* const parameters_input,
    const dealii::ParameterHandler& parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
void BurgersLimiter<dim, nstate>::set_initial_time_step(const unsigned int n_global_active_cells, const int poly_degree) const
{
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    double left = all_parameters_new.flow_solver_param.grid_left_bound;
    double right = all_parameters_new.flow_solver_param.grid_right_bound;

    double n_dofs_cfl = pow(n_global_active_cells, dim) * pow(poly_degree + 1.0, dim);
    double delta_x = (PHILIP_DIM == 2) ? (right - left) / pow(n_global_active_cells, (1.0 / dim)) : (right - left) / pow(n_dofs_cfl, (1.0 / dim));
    all_parameters_new.ode_solver_param.initial_time_step = (PHILIP_DIM == 2) ? (1.0 / 14.0) * pow(delta_x, 1.0/*(4.0/3.0)*/) : (1.0 / 24.0) * pow(delta_x, (1.0));
    pcout << "time_step:   " << all_parameters_new.ode_solver_param.initial_time_step << std::endl;
}

template <int dim, int nstate>
int BurgersLimiter<dim, nstate>::run_test() const
{
    pcout << " Running Burgers limiter test. " << std::endl;
    pcout << dim << "    " << nstate << std::endl;
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;

    int test_result = 1;

    if (!all_parameters_new.limiter_param.use_OOA) {
        test_result = run_burgers_lim();
    }
    else {
        test_result = run_burgers_lim_conv();
    }
    return test_result; //if got to here means passed the test, otherwise would've failed earlier
}

template <int dim, int nstate>
int BurgersLimiter<dim, nstate>::run_burgers_lim() const
{
    pcout << "\n" << "Creating FlowSolver" << std::endl;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);
    const unsigned int n_global_active_cells2 = flow_solver->dg->triangulation->n_global_active_cells();
    const int poly_degree = all_parameters_new.flow_solver_param.poly_degree;

    set_initial_time_step(n_global_active_cells2, poly_degree);

    flow_solver->run();

    return 0;
}

template <int dim, int nstate>
int BurgersLimiter<dim, nstate>::run_burgers_lim_conv() const
{
    std::cout << "correct condition" << std::endl;
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    PHiLiP::Parameters::ManufacturedConvergenceStudyParam manu_grid_conv_param = all_parameters_new.manufactured_convergence_study_param;

    const unsigned int n_grids = manu_grid_conv_param.number_of_grids;
    dealii::ConvergenceTable convergence_table;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error(n_grids);

    for (unsigned int igrid = 0; igrid < n_grids; igrid++) {

        pcout << "\n" << "Creating FlowSolver" << std::endl;

        Parameters::AllParameters param = *(TestsBase::all_parameters);
        param.flow_solver_param.number_of_mesh_refinements = igrid;
        std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);
        const unsigned int n_global_active_cells = flow_solver->dg->triangulation->n_global_active_cells();
        const int poly_degree = all_parameters_new.flow_solver_param.poly_degree;

        set_initial_time_step(n_global_active_cells, poly_degree);

        flow_solver->run();

        // output results
        const unsigned int n_dofs = flow_solver->dg->dof_handler.n_dofs();
        this->pcout << "Dimension: " << dim
            << "\t Polynomial degree p: " << poly_degree
            << std::endl
            << "Grid number: " << igrid + 1 << "/" << n_grids
            << ". Number of active cells: " << n_global_active_cells
            << ". Number of degrees of freedom: " << n_dofs
            << std::endl;

        // Overintegrate the error to make sure there is not integration error in the error estimate
        int overintegrate = 10;
        dealii::QGauss<dim> quad_extra(poly_degree + 1 + overintegrate);
        dealii::FEValues<dim, dim> fe_values_extra(*(flow_solver->dg->high_order_grid->mapping_fe_field), flow_solver->dg->fe_collection[poly_degree], quad_extra,
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
        const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
        std::array<double, nstate> soln_at_q;

        double l2error = 0.0;

        // Integrate every cell and compute L2
        const double pi = atan(1) * 4.0;
        std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
        for (auto cell = flow_solver->dg->dof_handler.begin_active(); cell != flow_solver->dg->dof_handler.end(); ++cell) {
            if (!cell->is_locally_owned()) continue;

            fe_values_extra.reinit(cell);
            cell->get_dof_indices(dofs_indices);

            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {

                std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
                    const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                    soln_at_q[istate] += flow_solver->dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                }

                for (int istate = 0; istate < nstate; ++istate) {
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    double uexact = 1.0;
                    for (int idim = 0; idim < dim; idim++) {
                        uexact *= cos(pi * (qpoint[idim] - all_parameters_new.flow_solver_param.final_time));//for grid 1-3
                    }
                    l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                }
            }

        }
        const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, this->mpi_communicator));

        // Convergence table
        const double dx = 1.0 / pow(n_dofs, (1.0 / dim));
        grid_size[igrid] = dx;
        soln_error[igrid] = l2error_mpi_sum;

        convergence_table.add_value("p", poly_degree);
        convergence_table.add_value("cells", n_global_active_cells);
        convergence_table.add_value("DoFs", n_dofs);
        convergence_table.add_value("dx", dx);
        convergence_table.add_value("soln_L2_error", l2error_mpi_sum);

        this->pcout << " Grid size h: " << dx
            << " L2-soln_error: " << l2error_mpi_sum
            << " Residual: " << flow_solver->ode_solver->residual_norm
            << std::endl;

        if (igrid > 0) {
            const double slope_soln_err = log(soln_error[igrid] / soln_error[igrid - 1])
                / log(grid_size[igrid] / grid_size[igrid - 1]);
            this->pcout << "From grid " << igrid - 1
                << "  to grid " << igrid
                << "  dimension: " << dim
                << "  polynomial degree p: " << poly_degree
                << std::endl
                << "  solution_error1 " << soln_error[igrid - 1]
                << "  solution_error2 " << soln_error[igrid]
                << "  slope " << slope_soln_err
                << std::endl;
        }

        this->pcout << " ********************************************"
            << std::endl
            << " Convergence rates for p = " << poly_degree
            << std::endl
            << " ********************************************"
            << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
        if (this->pcout.is_active()) convergence_table.write_text(this->pcout.get_stream());
    }//end of grid loop
    return 0;
}

//double finalTime = 2.0;
//double finalTime = (PHILIP_DIM == 2) ? 0.05 : 0.15;//Comparison with 2010 Zhang Shu Paper

template class BurgersLimiter<PHILIP_DIM,PHILIP_DIM>;

} // Tests namespace
} // PHiLiP namespace
