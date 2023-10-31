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
double BoundPreservingLimiterTests<dim, nstate>::calculate_uexact(const dealii::Point<dim> qpoint, const dealii::Tensor<1, 3, double> adv_speeds, double final_time) const
{
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case = all_parameters_new.flow_solver_param.flow_case_type;
    const double pi = atan(1) * 4.0;

    double uexact = 1.0;
    if (flow_case == Parameters::FlowSolverParam::FlowCaseType::low_density_2d && dim == 2) {
        uexact = 1.00 + 0.99 * sin(qpoint[0] + qpoint[1] - (2.00 * final_time));
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
double BoundPreservingLimiterTests<dim, nstate>::calculate_l2error(
    std::shared_ptr<DGBase<dim, double>> dg,
    const int poly_degree,
    const double final_time) const
{
    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(poly_degree + 1 + overintegrate);
    dealii::FEValues<dim, dim> fe_values_extra(*(flow_solver_dg->high_order_grid->mapping_fe_field), flow_solver_dg->fe_collection[poly_degree], quad_extra,
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double, nstate> soln_at_q;

    double l2error = 0.0;

    // Integrate every cell and compute L2
    std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
    const dealii::Tensor<1, 3, double> adv_speeds = Parameters::ManufacturedSolutionParam::get_default_advection_vector();
    for (auto cell = flow_solver_dg->dof_handler.begin_active(); cell != flow_solver_dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit(cell);
        cell->get_dof_indices(dofs_indices);

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += flow_solver_dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
            double uexact = calculate_uexact(qpoint, adv_speeds, final_time);
            l2error += pow(soln_at_q[0] - uexact, 2) * fe_values_extra.JxW(iquad);
        }
    }
    const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, this->mpi_communicator));
    return l2error_mpi_sum;
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
    const double pi = atan(1) * 4.0;
    if (flow_case == Parameters::FlowSolverParam::FlowCaseType::low_density_2d) {
        param.flow_solver_param.grid_left_bound = 0.0;
        param.flow_solver_param.grid_right_bound = 2.0 * pi;
    }

    std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);
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
    std::vector<double> soln_error(n_grids);

    for (unsigned int igrid = 0; igrid < n_grids; igrid++) {

        pcout << "\n" << "Creating FlowSolver" << std::endl;

        Parameters::AllParameters param = *(TestsBase::all_parameters);
        param.flow_solver_param.number_of_mesh_refinements = igrid;

        using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
        flow_case_enum flow_case = all_parameters_new.flow_solver_param.flow_case_type;
        const double pi = atan(1) * 4.0;

        if (flow_case == Parameters::FlowSolverParam::FlowCaseType::low_density_2d) {
            param.flow_solver_param.grid_left_bound = 0.0;
            param.flow_solver_param.grid_right_bound = 2.0 * pi;
        }

        std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim, nstate>::select_flow_case(&param, parameter_handler);
        const unsigned int n_global_active_cells = flow_solver->dg->triangulation->n_global_active_cells();
        const int poly_degree = all_parameters_new.flow_solver_param.poly_degree;
        const double final_time = all_parameters_new.flow_solver_param.final_time;

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

        const double l2error_mpi_sum = calculate_l2error(flow_solver->dg, poly_degree, final_time);

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

#if PHILIP_DIM==1
template class BoundPreservingLimiterTests<PHILIP_DIM, PHILIP_DIM>;
#elif PHILIP_DIM==2
template class BoundPreservingLimiterTests<PHILIP_DIM, PHILIP_DIM>;
template class BoundPreservingLimiterTests<PHILIP_DIM, PHILIP_DIM + 2>;
template class BoundPreservingLimiterTests<PHILIP_DIM, 1>;
#endif

} // Tests namespace
} // PHiLiP namespace
