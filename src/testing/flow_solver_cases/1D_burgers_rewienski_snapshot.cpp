#include "1D_burgers_rewienski_snapshot.h"
#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/vector.h>
#include "linear_solver/linear_solver.h"

namespace PHiLiP {

namespace Tests {

template <int dim, int nstate>
BurgersRewienskiSnapshot<dim, nstate>::BurgersRewienskiSnapshot(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_refinements(this->all_param.grid_refinement_study_param.num_refinements)
        , domain_left(this->all_param.grid_refinement_study_param.grid_left)
        , domain_right(this->all_param.grid_refinement_study_param.grid_right)
{
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> BurgersRewienskiSnapshot<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
    );
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void BurgersRewienskiSnapshot<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Display the information about the grid
    this->pcout << "\n- GRID INFORMATION:" << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << domain_left << std::endl;
    this->pcout << "- - Domain right: " << domain_right << std::endl;
    this->pcout << "- - Number of refinements:  " << number_of_refinements << std::endl;
}

template <int dim, int nstate>
void BurgersRewienskiSnapshot<dim, nstate>::compute_unsteady_data_and_write_to_table(
    const unsigned int current_iteration,
    const double current_time,
    const std::shared_ptr <DGBase<dim, double>> dg,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    if (this->all_param.ode_solver_param.output_solution_vector_modulo > 0) {
        if (current_iteration % this->all_param.ode_solver_param.output_solution_vector_modulo == 0) {
            for (unsigned int i = 0; i < dg->solution.size(); ++i) {
                unsteady_data_table->add_value(
                        "Time:" + std::to_string(current_time),
                        dg->solution[i]);
            }
            unsteady_data_table->set_precision("Time:" + std::to_string(current_time), 16);
            // Write to file
            std::ofstream unsteady_data_table_file(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt");
            unsteady_data_table->write_text(unsteady_data_table_file);
        }
    }
}

template <int dim, int nstate>
void BurgersRewienskiSnapshot<dim, nstate>::steady_state_postprocessing(std::shared_ptr <DGBase<dim, double>> dg) const
{
    this->pcout << "Computing sensitivity to parameter" << std::endl;
    int overintegrate = 0;
    dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->operators->fe_collection_basis[dg->max_degree], quad_extra,
                                              dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
    dealii::LinearAlgebra::distributed::Vector<double> sensitivity_dRdb(dg->n_dofs());
    sensitivity_dRdb*=0;
    for (auto cell : dg->dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit(cell);
        cell->get_dof_indices(dofs_indices);

        for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                double b = this->all_param.burgers_param.rewienski_b;
                const dealii::Point<dim, double> point = fe_values_extra.quadrature_point(iquad);
                sensitivity_dRdb[dofs_indices[idof]] += fe_values_extra.shape_value_component(idof, iquad, istate) * 0.02 * point[0] * exp(point[0] * b) * fe_values_extra.JxW(iquad);
            }
        }
    }

    //Apply inverse Jacobian
    dealii::LinearAlgebra::distributed::Vector<double> sensitivity_dWdb(dg->n_dofs());
    const bool compute_dRdW=true;
    const bool compute_dRdX=false;
    const bool compute_d2R=false;
    double flow_CFL_ = 0.0;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);
    dg->system_matrix *= -1.0;

    PHiLiP::Parameters::LinearSolverParam linear_solver_param;
    linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::direct;

    solve_linear(dg->system_matrix, sensitivity_dRdb, sensitivity_dWdb, linear_solver_param);

    dealii::TableHandler solutions_table;
    for (unsigned int i = 0; i < sensitivity_dWdb.size(); ++i) {
        solutions_table.add_value(
                "Sensitivity:",
                sensitivity_dWdb[i]);
    }
    solutions_table.set_precision("Sensitivity:", 16);
    std::ofstream out_file(this->all_param.flow_solver_param.sensitivity_table_filename + ".txt");
    solutions_table.write_text(out_file);
}

#if PHILIP_DIM==1
template class BurgersRewienskiSnapshot<PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace
