#include "1d_burgers_viscous_snapshot.h"
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
BurgersViscousSnapshot<dim, nstate>::BurgersViscousSnapshot(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_refinements(this->all_param.grid_refinement_study_param.num_refinements)
        , domain_left(this->all_param.grid_refinement_study_param.grid_left)
        , domain_right(this->all_param.grid_refinement_study_param.grid_right)
{
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> BurgersViscousSnapshot<dim,nstate>::generate_grid() const
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
void BurgersViscousSnapshot<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Display the information about the grid
    this->pcout << "\n- GRID INFORMATION:" << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << domain_left << std::endl;
    this->pcout << "- - Domain right: " << domain_right << std::endl;
    this->pcout << "- - Number of refinements:  " << number_of_refinements << std::endl;
}

template <int dim, int nstate>
void BurgersViscousSnapshot<dim, nstate>::compute_unsteady_data_and_write_to_table(
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

#if PHILIP_DIM==1
template class BurgersViscousSnapshot<PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace
