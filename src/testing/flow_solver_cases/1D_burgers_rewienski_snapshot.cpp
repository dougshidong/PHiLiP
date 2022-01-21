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

namespace PHiLiP {

namespace Tests {

template <int dim, int nstate>
BurgersRewienskiSnapshot<dim, nstate>::BurgersRewienskiSnapshot(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolver<dim,nstate>(parameters_input)
        , number_of_refinements(this->all_param.grid_refinement_study_param.num_refinements)
        , domain_left(this->all_param.grid_refinement_study_param.grid_left)
        , domain_right(this->all_param.grid_refinement_study_param.grid_right)
{
}

template <int dim, int nstate>
void BurgersRewienskiSnapshot<dim,nstate>::display_flow_solver_setup() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::burgers_rewienski)    {pde_string = "burgers_rewienski";}
    this->pcout << "- PDE Type: " << pde_string << std::endl;
    this->pcout << "- Polynomial degree: " << this->poly_degree << std::endl;
    this->pcout << "- Final time: " << this->final_time << std::endl;
}

template <int dim, int nstate>
void BurgersRewienskiSnapshot<dim,nstate>
::generate_grid(std::shared_ptr<Triangulation> grid) const
{
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    grid->refine_global(number_of_refinements);
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
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const
{
    if (this->ode_param.output_solution_vector_modulo > 0) {
        if (current_iteration % this->ode_param.output_solution_vector_modulo == 0) {
            for (unsigned int i = 0; i < dg->solution.size(); ++i) {
                unsteady_data_table->add_value(
                        "Time:" + std::to_string(current_time),
                        dg->solution[i]);
            }
            // Write to file
            std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
            unsteady_data_table->write_text(unsteady_data_table_file);
        }
    }
}

#if PHILIP_DIM==1
template class BurgersRewienskiSnapshot<PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace
