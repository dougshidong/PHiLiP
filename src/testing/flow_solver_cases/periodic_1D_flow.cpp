#include "periodic_1D_flow.h"
#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "mesh/grids/straight_periodic_cube.hpp"
#include <deal.II/base/table_handler.h>

namespace PHiLiP {

namespace Tests {
//=========================================================
// PERIODIC 1D DOMAIN FOR TIME REFINEMENT STUDY
//=========================================================
template <int dim, int nstate>
Periodic1DFlow<dim, nstate>::Periodic1DFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_cells_per_direction(this->all_param.grid_refinement_study_param.grid_size)
        , domain_left(this->all_param.grid_refinement_study_param.grid_left)
        , domain_right(this->all_param.grid_refinement_study_param.grid_right)
        , domain_volume(pow(domain_right - domain_left, dim))
        , unsteady_data_table_filename_with_extension(parameters_input->flow_solver_param.unsteady_data_table_filename+".txt")
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = parameters_input->flow_solver_param.flow_case_type;

    this->number_times_refined_by_half = 0;
}

template <int dim, int nstate>
void Periodic1DFlow<dim,nstate>::display_flow_solver_setup() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::advection)                {pde_string = "advection";}
    this->pcout << "- PDE Type: " << pde_string << std::endl;
    this->pcout << "- Polynomial degree: " << this->all_param.grid_refinement_study_param.poly_degree << std::endl;
    this->pcout << "- Constant time step size: " << this->all_param.ode_param.initial_time_step << std::endl;
    this->pcout << "- Final time: " << this->all_param.flow_solver_param.final_time << std::endl;

}

template <int dim, int nstate>
std::shared_ptr<Triangulation> Periodic1DFlow<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
    );
    // Grids::straight_periodic_cube<dim,dealii::parallel::distributed::Triangulation<dim>>(grid, domain_left, domain_right, number_of_cells_per_direction);
    // Should modify the straight_periodic_cube to allow 1D later

    n_refinements = 5; //currently hard-coded, SHOULD CHANGE
    colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    dealii::GridTools::collect_periodic_faces(*grid, 0,1,0. matched_pairs);
    grid->add_periodicity(matched_pairs);
    grid->refine_global(n_refinements);

    // Display the information about the grid
    this->pcout << "\n- GRID INFORMATION:" << std::endl;
    // pcout << "- - Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << domain_left << std::endl;
    this->pcout << "- - Domain right: " << domain_right << std::endl;
    this->pcout << "- - Number of cells in each direction: " << number_of_cells_per_direction << std::endl;
    this->pcout << "- - Domain volume: " << domain_volume << std::endl;

    return grid;
}

template <int dim, int nstate>
double Periodic1DFlow<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const double initial_time_step = all_parameters.ode_solver_param.initial_time_step;
    double constant_time_step = initial_time_step * pow(0.5, this->number_of_times_refined_by_half);
    (this->number_of_times_refined_by_half)++
    return constant_time_step;
}

template <int dim, int nstate>
void Periodic1DFlow<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const
{

    // Pretty sure this doesn't actually need to write anything
    
    
    if(this->mpi_rank==0) {
        unsteady_data_table->add_value("cells", number_of_cells_per_direction);
        unsteady_data_table->add_value("space_poly_degree", all_param.grid_refinement_study_param.poly_degree);
        unsteady_data_table->add_value("dt", all_param.ode_solver_param.initial_time_step); //maybe store this in the class
        unsteady_data_table->set_precision("dt",3);
        unsteady_data_table->set_scientific("dt", true);
        //ADD ERROR CALC BASED ON EXACT SOLUTION

    }

/*    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Energy: " << kinetic_energy
                << std::endl;
*/

}

//not confident this is correct
#if PHILIP_DIM==1
    template class Periodic1DFlow <PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace

