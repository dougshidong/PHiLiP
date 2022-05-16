#include "periodic_1D_unsteady.h"
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

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// PERIODIC 1D DOMAIN FOR UNSTEADY CALCULATIONS
//=========================================================

template <int dim, int nstate>
Periodic1DUnsteady<dim, nstate>::Periodic1DUnsteady(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicCubeFlow<dim, nstate>(parameters_input)
        , number_of_cells_per_direction(this->all_param.grid_refinement_study_param.grid_size)
        , domain_left(this->all_param.grid_refinement_study_param.grid_left)
        , domain_right(this->all_param.grid_refinement_study_param.grid_right)
        , domain_size(pow(this->domain_right - this->domain_left, dim))
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param. unsteady_data_table_filename+ ".txt")
{

}


template <int dim, int nstate>
std::shared_ptr<Triangulation> Periodic1DUnsteady<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
    );

    
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_case_type = this->all_param.flow_solver_param.flow_case_type;
    
    if (flow_case_type == FlowCaseEnum::advection_periodic){
        Grids::straight_periodic_cube<dim,Triangulation>(grid, domain_left, domain_right, number_of_cells_per_direction);
    }else if (flow_case_type == FlowCaseEnum::burgers_periodic){
        const int number_of_refinements = log(number_of_cells_per_direction)/log(2);

        //using hyper_cube and relying on use_periodic_bc flag
        dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, true);//colorize = true
        grid->refine_global(number_of_refinements);
    }

    return grid;
}

template <int dim, int nstate>
double Periodic1DUnsteady<dim, nstate>::compute_energy(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
    double energy = 0.0;
    for (unsigned int i = 0; i < dg->solution.size(); ++i)
    {
        energy += 1./(dg->global_inverse_mass_matrix.diag_element(i)) * dg->solution(i) * dg->solution(i);
    }
    return energy;
}

template <int dim, int nstate>
void Periodic1DUnsteady<dim, nstate>::compute_unsteady_data_and_write_to_table(
       const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg ,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table )
{
    double dt = this->all_param.ode_solver_param.initial_time_step;
    int output_solution_every_n_iterations = round(this->all_param.ode_solver_param.output_solution_every_dt_time_intervals/dt);
 
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    if (flow_case_type == FlowCaseEnum::advection_periodic){
        if ((current_iteration % output_solution_every_n_iterations) == 0){
            this->pcout << "    Iter: " << current_iteration
                        << "    Time: " << current_time
                        << std::endl;
        }
        (void) dg;
        (void) unsteady_data_table;
    }
    else if (flow_case_type == FlowCaseEnum::burgers_periodic){
        double energy = this->compute_energy(dg);
    
        if ((current_iteration % output_solution_every_n_iterations) == 0){
            this->pcout << "    Iter: " << current_iteration
                        << "    Time: " << current_time
                        << "    Energy: " << energy
                        << std::endl;
        }
    
        //detecting if the current run is calculating a reference solution
        int number_timesteps_ref = this->all_param.flow_solver_param.number_of_timesteps_for_reference_solution;
        double final_time = this->all_param.flow_solver_param.final_time;
        bool is_reference_solution = (dt < 2 * final_time/number_timesteps_ref);

        if(this->mpi_rank==0 && !is_reference_solution) {
            unsteady_data_table->add_value("iteration", current_iteration);
            this->add_value_to_data_table(current_time,"time",unsteady_data_table);
            this->add_value_to_data_table(energy,"energy",unsteady_data_table);
            // Write to file
            std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
            unsteady_data_table->write_text(unsteady_data_table_file);
        }
    }

}

#if PHILIP_DIM==1
template class Periodic1DUnsteady <PHILIP_DIM,PHILIP_DIM>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

