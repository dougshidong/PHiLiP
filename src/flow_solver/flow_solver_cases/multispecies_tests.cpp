#include "multispecies_tests.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/straight_periodic_cube.hpp"
#include "mesh/grids/positivity_preserving_tests_grid.h"
#include "mesh/gmsh_reader.hpp"

namespace PHiLiP {

namespace FlowSolver {
//==========================================================================
// FLOW SOLVER CASE FOR TESTS INVOLVING MULTISPECIES AND RELATED PARAMETERS
//==========================================================================
template <int dim, int nspecies, int nstate>
MultispeciesTests<dim, nspecies, nstate>::MultispeciesTests(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : CubeFlow_UniformGrid<dim, nspecies, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
        , number_of_cells_per_direction(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension)
        , domain_left(this->all_param.flow_solver_param.grid_left_bound)
        , domain_right(this->all_param.flow_solver_param.grid_right_bound)
        , domain_size(pow(this->domain_right - this->domain_left, dim))
{ }

template <int dim, int nspecies, int nstate>
std::shared_ptr<Triangulation> MultispeciesTests<dim,nspecies,nstate>::generate_grid() const
{
    
    this->pcout << "- Generating grid using dealii GridGenerator" << std::endl;
    
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
    #if PHILIP_DIM!=1
        this->mpi_communicator
    #endif
    );
        
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    if(dim==1 && flow_case_type == flow_case_enum::multi_species_sod_shock_tube) {
        Grids::shock_tube_1D_grid<dim>(*grid, &this->all_param.flow_solver_param);
    } else {
        Grids::straight_periodic_cube<dim, Triangulation>(grid, domain_left, domain_right,
                                                            number_of_cells_per_direction);
    }
    return grid;

}

template <int dim, int nspecies, int nstate>
void MultispeciesTests<dim,nspecies,nstate>::display_grid_parameters() const
{
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    std::string grid_type_string = "";
    if(flow_case_type == flow_case_enum::multi_species_sod_shock_tube) {
        grid_type_string = "1d_shock_tube";
    } else {
        grid_type_string = "straight_periodic_cube";
    }
    // Display the information about the grid
    this->pcout << "- Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Grid degree: " << this->all_param.flow_solver_param.grid_degree << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << this->domain_left << std::endl;
    this->pcout << "- - Domain right: " << this->domain_right << std::endl;

    int cells_in_each_dir = 0;
    if(this->number_of_cells_per_direction > this->all_param.flow_solver_param.number_of_grid_elements_x)
        cells_in_each_dir = this->number_of_cells_per_direction;
    else
        cells_in_each_dir = this->all_param.flow_solver_param.number_of_grid_elements_x;

    this->pcout << "- - Number of cells in each direction: " << cells_in_each_dir << std::endl;
    if constexpr(dim==1) this->pcout << "- - Domain length: " << this->domain_size << std::endl;
    if constexpr(dim==2) this->pcout << "- - Domain area: " << this->domain_size << std::endl;
    if constexpr(dim==3) this->pcout << "- - Domain volume: " << this->domain_size << std::endl;
}

template <int dim, int nspecies, int nstate>
void MultispeciesTests<dim,nspecies,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->display_grid_parameters();
}

template <int dim, int nspecies, int nstate>
void MultispeciesTests<dim,nspecies,nstate>::modify_dg_object(std::shared_ptr <DGBase<dim, nspecies, double>> dg) const
{
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;
    
    if (flow_case_type == flow_case_enum::multi_species_isentropic_vortex && this->all_param.flow_solver_param.final_time == 40.0) {
        this->pcout << "Reversing velocities to conduct flow reversal...";
        for (auto soln_cell : dg->dof_handler.active_cell_iterators()) {
            if (!soln_cell->is_locally_owned()) continue;

            std::vector<dealii::types::global_dof_index> current_dofs_indices;
            // Current reference element related to this physical cell
            const int i_fele = soln_cell->active_fe_index();
            const dealii::FESystem<dim, dim>& current_fe_ref = dg->fe_collection[i_fele];
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

            // Allocate solution dofs
            for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
                const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
                soln_coeff[istate][ishape] = dg->solution[current_dofs_indices[idof]];
            }

            // Write limited solution dofs to the global solution vector.
            for (int istate = 0; istate < nstate; istate++) {
                for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
                    if (istate == 1 || istate == 2)
                        soln_coeff[istate][ishape] *= -1;
                    const unsigned int idof = istate * n_shape_fns + ishape;
                    dg->solution[current_dofs_indices[idof]] = soln_coeff[istate][ishape]; //
                }
            }
        }
        this->pcout << "done." << std::endl;
    }
}

template <int dim, int nspecies, int nstate>
void MultispeciesTests<dim, nspecies, nstate>::compute_unsteady_data_and_write_to_table(
    const std::shared_ptr<ODE::ODESolverBase<dim, nspecies, double>> ode_solver,
    const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table,
    const bool do_write_unsteady_data_table_file)
{
    //unpack current iteration and current time from ode solver
    const unsigned int current_iteration = ode_solver->current_iteration;
    const double current_time = ode_solver->current_time;

    if (this->mpi_rank == 0) {

        unsteady_data_table->add_value("iteration", current_iteration);
        // Add values to data table
        this->add_value_to_data_table(current_time, "time", unsteady_data_table);
        // Write to file
        if(do_write_unsteady_data_table_file){
            std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
            unsteady_data_table->write_text(unsteady_data_table_file);
        }
    }

    if (current_iteration % this->all_param.ode_solver_param.print_iteration_modulo == 0) {
        // Print to console
        this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << std::setprecision(16) << current_time;

        this->pcout << std::endl;
    }

    // Update local maximum wave speed before calculating next time step
    update_maximum_local_wave_speed(*dg);
}

template class MultispeciesTests <PHILIP_DIM, PHILIP_SPECIES,PHILIP_DIM+PHILIP_SPECIES+1>;
} // FlowSolver namespace
} // PHiLiP namespace

