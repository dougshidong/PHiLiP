#include "periodic_1D_unsteady.h"

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// PERIODIC 1D DOMAIN FOR UNSTEADY CALCULATIONS
//=========================================================

template <int dim, int nstate>
Periodic1DUnsteady<dim, nstate>::Periodic1DUnsteady(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicCubeFlow<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{

}


template <int dim, int nstate>
double Periodic1DUnsteady<dim, nstate>::compute_energy_collocated(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
    // Intention is to eventually calculate energy in physics and generalize to arbitrary nodes
    // For now, using collocated energy calculation
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
    const double dt = this->all_param.ode_solver_param.initial_time_step;
    int output_solution_every_n_iterations = round(this->all_param.ode_solver_param.output_solution_every_dt_time_intervals/dt);
 
    using PDEEnum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDEEnum pde_type = this->all_param.pde_type;

    if (pde_type == PDEEnum::advection){
        if ((current_iteration % output_solution_every_n_iterations) == 0){
            this->pcout << "    Iter: " << current_iteration
                        << "    Time: " << current_time
                        << std::endl;
        }
        (void) dg;
        (void) unsteady_data_table;
    }
    else if ((pde_type == PDEEnum::burgers_inviscid)&&(this->all_param.use_collocated_nodes)){
        const double energy = this->compute_energy_collocated(dg);
    
        if ((current_iteration % output_solution_every_n_iterations) == 0){
            this->pcout << "    Iter: " << current_iteration
                        << "    Time: " << current_time
                        << "    Energy: " << energy
                        << std::endl;
        }
    
        //detecting if the current run is calculating a reference solution 
        const int number_timesteps_ref = this->all_param.time_refinement_study_param.number_of_timesteps_for_reference_solution;
        const double final_time = this->all_param.flow_solver_param.final_time;
        const bool is_reference_solution = (dt < 2 * final_time/number_timesteps_ref);

        if(this->mpi_rank==0 && !is_reference_solution) {
            //omit writing if current calculation is for a reference solution
            unsteady_data_table->add_value("iteration", current_iteration);
            this->add_value_to_data_table(current_time,"time",unsteady_data_table);
            this->add_value_to_data_table(energy,"energy",unsteady_data_table);
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

