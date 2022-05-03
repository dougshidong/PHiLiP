#include "time_refinement_study_advection.h"
#include "flow_solver.h"
#include "flow_solver_cases/periodic_1D_flow.h"
#include "cmath"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TimeRefinementStudyAdvection<dim, nstate>::TimeRefinementStudyAdvection(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input)
{}



template <int dim, int nstate>
 Parameters::AllParameters TimeRefinementStudyAdvection<dim, nstate>::reinit_params_and_refine_timestep(int refinement) const{
     dealii::ParameterHandler parameter_handler;
     PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
     PHiLiP::Parameters::AllParameters parameters;
     parameters.parse_parameters(parameter_handler);


     //parameters = this->all_parameters; //This doesn't work, operator = not defined. Block below copies params one at a time.

     // Copy all parameters
     parameters.manufactured_convergence_study_param = this->all_parameters->manufactured_convergence_study_param;
     parameters.ode_solver_param = this->all_parameters->ode_solver_param;
     parameters.linear_solver_param = this->all_parameters->linear_solver_param;
     parameters.euler_param = this->all_parameters->euler_param;
     parameters.navier_stokes_param = this->all_parameters->navier_stokes_param;
     parameters.reduced_order_param= this->all_parameters->reduced_order_param;
     parameters.burgers_param = this->all_parameters->burgers_param;
     parameters.grid_refinement_study_param = this->all_parameters->grid_refinement_study_param;
     parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
     parameters.flow_solver_param = this->all_parameters->flow_solver_param;
     parameters.mesh_adaptation_param = this->all_parameters->mesh_adaptation_param;
     parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
     parameters.dimension = this->all_parameters->dimension;
     parameters.pde_type = this->all_parameters->pde_type;
     parameters.test_type = this->all_parameters->test_type;
     parameters.use_weak_form = this->all_parameters->use_weak_form;
     parameters.use_collocated_nodes = this->all_parameters->use_collocated_nodes;
     parameters.use_periodic_bc = this->all_parameters->use_periodic_bc;
     
     parameters.ode_solver_param.initial_time_step *= pow(refine_ratio,refinement);
     //ADD : flow_solver params: change file name
     return parameters;
 }


/*
template <int dim, int nstate>
void TimeRefinementStudyAdvection<dim, nstate>::write_convergence_summary(int refinement, double dt) {
    
    //convergence_table.add_value("cells",grid->n_active_cells());
    //convergence_table.add_value("space_poly_deg", space_poly_degree);
    convergence_table.add_value("refinement", refinement);
    convergence_table.add_value("dt", dt );
    //convergence_table.add_value("L2_error",L2_error );
    //convergence_table.set_precision("L2_error", 4);
    //convergence_table.set_scientific("L2_error", true);
    convergence_table.set_precision("dt", 3);
    convergence_table.set_scientific("dt", true);
    //convergence_table.evaluate_convergence_rates("L2_error", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);


}
*/

template <int dim, int nstate>
int TimeRefinementStudyAdvection<dim, nstate>::run_test() const
{
    dealii::ConvergenceTable convergence_table; 

    for (int refinement = 0; refinement < n_time_calculations; ++refinement){
        
        pcout << "\n\n---------------------------------------------\n Refinement number " << refinement <<
            " of " << n_time_calculations - 1 << std::endl << "---------------------------------------------" << std::endl;
        Parameters::AllParameters params = reinit_params_and_refine_timestep(refinement);

        std::unique_ptr<FlowSolver<dim,nstate>> flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params, parameter_handler);
        static_cast<void>(flow_solver->run_test());
        
        //check L2 error
        //write to data table
        //this->write_convergence_summary(refinement, params.ode_solver_param.initial_time_step);

        const double dt =  params.ode_solver_param.initial_time_step;
        convergence_table.add_value("refinement", refinement);
        convergence_table.add_value("dt", dt );
        convergence_table.set_precision("dt", 3);
        convergence_table.set_scientific("dt", true);
    }

    //Printing and writing convergence table
    pcout << std::endl;
    convergence_table.write_text(std::cout); //pcout gives an error. Shouldn't be an issue as this is 1D and doesn't use MPI

     std::ofstream conv_tab_file;
     const char fname[25] = "convergence_table_1D.txt";
     conv_tab_file.open(fname);
     convergence_table.write_text(conv_tab_file);
     conv_tab_file.close();



    //PASS/FAIL CHECK

    return 0;
}

#if PHILIP_DIM==1
    template class TimeRefinementStudyAdvection<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
