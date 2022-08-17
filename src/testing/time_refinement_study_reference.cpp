#include "time_refinement_study_reference.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_1D_unsteady.h"
#include "physics/exact_solutions/exact_solution.h"
#include "cmath"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TimeRefinementStudyReference<dim, nstate>::TimeRefinementStudyReference(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input),
         n_time_calculations(parameters_input->time_refinement_study_param.number_of_times_to_solve),
         refine_ratio(parameters_input->time_refinement_study_param.refinement_ratio)
{}

template <int dim, int nstate>
Parameters::AllParameters TimeRefinementStudyReference<dim,nstate>::reinit_params_for_reference_solution(int number_of_timesteps, double final_time) const
{
     PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

     double dt = final_time/number_of_timesteps;     
     parameters.ode_solver_param.initial_time_step = dt;
     
     pcout << "Using timestep size dt = " << dt << " for reference solution." << std::endl;
     
     return parameters;
}

template <int dim, int nstate>
Parameters::AllParameters TimeRefinementStudyReference<dim,nstate>::reinit_params_and_refine_timestep(int refinement) const
{
     PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
     
     parameters.ode_solver_param.initial_time_step *= pow(refine_ratio,refinement);
     
     return parameters;
}

template <int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> TimeRefinementStudyReference<dim,nstate>::calculate_reference_solution(
        double final_time) const
{
    int number_of_timesteps_for_reference_solution = this->all_parameters->time_refinement_study_param.number_of_timesteps_for_reference_solution;
    const Parameters::AllParameters params_reference = reinit_params_for_reference_solution(number_of_timesteps_for_reference_solution, final_time);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_reference = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params_reference, parameter_handler);
    static_cast<void>(flow_solver_reference->run());

    return flow_solver_reference->dg->solution;
}

template <int dim, int nstate>
double TimeRefinementStudyReference<dim,nstate>::calculate_L2_error_at_final_time_wrt_reference(
            std::shared_ptr<DGBase<dim,double>> dg,
            const Parameters::AllParameters parameters, 
            double final_time_actual,
            dealii::LinearAlgebra::distributed::Vector<double> reference_solution) const
{
    const double final_time_target = parameters.flow_solver_param.final_time;

    if (abs(final_time_target-final_time_actual)<1E-13){
        
        pcout << "Comparing to reference solution at target final_time = " << final_time_target << " ..."  << std::endl;

        //calculate L2 norm of error
        dealii::LinearAlgebra::distributed::Vector<double> cellwise_difference(reference_solution); 
        cellwise_difference.add(-1.0, dg->solution);
        double L2_error = cellwise_difference.l2_norm();
        return L2_error;
    }else{
        //recompute reference solution at actual end time
        //intended to be used when using ode_solver = rrk_explicit_solver

        pcout << "    -------------------------------------------------------" << std::endl;
        pcout << "    Calculating reference solution at actual final_time = " << final_time_actual << " ..."<<std::endl;
        pcout << "    -------------------------------------------------------" << std::endl;
        const dealii::LinearAlgebra::distributed::Vector<double> reference_solution_actual = calculate_reference_solution(final_time_actual);

        dealii::LinearAlgebra::distributed::Vector<double> cellwise_difference(reference_solution_actual); 
        cellwise_difference.add(-1.0, dg->solution);
        double L2_error = cellwise_difference.l2_norm();
        return L2_error;
    }
    
}

template <int dim, int nstate>
int TimeRefinementStudyReference<dim, nstate>::run_test() const
{
    using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;

    double final_time = this->all_parameters->flow_solver_param.final_time;
    double initial_time_step = this->all_parameters->ode_solver_param.initial_time_step;
    int n_steps = round(final_time/initial_time_step);
    if (n_steps * initial_time_step != final_time){
        pcout << "Error: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, initial_time_step)
              << ". Modify parameters to run this test." << std::endl;
        std::abort();
    }

    int testfail = 0;
    double expected_order =(double) this->all_parameters->ode_solver_param.runge_kutta_order;
    double order_tolerance = 0.1;

    //pointer to flow_solver_case for computing energy
    std::unique_ptr<FlowSolver::Periodic1DUnsteady<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::Periodic1DUnsteady<dim,nstate>>(this->all_parameters);

    pcout << "\n\n-------------------------------------------------------" << std::endl;
    pcout << "Calculating reference solution at target final_time = " << final_time << " ..."<<std::endl;
    pcout << "-------------------------------------------------------" << std::endl;
    
    double final_time_target = this->all_parameters->flow_solver_param.final_time;
    const dealii::LinearAlgebra::distributed::Vector<double> reference_solution = calculate_reference_solution(final_time_target);

    dealii::ConvergenceTable convergence_table;
    double L2_error_old = 0;
    double L2_error_conv_rate=0;

    for (int refinement = 0; refinement < n_time_calculations; ++refinement){
        
        pcout << "\n\n-------------------------------------------------------" << std::endl;
        pcout << "Refinement number " << refinement << " of " << n_time_calculations - 1 << std::endl;
        pcout << "-------------------------------------------------------" << std::endl;

        const Parameters::AllParameters params = reinit_params_and_refine_timestep(refinement);
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);
        const double energy_initial = flow_solver_case->compute_energy_collocated(flow_solver->dg);
        static_cast<void>(flow_solver->run());

        double final_time_actual = flow_solver->ode_solver->current_time;

        //check L2 error
        double L2_error = calculate_L2_error_at_final_time_wrt_reference(
                flow_solver->dg, 
                params, 
                final_time_actual,
                reference_solution);
        pcout << "Computed error is " << L2_error << std::endl;

        const double dt =  params.ode_solver_param.initial_time_step;
        convergence_table.add_value("refinement", refinement);
        convergence_table.add_value("dt", dt );
        convergence_table.set_precision("dt", 16);
        convergence_table.set_scientific("dt", true);
        convergence_table.add_value("final_time", final_time_actual );
        convergence_table.set_precision("final_time", 16);
        convergence_table.add_value("L2_error",L2_error);
        convergence_table.set_precision("L2_error", 16);
        convergence_table.evaluate_convergence_rates("L2_error", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);
        
        if (params.use_collocated_nodes){
            //current energy calculation is only valid for collocated nodes
            const double energy_end = flow_solver_case->compute_energy_collocated(flow_solver->dg);
            const double energy_change = energy_initial - energy_end;
            convergence_table.add_value("energy_change", energy_change);
            convergence_table.set_precision("energy_change", 16);
            convergence_table.evaluate_convergence_rates("energy_change", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);
        }
        
        if(params.ode_solver_param.ode_solver_type == ODESolverEnum::rrk_explicit_solver){
            //for burgers, this is the average gamma over the runtime
            double gamma_aggregate_m1 = (final_time_actual / final_time_target)-1;
            convergence_table.add_value("gamma_aggregate_m1", gamma_aggregate_m1);
            convergence_table.set_precision("gamma_aggregate_m1", 16);
            convergence_table.set_scientific("gamma_aggregate_m1", true);
            convergence_table.evaluate_convergence_rates("gamma_aggregate_m1", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);
        }

        //Checking convergence order
        if (refinement > 0) {
            L2_error_conv_rate = -log(L2_error_old/L2_error)/log(refine_ratio);
            pcout << "Order at " << refinement << " is " << L2_error_conv_rate << std::endl;
            if (abs(L2_error_conv_rate - expected_order) > order_tolerance){
                testfail = 1;
                pcout << "Expected convergence order was not reached at refinement " << refinement <<std::endl;
            }
        }
        L2_error_old = L2_error;
    }

    //Printing and writing convergence table
    pcout << std::endl;
    if (pcout.is_active()) {
        convergence_table.write_text(pcout.get_stream());

        std::ofstream conv_tab_file;
        const std::string fname = "temporal_convergence_table.txt";
        conv_tab_file.open(fname);
        convergence_table.write_text(conv_tab_file);
        conv_tab_file.close();
    }

    return testfail;
}

#if PHILIP_DIM==1
    template class TimeRefinementStudyReference<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
