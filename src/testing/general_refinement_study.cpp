#include "general_refinement_study.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_1D_unsteady.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"
#include "physics/exact_solutions/exact_solution.h"
#include "cmath"
//#include "ode_solver/runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
GeneralRefinementStudy<dim, nstate>::GeneralRefinementStudy(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        const RefinementType refinement_type_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input),
         refinement_type(refinement_type_input),
         n_calculations(parameters_input->time_refinement_study_param.number_of_times_to_solve),
         refine_ratio(parameters_input->time_refinement_study_param.refinement_ratio)
{}

template <int dim, int nstate>
Parameters::AllParameters GeneralRefinementStudy<dim,nstate>::reinit_params_and_refine(const Parameters::AllParameters *parameters_in, int refinement, const RefinementType how) const
{
    PHiLiP::Parameters::AllParameters parameters = *(parameters_in);

    if (how == RefinementType::timestep){
        parameters.ode_solver_param.initial_time_step *= pow(refine_ratio,refinement);
    } else if (how == RefinementType::h){
        if (abs(refine_ratio - 2.0 ) > 1E-13) { 
            this->pcout << "Warning: h refinement will use a refinement factor of 2." << std::endl
                        << "User input time_refinement_study_param.refinement_ratio will be ignored." << std::endl;
        }
         parameters.flow_solver_param.number_of_grid_elements_per_dimension *= pow(2, refinement);
    }

    //For RRK, do not end at exact time because of how relaxation parameter convergence is calculated
    using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
    if (parameters.ode_solver_param.ode_solver_type == ODESolverEnum::rrk_explicit_solver){
        parameters.flow_solver_param.end_exactly_at_final_time = false;
    }

    return parameters;
}

template <int dim, int nstate>
double GeneralRefinementStudy<dim,nstate>::calculate_Lp_error_at_final_time_wrt_function(std::shared_ptr<DGBase<dim,double>> dg, const Parameters::AllParameters parameters, double final_time, int norm_p) const
{
    //generate exact solution at final time
    std::shared_ptr<ExactSolutionFunction<dim,nstate,double>> exact_solution_function;
    exact_solution_function = ExactSolutionFactory<dim,nstate,double>::create_ExactSolutionFunction(parameters.flow_solver_param, final_time);
    this->pcout << "End time: " << final_time << std::endl;
    double Lp_error=0;
    int overintegrate = 10;

    int poly_degree = parameters.grid_refinement_study_param.poly_degree;
    dealii::Vector<double> difference_per_cell(dg->solution.size());

    dealii::VectorTools::NormType norm_type;
    if (norm_p == 1)       norm_type = dealii::VectorTools::L1_norm;
    else if (norm_p == 2)  norm_type = dealii::VectorTools::L2_norm;
    else if (norm_p<0)     norm_type = dealii::VectorTools::Linfty_norm;
    else {
        pcout << "Norm not defined. Aborting... " << std::endl;
        std::abort();
    }

    dealii::VectorTools::integrate_difference(dg->dof_handler,    
                                              dg->solution,
                                              *exact_solution_function,
                                              difference_per_cell,
                                              dealii::QGauss<dim>(poly_degree + overintegrate), //overintegrating
                                              norm_type);
    Lp_error = dealii::VectorTools::compute_global_error(*dg->triangulation,
                                                              difference_per_cell,
                                                              norm_type);
    return Lp_error;    
}

template <int dim, int nstate>
int GeneralRefinementStudy<dim,nstate>::run_refinement_study_and_write_result(const Parameters::AllParameters *parameters_in, const double expected_order, const bool append_to_file) const{

    const double final_time = parameters_in->flow_solver_param.final_time;
    const double initial_time_step = parameters_in->ode_solver_param.initial_time_step;
    const int n_steps = floor(final_time/initial_time_step);
    if (n_steps * initial_time_step != final_time){
        pcout << "WARNING: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, initial_time_step)
              << ". Consider modifying parameters." << std::endl;
    }

    int testfail = 0;

    dealii::ConvergenceTable convergence_table;
    double L2_error_old = 0;
    double L2_error_conv_rate=0;

    for (int refinement = 0; refinement < n_calculations; ++refinement){
        
        pcout << "\n\n---------------------------------------------" << std::endl;
        pcout << "Refinement number " << refinement << " of " << n_calculations - 1 << std::endl;
        pcout << "---------------------------------------------" << std::endl;

        const Parameters::AllParameters params = reinit_params_and_refine(parameters_in,refinement, refinement_type);
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);
        static_cast<void>(flow_solver->run());
        
        pcout << "Finished flowsolver " << std::endl;

        const double final_time_actual = flow_solver->ode_solver->current_time;
        
        //check Lp error
        const double L1_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, 1);
        const double L2_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, 2);
        const double Linfty_error = calculate_Lp_error_at_final_time_wrt_function(flow_solver->dg, params,final_time_actual, -1);
        pcout << "Computed errors are: " << std::endl
              << "    L1:      " << L1_error << std::endl
              << "    L2:      " << L2_error << std::endl
              << "    Linfty:  " << Linfty_error << std::endl;



        // hard-coded for LSRK test
        if (params.ode_solver_param.runge_kutta_method == PHiLiP::Parameters::ODESolverParam::RK3_2_5F_3SStarPlus 
            && params.ode_solver_param.atol == 1e-4 && params.ode_solver_param.rtol == 1e-4 
            && params.time_refinement_study_param.number_of_times_to_solve == 1){
            double L2_error_expected = 2.14808703658e-5; 
            pcout << " Expected L2 error is: " << L2_error_expected << std::endl;
            if (L2_error > L2_error_expected + 1e-9 || L2_error < L2_error_expected - 1e-9){
                testfail = 1;
                pcout << "Expected L2 error for RK3(2)5F[3S*+] using an atol = rtol = 1e-4 was not reached " << refinement <<std::endl;
            }
        }
        
        std::string step_string; 
        double step=1.0;
        if (refinement_type == RefinementType::timestep){
            step_string = "dt";
            step = params.ode_solver_param.initial_time_step;
        }else if (refinement_type == RefinementType::h){
            step_string = "h";
            step = (params.flow_solver_param.grid_right_bound - params.flow_solver_param.grid_left_bound) / (params.flow_solver_param.number_of_grid_elements_per_dimension);
        }
        convergence_table.add_value("refinement", refinement);
        convergence_table.add_value(step_string, step );
        convergence_table.set_precision(step_string, 16);
        convergence_table.add_value("n_cells_per_dim",params.flow_solver_param.number_of_grid_elements_per_dimension); 
        convergence_table.set_scientific(step_string, true);
        convergence_table.add_value("L1_error",L1_error);
        convergence_table.set_precision("L1_error", 16);
        convergence_table.evaluate_convergence_rates("L1_error", step_string, dealii::ConvergenceTable::reduction_rate_log2, 1);
        convergence_table.add_value("L2_error",L2_error);
        convergence_table.set_precision("L2_error", 16);
        convergence_table.evaluate_convergence_rates("L2_error", step_string, dealii::ConvergenceTable::reduction_rate_log2, 1);
        convergence_table.add_value("Linfty_error",Linfty_error);
        convergence_table.set_precision("Linfty_error", 16);
        convergence_table.evaluate_convergence_rates("Linfty_error", step_string, dealii::ConvergenceTable::reduction_rate_log2, 1);
        
        using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
        if(params.ode_solver_param.ode_solver_type == ODESolverEnum::rrk_explicit_solver){
            const double dt =  params.ode_solver_param.initial_time_step;
            const int n_timesteps= flow_solver->ode_solver->current_iteration;
            pcout << " at dt = " << dt << std::endl;
            //for burgers, this is the average gamma over the runtime
            const double gamma_aggregate_m1 = final_time_actual / (n_timesteps * dt)-1;
            convergence_table.add_value("gamma_aggregate_m1", gamma_aggregate_m1);
            convergence_table.set_precision("gamma_aggregate_m1", 16);
            convergence_table.set_scientific("gamma_aggregate_m1", true);
            convergence_table.evaluate_convergence_rates("gamma_aggregate_m1", step_string, dealii::ConvergenceTable::reduction_rate_log2, 1);
        }
 
        const double order_tolerance = 0.1;
        if (refinement > 0) {
            L2_error_conv_rate = abs(log(L2_error_old/L2_error)/log(refine_ratio));
            pcout << "L2 order at " << refinement << " is " << L2_error_conv_rate << std::endl;
            if ((L2_error_conv_rate ) < expected_order - order_tolerance){
                // Fail if the found convergence order is lower than the expected order.
                testfail = 1;
                pcout << "Expected convergence order was not reached at refinement " << refinement <<std::endl;
            }
           
            // output current time refinement results to console 
            if (refinement < n_calculations-1 && pcout.is_active()) convergence_table.write_text(pcout.get_stream());
        }
        L2_error_old = L2_error;
    }

    //Printing and writing convergence table
    pcout << std::endl;
    if (pcout.is_active()){ 
        convergence_table.write_text(pcout.get_stream());

        std::ofstream conv_tab_file;
        const std::string fname = "convergence_table.txt";
        if (append_to_file == true) {
            conv_tab_file.open(fname, std::ios::app);
        } else{
            conv_tab_file.open(fname);
        }
        convergence_table.write_text(conv_tab_file);
        conv_tab_file.close();
    }


    return testfail;
}

template <int dim, int nstate>
int GeneralRefinementStudy<dim, nstate>::run_test() const
{
    // setting expected order
    double expected_order_=0;
    if (refinement_type == RefinementType::timestep){
        expected_order_ = this->all_parameters->ode_solver_param.rk_order;
    } else if (refinement_type == RefinementType::h){
        expected_order_ = this->all_parameters->flow_solver_param.poly_degree + 1; 
    }
    const double expected_order = expected_order_;
    
    double testfail = this->run_refinement_study_and_write_result(this->all_parameters, expected_order);

    return testfail;
}

#if PHILIP_DIM==1
    template class GeneralRefinementStudy<PHILIP_DIM,PHILIP_DIM>;
#else
    template class GeneralRefinementStudy<PHILIP_DIM,1>;
    template class GeneralRefinementStudy<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
