#include "time_refinement_study.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_1D_unsteady.h"
#include "physics/exact_solutions/exact_solution.h"
#include "cmath"
//#include "ode_solver/runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
TimeRefinementStudy<dim, nstate>::TimeRefinementStudy(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input)  
        : TestsBase::TestsBase(parameters_input),
         parameter_handler(parameter_handler_input),
         n_time_calculations(parameters_input->time_refinement_study_param.number_of_times_to_solve),
         refine_ratio(parameters_input->time_refinement_study_param.refinement_ratio)
{}



template <int dim, int nstate>
Parameters::AllParameters TimeRefinementStudy<dim,nstate>::reinit_params_and_refine_timestep(int refinement) const
{
     PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
     
     parameters.ode_solver_param.initial_time_step *= pow(refine_ratio,refinement);
     
     return parameters;
}

template <int dim, int nstate>
double TimeRefinementStudy<dim,nstate>::calculate_L2_error_at_final_time_wrt_function(std::shared_ptr<DGBase<dim,double>> dg, const Parameters::AllParameters parameters, double final_time) const
{

    //generate exact solution at final time
    std::shared_ptr<ExactSolutionFunction<dim,nstate,double>> exact_solution_function;
    exact_solution_function = ExactSolutionFactory<dim,nstate,double>::create_ExactSolutionFunction(parameters.flow_solver_param, final_time);
    int poly_degree = parameters.grid_refinement_study_param.poly_degree;
    dealii::Vector<double> difference_per_cell(dg->solution.size());
    
    dealii::VectorTools::integrate_difference(dg->dof_handler,    
                                              dg->solution,
                                              *exact_solution_function,
                                              difference_per_cell,
                                              dealii::QGauss<dim>(poly_degree+10), //overintegrating by 10
                                              dealii::VectorTools::L2_norm);

    double L2_error = dealii::VectorTools::compute_global_error(*dg->triangulation,
                                                              difference_per_cell,
                                                              dealii::VectorTools::L2_norm);
    return L2_error;    

}


template <int dim, int nstate>
int TimeRefinementStudy<dim, nstate>::run_test() const
{

    const double final_time = this->all_parameters->flow_solver_param.final_time;
    const double initial_time_step = this->all_parameters->ode_solver_param.initial_time_step;
    const int n_steps = round(final_time/initial_time_step);
    if (n_steps * initial_time_step != final_time){
        pcout << "Error: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, initial_time_step)
              << ". Modify parameters to run this test." << std::endl;
        std::abort();
    }

    int testfail = 0;

    dealii::ConvergenceTable convergence_table;
    double L2_error_old = 0;
    double L2_error_conv_rate=0;

    for (int refinement = 0; refinement < n_time_calculations; ++refinement){
        
        pcout << "\n\n---------------------------------------------" << std::endl;
        pcout << "Refinement number " << refinement << " of " << n_time_calculations - 1 << std::endl;
        pcout << "---------------------------------------------" << std::endl;

        const Parameters::AllParameters params = reinit_params_and_refine_timestep(refinement);
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);
        static_cast<void>(flow_solver->run());
        
        //check L2 error
        const double L2_error = calculate_L2_error_at_final_time_wrt_function(flow_solver->dg, params, flow_solver->ode_solver->current_time);
        pcout << "Computed error is " << L2_error << std::endl;

        const double dt =  params.ode_solver_param.initial_time_step;
        convergence_table.add_value("refinement", refinement);
        convergence_table.add_value("dt", dt );
        convergence_table.set_precision("dt", 16);
        convergence_table.set_scientific("dt", true);
        convergence_table.add_value("L2_error",L2_error);
        convergence_table.set_precision("L2_error", 16);
        convergence_table.evaluate_convergence_rates("L2_error", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);

        //Checking convergence order
        const double expected_order = params.ode_solver_param.rk_order;
        const double order_tolerance = 0.1;
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
    if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
    //convergence_table.write_text(std::cout);

    std::ofstream conv_tab_file;
    const std::string fname = "temporal_convergence_table.txt";
    conv_tab_file.open(fname);
    convergence_table.write_text(conv_tab_file);
    conv_tab_file.close();

    return testfail;
}

#if PHILIP_DIM==1
    template class TimeRefinementStudy<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
