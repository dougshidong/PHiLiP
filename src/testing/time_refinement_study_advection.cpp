#include "time_refinement_study_advection.h"
#include "flow_solver.h"
#include "flow_solver_cases/periodic_cube_flow.h"
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

     PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
     
     parameters.ode_solver_param.initial_time_step *= pow(refine_ratio,refinement);
     
     return parameters;
 }


template <int dim, int nstate>
int TimeRefinementStudyAdvection<dim, nstate>::run_test() const
{
    int testfail = 0;
    double expected_order =(double) this->all_parameters->ode_solver_param.runge_kutta_order;
    double order_tolerance = 0.1;

    dealii::ConvergenceTable convergence_table;
    double L2_error_old = 0;
    double L2_error_conv_rate=0;

    for (int refinement = 0; refinement < n_time_calculations; ++refinement){
        
        pcout << "\n\n---------------------------------------------\n Refinement number " << refinement <<
            " of " << n_time_calculations - 1 << std::endl << "---------------------------------------------" << std::endl;
        Parameters::AllParameters params = reinit_params_and_refine_timestep(refinement);
        std::unique_ptr<FlowSolver<dim,nstate>> flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params, parameter_handler);
        static_cast<void>(flow_solver->run_test());
        
        //check L2 error
        double L2_error = flow_solver->calculate_L2_error_at_final_time_wrt_function();

        const double dt =  params.ode_solver_param.initial_time_step;
        convergence_table.add_value("refinement", refinement);
        convergence_table.add_value("dt", dt );
        convergence_table.set_precision("dt", 3);
        convergence_table.set_scientific("dt", true);
        convergence_table.add_value("L2_error",L2_error);
        convergence_table.set_precision("L2_error", 4);
        convergence_table.set_scientific("L2_error", true);
        convergence_table.evaluate_convergence_rates("L2_error", "dt", dealii::ConvergenceTable::reduction_rate_log2, 1);

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
    convergence_table.write_text(std::cout); //pcout gives an error. Shouldn't be an issue as this is 1D and doesn't use MPI

    std::ofstream conv_tab_file;
    const char fname[25] = "convergence_table_1D.txt";
    conv_tab_file.open(fname);
    convergence_table.write_text(conv_tab_file);
    conv_tab_file.close();

    return testfail;
}

#if PHILIP_DIM==1
    template class TimeRefinementStudyAdvection<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
