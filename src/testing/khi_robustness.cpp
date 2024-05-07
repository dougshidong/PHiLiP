#include "khi_robustness.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
KHIRobustness<dim, nstate>::KHIRobustness(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters KHIRobustness<dim,nstate>::reinit_params(double atwood_number) const
{
     PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
     
     parameters.flow_solver_param.atwood_number = atwood_number;
     parameters.flow_solver_param.unsteady_data_table_filename+=std::to_string(atwood_number);
     
     return parameters;
}

template <int dim, int nstate>
int KHIRobustness<dim, nstate>::run_test() const
{
    const unsigned int n_runs = 2;
    // Range of Atwood numbers to test
    const double A_range[n_runs] = {0.8,0.9};
    double end_times[n_runs] = {0};

    for (unsigned int i_run = 0; i_run < n_runs; ++i_run){
        this->pcout << "--------------------------------------------------------------------" << std::endl
                    << "  Starting run for A = " << A_range[i_run] << std::endl
                    << "--------------------------------------------------------------------" << std::endl;
        //Define new parameters according to Atwood number
        const Parameters::AllParameters params = reinit_params(A_range[i_run]);

        // Initialize flow_solver
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate,1>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate,1>::select_flow_case(&params, parameter_handler);

        // Solve flow case 
        try{
            static_cast<void>(flow_solver->run());
        }
        catch (double end) {
            this->pcout << "WARNING: Flow simulation did not reach end_time. Crash at t = " << end << std::endl;
        }

        end_times[i_run] = flow_solver->ode_solver->current_time;

        this->pcout << "End times for all runs so far:" << std::endl;
        for (unsigned int j = 0; j < i_run+1; ++j){
            this->pcout << "    A = " << A_range[j] << "    end_time = " << end_times[j] << std::endl;
        }
        this->pcout << std::endl << std::endl;

        if (mpi_rank==0){
            std::ofstream current_end_times;
            current_end_times.open("khi_end_times.txt");
            for (unsigned int j = 0; j < i_run+1; ++j){
                current_end_times << A_range[j] << " "  << end_times[j] << std::endl;
            }
            current_end_times.close();
        }
    }

    return 0; //Always pass as the flow_solver runs are expected to crash
}

#if PHILIP_DIM==2
    template class KHIRobustness<PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
