#include <deal.II/base/utilities.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <fenv.h> // catch nan
#include <iostream>
#include <fstream>

#include "testing/tests.h"
#include "flow_solver/flow_solver_factory.h"
#include "parameters/all_parameters.h"

#include "global_counter.hpp"

int main (int argc, char *argv[])
{
// #if !defined(__APPLE__)
//     feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
// #endif

    n_vmult = 0;
    dRdW_form = 0;
    dRdW_mult = 0;
    dRdX_mult = 0;
    d2R_mult = 0;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int n_mpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (n_mpi==1 || mpi_rank==0) {
        dealii::deallog.depth_console(99);
    } else {
        dealii::deallog.depth_console(0);
    }

    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    pcout << "Starting program with " << n_mpi << " processors..." << std::endl;
    if ((PHILIP_DIM==1) && !(n_mpi==1)) {
        std::cout << "********************************************************" << std::endl;
        std::cout << "Can't use mpirun -np X, where X>1, for 1D." << std::endl
                  << "Currently using " << n_mpi << " processors." << std::endl
                  << "Aborting..." << std::endl;
        std::cout << "********************************************************" << std::endl;
        std::abort();
    }
    int run_error = 1;
    const int number_of_parameter_file = argc-2;
    if (number_of_parameter_file<=0) {
        pout << "Input file is required..." << std::endl;
        std::abort();
    }
    try
    {
        // Declare possible inputs
        std::vector<dealii::ParameterHandler> parameter_handler(number_of_parameter_file);
        for (int i=0;i<number_of_parameter_file;++i){
            PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler[i]);
        }
        PHiLiP::Parameters::parse_command_line (argc, argv, parameter_handler);

        // Read inputs from parameter files and set those values in AllParameters objects
        std::vector<PHiLiP::Parameters::AllParameters> all_parameters(number_of_parameter_file);
        std::vector<PHiLiP::Parameters::AllParameters*> all_parameters_pointer(number_of_parameter_file);
        pcout << "Reading input..." << std::endl;
        for (int i=0;i<number_of_parameter_file;++i){
            all_parameters[i].parse_parameters (parameter_handler[i]); 
            AssertDimension(all_parameters[i].dimension, PHILIP_DIM);
            all_parameters_pointer[i] = &all_parameters[i];
        }

        const int max_dim = PHILIP_DIM;
        const int max_nstate = 5;

        if(all_parameters[0].run_type == PHiLiP::Parameters::AllParameters::RunType::flow_simulation) {
            std::unique_ptr<PHiLiP::FlowSolver::FlowSolverBase> flow_solver = PHiLiP::FlowSolver::FlowSolverFactory<max_dim,max_nstate>::create_flow_solver(all_parameters_pointer,parameter_handler);
            run_error = flow_solver->run();
            pcout << "Flow simulation complete with run error code: " << run_error << std::endl;
        }
        else if(all_parameters[0].run_type == PHiLiP::Parameters::AllParameters::RunType::integration_test) {
            std::unique_ptr<PHiLiP::Tests::TestsBase> test = PHiLiP::Tests::TestsFactory<max_dim,max_nstate>::create_test(&all_parameters[0],parameter_handler);
            run_error = test->run_test();
            pcout << "Finished integration test with run error code: " << run_error << std::endl;
        }
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl
                  << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl
                  << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    //std::cout << "MPI process " << mpi_rank+1 << " out of " << n_mpi << "reached end of program." << std::endl;
    pcout << "End of program" << std::endl;
    return run_error;
}
