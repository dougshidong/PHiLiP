#include <deal.II/base/utilities.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <fenv.h> // catch nan
#include <iostream>
#include <fstream>

#include "tests/tests.h"
#include "tests/grid_study.h"
#include "ode_solver/ode_solver.h"
#include "parameters/all_parameters.h"


int main (int argc, char *argv[])
{
    //feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
    dealii::deallog.depth_console(99);
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int test_error = 1;
    try
    {
        // Declare possible inputs
        dealii::ParameterHandler parameter_handler;
        PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
        PHiLiP::Parameters::parse_command_line (argc, argv, parameter_handler);

        // Read inputs from parameter file and set those values in AllParameters object
        PHiLiP::Parameters::AllParameters all_parameters;
        std::cout << "Reading input..." << std::endl;
        all_parameters.parse_parameters (parameter_handler);

        AssertDimension(all_parameters.dimension, PHILIP_DIM);

        std::cout << "Starting program..." << std::endl;

        const int max_dim = PHILIP_DIM;
        const int max_nstate = 5;
        std::unique_ptr<PHiLiP::Tests::TestsBase> test = PHiLiP::Tests::TestsFactory<max_dim,max_nstate>::create_test(&all_parameters);
        test_error = test->run_test();

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
    std::cout << "End of program." << std::endl;
    return test_error;
}
