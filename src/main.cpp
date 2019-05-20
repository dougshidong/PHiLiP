#include <deal.II/base/utilities.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <iostream>
#include <fstream>

#include "grid_study.h"
#include "ode_solver.h"
#include "parameters/all_parameters.h"


int main (int argc, char *argv[])
{
    dealii::deallog.depth_console(99);
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    //dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    //    argc, argv, dealii::numbers::invalid_unsigned_int);
    int test = 1;
    try
    {
        std::cout << "Reading input..." << std::endl;
        dealii::ParameterHandler parameter_handler;
        Parameters::AllParameters::declare_parameters (parameter_handler);
        Parameters::parse_command_line (argc, argv, parameter_handler);


        Parameters::AllParameters all_parameters;
        all_parameters.parse_parameters (parameter_handler);

        std::cout << "Starting program..." << std::endl;


        test = PHiLiP::manufactured_grid_convergence<PHILIP_DIM> (all_parameters);

        //const unsigned int np_1d = 6;
        //const unsigned int np_2d = 4;
        //const unsigned int np_3d = 3;
        //for (unsigned int poly_degree = 0; poly_degree <= np_2d; ++poly_degree) {
        //    PHiLiP::DiscontinuousGalerkin<2, double> adv(poly_degree, all_parameters);
        //    const int failure = adv.grid_convergence_explicit();
        //    if (failure) return 1;
        //}
        // Too long to grid_convergence_explicit
        //for (unsigned int poly_degree = 0; poly_degree <= np_3d; ++poly_degree) {
        //    PHiLiP::DiscontinuousGalerkin<3, double> adv(poly_degree);
        //    const int failure = adv.run();
        //    if (failure) return 1;
        //}
                                                                 

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
    return test;
}
