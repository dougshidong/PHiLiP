#include "dg.h"
#include "parameters.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <iostream>
#include <fstream>

template <int dim>
int runDG (Parameters::AllParameters &parameters, const unsigned int poly_degree)
{
    PHiLiP::DiscontinuousGalerkin<dim, double> dg(&parameters, poly_degree);
    return dg.grid_convergence();
}

int main (int argc, char *argv[])
{
    try
    {
        std::cout << "Reading input..." << std::endl;
        dealii::ParameterHandler parameter_handler;
        Parameters::AllParameters::declare_parameters (parameter_handler);
        Parameters::parse_command_line (argc, argv, parameter_handler);


        Parameters::AllParameters parameters;
        parameters.parse_parameters (parameter_handler);

        std::cout << "Starting program..." << std::endl;


        const unsigned int dim = parameters.dimension;
        const unsigned int p_start = parameters.degree_start;
        const unsigned int p_end = parameters.degree_end;
        for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {
            int failure;
            if (dim == 1) failure = runDG<1>(parameters, poly_degree);
            if (dim == 2) failure = runDG<2>(parameters, poly_degree);
            if (dim == 3) failure = runDG<3>(parameters, poly_degree);
            if (failure) return 1;
        }
        //const unsigned int np_1d = 6;
        //const unsigned int np_2d = 4;
        //const unsigned int np_3d = 3;
        //for (unsigned int poly_degree = 0; poly_degree <= np_2d; ++poly_degree) {
        //    PHiLiP::DiscontinuousGalerkin<2, double> adv(poly_degree, parameters);
        //    const int failure = adv.grid_convergence();
        //    if (failure) return 1;
        //}
        // Too long to grid_convergence
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
    return 0;
}
