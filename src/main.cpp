

#include "dg.h"

#include <deal.II/base/logstream.h>
#include <iostream>
#include <fstream>
int main()
{
    try
    {
        std::cout << "Starting program..." << std::endl;

        const unsigned int np_1d = 6;
        const unsigned int np_2d = 4;
        const unsigned int np_3d = 3;
        for (unsigned int poly_degree = 0; poly_degree <= np_1d; ++poly_degree) {
            PHiLiP::discontinuous_galerkin<1, double> adv(poly_degree);
            const int failure = adv.run();
            if (failure) return 1;
        }
        for (unsigned int poly_degree = 0; poly_degree <= np_2d; ++poly_degree) {
            PHiLiP::discontinuous_galerkin<2, double> adv(poly_degree);
            const int failure = adv.run();
            if (failure) return 1;
        }
        // Too long to run
        //for (unsigned int poly_degree = 0; poly_degree <= np_3d; ++poly_degree) {
        //    PHiLiP::discontinuous_galerkin<3, double> adv(poly_degree);
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
