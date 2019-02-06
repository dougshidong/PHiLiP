

#include "advection_explicit.h"

#include <deal.II/base/logstream.h>
#include <iostream>
#include <fstream>
int main()
{
    try
    {
        std::cout << "Starting program..." << std::endl;

        PHiLiP::PDE<1, double> adv1_0(0); const int failure1_0 = adv1_0.run();
        PHiLiP::PDE<1, double> adv1_1(1); const int failure1_1 = adv1_1.run();
        PHiLiP::PDE<1, double> adv1_2(2); const int failure1_2 = adv1_2.run();
        PHiLiP::PDE<1, double> adv1_3(3); const int failure1_3 = adv1_3.run();
                                                                 
        PHiLiP::PDE<2, double> adv2_0(0); const int failure2_0 = adv2_0.run();
        PHiLiP::PDE<2, double> adv2_1(1); const int failure2_1 = adv2_1.run();
        
        if (failure1_0) return 1;
        if (failure1_1) return 1;
        if (failure1_2) return 1;
        if (failure1_3) return 1;

        if (failure2_0) return 1;
        if (failure2_1) return 1;
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
