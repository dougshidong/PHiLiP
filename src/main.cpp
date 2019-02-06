

#include "advection_explicit.h"

#include <deal.II/base/logstream.h>
#include <iostream>
#include <fstream>
int main()
{
    try
    {
        std::cout << "Starting program..." << std::endl;

        PHiLiP::PDE<1, double> adv11(1); adv11.run();
        PHiLiP::PDE<1, double> adv12(2); adv12.run();
        PHiLiP::PDE<1, double> adv13(3); adv13.run();
        //PHiLiP::PDE<2, double> adv21(1); adv21.run();
        //PHiLiP::PDE<2, double> adv22(2); adv22.run();
        //PHiLiP::PDE<2, double> adv23(3); adv23.run();
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
