#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include "all_real_gas_constants.h"
#include "readspecies.h"
#include "ReactiveVar.h"

namespace PHiLiP {
namespace RealGasConstants {

AllRealGasConstants::AllRealGasConstants ()
    : //manufactured_convergence_study_param(ManufacturedConvergenceStudyParam())
    // , reactive_var(EulerParam())
    // , nonequilibrium_var(NavierStokesParam())
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{ }

void AllRealGasConstants::read_species()
{
    pcout << "Declaring inputs." << std::endl;

    readspecies(namechem);
    
    pcout << "Done declaring inputs." << std::endl;
}

} // RealGasConstants namespace
} // PHiLiP namespace
