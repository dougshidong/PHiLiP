#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include "parameters/all_real_gas_constants.h"
#include "combustion/readspecies.h"

//for checking output directories
#include <sys/types.h>
#include <sys/stat.h>

namespace PHiLiP {
namespace RealGasConstants {

AllRealGasConstants::AllRealGasConstants ()
    : //manufactured_convergence_study_param(ManufacturedConvergenceStudyParam())
    // , reactive_var(EulerParam())
    // , nonequilibrium_var(NavierStokesParam())
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{ }

void AllRealGasConstants::read_species()
{
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    pcout << "Declaring inputs." << std::endl;

    readspecies(namechem);
    
    pcout << "Done declaring inputs." << std::endl;
}

} // RealGasConstants namespace
} // PHiLiP namespace
