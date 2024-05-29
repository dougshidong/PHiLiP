#ifndef __ROM_TEST_LOCATION__
#define __ROM_TEST_LOCATION__

#include "parameters/all_parameters.h"
#include "pod_basis_base.h"
#include "reduced_order_solution.h"
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::RowVectorXd;

/// Class to compute and store adjoint-based error estimates
template <int dim, int nstate>
class ROMTestLocation
{
public:
    /// Constructor
    ROMTestLocation(const RowVectorXd& parameter, std::unique_ptr<ROMSolution < dim, nstate>> rom_solution);

    /// Compute adjoint error estimate between FOM and initial ROM
    void compute_FOM_to_initial_ROM_error();

    /// Compute error between initial ROM and final ROM
    void compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod_updated);

    /// Compute total error between final ROM and FOM
    void compute_total_error();

    /// Parameter
    RowVectorXd parameter;

    /// ROM solution
    std::unique_ptr<ROMSolution<dim, nstate>> rom_solution;

    /// Error between FOM and initial ROM
    double fom_to_initial_rom_error;

    /// Error from initial ROM to final ROM
    double initial_rom_to_final_rom_error;

    /// Total error
    double total_error;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.

    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;

};

}
}


#endif