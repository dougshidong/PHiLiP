#ifndef __HROM_TEST_LOCATION__
#define __HROM_TEST_LOCATION__

#include "parameters/all_parameters.h"
#include "pod_basis_base.h"
#include "reduced_order_solution.h"
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::RowVectorXd;

/// Class to compute and store adjoint-based error estimates with hyperreduction
// Based very closely on the ROMTestLocation class

/*
Based on the work in Donovan Blais' thesis:
Goal-Oriented Adaptive Sampling for Projection-Based Reduced-Order Models, 2022

Details on the ROM points/errors can be found in sections 5 and 6

Derivation of the new error indicator will likely be detailed in Calista Biondic's thesis
*/
template <int dim, int nstate>
class HROMTestLocation
{
public:
    /// Constructor
    HROMTestLocation(const RowVectorXd& parameter, std::unique_ptr<ROMSolution < dim, nstate>> rom_solution, std::shared_ptr< DGBase<dim, double> > dg_input, Epetra_Vector weights);

    /// Compute adjoint error estimate between FOM and initial ROM
    void compute_FOM_to_initial_ROM_error();

    /// Compute error between initial ROM and final ROM
    void compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod_updated);

    /// Compute total error between final ROM and FOM
    void compute_total_error();

    /// Generate hyper-reduced jacobian matrix
    std::shared_ptr<Epetra_CrsMatrix> generate_hyper_reduced_jacobian(const Epetra_CrsMatrix &system_matrix);

    /// Generate test basis
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &pod_basis);

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

    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,double>> dg;

    /// ECSW hyper-reduction weights
    Epetra_Vector ECSW_weights;

};

}
}


#endif