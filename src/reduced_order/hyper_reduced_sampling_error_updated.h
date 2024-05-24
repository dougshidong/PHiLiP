#ifndef __HYPER_REDUCED_SAMPLING_ERROR_UPDATED__
#define __HYPER_REDUCED_SAMPLING_ERROR_UPDATED__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "pod_basis_online.h"
#include "hrom_test_location.h"
#include <eigen/Eigen/Dense>
#include "nearest_neighbors.h"

namespace PHiLiP {
using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// Hyperreduced Adaptive Sampling with the updated error indicator
// Currently seperate from the adaptive sampling base as the pointer of ROM locations has
// been changed to a have type HROMTestLocation to update DWR errors
template <int dim, int nstate>
class HyperreducedSamplingErrorUpdated
{
public:
    /// Constructor
    HyperreducedSamplingErrorUpdated(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~HyperreducedSamplingErrorUpdated() {};

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    /// Vector of parameter-ROMTestLocation pairs
    mutable std::vector<std::unique_ptr<ProperOrthogonalDecomposition::HROMTestLocation<dim,nstate>>> rom_locations;

    /// Maximum error
    mutable double max_error;

    /// Most up to date POD basis
    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> current_pod;

    /// Nearest neighbors of snapshots
    std::shared_ptr<ProperOrthogonalDecomposition::NearestNeighbors> nearest_neighbors;

    /// Ptr vector of ECSW Weights
    mutable std::shared_ptr<Epetra_Vector> ptr_weights;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.

    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;

    /// Run test
    int run_sampling () const;

    /// Placement of initial snapshots
    void placeInitialSnapshots() const;

    /// Placement of ROMs
    bool placeROMLocations(const MatrixXd& rom_points, Epetra_Vector weights) const;

    /// Updates nearest ROM points to snapshot if error discrepancy is above tolerance
    void updateNearestExistingROMs(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Compute RBF and find max error
    RowVectorXd getMaxErrorROM() const;

    /// Solve full-order snapshot
    dealii::LinearAlgebra::distributed::Vector<double> solveSnapshotFOM(const RowVectorXd& parameter) const;

    /// Solve reduced-order solution
    std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Reinitialize parameters
    Parameters::AllParameters reinitParams(const RowVectorXd& parameter) const;

    /// Set up parameter space depending on test case
    void configureInitialParameterSpace() const;

    /// Output for each iteration
    void outputIterationData(std::string iteration) const;
};

}


#endif