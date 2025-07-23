#ifndef __ADAPTIVE_SAMPLING_BASE__
#define __ADAPTIVE_SAMPLING_BASE__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "pod_basis_online.h"
#include "rom_test_location.h"
#include <eigen/Eigen/Dense>
#include "nearest_neighbors.h"

namespace PHiLiP {
using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// Adaptive sampling base class
/// Can then be built with or without hyperreduction

/*
Based on the work in Donovan Blais' thesis:
Goal-Oriented Adaptive Sampling for Projection-Based Reduced-Order Models, 2022
*/
template <int dim, int nstate>
class AdaptiveSamplingBase
{
public:
    /// Default constructor that will set the constants.
    AdaptiveSamplingBase(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Virtual destructor
    virtual ~AdaptiveSamplingBase() = default;

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    /// Vector of parameter-ROMTestLocation pairs
    mutable std::vector<std::unique_ptr<ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>>> rom_locations;

    /// Vector of parameter-ROMTestLocation pairs
    mutable std::vector<dealii::LinearAlgebra::distributed::Vector<double>> fom_locations;

    /// Maximum error
    mutable double max_error;

    /// Most up to date POD basis
    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> current_pod;

    /// Nearest neighbors of snapshots
    std::shared_ptr<ProperOrthogonalDecomposition::NearestNeighbors> nearest_neighbors;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.

    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;

    /// Output for each iteration
    virtual void outputIterationData(std::string iteration) const;
    
    /// Find point to solve for functional from param file
    RowVectorXd readROMFunctionalPoint() const;

    /// Run Sampling Procedure
    virtual int run_sampling () const = 0;

    /// Placement of initial snapshots
    void placeInitialSnapshots() const;

    /// Compute RBF and find max error
    virtual RowVectorXd getMaxErrorROM() const;

    /// Solve full-order snapshot
    dealii::LinearAlgebra::distributed::Vector<double> solveSnapshotFOM(const RowVectorXd& parameter) const;

    /// Reinitialize parameters
    Parameters::AllParameters reinit_params(const RowVectorXd& parameter) const;

    /// Set up parameter space depending on test case
    void configureInitialParameterSpace() const;

};

}


#endif