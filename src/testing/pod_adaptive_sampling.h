#ifndef __POD_ADAPTIVE_SAMPLING__
#define __POD_ADAPTIVE_SAMPLING__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "reduced_order/pod_basis_online.h"
#include "reduced_order/rom_test_location.h"
#include <eigen/Eigen/Dense>
#include "reduced_order/nearest_neighbors.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// POD adaptive sampling
template <int dim, int nstate>
class AdaptiveSampling: public TestsBase
{
public:
    /// Constructor
    AdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~AdaptiveSampling() {};

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    /// Vector of parameter-ROMTestLocation pairs
    mutable std::vector<std::unique_ptr<ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>>> rom_locations;

    /// Maximum error
    mutable double max_error;

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Most up to date POD basis
    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> current_pod;

    /// Nearest neighbors of snapshots
    std::shared_ptr<ProperOrthogonalDecomposition::NearestNeighbors> nearest_neighbors;

    /// Run test
    int run_test () const override;

    /// Placement of initial snapshots
    void placeInitialSnapshots() const;

    /// Placement of ROMs
    bool placeROMLocations(const MatrixXd& rom_points) const;

    /// Updates nearest ROM points to snapshot if error discrepancy is above tolerance
    void updateNearestExistingROMs(const RowVectorXd& parameter) const;

    /// Compute RBF and find max error
    RowVectorXd getMaxErrorROM() const;

    /// Solve full-order snapshot
    dealii::LinearAlgebra::distributed::Vector<double> solveSnapshotFOM(const RowVectorXd& parameter) const;

    /// Solve reduced-order solution
    std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(const RowVectorXd& parameter) const;

    /// Reinitialize parameters
    Parameters::AllParameters reinitParams(const RowVectorXd& parameter) const;

    /// Set up parameter space depending on test case
    void configureInitialParameterSpace() const;

    /// Output for each iteration
    void outputIterationData(int iteration) const;
};

}
}


#endif