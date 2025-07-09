#ifndef __NEAREST_NEIGHBORS__
#define __NEAREST_NEIGHBORS__

#include <eigen/Eigen/Dense>
#include <deal.II/lac/la_parallel_vector.h>
#include "min_max_scaler.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/// Nearest neighbors
class NearestNeighbors
{
public:
    /// Constructor
    NearestNeighbors();

    /// Add snapshot
    void update_snapshots(const MatrixXd& snapshot_parameters, dealii::LinearAlgebra::distributed::Vector<double> snapshot);

    /// Find midpoint of all snapshot locations
    MatrixXd kPairwiseNearestNeighborsMidpoint();

    ///Given a point, returns midpoint between point and k nearest snapshots, where k is 1+num_parameters
    MatrixXd kNearestNeighborsMidpoint(const RowVectorXd& point);

    //Given a point, return the index of nearest snapshot_parameter
    dealii::LinearAlgebra::distributed::Vector<double> nearestNeighborMidpointSolution(const RowVectorXd& point);

    /// Snapshot parameters
    MatrixXd snapshot_params;

    /// Scaled snapshot parameters
    MatrixXd scaled_snapshot_params;

    /// Scaler
    MinMaxScaler scaler;

    /// Vector containing all snapshots
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> snapshots;

};

}
}

#endif
