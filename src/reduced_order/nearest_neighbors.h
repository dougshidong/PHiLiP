#ifndef __NEAREST_NEIGHBORS__
#define __NEAREST_NEIGHBORS__

#include <Eigen/Dense>
#include <Eigen/LU>
#include <iostream>
#include <algorithm>

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
    NearestNeighbors(const MatrixXd& snapshot_parameters);

    /// Destructor
    ~NearestNeighbors() {};

    MatrixXd kPairwiseNearestNeighborsMidpoint();

    ///Given a point, returns midpoint between point and k nearest snapshots, where k is 1+num_parameters
    MatrixXd kNearestNeighborsMidpoint(const RowVectorXd& point);

    //Given a point, return the index of nearest snapshot_parameter
    Eigen::Index nearestNeighbor(const RowVectorXd& point);

    void updateSnapshotParameters(const MatrixXd& snapshot_parameters);

    MatrixXd snapshot_params;

};

}
}

#endif
