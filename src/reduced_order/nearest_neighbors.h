#ifndef __NEAREST_NEIGHBORS__
#define __NEAREST_NEIGHBORS__

#include <eigen/Eigen/Dense>
#include <eigen/Eigen/LU>
#include <iostream>
#include <algorithm>
#include <numeric>
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

    /// Destructor
    ~NearestNeighbors() {};

    void updateSnapshots(const MatrixXd& snapshot_parameters, dealii::LinearAlgebra::distributed::Vector<double> snapshot);

    MatrixXd kPairwiseNearestNeighborsMidpoint();

    ///Given a point, returns midpoint between point and k nearest snapshots, where k is 1+num_parameters
    MatrixXd kNearestNeighborsMidpoint(const RowVectorXd& point);

    //Given a point, return the index of nearest snapshot_parameter
    dealii::LinearAlgebra::distributed::Vector<double> nearestNeighborMidpointSolution(const RowVectorXd& point);

    MatrixXd snapshot_params;

    MinMaxScaler scaler;

    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> snapshots;

    MatrixXd scaled_snapshot_params;

};

}
}

#endif
