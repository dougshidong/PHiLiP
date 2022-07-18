#ifndef __MIN_MAX_SCALER__
#define __MIN_MAX_SCALER__

#include <eigen/Eigen/Dense>
#include <iostream>
#include <algorithm>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/// Scale data between 0 and 1
class MinMaxScaler
{
public:

    /// Constructor
    MinMaxScaler() = default;

    /// Destructor
    ~MinMaxScaler() {};

    /// Fit and transform data
    MatrixXd fit_transform(const MatrixXd& snapshot_parameters);

    /// Transform data to previously fitted dataset
    MatrixXd transform(const MatrixXd& snapshot_parameters);

    /// Unscale data
    MatrixXd inverse_transform(const MatrixXd& snapshot_parameters);

    /// Minimum values
    RowVectorXd min;

    /// Maximum values
    RowVectorXd max;

};

}
}

#endif
