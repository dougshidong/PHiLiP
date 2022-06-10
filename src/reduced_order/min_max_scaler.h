#ifndef __MIN_MAX_SCALER__
#define __MIN_MAX_SCALER__

#include <Eigen/Dense>
#include <iostream>
#include <algorithm>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/// Nearest neighbors
class MinMaxScaler
{
public:

    /// Constructor
    MinMaxScaler() = default;

    /// Destructor
    ~MinMaxScaler() {};

    MatrixXd fit_transform(const MatrixXd& snapshot_parameters);

    MatrixXd transform(const MatrixXd& snapshot_parameters);

    MatrixXd inverse_transform(const MatrixXd& snapshot_parameters);

    RowVectorXd min;

    RowVectorXd max;

};

}
}

#endif
