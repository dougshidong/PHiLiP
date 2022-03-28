#ifndef __RBF_INTERPOLATION__
#define __RBF_INTERPOLATION__

#include <Eigen/Dense>
#include <Eigen/LU>
#include <iostream>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/// Radial basis function interpolation
class RBFInterpolation
{
public:
    /// Constructor
    RBFInterpolation(MatrixXd data_coordinates, VectorXd data_values, std::string kernel);

    /// Destructor
    ~RBFInterpolation () {};

    void computeWeights();

    double radialBasisFunction(double r) const;

    VectorXd evaluate(RowVectorXd evaluate_coordinate) const;

    VectorXd weights;

    const MatrixXd data_coordinates;

    const VectorXd data_values;

    const std::string kernel;

    /******************For use as a Functor to use with Eigen's minimizer*****************************
    *************************************************************************************************/
    typedef double Scalar;

    typedef Eigen::VectorXd InputType;
    typedef Eigen::VectorXd ValueType;
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;

    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };

    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;

    int inputs() const;

    int values() const;

};

}
}


#endif