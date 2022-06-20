#ifndef __RBF_INTERPOLATION__
#define __RBF_INTERPOLATION__

#include <Eigen/Dense>
#include <Eigen/LU>
#include <iostream>
#include "ROL_OptimizationProblem.hpp"
#include "ROL_StdVector.hpp"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/// Radial basis function interpolation
class RBFInterpolation: public ROL::Objective<double>
{
public:
    /// Constructor
    RBFInterpolation(const MatrixXd& data_coordinates, const VectorXd& data_values, std::string kernel);

    RBFInterpolation() = default;

    /// Destructor
    ~RBFInterpolation () {};

    void computeWeights();

    double radialBasisFunction(double r) const;

    VectorXd evaluate(const RowVectorXd& evaluate_coordinate) const;

    VectorXd weights;

    const MatrixXd data_coordinates;

    const VectorXd data_values;

    const std::string kernel;

    //ROL

    typedef std::vector<double> vector;
    typedef ROL::Vector<double>      V;

    template<class VectorType>
    ROL::Ptr<const vector> getVector( const V& x ) {
        return dynamic_cast<const VectorType&>((x)).getVector();
    }

    double value(const ROL::Vector<double> &x, double &/*tol*/ );

};

}
}


#endif