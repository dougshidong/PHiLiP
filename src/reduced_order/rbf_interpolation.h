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
    RBFInterpolation(MatrixXd data_coordinates, VectorXd data_values, std::string kernel);

    RBFInterpolation() = default;

    /// Destructor
    ~RBFInterpolation () {};

    void computeWeights();

    double radialBasisFunction(double r) const;

    VectorXd evaluate(RowVectorXd evaluate_coordinate) const;

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

    double value(const ROL::Vector<double> &x, double &/*tol*/ ) {
        ROL::Ptr<const vector> xp = getVector<ROL::StdVector<double>>(x);
        double val = 100 * pow(pow((*xp)[0],2) - (*xp)[1], 2) + pow((*xp)[0] - 1.0, 2);
        return val;
    }

};

}
}


#endif