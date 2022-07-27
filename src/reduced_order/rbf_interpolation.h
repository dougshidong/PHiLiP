#ifndef __RBF_INTERPOLATION__
#define __RBF_INTERPOLATION__

#include <eigen/Eigen/Dense>
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

    /// Constructor
    RBFInterpolation() = default;

    /// Destructor
    ~RBFInterpolation () {};

    /// Compute RBF interpolation weights
    void computeWeights();

    /// Choose radial basis function
    double radialBasisFunction(double r) const;

    /// Evaluate RBF
    double evaluate(const RowVectorXd& evaluate_coordinate) const;

    /// RBF weights
    VectorXd weights;

    /// Data coordinates
    const MatrixXd data_coordinates;

    /// Data values
    const VectorXd data_values;

    /// RBF kernel
    const std::string kernel;

    /// ROL required
    typedef std::vector<double> vector;
    /// ROL required
    typedef ROL::Vector<double>      V;

    /// ROL required
    template<class VectorType>
    ROL::Ptr<const vector> getVector( const V& x ) {
        return dynamic_cast<const VectorType&>((x)).getVector();
    }

    /// ROL evaluate value
    double value(const ROL::Vector<double> &x, double &/*tol*/ );

};

}
}


#endif