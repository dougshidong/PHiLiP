#include "rbf_interpolation.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {


RBFInterpolation::RBFInterpolation(MatrixXd data_coordinates, VectorXd data_values, std::string kernel)
        : data_coordinates(data_coordinates)
        , data_values(data_values)
        , kernel(kernel)
{
    computeWeights();
}

void RBFInterpolation::computeWeights() {
    long N = data_coordinates.rows();

    MatrixXd A;
    A.resize(N,N);

    for(unsigned int i = 0 ; i < N ; i++){
        for(unsigned int j = i ; j < N ; j++){
            double point = (data_coordinates.row(i) - data_coordinates.row(j)).norm();
            std::cout << point << std::endl;
            A(i,j) = radialBasisFunction(point);
            A(j,i) = A(i,j);
        }
    }

    std::cout << A << std::endl;

    weights = A.lu().solve(data_values);

    std::cout << weights << std::endl;
}

double RBFInterpolation::radialBasisFunction(double r) const{
    if(kernel == "thin_plate_spline"){
        if(r > 0){
            return std::pow(r, 2) * std::log(r);
        }
        else{
            return 0;
        }
    }
    else{
        return std::pow(r, 2) * std::log(r);
    }
}

VectorXd RBFInterpolation::evaluate(RowVectorXd evaluate_coordinate) const {
    long N = data_coordinates.rows();

    RowVectorXd s(1,N);

    for(unsigned int i = 0 ; i < N ; i++){
        double point = (evaluate_coordinate - data_coordinates.row(i)).norm();
        s(0,i) = radialBasisFunction(point);
    }

    return s*weights;
}

/******************For use as a Functor to use with Eigen's minimizer*****************************
*************************************************************************************************/

int RBFInterpolation::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
{
    // IMPORTANT: The Eigen Levenberg-Marquardt algorithm will square fvec internally (it assumes that the overall error
    // function is made of the sum of squares of fvec components). Therefore, for this rbf, taking the inverse
    // of the absolute value will give the right minimum.
    std::cout << "Computing RBF at: " << x << std::endl;
    RowVectorXd rbf_x = evaluate(x.transpose());
    std::cout << "RBF value: " << rbf_x << std::endl;
    fvec(0) = rbf_x.transpose().cwiseAbs().cwiseInverse().value();
    //fvec(1) = 0;

    std::cout << "fvec value: " << fvec << std::endl;
    return 0;
}

int RBFInterpolation::inputs() const { return 2; }// inputs is the dimension of x.

int RBFInterpolation::values() const { return 2; } // "values" is the number of f_i and


}
}