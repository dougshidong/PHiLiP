#include "rbf_interpolation.h"
#include <eigen/Eigen/Dense>
#include <eigen/Eigen/LU>
#include <iostream>
#include "ROL_StdVector.hpp"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

RBFInterpolation::RBFInterpolation(const MatrixXd& data_coordinates, const VectorXd& data_values, std::string kernel)
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
            //std::cout << point << std::endl;
            A(i,j) = radialBasisFunction(point);
            A(j,i) = A(i,j);
        }
    }

    weights = A.lu().solve(data_values);
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
    else if(kernel == "cubic"){
        return std::pow(r, 3);
    }
    else if(kernel == "linear"){
        return r;
    }
    else{
        return std::pow(r, 2) * std::log(r);
    }
}

double RBFInterpolation::evaluate(const RowVectorXd& evaluate_coordinate) const {
    long N = data_coordinates.rows();

    RowVectorXd s(1,N);

    for(unsigned int i = 0 ; i < N ; i++){
        double point = (evaluate_coordinate - data_coordinates.row(i)).norm();
        s(0,i) = radialBasisFunction(point);
    }

    //std::cout << s*weights << std::endl;

    return s*weights;
}

double RBFInterpolation::value(const ROL::Vector<double> &x, double &/*tol*/ ) {
    ROL::Ptr<const vector> xp = getVector<ROL::StdVector<double>>(x);
    RowVectorXd evaluate_coordinate(2);
    evaluate_coordinate(0) = (*xp)[0];
    evaluate_coordinate(1) = (*xp)[1];
    double val = evaluate(evaluate_coordinate);

    //For optimization, return -abs(val) to consider only magnitude of error, not sign
    return -std::abs(val);
}

}
}