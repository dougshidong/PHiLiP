#include "min_max_scaler.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

MatrixXd MinMaxScaler::fit_transform(const MatrixXd &parameters){
    min.resize(parameters.cols());
    max.resize(parameters.cols());
    MatrixXd parameters_scaled(parameters.rows(), parameters.cols());
    for(int j = 0 ; j < parameters.cols() ; j++){
        min(j) = parameters.col(j).minCoeff();
        max(j) = parameters.col(j).maxCoeff();
        if(max(j) == min(j)){
            std::cout << "Min and max are equal, causing the MinMaxScaler to divide by zero. Please ensure that min != max." << std::endl;
            std::abort();
        }
        for(int k = 0 ; k < parameters.rows() ; k++){
            parameters_scaled(k, j) = (parameters(k, j) - min(j)) / (max(j) - min(j));
        }
    }
    return parameters_scaled;
}

MatrixXd MinMaxScaler::transform(const MatrixXd &parameters){
    MatrixXd parameters_scaled(parameters.rows(), parameters.cols());
    for(int j = 0 ; j < parameters.cols() ; j++){
        for(int k = 0 ; k < parameters.rows() ; k++){
            parameters_scaled(k, j) = (parameters(k, j) - min(j)) / (max(j) - min(j));
        }
    }
    return parameters_scaled;
}

MatrixXd MinMaxScaler::inverse_transform(const MatrixXd &parameters_scaled){
    MatrixXd parameters(parameters_scaled.rows(), parameters_scaled.cols());
    for(int j = 0 ; j < parameters_scaled.cols() ; j++){
        for(int k = 0 ; k < parameters_scaled.rows() ; k++){
            parameters(k, j) =  (parameters_scaled(k,j)*(max(j) - min(j))) + min(j);
        }
    }
    return parameters;
}

}
}


