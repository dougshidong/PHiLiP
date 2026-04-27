#include "nearest_neighbors.h"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

NearestNeighbors::NearestNeighbors()
        : snapshot_params()
        , scaler()
        , snapshots()
{}

void NearestNeighbors::update_snapshots(const MatrixXd &snapshot_parameters, dealii::LinearAlgebra::distributed::Vector<double> snapshot){
    snapshot_params = snapshot_parameters;
    snapshots.emplace_back(snapshot);
    if(snapshots.size() > 1){
        scaled_snapshot_params = scaler.fit_transform(snapshot_params);
    }
}

MatrixXd NearestNeighbors::kPairwiseNearestNeighborsMidpoint(){

    MatrixXd midpoints(0, snapshot_params.cols());

    for(auto snapshot : scaled_snapshot_params.rowwise()){

        VectorXd distances = (scaled_snapshot_params.rowwise() - snapshot).rowwise().squaredNorm();

        std::vector<int> index(distances.size());
        std::iota(index.begin(), index.end(), 0);

        std::sort(index.begin(), index.end(),
                  [&](const int& a, const int& b) {
                      return distances[a] < distances[b];
                  });

        for (int i = 1 ; i < snapshot.cols() + 2 ; i++) { //Ignore zeroth index as this would be the same point
            midpoints.conservativeResize(midpoints.rows()+1, midpoints.cols());
            midpoints.row(midpoints.rows()-1) = (snapshot_params.row(index[i])+scaler.inverse_transform(snapshot))/2;
        }
    }
    return midpoints;
}

MatrixXd NearestNeighbors::kNearestNeighborsMidpoint(const RowVectorXd& point){
    RowVectorXd scaled_point = scaler.transform(point);
    VectorXd distances = (scaled_snapshot_params.rowwise() - scaled_point).rowwise().squaredNorm();

    std::vector<int> index(distances.size());
    std::iota(index.begin(), index.end(), 0);

    std::sort(index.begin(), index.end(),
              [&](const int& a, const int& b) {
                  return distances[a] < distances[b];
              });


    MatrixXd midpoints(point.cols()+1, point.cols());
    for (int i = 0 ; i < point.cols()+1 ; i++) {
        midpoints.row(i) = (snapshot_params.row(index[i+1])+point)/2; //i+1 to ignore zeroth index as this would be the same point
    }

    return midpoints;
}

dealii::LinearAlgebra::distributed::Vector<double> NearestNeighbors::nearestNeighborMidpointSolution(const RowVectorXd& point){
    RowVectorXd scaled_point = scaler.transform(point);
    VectorXd distances = (scaled_snapshot_params.rowwise() - scaled_point).rowwise().squaredNorm();

    std::vector<int> index(distances.size());
    std::iota(index.begin(), index.end(), 0);

    std::sort(index.begin(), index.end(),
              [&](const int& a, const int& b) {
                  return distances[a] < distances[b];
              });

    dealii::LinearAlgebra::distributed::Vector<double> interpolated_solution = snapshots[index[0]];
    interpolated_solution += snapshots[index[1]];
    interpolated_solution /= 2;
    return interpolated_solution;
}

}
}