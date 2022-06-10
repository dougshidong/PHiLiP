#include <numeric>
#include "nearest_neighbors.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

NearestNeighbors::NearestNeighbors(const MatrixXd& snapshot_parameters)
        : snapshot_params(snapshot_parameters)
{}

MatrixXd NearestNeighbors::kPairwiseNearestNeighborsMidpoint(){

    MatrixXd midpoints(0, snapshot_params.cols());

    for(auto snapshot : snapshot_params.rowwise()){

        VectorXd distances = (snapshot_params.rowwise() - snapshot).rowwise().squaredNorm();

        std::vector<int> index(distances.size());
        std::iota(index.begin(), index.end(), 0);

        std::sort(index.begin(), index.end(),
                  [&](const int& a, const int& b) {
                      return distances[a] < distances[b];
                  });

        for (int i = 1 ; i < snapshot.cols() + 2 ; i++) { //Ignore zeroth index as this would be the same point
            midpoints.conservativeResize(midpoints.rows()+1, midpoints.cols());
            midpoints.row(midpoints.rows()-1) = (snapshot_params.row(index[i])+snapshot)/2;
        }
    }
    return midpoints;
}

MatrixXd NearestNeighbors::kNearestNeighborsMidpoint(const RowVectorXd& point){
    VectorXd distances = (snapshot_params.rowwise() - point).rowwise().squaredNorm();

    std::vector<int> index(distances.size());
    std::iota(index.begin(), index.end(), 0);

    std::sort(index.begin(), index.end(),
              [&](const int& a, const int& b) {
                  return distances[a] < distances[b];
              });


    MatrixXd midpoints(point.cols()+1, point.cols());
    for (int i = 0 ; i < point.cols()+1 ; i++) {
        midpoints.row(i) = (snapshot_params.row(index[i])+point)/2;
    }

    return midpoints;
}

Eigen::Index NearestNeighbors::nearestNeighbor(const RowVectorXd& point){
    Eigen::Index index;
    (snapshot_params.rowwise() - point).rowwise().squaredNorm().minCoeff(&index);
    std::cout << "Nearest neighbour is row " << index << ":" << std::endl;
    std::cout << snapshot_params.row(index) << std::endl;
    return index;
}

void NearestNeighbors::updateSnapshotParameters(const MatrixXd &snapshot_parameters){
    snapshot_params = snapshot_parameters;
}

}
}