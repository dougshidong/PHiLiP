#ifndef __POD_ADAPTIVE_SAMPLING__
#define __POD_ADAPTIVE_SAMPLING__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "pod_basis_online.h"
#include "rom_test_location.h"
#include <eigen/Eigen/Dense>
#include "nearest_neighbors.h"
#include "adaptive_sampling_base.h"

namespace PHiLiP {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// POD adaptive sampling
/*
Based on the work in Donovan Blais' thesis:
Goal-Oriented Adaptive Sampling for Projection-Based Reduced-Order Models, 2022
*/
template <int dim, int nstate>
class AdaptiveSampling: public AdaptiveSamplingBase<dim,nstate>
{
public:
    /// Constructor
    AdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~AdaptiveSampling() {};

    /// Run Sampling Procedure
    int run_sampling () const override;

    /// Placement of ROMs
    bool placeROMLocations(const MatrixXd& rom_points) const;

    /// Compute true/actual error at all ROM points (error in functional between FOM and ROM solution)
    void trueErrorROM(const MatrixXd& rom_points) const;

    /// Solve FOM and ROM, return error in functional between the models
    double solveSnapshotROMandFOM(const RowVectorXd& parameter) const;

    /// Solve ROM and track functional
    void solveFunctionalROM(const RowVectorXd& parameter) const;

    /// Updates nearest ROM points to snapshot if error discrepancy is above tolerance
    void updateNearestExistingROMs(const RowVectorXd& parameter) const;

    /// Solve reduced-order solution
    std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(const RowVectorXd& parameter) const;

    /// Functional value predicted by the rom at each sammpling iteration at parameter location specified in the inputs
    mutable std::vector<double> rom_functional;

};

}


#endif