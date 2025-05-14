#ifndef __HYPER_REDUCED_ADAPTIVE_SAMPLING__
#define __HYPER_REDUCED_ADAPTIVE_SAMPLING__

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

/// Hyperreduced adaptive sampling

/*
Based on the work in Donovan Blais' thesis:
Goal-Oriented Adaptive Sampling for Projection-Based Reduced-Order Models, 2022
 
and the ECSW hyperreduction technique:
"Mesh sampling and weighting for the hyperreduction of nonlinear Petrov–Galerkin reduced-order models with local reduced-order bases"
Sebastian Grimberg, Charbel Farhat, Radek Tezaur, Charbel Bou-Mosleh
International Journal for Numerical Methods in Engineering, 2020
https://onlinelibrary.wiley.com/doi/10.1002/nme.6603
*/
template <int dim, int nstate>
class HyperreducedAdaptiveSampling: public AdaptiveSamplingBase<dim,nstate>
{
public:
    /// Constructor
    HyperreducedAdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Run test
    int run_sampling () const override;
    
    /// Placement of ROMs
    bool placeROMLocations(const MatrixXd& rom_points, Epetra_Vector weights) const;

    /// Compute true/actual error at all ROM points (error in functional between FOM and ROM solution)
    void trueErrorROM(const MatrixXd& rom_points, Epetra_Vector weights) const;

    /// Solve FOM and ROM, return error in functional between the models
    double solveSnapshotROMandFOM(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Solve ROM and track functional
    void solveFunctionalHROM(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Updates nearest ROM points to snapshot if error discrepancy is above tolerance
    void updateNearestExistingROMs(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Solve reduced-order solution
    std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Copy all elements in matrix A to all cores
    Epetra_Vector allocateVectorToSingleCore(const Epetra_Vector &b) const;

    /// Ptr vector of ECSW Weights
    mutable std::shared_ptr<Epetra_Vector> ptr_weights;

    /// Functional value predicted by the rom at each sammpling iteration at parameter location specified in the inputs
    mutable std::vector<double> rom_functional;
};

}


#endif