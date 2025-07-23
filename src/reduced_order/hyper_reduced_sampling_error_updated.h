#ifndef __HYPER_REDUCED_SAMPLING_ERROR_UPDATED__
#define __HYPER_REDUCED_SAMPLING_ERROR_UPDATED__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "pod_basis_online.h"
#include "hrom_test_location.h"
#include <eigen/Eigen/Dense>
#include "nearest_neighbors.h"
#include "adaptive_sampling_base.h"

namespace PHiLiP {
using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

/// Hyperreduced Adaptive Sampling with the updated error indicator
// Currently seperate from the adaptive sampling base as the pointer of ROM locations has
// been changed to a have type HROMTestLocation to update DWR errors

/*
Based on the work in Donovan Blais' thesis:
Goal-Oriented Adaptive Sampling for Projection-Based Reduced-Order Models, 2022

Details on the ROM points/errors can be found in sections 5 and 6

Derivation of the new error indicator will likely be detailed in Calista Biondic's thesis
*/

template <int dim, int nstate>
class HyperreducedSamplingErrorUpdated: public AdaptiveSamplingBase<dim,nstate>
{
public:
    /// Constructor
    HyperreducedSamplingErrorUpdated(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Run test
    int run_sampling () const override;

    /// Compute RBF and find max error
    RowVectorXd getMaxErrorROM() const override;

    /// Placement of ROMs
    bool placeROMLocations(const MatrixXd& rom_points, Epetra_Vector weights) const;

    /// Compute true/actual error at all ROM points (error in functional between FOM and ROM solution)
    void trueErrorROM(const MatrixXd& rom_points, Epetra_Vector weights) const;

    /// Solve FOM and ROM, return error in functional between the models
    double solveSnapshotROMandFOM(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Solve HROM and track functional
    void solveFunctionalHROM(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Updates nearest ROM points to snapshot if error discrepancy is above tolerance
    void updateNearestExistingROMs(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Solve reduced-order solution
    std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Copy all elements in matrix A to all cores
    Epetra_Vector allocateVectorToSingleCore(const Epetra_Vector &b) const;

    /// Output for each iteration
    void outputIterationData(std::string iteration) const override;

    /// Vector of parameter-HROMTestLocation pairs
    mutable std::vector<std::unique_ptr<ProperOrthogonalDecomposition::HROMTestLocation<dim,nstate>>> hrom_locations;

    /// Ptr vector of ECSW Weights
    mutable std::shared_ptr<Epetra_Vector> ptr_weights;

    /// Functional value predicted by the rom at each sammpling iteration at parameter location specified in the inputs
    mutable std::vector<double> rom_functional;

};

}


#endif