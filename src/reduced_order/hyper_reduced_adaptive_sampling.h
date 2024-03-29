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

    /// Updates nearest ROM points to snapshot if error discrepancy is above tolerance
    void updateNearestExistingROMs(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Solve reduced-order solution
    std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> solveSnapshotROM(const RowVectorXd& parameter, Epetra_Vector weights) const;

    /// Ptr vector of ECSW Weights
    mutable std::shared_ptr<Epetra_Vector> ptr_weights;
};

}


#endif