#ifndef __HYPER_ERROR_POST_SAMPLING_H__
#define __HYPER_ERROR_POST_SAMPLING_H__

#include "tests.h"
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {

using Eigen::MatrixXd;

/// Hyper-reduction test, verifies accuracy of the solution with ECSW hyper-reduction of the residual and Jacobian
template <int dim, int nstate>
class HyperErrorPostSampling: public TestsBase
{
public:
    /// Constructor.
    HyperErrorPostSampling(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    
    /// Reinitialize parameters
    Parameters::AllParameters reinitParams(std::string path) const;
    
    bool getSnapshotParamsFromFile() const;

    bool getROMParamsFromFile() const;

    void getROMPoints() const;

    std::shared_ptr<Epetra_Vector> getWeightsFromFile() const;

    /// Run Hyper-reduction tes
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    mutable MatrixXd rom_points;
    
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
