#ifndef __ROM_ERROR_POST_SAMPLING_H__
#define __ROM_ERROR_POST_SAMPLING_H__

#include "tests.h"
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace Tests {

using Eigen::MatrixXd;

/// Find the "true" error between the FOM and a ROM (with no hyperreduction) built from the snapshots after
/// an adaptive sampling procedure has been run.
/// NOTE: The folder the test reads from should only contain the outputted files from the last iteration of
/// the adaptive sampling procedure. It should include one text file beginning with "snapshot_table" and one
/// beginning with "solution_snapshots".
template <int dim, int nstate>
class ROMErrorPostSampling: public TestsBase
{
public:
    /// Constructor.
    ROMErrorPostSampling(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    
    /// Reinitialize parameters
    Parameters::AllParameters reinit_params(std::string path) const;

    /// Evaluate and output the "true" error at ROM Points
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    /// Matrix of error sampling points
    mutable MatrixXd rom_points;
    
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
