#ifndef __ADAPTIVE_SAMPLING_TESTING__
#define __ADAPTIVE_SAMPLING_TESTING__

#include "tests.h"
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace Tests {

using Eigen::RowVectorXd;
using Eigen::RowVector2d;
using Eigen::VectorXd;
using Eigen::MatrixXd;

/// Adaptive Sampling Testing
template <int dim, int nstate>
class AdaptiveSamplingTesting: public TestsBase
{
public:
    /// Constructor.
    AdaptiveSamplingTesting(const Parameters::AllParameters *const parameters_input,
                            const dealii::ParameterHandler &parameter_handler_input);

    /// Run test
    int run_test () const override;

    /// Reinitialize parameters
    Parameters::AllParameters reinit_params(RowVector2d parameter) const;

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Output errors
    void outputErrors(int iteration) const;
};


} // End of Tests namespace
} // End of PHiLiP namespace

#endif
