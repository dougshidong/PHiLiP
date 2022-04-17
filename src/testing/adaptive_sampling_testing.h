#ifndef __ADAPTIVE_SAMPLING_TESTING__
#define __ADAPTIVE_SAMPLING_TESTING__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"
#include "functional/functional.h"
#include <Eigen/Dense>

namespace PHiLiP {
namespace Tests {

using Eigen::RowVectorXd;
using Eigen::RowVector2d;
using Eigen::VectorXd;
using Eigen::MatrixXd;

/// Burgers Rewienski snapshot
template <int dim, int nstate>
class AdaptiveSamplingTesting: public TestsBase
{
public:
    /// Constructor.
    AdaptiveSamplingTesting(const Parameters::AllParameters *const parameters_input);

    /// Run test
    int run_test () const override;

    Parameters::AllParameters reinitParams(RowVector2d parameter) const;

    void outputErrors(int iteration) const;
};


} // End of Tests namespace
} // End of PHiLiP namespace

#endif
