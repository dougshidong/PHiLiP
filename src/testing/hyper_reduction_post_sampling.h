#ifndef __HYPER_REDUCTION_POST_SAMPLING_H__
#define __HYPER_REDUCTION_POST_SAMPLING_H__

#include "tests.h"
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {

using Eigen::MatrixXd;

/// Evaluates HROM at ROM points from an adaptive sampling procedure run without hyperreduction
/// NOTE: The folder the test reads from should only contain the outputted files from the last iteration of
/// the adaptive sampling procedure. It should include one text file beginning with "snapshot_table", one
/// beginning with "solution_snapshots", and one beginning with "rom_table".
template <int dim, int nstate>
class HyperReductionPostSampling: public TestsBase
{
public:
    /// Constructor.
    HyperReductionPostSampling(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    
    /// Reinitialize parameters
    Parameters::AllParameters reinit_params(const int max_iter) const;

    /// Conduct hyperreduction and evaluate HROM at ROM points
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
    
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
