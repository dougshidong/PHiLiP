#ifndef __HYPER_REDUCTION_COMPARISON_H__
#define __HYPER_REDUCTION_COMPARISON_H__

#include "tests.h"
#include "parameters/all_parameters.h"
#include "dg/dg_base.hpp"
#include <eigen/Eigen/Dense>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {

using Eigen::MatrixXd;

/// Compare FOM, ROM, and HROM (when hyperreduction is conducted after the adaptive sampling procedure)
/// Check accuracy in the solution and the functional at one parameter location
template <int dim, int nstate>
class HyperReductionComparison: public TestsBase
{
public:
    /// Constructor.
    HyperReductionComparison(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    
    /// Reinitialize parameters
    Parameters::AllParameters reinit_params(const int max_iter) const;

    /// Read ECSW weights from the text file 
    bool getWeightsFromFile(std::shared_ptr<DGBase<dim,double>> &dg) const;

    /// Build three models and evaluate error measures
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;

    /// Matrix of snapshot parameters
    mutable MatrixXd snapshot_parameters;

    /// Ptr vector of ECSW Weights
    mutable std::shared_ptr<Epetra_Vector> ptr_weights;
};
} // End of Tests namespace
} // End of PHiLiP namespace

#endif
