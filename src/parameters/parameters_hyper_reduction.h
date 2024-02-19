#ifndef __PARAMETERS_HYPER_REDUCTION_H__
#define __PARAMETERS_HYPER_REDUCTION_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to reduced-order model
class HyperReductionParam
{
public:

    /// Tolerance for NNLS Solver
    double NNLS_tol;

    /// Maximum number of iterations for NNLS Solver
    int NNLS_max_iter;

    /// Training data (Residual-based vs Jacobian-based)
    std::string training_data;

    /// Maximum number of snapshots in the ECSW training
    int num_training_snaps;

    /// Run Adapative Sampling (Online POD) or use Snapshots in path_to_search in Reduced Order Params
    bool adapt_sampling_bool;

    /// Minimum Error for ROM sampling point to be included in post-sampling HROM analysis
    double ROM_error_tol;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // Parameters namespace
} // PHiLiP namespace
#endif