#include "parameters/parameters_hyper_reduction.h"
namespace PHiLiP {
namespace Parameters {

void HyperReductionParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("hyperreduction");
    {
        prm.declare_entry("NNLS_tol", "1E-6",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for the NNLS solver");
        prm.declare_entry("NNLS_max_iter", "5000",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Maximum number of iterations for the NNLS solver");
        prm.declare_entry("training_data", "jacobian",
                          dealii::Patterns::Selection(
                          " jacobian | "
                          " residual "
                          ),
                          "Training Data to be used for ECSW Weights"
                          "Choices are "
                          " <jacobian | "
                          "  residual>.");
        prm.declare_entry("num_training_snaps", "0",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Number of snapshots used for training");
        prm.declare_entry("adapt_sampling_bool","true",
                          dealii::Patterns::Bool(),
                          "Set as true by default (i.e. runs adaptive sampling procedure before hyperreduction). ");
        prm.declare_entry("ROM_error_tol", "0",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Minimum Error for ROM sampling point to be included in post-sampling HROM analysis");
    }
    prm.leave_subsection();
}

void HyperReductionParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("hyperreduction");
    {
        NNLS_tol = prm.get_double("NNLS_tol");
        NNLS_max_iter = prm.get_integer("NNLS_max_iter");
        training_data = prm.get("training_data");
        num_training_snaps = prm.get_integer("num_training_snaps");
        adapt_sampling_bool = prm.get_bool("adapt_sampling_bool");
        ROM_error_tol = prm.get_double("ROM_error_tol");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace