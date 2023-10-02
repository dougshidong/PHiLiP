#include "parameters/parameters_limiter.h"

namespace PHiLiP {
namespace Parameters {

// Limiter inputs
LimiterParam::LimiterParam() = default;

void LimiterParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("limiter");
    {
        prm.declare_entry("use_OOA", "false",
                          dealii::Patterns::Bool(),
                          "Does not perform Convergence Test by default. Otherwise, performs Convergence Test.");

        prm.declare_entry("bound_preserving_limiter", "none",
                           dealii::Patterns::Selection(
                           "none | "
                           "maximum_principle | "
                           "positivity_preservingZhang2010 | "
                           "positivity_preservingWang2012 "),
                           "The type of limiter we want to apply to the solution. "
                           "Choices are "
                           " <none | "
                           " maximum_principle | "
                           " positivity_preservingZhang2010 | "
                           " positivity_preservingWang2012>.");

        prm.declare_entry("min_density", "1e-13",
                          dealii::Patterns::Double(1e-20, 1e200),
                          "Lower bound for density used in Positivity-Preserving Limiter. Small value greater than zero, less than solution at all times.");

        prm.declare_entry("use_tvb_limiter", "false",
                          dealii::Patterns::Bool(),
                          "Applies TVB Limiter to solution. Tune M and h to obtain favourable results.");

        prm.declare_entry("max_delta_x", "1.0",
                          dealii::Patterns::Double(0, 1e200),
                          "Maximum delta_x.");

        using convert_tensor = dealii::Patterns::Tools::Convert<dealii::Tensor<1, 4, double>>;
        prm.declare_entry("tuning_parameter_for_each_state", "0,0,0,0", 
                          *convert_tensor::to_pattern(), 
                          "TVB Limiter tuning parameters for each state.");
    }
    prm.leave_subsection();
}

void LimiterParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("limiter");
    {
        use_OOA = prm.get_bool("use_OOA");

        const std::string bound_preserving_limiter_string = prm.get("bound_preserving_limiter");
        if (bound_preserving_limiter_string == "none")                               bound_preserving_limiter = LimiterType::none;
        if (bound_preserving_limiter_string == "maximum_principle")                  bound_preserving_limiter = LimiterType::maximum_principle;
        if (bound_preserving_limiter_string == "positivity_preservingZhang2010")     bound_preserving_limiter = LimiterType::positivity_preservingZhang2010;
        if (bound_preserving_limiter_string == "positivity_preservingWang2012")      bound_preserving_limiter = LimiterType::positivity_preservingWang2012;

        min_density = prm.get_double("min_density");

        use_tvb_limiter = prm.get_bool("use_tvb_limiter");
        max_delta_x = prm.get_double("max_delta_x");

        using convert_tensor = dealii::Patterns::Tools::Convert<dealii::Tensor<1, 4, double>>;
        tuning_parameter_for_each_state = convert_tensor::to_value(prm.get("tuning_parameter_for_each_state"));
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
