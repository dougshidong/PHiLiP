#include "parameters/parameters_reduced_order.h"
namespace PHiLiP {
namespace Parameters {

// Reduced Order Model inputs
ReducedOrderModelParam::ReducedOrderModelParam () {}

void ReducedOrderModelParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        prm.declare_entry("adaptation_tolerance", "1",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for POD adaptation");
        prm.declare_entry("path_to_search", ".",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Path to search for saved snapshots or POD basis.");
        prm.declare_entry("reduced_residual_tolerance", "1E-13",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for nonlinear reduced residual");
        prm.declare_entry("num_halton", "0",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Number of Halton sequence points to add to initial snapshot set");
        prm.declare_entry("recomputation_coefficient", "5",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Number of Halton sequence points to add to initial snapshot set");
        prm.declare_entry("parameter_names", "mach, alpha",
                          dealii::Patterns::List(dealii::Patterns::Anything(), 0, 10, ","),
                          "Names of parameters for adaptive sampling");
        prm.declare_entry("parameter_min_values", "0.4, 0",
                          dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ","),
                          "Minimum values for parameters");
        prm.declare_entry("parameter_max_values", "0.7, 4",
                          dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ","),
                          "Maximum values for parameters");
    }
    prm.leave_subsection();
}

void ReducedOrderModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        adaptation_tolerance = prm.get_double("adaptation_tolerance");
        reduced_residual_tolerance = prm.get_double("reduced_residual_tolerance");
        num_halton = prm.get_integer("num_halton");
        recomputation_coefficient = prm.get_integer("recomputation_coefficient");
        path_to_search = prm.get("path_to_search");

        std::string parameter_names_string = prm.get("parameter_names");
        const dealii::Patterns::List ListPatternNames = dealii::Patterns::List(dealii::Patterns::Anything(), 0, 10, ",");
        parameter_names = dealii::Patterns::Tools::Convert<decltype(parameter_names)>::to_value(parameter_names_string, ListPatternNames);

        std::string parameter_min_string = prm.get("parameter_min_values");
        const dealii::Patterns::List ListPatternMin = dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ","); 
        parameter_min_values = dealii::Patterns::Tools::Convert<decltype(parameter_min_values)>::to_value(parameter_min_string, ListPatternMin);

        std::string parameter_max_string = prm.get("parameter_max_values");
        const dealii::Patterns::List ListPatternMax = dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ","); 
        parameter_max_values = dealii::Patterns::Tools::Convert<decltype(parameter_max_values)>::to_value(parameter_max_string, ListPatternMax);
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
