#include "parameters/parameters_reduced_order.h"
namespace PHiLiP {
namespace Parameters {

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
        prm.declare_entry("file_path_for_snapshot_locations", "",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Path to search for lhs snapshots (should contain snapshot_table)");
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
        prm.declare_entry("FOM_error_linear_solver_type", "direct",
                          dealii::Patterns::Selection("direct|gmres"),
                          "Type of linear solver used for first adjoint problem (DWR between FOM and ROM)"
                          "Choices are <direct|gmres>.");
        prm.declare_entry("residual_error_bool", "false",
                          dealii::Patterns::Bool(),
                          "Use residual/reduced residual for error indicator instead of DWR. False by default.");
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
        file_path_for_snapshot_locations = prm.get("file_path_for_snapshot_locations");
        recomputation_coefficient = prm.get_integer("recomputation_coefficient");
        path_to_search = prm.get("path_to_search");

        std::string parameter_names_string = prm.get("parameter_names");
        std::unique_ptr<dealii::Patterns::PatternBase> ListPatternNames(new dealii::Patterns::List(dealii::Patterns::Anything(), 0, 10, ",")); //Note, in a future version of dealii, this may change from a unique_ptr to simply the object. Will need to use std::move(ListPattern) in next line.
        parameter_names = dealii::Patterns::Tools::Convert<decltype(parameter_names)>::to_value(parameter_names_string, ListPatternNames);

        std::string parameter_min_string = prm.get("parameter_min_values");
        std::unique_ptr<dealii::Patterns::PatternBase> ListPatternMin(new dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ",")); //Note, in a future version of dealii, this may change from a unique_ptr to simply the object. Will need to use std::move(ListPattern) in next line.
        parameter_min_values = dealii::Patterns::Tools::Convert<decltype(parameter_min_values)>::to_value(parameter_min_string, ListPatternMin);

        std::string parameter_max_string = prm.get("parameter_max_values");
        std::unique_ptr<dealii::Patterns::PatternBase> ListPatternMax(new dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ",")); //Note, in a future version of dealii, this may change from a unique_ptr to simply the object. Will need to use std::move(ListPattern) in next line.
        parameter_max_values = dealii::Patterns::Tools::Convert<decltype(parameter_max_values)>::to_value(parameter_max_string, ListPatternMax);


        const std::string solver_string = prm.get("FOM_error_linear_solver_type");
        if (solver_string == "direct") FOM_error_linear_solver_type = LinearSolverEnum::direct;
        if (solver_string == "gmres") FOM_error_linear_solver_type = LinearSolverEnum::gmres;
        residual_error_bool = prm.get_bool("residual_error_bool");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace