#include "parameters/parameters_reduced_order.h"
namespace PHiLiP {
namespace Parameters {

// Reduced Order Model inputs
ReducedOrderModelParam::ReducedOrderModelParam () {}

void ReducedOrderModelParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        prm.declare_entry("coarse_basis_dimension", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Initial dimension of the coarse POD basis");
        prm.declare_entry("fine_basis_dimension", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Initial dimension of the fine POD basis");
        prm.declare_entry("coarse_expanded_basis_dimension", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Initial dimension of the coarse expanded POD basis");
        prm.declare_entry("fine_expanded_basis_dimension", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Dimension of the fine expanded POD basis");
        prm.declare_entry("num_sensitivities", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Number of POD sensitivities to compute and append to state basis");
        prm.declare_entry("extrapolated_basis_dimension", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Initial dimension of the extrapolated POD basis");
        prm.declare_entry("extrapolated_parameter_delta", "0.0",
                          dealii::Patterns::Double(dealii::Patterns::Double::min_double_value, dealii::Patterns::Double::max_double_value),
                          "Change in parameter from base parameter value");
        prm.declare_entry("adapt_coarse_basis_constant", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Number of basis functions to add to coarse basis at each adaptation iteration. Set to 0 to turn off.");
        prm.declare_entry("adaptation_tolerance", "1",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for POD adaptation");
        prm.declare_entry("path_to_search", ".",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Path to search for saved snapshots or POD basis.");
        prm.declare_entry("method_of_snapshots", "false",
                          dealii::Patterns::Bool(),
                          "Use the method of snapshots to compute the POD basis. False by default.");
        prm.declare_entry("consider_error_sign", "false",
                          dealii::Patterns::Bool(),
                          "Consider the sign of the error estimate from the dual-weighted residual. False by default.");
        prm.declare_entry("reduced_residual_tolerance", "1E-13",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for nonlinear reduced residual");
    }
    prm.leave_subsection();
}

void ReducedOrderModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        coarse_basis_dimension = prm.get_integer("coarse_basis_dimension");
        fine_basis_dimension = prm.get_integer("fine_basis_dimension");
        coarse_expanded_basis_dimension = prm.get_integer("coarse_expanded_basis_dimension");
        fine_expanded_basis_dimension = prm.get_integer("fine_expanded_basis_dimension");
        num_sensitivities = prm.get_integer("num_sensitivities");
        extrapolated_basis_dimension = prm.get_integer("extrapolated_basis_dimension");
        extrapolated_parameter_delta = prm.get_double("extrapolated_parameter_delta");
        adapt_coarse_basis_constant = prm.get_integer("adapt_coarse_basis_constant");
        adaptation_tolerance = prm.get_double("adaptation_tolerance");
        reduced_residual_tolerance = prm.get_double("reduced_residual_tolerance");
        path_to_search = prm.get("path_to_search");
        method_of_snapshots = prm.get_bool("method_of_snapshots");
        consider_error_sign = prm.get_bool("consider_error_sign");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace