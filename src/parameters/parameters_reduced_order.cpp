#include "parameters/parameters_reduced_order.h"
namespace PHiLiP {
namespace Parameters {

// Reduced Order Model inputs
ReducedOrderModelParam::ReducedOrderModelParam () {}

void ReducedOrderModelParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        prm.declare_entry("rewienski_a", "2.2360679775", //sqrt(5)
                          dealii::Patterns::Double(1, 10),
                          "Burgers Rewienski parameter a");
        prm.declare_entry("rewienski_b", "0.02",
                          dealii::Patterns::Double(0.01, 0.1),
                          "Burgers Rewienski parameter b");
        prm.declare_entry("rewienski_manufactured_solution", "false",
                          dealii::Patterns::Bool(),
                          "Adds the manufactured solution source term to the PDE source term."
                          "Set as true for running a manufactured solution.");
        prm.declare_entry("coarse_basis_dimension", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Initial dimension of the coarse POD basis");
        prm.declare_entry("fine_basis_dimension", "0",
                          dealii::Patterns::Integer(0,dealii::Patterns::Integer::max_int_value),
                          "Initial dimension of the fine POD basis");
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
    }
    prm.leave_subsection();
}

void ReducedOrderModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        rewienski_a = prm.get_double("rewienski_a");
        rewienski_b = prm.get_double("rewienski_b");
        rewienski_manufactured_solution = prm.get_bool("rewienski_manufactured_solution");
        coarse_basis_dimension = prm.get_integer("coarse_basis_dimension");
        fine_basis_dimension = prm.get_integer("fine_basis_dimension");
        adapt_coarse_basis_constant = prm.get_integer("adapt_coarse_basis_constant");
        adaptation_tolerance = prm.get_double("adaptation_tolerance");
        path_to_search = prm.get("path_to_search");
        method_of_snapshots = prm.get_bool("method_of_snapshots");
        consider_error_sign = prm.get_bool("consider_error_sign");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
