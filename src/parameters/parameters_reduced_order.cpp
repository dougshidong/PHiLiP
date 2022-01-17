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
                          dealii::Patterns::Double(2, 10),
                          "Burgers Rewienski parameter a");
        prm.declare_entry("rewienski_b", "0.02",
                          dealii::Patterns::Double(0.01, 0.08),
                          "Burgers Rewienski parameter b");
        prm.declare_entry("final_time", "50",
                          dealii::Patterns::Double(0, 1000),
                          "Final solution time");
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
    }
    prm.leave_subsection();
}

void ReducedOrderModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        rewienski_a = prm.get_double("rewienski_a");
        rewienski_b = prm.get_double("rewienski_b");
        final_time = prm.get_double("final_time");
        rewienski_manufactured_solution = prm.get_bool("rewienski_manufactured_solution");
        coarse_basis_dimension = prm.get_integer("coarse_basis_dimension");
        fine_basis_dimension = prm.get_integer("fine_basis_dimension");
        adapt_coarse_basis_constant = prm.get_integer("adapt_coarse_basis_constant");
        adaptation_tolerance = prm.get_double("adaptation_tolerance");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
