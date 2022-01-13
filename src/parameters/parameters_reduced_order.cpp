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
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
