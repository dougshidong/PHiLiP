#include "parameters_flow_solver.h"

#include <string>

namespace PHiLiP {

namespace Parameters {

// Flow Solver inputs
FlowSolverParam::FlowSolverParam() {}

void FlowSolverParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("flow_solver");
    {
        prm.declare_entry("flow_case_type","inviscid_taylor_green_vortex",
                          dealii::Patterns::Selection(
                          " inviscid_taylor_green_vortex | "
                          " viscous_taylor_green_vortex"
                          ),
                          "The type of flow we want to simulate. "
                          "Choices are "
                          " <inviscid_taylor_green_vortex | "
                          "  viscous_taylor_green_vortex>.");

        prm.declare_entry("final_time", "1",
                          dealii::Patterns::Double(1e-15, 10000000),
                          "Final solution time.");

        prm.declare_entry("courant_friedrich_lewy_number", "1",
                          dealii::Patterns::Double(1e-15, 10000000),
                          "Courant-Friedrich-Lewy (CFL) number for constant time step.");
    }
    prm.leave_subsection();
}

void FlowSolverParam::parse_parameters(dealii::ParameterHandler &prm)
{   
    prm.enter_subsection("flow_solver");
    {
        const std::string flow_case_type_string = prm.get("flow_case_type");
        if     (flow_case_type_string == "inviscid_taylor_green_vortex")  {flow_case_type = inviscid_taylor_green_vortex;} 
        else if(flow_case_type_string == "viscous_taylor_green_vortex")   {flow_case_type = viscous_taylor_green_vortex;} 

        final_time = prm.get_double("final_time");
        courant_friedrich_lewy_number = prm.get_double("courant_friedrich_lewy_number");
    }
    prm.leave_subsection();
}

} // Parameters namespace

} // PHiLiP namespace
