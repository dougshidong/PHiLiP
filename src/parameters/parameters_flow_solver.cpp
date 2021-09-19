#include "parameters_flow_solver.h"

#include <string>

namespace PHiLiP {

namespace Parameters {

// Flow Solver inputs
FlowSolverParam::FlowSolverParam() {}

void FlowSolverParam::declare_parameters(dealii::ParameterHandler &prm)
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
}

void FlowSolverParam::parse_parameters(dealii::ParameterHandler &prm)
{   
    const std::string flow_case_type_string = prm.get("flow_case_type");
    if     (flow_case_type_string == "inviscid_taylor_green_vortex")  {flow_case_type = inviscid_taylor_green_vortex;} 
    else if(flow_case_type_string == "viscous_taylor_green_vortex")   {flow_case_type = viscous_taylor_green_vortex;} 
}

} // Parameters namespace

} // PHiLiP namespace
