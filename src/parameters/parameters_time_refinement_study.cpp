#include "parameters_time_refinement_study.h"

#include <string>

namespace PHiLiP {

namespace Parameters {

TimeRefinementStudyParam::TimeRefinementStudyParam() {}

void TimeRefinementStudyParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("time_refinement_study");
    {
        prm.declare_entry("number_of_times_to_solve", "4",
                          dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                          "Number of times to run the flow solver during a time refinement study.");
        prm.declare_entry("refinement_ratio", "0.5",
                          dealii::Patterns::Double(0, 1.0),
                          "Ratio between the next timestep size and the current one in a time refinement study, 0<r<1.");
        prm.declare_entry("number_of_timesteps_for_reference_solution", "100000",
                          dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                          "Number of times to run the flow solver during a time refinement study.");
    }
    prm.leave_subsection();
}

void TimeRefinementStudyParam::parse_parameters(dealii::ParameterHandler &prm)
{   
    prm.enter_subsection("time_refinement_study");
    {
        number_of_times_to_solve = prm.get_integer("number_of_times_to_solve");
        refinement_ratio = prm.get_double("refinement_ratio");
        number_of_timesteps_for_reference_solution = prm.get_integer("number_of_timesteps_for_reference_solution");
    }
    prm.leave_subsection();
}

} // Parameters namespace

} // PHiLiP namespace
