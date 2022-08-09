#ifndef __PARAMETERS_TIME_REFINEMENT_STUDY_H__
#define __PARAMETERS_TIME_REFINEMENT_STUDY_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace PHiLiP {

namespace Parameters {

/// Parameters related to time refinement studies
class TimeRefinementStudyParam
{
public:
    TimeRefinementStudyParam(); ///< Constructor

    int number_of_times_to_solve; ///<number of times to run the calculation
    double refinement_ratio; ///<ratio of next timestep size to current one, 0<r<1
    int number_of_timesteps_for_reference_solution; ///<For time refinement study with reference solution, number of steps for reference solution

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace

} // PHiLiP namespace

#endif

