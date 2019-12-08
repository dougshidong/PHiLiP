#include "parameters_grid_refinement_study.h"

namespace PHiLiP {

namespace Parameters {

GridRefinementStudyParam::GridRefinementStudyParam() : 
    functional_param(FunctionalParam()), 
    grid_refinement_param(GridRefinementParam()),
    manufactured_solution_param(ManufacturedSolutionParam()) {}

void GridRefinementStudyParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("grid refinement study");
    {
        Parameters::FunctionalParam::declare_parameters(prm);
        Parameters::GridRefinementParam::declare_parameters(prm);
        Parameters::ManufacturedSolutionParam::declare_parameters(prm);
    }
    prm.leave_subsection();
}

void GridRefinementStudyParam::parse_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("grid refinement study");
    {
        functional_param.parse_parameters(prm);
        grid_refinement_param.parse_parameters(prm);
        manufactured_solution_param.parse_parameters(prm);
    }
    prm.leave_subsection();
}

} // namespace Parameters

} // namespace PHiLiP