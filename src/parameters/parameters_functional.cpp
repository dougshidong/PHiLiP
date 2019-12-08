#include "parameters_functional.h"

namespace PHiLiP {

namespace Parameters {

FunctionalParam::FunctionalParam(){}

void FunctionalParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("functional");
    {
    }
    prm.leave_subsection();
}

void FunctionalParam::parse_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("functional");
    {
    }
    prm.leave_subsection();
}

} // namespace Parameters

} // namespace PHiLiP