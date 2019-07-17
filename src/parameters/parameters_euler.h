#ifndef __PARAMETERS_EULER_H__
#define __PARAMETERS_EULER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to the linear solver
class EulerParam
{
public:
    double ref_length;
    double mach_inf;
    double gamma_gas;
    double angle_of_attack;
    double side_slip_angle;

    EulerParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
