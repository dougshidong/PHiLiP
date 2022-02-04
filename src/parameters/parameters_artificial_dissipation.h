#ifndef __PARAMETERS_ARTIFICIAL_DISSIPATION_H__
#define __PARAMETERS_ARTIFICIAL_DISSIPATION_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to the artificial dissipation
class ArtificialDissipationParam
{
public:
    /// Flag to add artificial dissipation from Persson's shock capturing paper.
    /** This feature dissipates the Burger's invisid shock and shock over transonic euler gaussian bump.
    *   Some dissipation types reduce enthalpy significantly. However, the (p+1) order of convergence is lost for the case of Gaussian bump with shocks.
    */
    bool add_artificial_dissipation;
  
    /// Specified choices of artificial dissipation type.
    enum ArtificialDissipationType{
        laplacian, 
        physical, 
        enthalpy_conserving_laplacian
    };

    /// Selected artificial dissipation type specified in the input.
    ArtificialDissipationType artificial_dissipation_type;

    /// Specified choices of dissipation test types.
    enum ArtificialDissipationTestType{
        residual_convergence,
        discontinuity_sensor_activation,
        enthalpy_conservation,
        poly_order_convergence
    };

    /// Selected dissipation test type.
    ArtificialDissipationTestType artificial_dissipation_test_type;
    
    /// Parameter mu from Persson & Peraire, 2008.
    double mu_artificial_dissipation;

    /// Parameter kappa from Persson and Peraire, 2008.
    double kappa_artificial_dissipation;

    ///Flag to calculate enthalpy error 
    bool use_enthalpy_error;

    /// Constructor
    ArtificialDissipationParam();

    /// Function to declare parameters.
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Function to parse parameters.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
