#ifndef __PARAMETERS_ARTIFICIAL_DISSIPATION_H__
#define __PARAMETERS_ARTIFICIAL_DISSIPATION_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to the linear solver
class ArtificialDissipationParam
{
public:
    /// Flag to add artificial dissipation from Persson's shock capturing paper.
    /** This feature is currently not fully working. It dissipates the Burger's
     *  invisid shock, but loses all the order of accuracy for the Gaussian bump.
     */
    bool add_artificial_dissipation;

    bool entropy_error_discontinuity_sensor;
    
    enum ArtificialDissipationType{
        laplacian, 
        physical, 
        enthalpy_conserving_laplacian
    };
    ArtificialDissipationType artificial_dissipation_type;

    enum ArtificialDissipationTestType{
        residual_convergence,
        discontinuity_sensor_activation,
        poly_order_convergence
    };
    ArtificialDissipationTestType artificial_dissipation_test_type;
    
    double mu_artificial_dissipation;

    double kappa_artificial_dissipation;

    ///Flag to calculate enthalpy error
    bool use_enthalpy_error;

    ArtificialDissipationParam(); /// Constructor
    static void declare_parameters (dealii::ParameterHandler &prm);
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
