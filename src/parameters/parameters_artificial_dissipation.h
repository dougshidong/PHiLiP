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

    /// Parameter to freeze artificial dissipation below a certain residual.
    double freeze_artificial_dissipation_below_residual;

    ///Flag to calculate enthalpy error 
    bool use_enthalpy_error;

    /// Include artificial dissipation in Jacobian
    ///
    /// Note, it cannot be used with the C0-smoothed artificial dissipation.
    /// This is because the C0-smoothed artificial dissipation pre-computes the constant artificial dissipation coefficients, and 
    /// transforms them in a second cell loop. As a result, it is not evaluated within the DG loop assembly that includes the 
    /// automatic differentiation mechanics.
    bool include_artificial_dissipation_in_jacobian;

    /// Smooths out artificial dissipation through a C0-continuous function
    /// Note, it cannot be used when the artificial dissipation is included in the Jacobian.
    /// Note, it cannot be used with the Gegenbauer-smoothed artificial dissipation.
    ///
    /// Persson, P.-O., “Shock Capturing for High-Order Discontinuous Galerkin 
    /// Simulation of Transient Flow Problems,” 2013. https://doi.org/10.2514/6.2013-3061
    bool use_c0_smoothed_artificial_dissipation;

    /// Sets artificial dissipation to zero at the boundary.
    /// Defaults to true since it is unclear how to implement artificial dissipation boundary conditions
    /// for inviscid flows.
    bool zero_artificial_dissipation_at_boundary;

    /// Use Gegenbauer polynomial to smooth out artificial dissipation
    /// It should technically transform the polynomial using a Gegenbauer function,
    /// However, the implementation is such that the artificial dissipation is zero-ed out at every cell 
    /// face, which is equivalent to using a Gegenbauer polynomial with a tiny exponent (lambda).
    /// See the reference below for more details, Fig. 5.
    ///
    /// Note, it can not be used with the C0-smoothed artificial dissipation.
    ///
    /// Glaubitz, J., Nogueira, A. C., Almeida, J. L. S., Cantão, R. F., and Silva, C. A. C., 
    /// “Smooth and Compactly Supported Viscous Sub-Cell Shock Capturing for Discontinuous Galerkin Methods,”
    /// Journal of Scientific Computing, Vol. 79, No. 1, 2018, pp. 249–272.
    ///  https://doi.org/10.1007/s10915-018-0850-3
    bool use_gegenbauer_smoothed_artificial_dissipation;

  
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
