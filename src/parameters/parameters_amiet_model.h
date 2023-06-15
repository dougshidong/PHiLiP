#ifndef __PARAMETERS_AMIET_H__
#define __PARAMETERS_AMIET_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to aero-acoustic Amiet's model
class AmietParam
{
public:
    /// Choices for acoustic contribution types.
    enum AcousticContributionEnum{
        main,
        back,
        main_and_back
    };

    /// Choices for wall pressure spectral model types.
    enum WallPressureSpectralModelEnum{
        Goody,
        Rozenberg,
        Kamruzzaman
    };

    /// Acoustic contribution types input.
    AcousticContributionEnum acoustic_contribution_type;

    /// Wall pressure spectral model types input.
    WallPressureSpectralModelEnum wall_pressure_spectral_model_type;

    /// Lower limit of investigated frequency (rad/s) range.
    double omega_min;

    /// Upper limit of investigated frequency (rad/s) range.
    double omega_max;

    /// Interval of investigated frequency (rad/s) range.
    double omega_interval;

    /// Coordinate of farfield acoustic observer.
    /** Observer coordinate is based on reference coordinate built at the center of trailing edge as sketched:
     *                       ------------------
     *              /       /                /     z
     *             /       /                /      ^
     *            /       /                /       |
     *      b <- /       /--------------- /(0,0,0) ----> x 
     *          /       /       /        /        /
     *         /       /       /-> b/2  /        v 
     *        /       /       /        /        y
     *               ------------------
     *  Note: the observer coordinate is most likely different from the one used for flow solver 
     */
    /// The x coordinate of farfield acoustic observer
    double observer_coord_ref_x;

    /// The y coordinate of farfield acoustic observer
    double observer_coord_ref_y;

    /// The z coordinate of farfield acoustic observer
    double observer_coord_ref_z;

    /// Reference density. Units: [kg/m^3].
    double ref_density;

    /// Nondimensional chord length of airfoil.
    double chord_length;

    /// Nondimensional span length of airfoil.
    double span_length;

    /// Ratio of free-stream and convection speed of turbulence.
    double alpha;

    /// Specific gas constant R. Units: [J/(kg*K)].
    double R_specific;

    AmietParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif