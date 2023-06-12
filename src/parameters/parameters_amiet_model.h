#ifndef __PARAMETERS_AMIET_H__
#define __PARAMETERS_AMIET_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to aero-acoustic Amiet's model
class AmietParam
{
public:
    /// Choices for acoustic contribution types
    enum AcousticContributionEnum{
        main,
        back,
        main_and_back
    };

    /// Choices for wall pressure spectral model types
    enum WallPressureSpectralModelEnum{
        Goody,
        Rozenberg,
        Kamruzzaman
    };

    AcousticContributionEnum acoustic_contribution_type;

    WallPressureSpectralModelEnum wall_pressure_spectral_model_type;

    double omega_min;
    double omega_max;
    double omega_interval;

    /** Observer coordinate is based on reference coordinate built at the center of trailing edge as sketched                                     
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
    double observer_coord_ref_x;
    double observer_coord_ref_y;
    double observer_coord_ref_z;

    //double ref_U;
    double ref_density;
    //double ref_viscosity;

    double chord_length;
    double span_length;
    double alpha;

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