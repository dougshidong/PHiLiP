#ifndef __PARAMETERS_PHYSICS_MODEL_H__
#define __PARAMETERS_PHYSICS_MODEL_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to Physics Models
class PhysicsModelParam
{
public:
    /// Constructor
    PhysicsModelParam ();

    /** Set as false by default. 
      * If true, sets the baseline physics to the Euler equations
      */
    bool euler_turbulence;

    /// Types of sub-grid scale (SGS) models that can be used.
    enum SubGridScaleModel { 
        smagorinsky
       ,wall_adaptive_local_eddy_viscosity
       ,vreman
       ,shear_improved_smagorinsky
       ,dynamic_smagorinsky
       ,small_small_variational_multiscale
       ,all_all_variational_multiscale
    };
    /// Store the SubGridScale (SGS) model type
    SubGridScaleModel SGS_model_type;

    /// Types of Reynolds-averaged Navier-Stokes (RANS) models that can be used.
    enum ReynoldsAveragedNavierStokesModel {
        SA, 
        SA_negative
    };
    /// Store the Reynolds-averaged Navier-Stokes (RANS) model type
    ReynoldsAveragedNavierStokesModel RANS_model_type;

    /// Turbulent flow characteristics:
    double turbulent_prandtl_number; ///< Turbulent Prandtl number

    /// Eddy-viscosity model constants:
    double smagorinsky_model_constant; ///< Smagorinsky eddy viscosity model Constant
    double WALE_model_constant; ///< WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model constant
    double vreman_model_constant; ///< Vreman eddy viscosity model constant
    double ratio_of_filter_width_to_cell_size; ///< Ratio of the large eddy simulation filter width to the cell size
    bool do_compute_filtered_solution; ///< Flag to compute the filtered solution
    bool apply_modal_high_pass_filter_on_filtered_solution; ///< Flag to apply modal high pass filter on the filtered solution
    unsigned int poly_degree_max_large_scales; ///< Max poly degree representing the large scales for LES VMS filtering
    double dynamic_smagorinsky_model_constant_clipping_limit; ///< Clipping limit for the Dynamic Smagorinsky model constant
    bool apply_low_reynolds_number_eddy_viscosity_correction; ///< Flag for applying the low Reynolds number eddy viscosity correction

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
