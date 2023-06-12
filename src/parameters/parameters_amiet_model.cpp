#include "parameters/parameters_amiet_model.h"

namespace PHiLiP {
namespace Parameters {

// Amiet model inputs
AmietParam::AmietParam () {}

void AmietParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("amiet_model");
    {
        prm.declare_entry("acoustic_contribution_type", "main_and_back",
                          dealii::Patterns::Selection(
                          "main | back | main_and_back"),
                          "Acoustic contribution type. "
                          "Choices are <main | back | main_and_back>.");

        prm.declare_entry("wall_pressure_spectral_model_type", "Goody",
                          dealii::Patterns::Selection(
                          "Goody | Rozenberg | Kamruzzaman"),
                          "Wall pressure spectral model type. "
                          "Choices are <Goody | Rozenberg | Kamruzzaman>.");

        prm.declare_entry("omega_min", "1.0",
                          dealii::Patterns::Double(1.0, 100000000.0),
                          "The lower limit of investigated frequency (rad/s) range. Default value is 1.0. ");
        prm.declare_entry("omega_max", "1000000.0",
                          dealii::Patterns::Double(1.0, 100000000.0),
                          "The upper limit of investigated frequency (rad/s) range. Default value is 1000000.0. ");
        prm.declare_entry("omega_interval", "1.0",
                          dealii::Patterns::Double(1.0, 100000000.0),
                          "The interval of investigated frequency (rad/s) range. Default value is 1.0. ");

        prm.declare_entry("observer_coord_ref_x", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The x coordinate of farfield acoustic observer. Default value is 0.0. ");
        prm.declare_entry("observer_coord_ref_y", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The x coordinate of farfield acoustic observer. Default value is 0.0. ");
        prm.declare_entry("observer_coord_ref_z", "0.0",
                          dealii::Patterns::Double(-1000.0, 1000.0),
                          "The x coordinate of farfield acoustic observer. Default value is 0.0. ");

        //prm.declare_entry("ref_U", "1.0",
        //                  dealii::Patterns::Double(0.0, 100000.0),
        //                  "The reference flow speed. Default value is 1.0. ");
        prm.declare_entry("ref_density", "1.204",
                          dealii::Patterns::Double(0.0, 100000.0),
                          "The reference fluid density. Default value is 1.204. ");
        //prm.declare_entry("ref_viscosity", "1.825e-5",
        //                  dealii::Patterns::Double(0.0, 100000.0),
        //                  "The reference viscosity. Default value is 1.825e-5. ");

        prm.declare_entry("chord_length", "1.0",
                          dealii::Patterns::Double(0.0, 100000.0),
                          "The chord length of airfoil. Default value is 1.0. ");
        prm.declare_entry("span_length", "1.0",
                          dealii::Patterns::Double(0.0, 100000.0),
                          "The span length of airfoil. Default value is 1.0. ");
        prm.declare_entry("alpha", "1.43",
                          dealii::Patterns::Double(0.0, 100.0),
                          "The Ratio of free-stream and convection speed of turbulence. Default value is 1.43. ");

        prm.declare_entry("R_specific", "287.05",
                          dealii::Patterns::Double(0.0, 100000.0),
                          "The specific gas constant R. Default value is 287.05. ");
    }
    prm.leave_subsection();
}

void AmietParam ::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("amiet_model");
    {
        const std::string acoustic_contribution_type_string = prm.get("acoustic_contribution_type");
        if (acoustic_contribution_type_string == "main") acoustic_contribution_type = AcousticContributionEnum::main;
        if (acoustic_contribution_type_string == "back") acoustic_contribution_type = AcousticContributionEnum::back;
        if (acoustic_contribution_type_string == "main_and_back") acoustic_contribution_type = AcousticContributionEnum::main_and_back;

        const std::string wall_pressure_spectral_model_type_string = prm.get("wall_pressure_spectral_model_type");
        if (wall_pressure_spectral_model_type_string == "Goody") wall_pressure_spectral_model_type = WallPressureSpectralModelEnum::Goody;
        if (wall_pressure_spectral_model_type_string == "Rozenberg") wall_pressure_spectral_model_type = WallPressureSpectralModelEnum::Rozenberg;
        if (wall_pressure_spectral_model_type_string == "Kamruzzaman") wall_pressure_spectral_model_type = WallPressureSpectralModelEnum::Kamruzzaman;

        omega_min = prm.get_double("omega_min");
        omega_max = prm.get_double("omega_max");
        omega_interval = prm.get_double("omega_interval");

        observer_coord_ref_x = prm.get_double("observer_coord_ref_x");
        observer_coord_ref_y = prm.get_double("observer_coord_ref_y");
        observer_coord_ref_z = prm.get_double("observer_coord_ref_z");

        //ref_U = prm.get_double("ref_U");
        ref_density = prm.get_double("ref_density");
        //ref_viscosity = prm.get_double("ref_viscosity");

        chord_length = prm.get_double("chord_length");
        span_length = prm.get_double("span_length");
        alpha = prm.get_double("alpha");
        R_specific = prm.get_double("R_specific");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace