#include "parameters/parameters_physics_model.h"

namespace PHiLiP {
namespace Parameters {
    
// Models inputs
PhysicsModelParam::PhysicsModelParam () {}

void PhysicsModelParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("physics_model");
    {
        prm.declare_entry("euler_turbulence","false",
                              dealii::Patterns::Bool(),
                              "Set as false by default (i.e. Navier-Stokes is the baseline physics). " 
                              "If true, sets the baseline physics to the Euler equations.");

        prm.enter_subsection("large_eddy_simulation");
        {
            prm.declare_entry("SGS_model_type", "smagorinsky",
                              dealii::Patterns::Selection(
                              " smagorinsky | "
                              " wall_adaptive_local_eddy_viscosity | "
                              " vreman | "
                              " shear_improved_smagorinsky | "
                              " dynamic_smagorinsky | "
                              " small_small_variational_multiscale | "
                              " all_all_variational_multiscale "),
                              "Enum of sub-grid scale models."
                              "Choices are "
                              " <smagorinsky | "
                              "  wall_adaptive_local_eddy_viscosity | "
                              "  vreman | "
                              "  shear_improved_smagorinsky | "
                              "  dynamic_smagorinsky | "
                              "  small_small_variational_multiscale | "
                              "  all_all_variational_multiscale>.");

            prm.declare_entry("turbulent_prandtl_number", "0.6",
                              dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                              "Turbulent Prandlt number (default is 0.6)");

            prm.declare_entry("smagorinsky_model_constant", "0.1",
                              dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                              "Smagorinsky model constant (default is 0.1)");

            prm.declare_entry("WALE_model_constant", "0.325",
                              dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                              "WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model constant (default is 0.325)");

            prm.declare_entry("vreman_model_constant", "0.025",
                              dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                              "Vreman eddy viscosity model constant (default is 0.025)");
            
            prm.declare_entry("ratio_of_filter_width_to_cell_size", "1.0",
                              dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                              "Ratio of the large eddy simulation filter width to the cell size (default is 1)");

            prm.declare_entry("do_compute_filtered_solution", "false",
                              dealii::Patterns::Bool(),
                              "Flag to compute the filtered solution. By default, false.");

            prm.declare_entry("apply_modal_high_pass_filter_on_filtered_solution", "false",
                              dealii::Patterns::Bool(),
                              "Flag to apply a modal high pass filter the filtered solution. By default, false.");

            prm.declare_entry("poly_degree_max_large_scales", "0",
                              dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                              "Used for variational-multiscale (VMS) filtering of the solution. "
                              "This represents the maximum polynomial order for the large scales. "
                              "Warning: This must be less than the poly_degree of the solution.");

            prm.declare_entry("dynamic_smagorinsky_model_constant_clipping_limit", "0.01",
                              dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                              "Clipping limit for the Dynamic Smagorinsky model constant (default is 0.01).");

        }
        prm.leave_subsection();

        prm.enter_subsection("reynolds_averaged_navier_stokes");
        {
            prm.declare_entry("euler_turbulence","false",
                              dealii::Patterns::Bool(),
                              "Set as false by default (i.e. Navier-Stokes is the baseline physics). " 
                              "If true, sets the baseline physics to the Euler equations.");

            prm.declare_entry("RANS_model_type", "SA_negative",
                              dealii::Patterns::Selection(
                              " SA_negative "),
                              "Enum of reynolds_averaged_navier_stokes models."
                              "Choices are "
                              " <SA_negative>");

            prm.declare_entry("turbulent_prandtl_number", "0.6",
                              dealii::Patterns::Double(1e-15, dealii::Patterns::Double::max_double_value),
                              "Turbulent Prandlt number (default is 0.6)");

        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

void PhysicsModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("physics_model");
    {
        euler_turbulence = prm.get_bool("euler_turbulence");

        prm.enter_subsection("large_eddy_simulation");
        {
            const std::string SGS_model_type_string = prm.get("SGS_model_type");
            if(SGS_model_type_string == "smagorinsky")                        SGS_model_type = smagorinsky;
            if(SGS_model_type_string == "wall_adaptive_local_eddy_viscosity") SGS_model_type = wall_adaptive_local_eddy_viscosity;
            if(SGS_model_type_string == "vreman")                             SGS_model_type = vreman;
            if(SGS_model_type_string == "shear_improved_smagorinsky")         SGS_model_type = shear_improved_smagorinsky;
            if(SGS_model_type_string == "dynamic_smagorinsky")                SGS_model_type = dynamic_smagorinsky;
            if(SGS_model_type_string == "small_small_variational_multiscale") SGS_model_type = small_small_variational_multiscale;
            if(SGS_model_type_string == "all_all_variational_multiscale")     SGS_model_type = all_all_variational_multiscale;

            turbulent_prandtl_number           = prm.get_double("turbulent_prandtl_number");
            smagorinsky_model_constant         = prm.get_double("smagorinsky_model_constant");
            WALE_model_constant                = prm.get_double("WALE_model_constant");
            vreman_model_constant              = prm.get_double("vreman_model_constant");
            ratio_of_filter_width_to_cell_size = prm.get_double("ratio_of_filter_width_to_cell_size");
            do_compute_filtered_solution       = prm.get_bool("do_compute_filtered_solution");
            apply_modal_high_pass_filter_on_filtered_solution 
                                               = prm.get_bool("apply_modal_high_pass_filter_on_filtered_solution");
            poly_degree_max_large_scales       = prm.get_integer("poly_degree_max_large_scales");
            dynamic_smagorinsky_model_constant_clipping_limit 
                                               = prm.get_double("dynamic_smagorinsky_model_constant_clipping_limit");
        }
        prm.leave_subsection();

        prm.enter_subsection("reynolds_averaged_navier_stokes");
        {
            euler_turbulence = prm.get_bool("euler_turbulence");

            const std::string RANS_model_type_string = prm.get("RANS_model_type");
            if(RANS_model_type_string == "SA_negative") RANS_model_type = SA_negative;

            turbulent_prandtl_number = prm.get_double("turbulent_prandtl_number");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
