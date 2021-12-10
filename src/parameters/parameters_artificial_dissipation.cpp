#include "parameters/parameters_artificial_dissipation.h"

namespace PHiLiP {
namespace Parameters {

// Artificial Dissipation inputs
ArtificialDissipationParam::ArtificialDissipationParam () {}

void ArtificialDissipationParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("artificial dissipation");
    {
 
    prm.declare_entry("add_artificial_dissipation", "false",
                      dealii::Patterns::Bool(),
                      "Persson's subscell shock capturing artificial dissipation.");
    
    prm.declare_entry("artificial_dissipation_type", "laplacian",
                      dealii::Patterns::Selection(
                      "laplacian |"
                      "physical |"
                      "enthalpy_conserving_laplacian |"),
                      "Type of artificial dissipation we want to implement. Choices are laplacian, physical and enthalpy_conserving_laplacian");
    
    prm.declare_entry("artificial_dissipation_test_type", "poly_order_convergence",
                      dealii::Patterns::Selection(
                      "residual_convergence |"
                      "discontinuity_sensor_activation |"
                      "enthalpy_conservation |"
                      "poly_order_convergence |"),
                      "Type of artificial dissipation test type we want to implement. Choices are residual_convergence, discontinuity_sensor_activation, poly_order_convergence");
    
    prm.declare_entry("mu_artificial_dissipation", "1.0",
                      dealii::Patterns::Double(-1e20,1e20),
                      "Mu (viscosity) from Persson's subcell shock capturing.");
                      
    prm.declare_entry("kappa_artificial_dissipation", "1.0",
                      dealii::Patterns::Double(-1e20,1e20),
                      "Kappa from Persson's subcell shock capturing");

    prm.declare_entry("use_enthalpy_error", "false",
                      dealii::Patterns::Bool(),
                      "By default we calculate the entropy error from the conservative variables. Otherwise, compute the enthalpy error. An example is in Euler Gaussian bump.");

    }
    prm.leave_subsection();
}

void ArtificialDissipationParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("artificial dissipation");
    {

        add_artificial_dissipation = prm.get_bool("add_artificial_dissipation");
        use_enthalpy_error = prm.get_bool("use_enthalpy_error");

        const std::string artificial_dissipation_string = prm.get("artificial_dissipation_type");
        if (artificial_dissipation_string == "laplacian")
        { 
            artificial_dissipation_type = laplacian;
        }
        else if (artificial_dissipation_string == "physical")
        { 
            artificial_dissipation_type = physical;
        }
        else if (artificial_dissipation_string == "enthalpy_conserving_laplacian")
        { 
            artificial_dissipation_type = enthalpy_conserving_laplacian;
        }

        const std::string artificial_dissipation_test_string = prm.get("artificial_dissipation_test_type");
        if (artificial_dissipation_test_string == "residual_convergence")
        { 
            artificial_dissipation_test_type = residual_convergence;
        }
        else if (artificial_dissipation_test_string == "discontinuity_sensor_activation")
        {
            artificial_dissipation_test_type = discontinuity_sensor_activation;
        } 
        else if (artificial_dissipation_test_string == "enthalpy_conservation")
        {
            artificial_dissipation_test_type = enthalpy_conservation;
        } 
        else if (artificial_dissipation_test_string == "poly_order_convergence")
        {
            artificial_dissipation_test_type = poly_order_convergence;
        }

        mu_artificial_dissipation = prm.get_double("mu_artificial_dissipation");
        kappa_artificial_dissipation = prm.get_double("kappa_artificial_dissipation");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
