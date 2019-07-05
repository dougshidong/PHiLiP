#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Parameters {

AllParameters::AllParameters ()
    :
    manufactured_convergence_study_param(ManufacturedConvergenceStudyParam()),
    ode_solver_param(ODESolverParam()),
    linear_solver_param(LinearSolverParam())
{ }
void AllParameters::declare_parameters (dealii::ParameterHandler &prm)
{
    std::cout << "Declaring inputs." << std::endl;
    prm.declare_entry("dimension", "1",
                      dealii::Patterns::Integer(),
                      "Number of dimensions");

    prm.declare_entry("use_weak_form", "true",
                      dealii::Patterns::Bool(),
                      "Use weak form by default. If false, use strong form.");
    prm.declare_entry("test_type", "run_control",
                      dealii::Patterns::Selection(
                      " run_control | "
                      " euler_gaussian_bump | "
                      " numerical_flux_convervation | "
                      " jacobian_regression "),
                      "The type of test we want to solve. "
                      "Choices are (only run control has been coded up for now)" 
                      " <run_control | " 
                      "  euler_gaussian_bump | "
                      "  numerical_flux_convervation | "
                      "  jacobian_regression>.");

    prm.declare_entry("pde_type", "advection",
                      dealii::Patterns::Selection(
                          " advection | "
                          " diffusion | "
                          " convection_diffusion | "
                          " advection_vector | "
                          " burgers_inviscid | "
                          " euler"),
                      "The PDE we want to solve. "
                      "Choices are " 
                      " <advection | " 
                      "  diffusion | "
                      "  convection_diffusion | "
                      "  advection_vector | "
                      "  burgers_inviscid | "
                      "  euler>.");
    prm.declare_entry("conv_num_flux", "lax_friedrichs",
                      dealii::Patterns::Selection("lax_friedrichs"),
                      "Convective numerical flux. "
                      "Choices are <lax_friedrichs>.");
    prm.declare_entry("diss_num_flux", "symm_internal_penalty",
                      dealii::Patterns::Selection("symm_internal_penalty"),
                      "Dissipative numerical flux. "
                      "Choices are <symm_internal_penalty>.");

    Parameters::LinearSolverParam::declare_parameters (prm);
    Parameters::ManufacturedConvergenceStudyParam::declare_parameters (prm);
    Parameters::ODESolverParam::declare_parameters (prm);

    std::cout << "Done declaring inputs." << std::endl;
}

void AllParameters::parse_parameters (dealii::ParameterHandler &prm)
{
    std::cout << "Parsing main input..." << std::endl;

    dimension                   = prm.get_integer("dimension");

    const std::string test_string = prm.get("test_type");
    if (test_string == "run_control") { test_type = run_control; }
    else if (test_string == "euler_gaussian_bump") { test_type = euler_gaussian_bump; }
    else if (test_string == "numerical_flux_convervation") { test_type = numerical_flux_convervation; }
    else if (test_string == "jacobian_regression") { test_type = jacobian_regression; }

    const std::string pde_string = prm.get("pde_type");
    if (pde_string == "advection") {
        pde_type = advection;
        nstate = 1;
    } else if (pde_string == "advection_vector") {
        pde_type = advection_vector;
        nstate = 2;
    } else if (pde_string == "diffusion") {
        pde_type = diffusion;
        nstate = 1;
    } else if (pde_string == "convection_diffusion") {
        pde_type = convection_diffusion;
        nstate = 1;
    } else if (pde_string == "burgers_inviscid") {
        pde_type = burgers_inviscid;
        nstate = dimension;
    } else if (pde_string == "euler") {
        pde_type = euler;
        nstate = dimension+2;
    }
    use_weak_form = prm.get_bool("use_weak_form");

    const std::string conv_num_flux_string = prm.get("conv_num_flux");
    if (conv_num_flux_string == "lax_friedrichs") conv_num_flux_type = lax_friedrichs;

    const std::string diss_num_flux_string = prm.get("diss_num_flux");
    if (diss_num_flux_string == "symm_internal_penalty") diss_num_flux_type = symm_internal_penalty;


    std::cout << "Parsing linear solver subsection..." << std::endl;
    linear_solver_param.parse_parameters (prm);

    std::cout << "Parsing ODE solver subsection..." << std::endl;
    ode_solver_param.parse_parameters (prm);

    std::cout << "Parsing manufactured convergence study subsection..." << std::endl;
    manufactured_convergence_study_param.parse_parameters (prm);

    std::cout << "Done parsing." << std::endl;
}

} // Parameters namespace
} // PHiLiP namespace
