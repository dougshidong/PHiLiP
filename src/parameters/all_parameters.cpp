#include "parameters/all_parameters.h"
namespace Parameters
{
    using namespace dealii;
    AllParameters::AllParameters ()
    :
    manufactured_convergence_study_param(ManufacturedConvergenceStudyParam()),
    ode_solver_param(ODESolverParam()),
    linear_solver_param(LinearSolverParam())
    {
    }
    void AllParameters::declare_parameters (ParameterHandler &prm)
    {
        std::cout << "Declaring inputs." << std::endl;
        prm.declare_entry("dimension", "1",
                          Patterns::Integer(),
                          "Number of dimensions");
        prm.declare_entry("pde_type", "advection",
                          Patterns::Selection(
                              " advection | "
                              " diffusion | "
                              " convection_diffusion | "
                              " advection_vector"),
                          "The PDE we want to solve. "
                          "Choices are " 
                          " <advection | " 
                          "  diffusion | "
                          "  convection_diffusion | "
                          "  advection_vector>.");
        prm.declare_entry("conv_num_flux", "lax_friedrichs",
                          Patterns::Selection("lax_friedrichs"),
                          "Convective numerical flux. "
                          "Choices are <lax_friedrichs>.");
        prm.declare_entry("diss_num_flux", "symm_internal_penalty",
                          Patterns::Selection("symm_internal_penalty"),
                          "Dissipative numerical flux. "
                          "Choices are <symm_internal_penalty>.");

        Parameters::LinearSolverParam::declare_parameters (prm);
        Parameters::ManufacturedConvergenceStudyParam::declare_parameters (prm);
        Parameters::ODESolverParam::declare_parameters (prm);

        std::cout << "Done declaring inputs." << std::endl;
    }
    void AllParameters::parse_parameters (ParameterHandler &prm)
    {
        std::cout << "Parsing main input..." << std::endl;

        dimension                   = prm.get_integer("dimension");

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
        }

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
}
