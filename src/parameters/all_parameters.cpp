#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Parameters {

AllParameters::AllParameters ()
    : manufactured_convergence_study_param(ManufacturedConvergenceStudyParam())
    , ode_solver_param(ODESolverParam())
    , linear_solver_param(LinearSolverParam())
    , euler_param(EulerParam())
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{ }
void AllParameters::declare_parameters (dealii::ParameterHandler &prm)
{
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    pcout << "Declaring inputs." << std::endl;
    prm.declare_entry("dimension", "1",
                      dealii::Patterns::Integer(),
                      "Number of dimensions");

    prm.declare_entry("use_weak_form", "true",
                      dealii::Patterns::Bool(),
                      "Use weak form by default. If false, use strong form.");

    prm.declare_entry("use_collocated_nodes", "false",
                      dealii::Patterns::Bool(),
                      "Use Gauss-Legendre by default. Otherwise, use Gauss-Lobatto to collocate.");

    prm.declare_entry("use_split_form", "false",
                      dealii::Patterns::Bool(),
                      "Use original form by defualt. Otherwise, split the fluxes.");

    prm.declare_entry("use_periodic_bc", "false",
                      dealii::Patterns::Bool(),
                      "Use other boundary conditions by default. Otherwise use periodic (for 1d burgers only");

    prm.declare_entry("use_energy", "false",
                      dealii::Patterns::Bool(),
                      "Not calculate energy by default. Otherwise, get energy per iteration.");

    prm.declare_entry("use_L2_norm", "false",
                      dealii::Patterns::Bool(),
                      "Not calculate L2 norm by default (M+K). Otherwise, get L2 norm per iteration.");

    prm.declare_entry("use_classical_Flux_Reconstruction", "false",
                      dealii::Patterns::Bool(),
                      "Not use Classical Flux Reconstruction by default. Otherwise, use Classical Flux Reconstruction.");

    prm.declare_entry("use_skew_sym_deriv", "false",
                      dealii::Patterns::Bool(),
                      "Not use Skew Symmetric Derivative by default. Otherwise, use Skew Symmetric Derivative (Metric Split form).");

    prm.declare_entry("use_jac_sol_points", "false",
                      dealii::Patterns::Bool(),
                      "Jacobian at quadrature points default, otherwise soln points.");

    prm.declare_entry("use_projected_flux", "true",
                      dealii::Patterns::Bool(),
                      "Use projected nonlinear flux to p+1 by default. If false, use unprojected nonlinear flux.");

    prm.declare_entry("test_type", "run_control",
                      dealii::Patterns::Selection(
                      " run_control | "
                      " burgers_energy_stability | "
                      " euler_gaussian_bump | "
                      " euler_cylinder | "
                      " euler_vortex | "
                      " euler_entropy_waves | "
                      " numerical_flux_convervation | "
                      " jacobian_regression |"
                      " advection_periodicity |"
                      " convection_diffusion_periodicity |"
                      " euler_split_taylor_green"),
                      "The type of test we want to solve. "
                      "Choices are (only run control has been coded up for now)" 
                      " <run_control | " 
                      "  burgers_energy_stability | "
                      "  euler_gaussian_bump | "
                      "  euler_cylinder | "
                      "  euler_vortex | "
                      "  euler_entropy_waves | "
                      "  numerical_flux_convervation | "
                      "  jacobian_regression |"
		      "  euler_split_taylor_green |"
                      " convection_diffusion_periodicity |"
		      "  advection_periodicity >.");

    prm.declare_entry("pde_type", "advection",
                      dealii::Patterns::Selection(
                          " advection | "
                          " diffusion | "
                          " convection_diffusion | "
                          " advection_vector | "
                          " burgers_inviscid | "
                          " euler |"
                          " mhd"),
                      "The PDE we want to solve. "
                      "Choices are " 
                      " <advection | " 
                      "  diffusion | "
                      "  convection_diffusion | "
                      "  advection_vector | "
                      "  burgers_inviscid | "
                      "  euler | "
                      "  mhd>.");
    prm.declare_entry("conv_num_flux", "lax_friedrichs",
                      dealii::Patterns::Selection("lax_friedrichs | roe | split_form"),
                      "Convective numerical flux. "
                      "Choices are <lax_friedrichs | roe | split_form>.");

    prm.declare_entry("flux_reconstruction", "cDG",
                      dealii::Patterns::Selection("cDG | cSD | cHU | cNegative | cNegative2 | cPlus | c10Thousand"),
                      "Flux Reconstruction. "
                      "Choices are <cDG | cSD | cHU | cNegative | cNegative2 | cPlus | c10Thousand>.");

    prm.declare_entry("flux_reconstruction_aux", "kDG",
                      dealii::Patterns::Selection("kDG | kSD | kHU | kNegative | kNegative2 | kPlus | k10Thousand"),
                      "Flux Reconstruction for Auxiliary Equation. "
                      "Choices are <kDG | kSD | kHU | kNegative | kNegative2 | kPlus | k10Thousand>.");

    prm.declare_entry("diss_num_flux", "symm_internal_penalty",
                      dealii::Patterns::Selection("symm_internal_penalty | BR2"),
                      "Dissipative numerical flux. "
                      "Choices are <symm_internal_penalty | BR2>.");

    Parameters::LinearSolverParam::declare_parameters (prm);
    Parameters::ManufacturedConvergenceStudyParam::declare_parameters (prm);
    Parameters::ODESolverParam::declare_parameters (prm);

    Parameters::EulerParam::declare_parameters (prm);

    pcout << "Done declaring inputs." << std::endl;
}

void AllParameters::parse_parameters (dealii::ParameterHandler &prm)
{
    pcout << "Parsing main input..." << std::endl;

    dimension                   = prm.get_integer("dimension");

    const std::string test_string = prm.get("test_type");
    if (test_string == "run_control") { test_type = run_control; }
    else if (test_string == "burgers_energy_stability") { test_type = burgers_energy_stability; }
    else if (test_string == "euler_gaussian_bump") { test_type = euler_gaussian_bump; }
    else if (test_string == "euler_cylinder") { test_type = euler_cylinder; }
    else if (test_string == "euler_vortex") { test_type = euler_vortex; }
    else if (test_string == "euler_entropy_waves") { test_type = euler_entropy_waves; }
    else if (test_string == "numerical_flux_convervation") { test_type = numerical_flux_convervation; }
    else if (test_string == "jacobian_regression") { test_type = jacobian_regression; }
    else if (test_string == "advection_periodicity") {test_type = advection_periodicity; }
    else if (test_string == "convection_diffusion_periodicity") {test_type = convection_diffusion_periodicity; }
    else if (test_string == "euler_split_taylor_green") {test_type = euler_split_taylor_green;}

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
    use_collocated_nodes = prm.get_bool("use_collocated_nodes");
    use_split_form = prm.get_bool("use_split_form");
    use_periodic_bc = prm.get_bool("use_periodic_bc");
    use_energy = prm.get_bool("use_energy");
    use_L2_norm = prm.get_bool("use_L2_norm");
    use_classical_FR = prm.get_bool("use_classical_Flux_Reconstruction");
    use_skew_sym_deriv = prm.get_bool("use_skew_sym_deriv");
    use_jac_sol_points = prm.get_bool("use_jac_sol_points");
    use_projected_flux = prm.get_bool("use_projected_flux");

    const std::string conv_num_flux_string = prm.get("conv_num_flux");
    if (conv_num_flux_string == "lax_friedrichs") conv_num_flux_type = lax_friedrichs;
    if (conv_num_flux_string == "split_form") conv_num_flux_type = split_form;
    if (conv_num_flux_string == "roe") conv_num_flux_type = roe;

    const std::string flux_reconstruction_string = prm.get("flux_reconstruction");
    if (flux_reconstruction_string == "cDG") flux_reconstruction_type = cDG;
    if (flux_reconstruction_string == "cSD") flux_reconstruction_type = cSD;
    if (flux_reconstruction_string == "cHU") flux_reconstruction_type = cHU;
    if (flux_reconstruction_string == "cNegative") flux_reconstruction_type = cNegative;
    if (flux_reconstruction_string == "cNegative2") flux_reconstruction_type = cNegative2;
    if (flux_reconstruction_string == "cPlus") flux_reconstruction_type = cPlus;
    if (flux_reconstruction_string == "c10Thousand") flux_reconstruction_type = c10Thousand;

    const std::string flux_reconstruction_aux_string = prm.get("flux_reconstruction_aux");
    if (flux_reconstruction_aux_string == "kDG") flux_reconstruction_aux_type = kDG;
    if (flux_reconstruction_aux_string == "kSD") flux_reconstruction_aux_type = kSD;
    if (flux_reconstruction_aux_string == "kHU") flux_reconstruction_aux_type = kHU;
    if (flux_reconstruction_aux_string == "kNegative") flux_reconstruction_aux_type = kNegative;
    if (flux_reconstruction_aux_string == "kNegative2") flux_reconstruction_aux_type = kNegative2;
    if (flux_reconstruction_aux_string == "kPlus") flux_reconstruction_aux_type = kPlus;
    if (flux_reconstruction_aux_string == "k10Thousand") flux_reconstruction_aux_type = k10Thousand;

    const std::string diss_num_flux_string = prm.get("diss_num_flux");
    if (diss_num_flux_string == "symm_internal_penalty") diss_num_flux_type = symm_internal_penalty;
    if (diss_num_flux_string == "BR2") diss_num_flux_type = BR2;


    pcout << "Parsing linear solver subsection..." << std::endl;
    linear_solver_param.parse_parameters (prm);

    pcout << "Parsing ODE solver subsection..." << std::endl;
    ode_solver_param.parse_parameters (prm);

    pcout << "Parsing manufactured convergence study subsection..." << std::endl;
    manufactured_convergence_study_param.parse_parameters (prm);

    pcout << "Parsing euler subsection..." << std::endl;
    euler_param.parse_parameters (prm);

    pcout << "Done parsing." << std::endl;
}

} // Parameters namespace
} // PHiLiP namespace
