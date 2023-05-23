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
    , navier_stokes_param(NavierStokesParam())
    , reduced_order_param(ReducedOrderModelParam())
    , burgers_param(BurgersParam())
    , physics_model_param(PhysicsModelParam())
    , grid_refinement_study_param(GridRefinementStudyParam())
    , artificial_dissipation_param(ArtificialDissipationParam())
    , flow_solver_param(FlowSolverParam())
    , mesh_adaptation_param(MeshAdaptationParam())
    , functional_param(FunctionalParam())
    , time_refinement_study_param(TimeRefinementStudyParam())
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{ }

void AllParameters::declare_parameters (dealii::ParameterHandler &prm)
{
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    pcout << "Declaring inputs." << std::endl;
    prm.declare_entry("dimension", "-1",
                      dealii::Patterns::Integer(),
                      "Number of dimensions");

    prm.declare_entry("run_type", "integration_test",
                      dealii::Patterns::Selection(
                      " integration_test | "
                      " flow_simulation"),
                      "Type of run (default is integration_test). "
                      "Choices are  <integration_test | flow_simulation>.");

    prm.declare_entry("mesh_type", "default_triangulation",
                      dealii::Patterns::Selection(
                      " default_triangulation | "
                      " triangulation | "
                      " parallel_shared_triangulation | "
                      " parallel_distributed_triangulation"),
                      "Type of triangulation to be used."
                      "Note: parralel_distributed_triangulation not availible int 1D."
                      " <default_triangulation | "
                      "  triangulation | "
                      "  parallel_shared_triangulation |"
                      "  parallel_distributed_triangulation>.");
                      
    prm.declare_entry("overintegration", "0",
                      dealii::Patterns::Integer(),
                      "Number of extra quadrature points to use."
                      "If overintegration=0, then we use n_quad = soln_degree + 1.");

    prm.declare_entry("use_weak_form", "true",
                      dealii::Patterns::Bool(),
                      "Use weak form by default. If false, use strong form.");

    prm.declare_entry("flux_nodes_type", "GL",
                      dealii::Patterns::Selection(
                      "GL | GLL"),
                      "Flux nodes type, default is GL for uncollocated. NOTE: Solution nodes are type GLL."
                      "Choices are <GL | GLL>.");

    prm.declare_entry("use_split_form", "false",
                      dealii::Patterns::Bool(),
                      "Use original form by defualt. Otherwise, split the fluxes.");

    prm.declare_entry("two_point_num_flux_type", "KG",
                      dealii::Patterns::Selection(
                      "KG | IR | CH | Ra"),
                      "Two point flux type. "
                      "Choices are <KG | IR | CH | Ra>.");

    prm.declare_entry("use_curvilinear_split_form", "false",
                      dealii::Patterns::Bool(),
                      "Use original form by defualt. Otherwise, split the curvilinear fluxes.");

    prm.declare_entry("store_residual_cpu_time", "false",
                      dealii::Patterns::Bool(),
                      "Do not store the residual local processor cpu time by default. Store the residual cpu time if true.");

    prm.declare_entry("use_weight_adjusted_mass", "false",
                      dealii::Patterns::Bool(),
                      "Use original form by defualt. Otherwise, use the weight adjusted low storage mass matrix for curvilinear.");

    prm.declare_entry("use_periodic_bc", "false",
                      dealii::Patterns::Bool(),
                      "Use other boundary conditions by default. Otherwise use periodic (for 1d burgers only");

    prm.declare_entry("use_curvilinear_grid", "false",
                      dealii::Patterns::Bool(),
                      "Use straight grid by default. Curvilinear is true. Only used in taylor_green_scaling test.");

    prm.declare_entry("use_energy", "false",
                      dealii::Patterns::Bool(),
                      "Not calculate energy by default. Otherwise, get energy per iteration.");

    prm.declare_entry("use_L2_norm", "false",
                      dealii::Patterns::Bool(),
                      "Not calculate L2 norm by default (M+K). Otherwise, get L2 norm per iteration.");

    prm.declare_entry("use_classical_FR", "false",
                      dealii::Patterns::Bool(),
                      "Not use Classical Flux Reconstruction by default. Otherwise, use Classical Flux Reconstruction.");

    prm.declare_entry("flux_reconstruction", "cDG",
                      dealii::Patterns::Selection(
                      "cDG | cSD | cHU | cNegative | cNegative2 | cPlus | c10Thousand | cHULumped"),
                      "Flux Reconstruction. "
                      "Choices are "
                      " <cDG | cSD | cHU | cNegative | cNegative2 | cPlus | c10Thousand | cHULumped>.");

    prm.declare_entry("flux_reconstruction_aux", "kDG",
                      dealii::Patterns::Selection(
                      "kDG | kSD | kHU | kNegative | kNegative2 | kPlus | k10Thousand"),
                      "Flux Reconstruction for Auxiliary Equation. "
                      "Choices are <kDG | kSD | kHU | kNegative | kNegative2 | kPlus | k10Thousand>.");

    prm.declare_entry("sipg_penalty_factor", "1.0",
                      dealii::Patterns::Double(1.0,1e200),
                      "Scaling of Symmetric Interior Penalty term to ensure coercivity.");

    prm.declare_entry("use_invariant_curl_form", "false",
                      dealii::Patterns::Bool(),
                      "Use conservative curl form for metric cofactor by default. If true, then use invariant curl form.");

    prm.declare_entry("use_inverse_mass_on_the_fly", "false",
                      dealii::Patterns::Bool(),
                      "Build global mass inverse matrix and apply it. Otherwise, use inverse mass on-the-fly by default for explicit timestepping.");

    prm.declare_entry("check_valid_metric_Jacobian", "true",
                      dealii::Patterns::Bool(),
                      "Check validty of metric Jacobian when high-order grid is constructed by default. Do not check if false. Not checking is useful if the metric terms are built on the fly with operators, it reduces the memory cost for high polynomial grids. The metric Jacobian is never checked for strong form, regardless of the user input.");

    prm.declare_entry("energy_file", "energy_file",
                      dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                      "Input file for energy test.");

    prm.declare_entry("test_type", "run_control",
                      dealii::Patterns::Selection(
                      " run_control | "
                      " grid_refinement_study | "
                      " burgers_energy_stability | "
                      " diffusion_exact_adjoint | "
                      " optimization_inverse_manufactured | "
                      " euler_gaussian_bump | "
                      " euler_gaussian_bump_enthalpy | "
                      " euler_gaussian_bump_adjoint | "
                      " euler_cylinder | "
                      " euler_cylinder_adjoint | "
                      " euler_vortex | "
                      " euler_entropy_waves | "
                      " euler_split_taylor_green | "
                      " taylor_green_scaling | "
                      " euler_bump_optimization | "
                      " euler_naca_optimization | "
                      " shock_1d | "
                      " euler_naca0012 | "
                      " reduced_order | "
                      " convection_diffusion_periodicity |"
                      " POD_adaptation | "
                      " POD_adaptive_sampling | "
                      " adaptive_sampling_testing | "
                      " finite_difference_sensitivity | "
                      " advection_periodicity | "
                      " dual_weighted_residual_mesh_adaptation | "
                      " anisotropic_mesh_adaptation | "
                      " taylor_green_vortex_energy_check | "
                      " taylor_green_vortex_restart_check | "
                      " homogeneous_isotropic_turbulence_initialization_check | "
                      " time_refinement_study | "
                      " time_refinement_study_reference | "
                      " burgers_energy_conservation_rrk | "
                      " euler_entropy_conserving_split_forms_check | "
                      " h_refinement_study_isentropic_vortex | "
                      " khi_robustness"),
                      "The type of test we want to solve. "
                      "Choices are " 
                      " <run_control | " 
                      "  grid_refinement_study | "
                      "  burgers_energy_stability | "
                      "  diffusion_exact_adjoint | "
                      "  optimization_inverse_manufactured | "
                      "  euler_gaussian_bump | "
                      "  euler_gaussian_bump_enthalpy | "
                      "  euler_gaussian_bump_adjoint | "
                      "  euler_cylinder | "
                      "  euler_cylinder_adjoint | "
                      "  euler_vortex | "
                      "  euler_entropy_waves | "
                      "  euler_split_taylor_green |"
                      " taylor_green_scaling | "
                      "  euler_bump_optimization | "
                      "  euler_naca_optimization | "
                      "  shock_1d | "
                      "  euler_naca0012 | "
                      "  convection_diffusion_periodicity |"
                      "  reduced_order | "
                      "  POD_adaptation | "
                      "  POD_adaptive_sampling | "
                      "  adaptive_sampling_testing | "
                      "  finite_difference_sensitivity | "
                      "  advection_periodicity | "
                      "  dual_weighted_residual_mesh_adaptation | "
                      "  anisotropic_mesh_adaptation | "
                      "  taylor_green_vortex_energy_check | "
                      "  taylor_green_vortex_restart_check | "
                      "  homogeneous_isotropic_turbulence_initialization_check | "
                      "  time_refinement_study | "
                      "  time_refinement_study_reference | "
                      "  burgers_energy_conservation_rrk | "
                      "  euler_entropy_conserving_split_forms_check | "
                      "  h_refinement_study_isentropic_vortex | "
                      "  khi_robustness>.");

    prm.declare_entry("pde_type", "advection",
                      dealii::Patterns::Selection(
                      " advection | "
                      " diffusion | "
                      " convection_diffusion | "
                      " advection_vector | "
                      " burgers_inviscid | "
                      " burgers_viscous | "
                      " burgers_rewienski | "
                      " euler |"
                      " mhd |"
                      " navier_stokes |"
                      " physics_model"),
                      "The PDE we want to solve. "
                      "Choices are " 
                      " <advection | " 
                      "  diffusion | "
                      "  convection_diffusion | "
                      "  advection_vector | "
                      "  burgers_inviscid | "
                      "  burgers_viscous | "
                      "  burgers_rewienski | "
                      "  euler | "
                      "  mhd |"
                      "  navier_stokes |"
                      "  physics_model>.");

    prm.declare_entry("model_type", "large_eddy_simulation",
                      dealii::Patterns::Selection(
                      "large_eddy_simulation | reynolds_averaged_navier_stokes"),
                      "Enum of physics models "
                      "(i.e. model equations and/or terms additional to Navier-Stokes or a chosen underlying baseline physics)."
                      "Choices are "
                      " <large_eddy_simulation | reynolds_averaged_navier_stokes>.");

    prm.declare_entry("conv_num_flux", "lax_friedrichs",
                      dealii::Patterns::Selection(
                      " lax_friedrichs | "
                      " roe | "
                      " l2roe | "
                      " central_flux | "
                      " two_point_flux | "
                      " two_point_flux_with_lax_friedrichs_dissipation | "
                      " two_point_flux_with_roe_dissipation | "
                      " two_point_flux_with_l2roe_dissipation"),
                      "Convective numerical flux. "
                      "Choices are "
                      " <lax_friedrichs | "
                      " roe | "
                      " l2roe | "
                      " central_flux | "
                      " two_point_flux | "
                      " two_point_flux_with_lax_friedrichs_dissipation | "
                      " two_point_flux_with_roe_dissipation | "
                      " two_point_flux_with_l2roe_dissipation>.");

    prm.declare_entry("diss_num_flux", "symm_internal_penalty",
                      dealii::Patterns::Selection("symm_internal_penalty | bassi_rebay_2 | central_visc_flux"),
                      "Dissipative numerical flux. "
                      "Choices are <symm_internal_penalty | bassi_rebay_2 | central_visc_flux>.");

    prm.declare_entry("solution_vtk_files_directory_name", ".",
                      dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                      "Name of directory for writing solution vtk files. Current directory by default.");

    prm.declare_entry("output_high_order_grid", "false",
                      dealii::Patterns::Bool(),
                      "Outputs the high-order mesh vtu files. False by default");

    prm.declare_entry("enable_higher_order_vtk_output", "true",
                      dealii::Patterns::Bool(),
                      "Enable writing of higher-order vtk files. True by default; "
                      "number of subdivisions is chosen according to the max of grid_degree and poly_degree.");

    prm.declare_entry("output_face_results_vtk", "false",
                      dealii::Patterns::Bool(),
                      "Outputs the surface solution vtk files. False by default");

    prm.declare_entry("do_renumber_dofs", "true",
                      dealii::Patterns::Bool(),
                      "Flag for renumbering DOFs using Cuthill-McKee renumbering. True by default. Set to false if doing 3D unsteady flow simulations.");

    prm.declare_entry("renumber_dofs_type", "CuthillMckee",
                      dealii::Patterns::Selection(
                      "CuthillMckee"),
                      "Renumber the dof handler type. Currently the only choice is Cuthill-Mckee.");

    Parameters::LinearSolverParam::declare_parameters (prm);
    Parameters::ManufacturedConvergenceStudyParam::declare_parameters (prm);
    Parameters::ODESolverParam::declare_parameters (prm);
    Parameters::EulerParam::declare_parameters (prm);
    Parameters::NavierStokesParam::declare_parameters (prm);
    Parameters::PhysicsModelParam::declare_parameters (prm);
    Parameters::ReducedOrderModelParam::declare_parameters (prm);
    Parameters::BurgersParam::declare_parameters (prm);
    Parameters::GridRefinementStudyParam::declare_parameters (prm);
    Parameters::ArtificialDissipationParam::declare_parameters (prm);
    Parameters::MeshAdaptationParam::declare_parameters (prm);
    Parameters::FlowSolverParam::declare_parameters (prm);
    Parameters::FunctionalParam::declare_parameters (prm);
    Parameters::TimeRefinementStudyParam::declare_parameters (prm);

    pcout << "Done declaring inputs." << std::endl;
}

void AllParameters::parse_parameters (dealii::ParameterHandler &prm)
{
    pcout << "Parsing main input..." << std::endl;

    dimension = prm.get_integer("dimension");

    const std::string run_type_string = prm.get("run_type");
    if      (run_type_string == "integration_test") { run_type = integration_test; }
    else if (run_type_string == "flow_simulation")  { run_type = flow_simulation; }

    const std::string mesh_type_string = prm.get("mesh_type");
    if      (mesh_type_string == "default_triangulation")              { mesh_type = default_triangulation; }
    else if (mesh_type_string == "triangulation")                      { mesh_type = triangulation; }
    else if (mesh_type_string == "parallel_shared_triangulation")      { mesh_type = parallel_shared_triangulation; }
    else if (mesh_type_string == "parallel_distributed_triangulation") { mesh_type = parallel_distributed_triangulation; }

    const std::string test_string = prm.get("test_type");
    if      (test_string == "run_control")                              { test_type = run_control; }
    else if (test_string == "grid_refinement_study")                    { test_type = grid_refinement_study; }
    else if (test_string == "burgers_energy_stability")                 { test_type = burgers_energy_stability; }
    else if (test_string == "diffusion_exact_adjoint")                  { test_type = diffusion_exact_adjoint; }
    else if (test_string == "euler_gaussian_bump")                      { test_type = euler_gaussian_bump; }
    else if (test_string == "euler_gaussian_bump_enthalpy")             { test_type = euler_gaussian_bump_enthalpy; }
    else if (test_string == "euler_gaussian_bump_adjoint")              { test_type = euler_gaussian_bump_adjoint; }
    else if (test_string == "euler_cylinder")                           { test_type = euler_cylinder; }
    else if (test_string == "euler_cylinder_adjoint")                   { test_type = euler_cylinder_adjoint; }
    else if (test_string == "euler_vortex")                             { test_type = euler_vortex; }
    else if (test_string == "euler_entropy_waves")                      { test_type = euler_entropy_waves; }
    else if (test_string == "advection_periodicity")                    { test_type = advection_periodicity; }
    else if (test_string == "convection_diffusion_periodicity")         { test_type = convection_diffusion_periodicity; }
    else if (test_string == "euler_split_taylor_green")                 { test_type = euler_split_taylor_green; }
    else if (test_string == "taylor_green_scaling")                     { test_type = taylor_green_scaling; }
    else if (test_string == "euler_bump_optimization")                  { test_type = euler_bump_optimization; }
    else if (test_string == "euler_naca_optimization")                  { test_type = euler_naca_optimization; }
    else if (test_string == "shock_1d")                                 { test_type = shock_1d; }
    else if (test_string == "reduced_order")                            { test_type = reduced_order; }
    else if (test_string == "POD_adaptation")                           { test_type = POD_adaptation; }
    else if (test_string == "POD_adaptive_sampling")                    { test_type = POD_adaptive_sampling; }
    else if (test_string == "adaptive_sampling_testing")                { test_type = adaptive_sampling_testing; }
    else if (test_string == "finite_difference_sensitivity")            { test_type = finite_difference_sensitivity; }
    else if (test_string == "euler_naca0012")                           { test_type = euler_naca0012; }
    else if (test_string == "optimization_inverse_manufactured")        { test_type = optimization_inverse_manufactured; }
    else if (test_string == "dual_weighted_residual_mesh_adaptation")   { test_type = dual_weighted_residual_mesh_adaptation; }
    else if (test_string == "anisotropic_mesh_adaptation")              { test_type = anisotropic_mesh_adaptation; }
    else if (test_string == "taylor_green_vortex_energy_check")         { test_type = taylor_green_vortex_energy_check; }
    else if (test_string == "taylor_green_vortex_restart_check")        { test_type = taylor_green_vortex_restart_check; }
    else if (test_string == "homogeneous_isotropic_turbulence_initialization_check")
                                                                        { test_type = homogeneous_isotropic_turbulence_initialization_check; }
    else if (test_string == "time_refinement_study")                    { test_type = time_refinement_study; }
    else if (test_string == "time_refinement_study_reference")          { test_type = time_refinement_study_reference; }
    else if (test_string == "burgers_energy_conservation_rrk")          { test_type = burgers_energy_conservation_rrk; }
    else if (test_string == "euler_entropy_conserving_split_forms_check") 
                                                                        { test_type = euler_entropy_conserving_split_forms_check; }
    else if (test_string == "h_refinement_study_isentropic_vortex")     { test_type = h_refinement_study_isentropic_vortex; }
    else if (test_string == "khi_robustness")                           { test_type = khi_robustness; }
    
    overintegration = prm.get_integer("overintegration");

    use_weak_form = prm.get_bool("use_weak_form");
    
    const std::string flux_nodes_string = prm.get("flux_nodes_type");
    if (flux_nodes_string == "GL") { flux_nodes_type = FluxNodes::GL; }
    if (flux_nodes_string == "GLL") { flux_nodes_type = FluxNodes::GLL; }

    use_collocated_nodes = (flux_nodes_type==FluxNodes::GLL) && (overintegration==0);

    use_split_form = prm.get_bool("use_split_form");

    const std::string two_point_num_flux_string = prm.get("two_point_num_flux_type");
    if (two_point_num_flux_string == "KG") { two_point_num_flux_type = TwoPointNumericalFlux::KG; }
    if (two_point_num_flux_string == "IR") { two_point_num_flux_type = TwoPointNumericalFlux::IR; }
    if (two_point_num_flux_string == "CH") { two_point_num_flux_type = TwoPointNumericalFlux::CH; }
    if (two_point_num_flux_string == "Ra") { two_point_num_flux_type = TwoPointNumericalFlux::Ra; }

    use_curvilinear_split_form = prm.get_bool("use_curvilinear_split_form");
    use_curvilinear_grid = prm.get_bool("use_curvilinear_grid");
    store_residual_cpu_time = prm.get_bool("store_residual_cpu_time");
    use_weight_adjusted_mass = prm.get_bool("use_weight_adjusted_mass");
    use_periodic_bc = prm.get_bool("use_periodic_bc");
    use_energy = prm.get_bool("use_energy");
    use_L2_norm = prm.get_bool("use_L2_norm");
    use_classical_FR = prm.get_bool("use_classical_FR");
    sipg_penalty_factor = prm.get_double("sipg_penalty_factor");
    use_invariant_curl_form = prm.get_bool("use_invariant_curl_form");
    use_inverse_mass_on_the_fly = prm.get_bool("use_inverse_mass_on_the_fly");
    check_valid_metric_Jacobian = prm.get_bool("check_valid_metric_Jacobian");
    if(!use_weak_form){
        check_valid_metric_Jacobian = false;
    }

    energy_file = prm.get("energy_file");

    const std::string conv_num_flux_string = prm.get("conv_num_flux");
    if (conv_num_flux_string == "lax_friedrichs")                                          { conv_num_flux_type = ConvectiveNumericalFlux::lax_friedrichs; }
    if (conv_num_flux_string == "roe")                                                     { conv_num_flux_type = ConvectiveNumericalFlux::roe; }
    if (conv_num_flux_string == "l2roe")                                                   { conv_num_flux_type = ConvectiveNumericalFlux::l2roe; }
    if (conv_num_flux_string == "central_flux")                                            { conv_num_flux_type = ConvectiveNumericalFlux::central_flux; }
    if (conv_num_flux_string == "two_point_flux")                                 { conv_num_flux_type = ConvectiveNumericalFlux::two_point_flux; }
    if (conv_num_flux_string == "two_point_flux_with_lax_friedrichs_dissipation") { conv_num_flux_type = ConvectiveNumericalFlux::two_point_flux_with_lax_friedrichs_dissipation; }
    if (conv_num_flux_string == "two_point_flux_with_roe_dissipation")            { conv_num_flux_type = ConvectiveNumericalFlux::two_point_flux_with_roe_dissipation; }
    if (conv_num_flux_string == "two_point_flux_with_l2roe_dissipation")          { conv_num_flux_type = ConvectiveNumericalFlux::two_point_flux_with_l2roe_dissipation; }

    const std::string diss_num_flux_string = prm.get("diss_num_flux");
    if (diss_num_flux_string == "symm_internal_penalty") { diss_num_flux_type = symm_internal_penalty; }
    if (diss_num_flux_string == "bassi_rebay_2") {
        diss_num_flux_type = bassi_rebay_2;
        sipg_penalty_factor = 0.0;
    }
    if (diss_num_flux_string == "central_visc_flux") diss_num_flux_type = central_visc_flux;

    const std::string flux_reconstruction_string = prm.get("flux_reconstruction");
    if (flux_reconstruction_string == "cDG")         { flux_reconstruction_type = cDG; }
    if (flux_reconstruction_string == "cSD")         { flux_reconstruction_type = cSD; }
    if (flux_reconstruction_string == "cHU")         { flux_reconstruction_type = cHU; }
    if (flux_reconstruction_string == "cNegative")   { flux_reconstruction_type = cNegative; }
    if (flux_reconstruction_string == "cNegative2")  { flux_reconstruction_type = cNegative2; }
    if (flux_reconstruction_string == "cPlus")       { flux_reconstruction_type = cPlus; }
    if (flux_reconstruction_string == "c10Thousand") { flux_reconstruction_type = c10Thousand; }
    if (flux_reconstruction_string == "cHULumped")   { flux_reconstruction_type = cHULumped; }

    const std::string flux_reconstruction_aux_string = prm.get("flux_reconstruction_aux");
    if (flux_reconstruction_aux_string == "kDG")         { flux_reconstruction_aux_type = kDG; }
    if (flux_reconstruction_aux_string == "kSD")         { flux_reconstruction_aux_type = kSD; }
    if (flux_reconstruction_aux_string == "kHU")         { flux_reconstruction_aux_type = kHU; }
    if (flux_reconstruction_aux_string == "kNegative")   { flux_reconstruction_aux_type = kNegative; }
    if (flux_reconstruction_aux_string == "kNegative2")  { flux_reconstruction_aux_type = kNegative2; }
    if (flux_reconstruction_aux_string == "kPlus")       { flux_reconstruction_aux_type = kPlus; }
    if (flux_reconstruction_aux_string == "k10Thousand") { flux_reconstruction_aux_type = k10Thousand; }

    solution_vtk_files_directory_name = prm.get("solution_vtk_files_directory_name");
    output_high_order_grid = prm.get_bool("output_high_order_grid");
    enable_higher_order_vtk_output = prm.get_bool("enable_higher_order_vtk_output");
    output_face_results_vtk = prm.get_bool("output_face_results_vtk");
    do_renumber_dofs = prm.get_bool("do_renumber_dofs");

    const std::string renumber_dofs_type_string = prm.get("renumber_dofs_type");
    if (renumber_dofs_type_string == "CuthillMckee") { renumber_dofs_type = RenumberDofsType::CuthillMckee; }

    output_high_order_grid = prm.get_bool("output_high_order_grid");

    pcout << "Parsing linear solver subsection..." << std::endl;
    linear_solver_param.parse_parameters (prm);

    pcout << "Parsing ODE solver subsection..." << std::endl;
    ode_solver_param.parse_parameters (prm);

    pcout << "Parsing manufactured convergence study subsection..." << std::endl;
    manufactured_convergence_study_param.parse_parameters (prm);

    pcout << "Parsing euler subsection..." << std::endl;
    euler_param.parse_parameters (prm);

    pcout << "Parsing navier stokes subsection..." << std::endl;
    navier_stokes_param.parse_parameters (prm);

    pcout << "Parsing reduced order subsection..." << std::endl;
    reduced_order_param.parse_parameters (prm);

    pcout << "Parsing Burgers subsection..." << std::endl;
    burgers_param.parse_parameters (prm);

    pcout << "Parsing physics model subsection..." << std::endl;
    physics_model_param.parse_parameters (prm);

    pcout << "Parsing grid refinement study subsection..." << std::endl;
    grid_refinement_study_param.parse_parameters (prm);

    pcout << "Parsing artificial dissipation subsection..." << std::endl;
    artificial_dissipation_param.parse_parameters (prm);
    
    pcout << "Parsing flow solver subsection..." << std::endl;
    flow_solver_param.parse_parameters (prm);

    pcout << "Parsing mesh adaptation subsection..." << std::endl;
    mesh_adaptation_param.parse_parameters (prm);
    
    pcout << "Parsing functional subsection..." << std::endl;
    functional_param.parse_parameters (prm);

    // WARNING: Must assign model_type before pde_type
    const std::string model_string = prm.get("model_type");
    if (model_string == "large_eddy_simulation") { model_type = large_eddy_simulation; }
    else if (model_string == "reynolds_averaged_navier_stokes") { model_type = reynolds_averaged_navier_stokes; }

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
    } else if (pde_string == "burgers_viscous") {
        pde_type = burgers_viscous;
        nstate = dimension;
    } else if (pde_string == "burgers_rewienski") {
        pde_type = burgers_rewienski;
        nstate = dimension;
    } else if (pde_string == "euler") {
        pde_type = euler;
        nstate = dimension+2;
    }
    else if (pde_string == "navier_stokes") {
        pde_type = navier_stokes;
        nstate = dimension+2;
    }
    else if (pde_string == "physics_model") {
        pde_type = physics_model;
        if (model_type == large_eddy_simulation)
        {
            nstate = dimension+2;
        }
        else if (model_type == reynolds_averaged_navier_stokes)
        {
            if(physics_model_param.RANS_model_type == Parameters::PhysicsModelParam::ReynoldsAveragedNavierStokesModel::SA_negative)
              nstate = dimension+3;
        }
    }
    
    pcout << "Parsing time refinement study subsection..." << std::endl;
    time_refinement_study_param.parse_parameters (prm);
    
    pcout << "Done parsing." << std::endl;
}

} // Parameters namespace
} // PHiLiP namespace
