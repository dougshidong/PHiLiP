#ifndef __ALL_PARAMETERS_H__
#define __ALL_PARAMETERS_H__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include "parameters.h"
#include "parameters/parameters_ode_solver.h"
#include "parameters/parameters_linear_solver.h"
#include "parameters/parameters_manufactured_convergence_study.h"

#include "parameters/parameters_euler.h"
#include "parameters/parameters_navier_stokes.h"
#include "parameters/parameters_physics_model.h"

#include "parameters/parameters_reduced_order.h"
#include "parameters/parameters_burgers.h"
#include "parameters/parameters_grid_refinement_study.h"
#include "parameters/parameters_grid_refinement.h"
#include "parameters/parameters_artificial_dissipation.h"
#include "parameters/parameters_flow_solver.h"
#include "parameters/parameters_mesh_adaptation.h"
#include "parameters/parameters_functional.h"
#include "parameters/parameters_time_refinement_study.h"

namespace PHiLiP {
namespace Parameters {

/// Main parameter class that contains the various other sub-parameter classes.
class AllParameters
{
public:
    /// Constructor
    AllParameters();

    /// Contains parameters for manufactured convergence study
    ManufacturedConvergenceStudyParam manufactured_convergence_study_param;
    /// Contains parameters for ODE solver
    ODESolverParam ode_solver_param;
    /// Contains parameters for linear solver
    LinearSolverParam linear_solver_param;
    /// Contains parameters for the Euler equations non-dimensionalization
    EulerParam euler_param;
    /// Contains parameters for the Navier-Stokes equations non-dimensionalization
    NavierStokesParam navier_stokes_param;
    /// Contains parameters for the Reduced-Order model
    ReducedOrderModelParam reduced_order_param;
    /// Contains parameters for Burgers equation
    BurgersParam burgers_param;
    /// Contains parameters for Physics Model
    PhysicsModelParam physics_model_param;
    /// Contains the parameters for grid refinement study
    GridRefinementStudyParam grid_refinement_study_param;
    /// Contains parameters for artificial dissipation
    ArtificialDissipationParam artificial_dissipation_param;
    /// Contains the parameters for simulation cases (flow solver test)
    FlowSolverParam flow_solver_param;
    /// Constains parameters for mesh adaptation
    MeshAdaptationParam mesh_adaptation_param;
    /// Contains parameters for functional
    FunctionalParam functional_param;
    /// Contains the parameters for time refinement study
    TimeRefinementStudyParam time_refinement_study_param;

    /// Number of dimensions. Note that it has to match the executable PHiLiP_xD
    unsigned int dimension;

    /// Run type
    enum RunType {
        integration_test,
        flow_simulation
        };
    RunType run_type; ///< Selected RunType from the input file

    /// Mesh type to be used in defining the triangulation
    enum MeshType {
        default_triangulation,
        triangulation,
        parallel_shared_triangulation,
        parallel_distributed_triangulation,
        };
    /// Store selected MeshType from the input file
    MeshType mesh_type;
    
    /// Number of additional quadrature points to use.
    /** overintegration = 0 leads to number_quad_points = dg_solution_degree + 1
     */
    int overintegration;

    /// Flag to use weak or strong form of DG
    bool use_weak_form;

    /// Flux nodes type
    enum FluxNodes { GL, GLL };
    /// Store selected FluxNodes from the input file
    FluxNodes flux_nodes_type;

    /// Flag for using collocated nodes; determined based on flux_nodes_type and overintegration input parameters
    bool use_collocated_nodes;

    /// Flag to use split form.
    bool use_split_form;

    /// Two point numerical flux type for split form
    enum TwoPointNumericalFlux { KG, IR, CH, Ra };
    /// Store selected TwoPointNumericalFlux from the input file
    TwoPointNumericalFlux two_point_num_flux_type;

    /// Flag to use curvilinear metric split form.
    bool use_curvilinear_split_form;

    /// Flag to use weight-adjusted Mass Matrix for curvilinear elements.
    bool use_weight_adjusted_mass;

    /// Flag to store the residual local processor cput time.
    bool store_residual_cpu_time;

    /// Flag to use periodic BC.
    /** Not fully tested.
     */
    bool use_periodic_bc;

    /// Flag to use curvilinear grid.
    bool use_curvilinear_grid;

    ///Flag to use an energy monotonicity test
    bool use_energy;

    ///Flag to use an L2 energy monotonicity test (for FR)
    bool use_L2_norm;

    ///Flag to use a Classical ESFR scheme where only the surface is reconstructed
    //The default ESFR scheme is the Nonlinearly Stable FR where the volume is also reconstructed
    bool use_classical_FR;

    ///Flag to store global mass matrix
    bool store_global_mass_matrix;

    /// Scaling of Symmetric Interior Penalty term to ensure coercivity.
    double sipg_penalty_factor;

    /// Flag to use invariant curl form for metric cofactor operator.
    bool use_invariant_curl_form;

    /// Flag to use inverse mass matrix on-the-fly for explicit solves.
    bool use_inverse_mass_on_the_fly;

    /// Flag to check if the metric Jacobian is valid when high-order grid is constructed.
    bool check_valid_metric_Jacobian;

    /// Energy file.
    std::string energy_file;

    /// Number of state variables. Will depend on PDE
    int nstate;

    /// Possible integration tests to run
    enum TestType {
        run_control,
        grid_refinement_study,
        burgers_energy_stability,
        diffusion_exact_adjoint,
        euler_gaussian_bump,
        euler_gaussian_bump_enthalpy,
        euler_gaussian_bump_adjoint,
        euler_cylinder,
        euler_cylinder_adjoint,
        euler_vortex,
        euler_entropy_waves,
        euler_split_taylor_green,
        taylor_green_scaling,
        burgers_split_form,
        optimization_inverse_manufactured,
        euler_bump_optimization,
        euler_naca_optimization,
        shock_1d,
        euler_naca0012,
        reduced_order,
        convection_diffusion_periodicity,
        POD_adaptation,
        POD_adaptive_sampling,
        adaptive_sampling_testing,
        finite_difference_sensitivity,
        advection_periodicity,
        dual_weighted_residual_mesh_adaptation,
        anisotropic_mesh_adaptation,
        taylor_green_vortex_energy_check,
        taylor_green_vortex_restart_check,
        time_refinement_study,
        time_refinement_study_reference,
        burgers_energy_conservation_rrk,
        euler_entropy_conserving_split_forms_check,
        h_refinement_study_isentropic_vortex,
        khi_robustness,
        homogeneous_isotropic_turbulence_initialization_check,
        build_NNLS_problem,
        hyper_reduction_comparison,
        hyper_adaptive_sampling_test,
        hyper_reduction_post_sampling
    };
    /// Store selected TestType from the input file.
    TestType test_type;

    /// Possible Partial Differential Equations to solve
    enum PartialDifferentialEquation {
        advection,
        diffusion,
        convection_diffusion,
        advection_vector,
        burgers_inviscid,
        burgers_viscous,
        burgers_rewienski,
        euler,
        mhd,
        navier_stokes,
        physics_model,
    };
    /// Store the PDE type to be solved
    PartialDifferentialEquation pde_type;

    /// Types of models available.
    enum ModelType {
        large_eddy_simulation,
        reynolds_averaged_navier_stokes,
    };
    /// Store the model type
    ModelType model_type;

    /// Possible boundary types, NOT IMPLEMENTED YET
    enum BoundaryType {
        manufactured_dirichlet,
        manufactured_neumann,
        manufactured_inout_flow,
    };

    /// Possible source terms, NOT IMPLEMENTED YET
    enum SourceTerm {
        zero,
        manufactured,
    };

    /// Possible convective numerical flux types
    enum ConvectiveNumericalFlux { 
        lax_friedrichs, 
        roe, 
        l2roe, 
        central_flux,
        two_point_flux,
        two_point_flux_with_lax_friedrichs_dissipation,
        two_point_flux_with_roe_dissipation,
        two_point_flux_with_l2roe_dissipation
    };
    /// Store convective flux type
    ConvectiveNumericalFlux conv_num_flux_type;

    /// Possible dissipative numerical flux types
    enum DissipativeNumericalFlux { symm_internal_penalty, bassi_rebay_2, central_visc_flux };
    /// Store diffusive flux type
    DissipativeNumericalFlux diss_num_flux_type;

    /// Type of correction in Flux Reconstruction
    enum Flux_Reconstruction {cDG, cSD, cHU, cNegative, cNegative2, cPlus, c10Thousand, cHULumped};
    /// Store flux reconstruction type
    Flux_Reconstruction flux_reconstruction_type;

    /// Type of correction in Flux Reconstruction for the auxiliary variables
    enum Flux_Reconstruction_Aux {kDG, kSD, kHU, kNegative, kNegative2, kPlus, k10Thousand};
    /// Store flux reconstruction type for the auxiliary variables
    Flux_Reconstruction_Aux flux_reconstruction_aux_type;

    /// Enum of nonphysical behavior
    enum NonPhysicalBehaviorEnum {return_big_number, abort_run, print_warning};
    /// Specify behavior on nonphysical results
    NonPhysicalBehaviorEnum non_physical_behavior_type;

    /// Name of directory for writing solution vtk files
    std::string solution_vtk_files_directory_name;

    /// Flag for outputting the high-order grid vtk files
    bool output_high_order_grid;

    /// Enable writing of higher-order vtk results
    bool enable_higher_order_vtk_output;

    /// Flag for outputting the surface solution vtk files
    bool output_face_results_vtk;

    /// Flag for renumbering DOFs
    bool do_renumber_dofs;
    /// Renumber dofs type.
    enum RenumberDofsType { CuthillMckee };
    /// Store selected RenumberDofsType from the input file.
    RenumberDofsType renumber_dofs_type;
    
    /// Declare parameters that can be set as inputs and set up the default options
    /** This subroutine should call the sub-parameter classes static declare_parameters()
      * such that each sub-parameter class is responsible to declare their own parameters.
      */
    static void declare_parameters (dealii::ParameterHandler &prm);

    /// Retrieve parameters from dealii::ParameterHandler
    /** This subroutine should call the sub-parameter classes static parse_parameters()
      * such that each sub-parameter class is responsible to parse their own parameters.
      */
    void parse_parameters (dealii::ParameterHandler &prm);

    //FunctionParser<dim> initial_conditions;
    //BoundaryConditions  boundary_conditions[max_n_boundaries];
protected:
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};  

} // Parameters namespace
} // PHiLiP namespace

#endif

