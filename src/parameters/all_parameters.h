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

#include "parameters/parameters_reduced_order.h"
#include "parameters/parameters_burgers.h"
#include "parameters/parameters_grid_refinement_study.h"
#include "parameters/parameters_grid_refinement.h"
#include "parameters/parameters_artificial_dissipation.h"
#include "parameters/parameters_flow_solver.h"
#include "parameters/parameters_mesh_adaptation.h"
#include "parameters/parameters_functional.h"

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

    /// Number of dimensions. Note that it has to match the executable PHiLiP_xD
    unsigned int dimension;

    /// Mesh type to be used in defining the triangulation
    enum MeshType {
        default_triangulation,
        triangulation,
        parallel_shared_triangulation,
        parallel_distributed_triangulation,
        };
    MeshType mesh_type; ///< Selected MeshType from the input file

    /// Number of additional quadrature points to use.
    /** overintegration = 0 leads to number_quad_points = dg_solution_degree + 1
     */
    int overintegration;

    /// Flag to use weak or strong form of DG
    bool use_weak_form;

    /// Flag to use Gauss-Lobatto Nodes;
    bool use_collocated_nodes;

    /// Flag to use split form.
    bool use_split_form;

    /// Flag to use curvilinear metric split form.
    bool use_curvilinear_split_form;

    /// Flag to use weight-adjusted Mass Matrix for curvilinear elements.
    bool use_weight_adjusted_mass;

    /// Flag to use periodic BC.
    /** Not fully tested.
     */
    bool use_periodic_bc;

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

    /// Number of state variables. Will depend on PDE
    int nstate;

    /// Currently allows to solve advection, diffusion, convection-diffusion
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
        burgers_split_form,
        optimization_inverse_manufactured,
        euler_bump_optimization,
        euler_naca_optimization,
        shock_1d,
        euler_naca0012,
        reduced_order,
        convection_diffusion_periodicity,
        POD_adaptation,
        finite_difference_sensitivity,
        advection_periodicity,
        flow_solver,
        dual_weighted_residual_mesh_adaptation,
        taylor_green_vortex_energy_check,
        taylor_green_vortex_restart_check,
    };
    TestType test_type; ///< Selected TestType from the input file.

    /// Currently allows to solve advection, diffusion, convection-diffusion
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
    };

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

    /// Store the PDE type to be solved
    PartialDifferentialEquation pde_type;

    /// Currently only Lax-Friedrichs, roe, and split_form can be used as an input parameter
    enum ConvectiveNumericalFlux { 
        lax_friedrichs, 
        roe, 
        l2roe, 
        split_form, 
        central_flux,
        entropy_cons_flux};

    /// Store convective flux type
    ConvectiveNumericalFlux conv_num_flux_type;

    /// Currently only symmetric internal penalty can be used as an input parameter
    enum DissipativeNumericalFlux { symm_internal_penalty, bassi_rebay_2 };
    /// Store diffusive flux type
    DissipativeNumericalFlux diss_num_flux_type;

    /// Type of correction in Flux Reconstruction
    enum Flux_Reconstruction {cDG, cSD, cHU, cNegative, cNegative2, cPlus, c10Thousand, cHULumped};

    /// Store flux reconstruction type
    Flux_Reconstruction flux_reconstruction_type;

    /// Type of correction in Flux Reconstruction for the auxiliary variables
    enum Flux_Reconstruction_Aux {kDG, kSD, kHU, kNegative, kNegative2, kPlus, k10Thousand};

    /// Store flux reconstruction type
    Flux_Reconstruction_Aux flux_reconstruction_aux_type;

    /// Name of directory for writing solution vtk files
    std::string solution_vtk_files_directory_name;

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


