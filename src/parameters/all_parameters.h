#ifndef __ALL_PARAMETERS_H__
#define __ALL_PARAMETERS_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters.h"
#include "parameters/parameters_ode_solver.h"
#include "parameters/parameters_linear_solver.h"
#include "parameters/parameters_manufactured_convergence_study.h"

#include "parameters/parameters_euler.h"

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

    /// Number of dimensions. Note that it has to match the executable PHiLiP_xD
    unsigned int dimension;

    /// Flag to use weak or strong form of DG
    bool use_weak_form;

    /// Number of state variables. Will depend on PDE
    int nstate;

    /// Currently allows to solve advection, diffusion, convection-diffusion
    enum TestType { 
        run_control,
        euler_gaussian_bump,
        euler_cylinder,
        euler_vortex,
        euler_entropy_waves,
        numerical_flux_convervation,
        jacobian_regression};
    TestType test_type;

    /// Currently allows to solve advection, diffusion, convection-diffusion
    enum PartialDifferentialEquation { 
        advection,
        diffusion,
        convection_diffusion,
        advection_vector,
        burgers_inviscid,
        euler};

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

    /// Currently only Lax-Friedrichs and roe can be used as an input parameter
    enum ConvectiveNumericalFlux { lax_friedrichs, roe };
    /// Store convective flux type
    ConvectiveNumericalFlux conv_num_flux_type;

    /// Currently only symmetric internal penalty can be used as an input parameter
    enum DissipativeNumericalFlux { symm_internal_penalty };
    /// Store diffusive flux type
    DissipativeNumericalFlux diss_num_flux_type;

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
};  

} // Parameters namespace
} // PHiLiP namespace

#endif

