#ifndef __ALL_PARAMETERS_H__
#define __ALL_PARAMETERS_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters.h"
#include "parameters/parameters_ode_solver.h"
#include "parameters/parameters_linear_solver.h"
#include "parameters/parameters_manufactured_convergence_study.h"


namespace Parameters
{
    using namespace dealii;

    /// Main parameter class that contains the various other sub-parameter classes
    class AllParameters
    {
    public:

        ManufacturedConvergenceStudyParam manufactured_convergence_study_param;
        ODESolverParam ode_solver_param;
        LinearSolverParam linear_solver_param;

        unsigned int dimension;
        enum PartialDifferentialEquation { advection, diffusion, convection_diffusion };
        PartialDifferentialEquation pde_type;

        enum ConvectiveNumericalFlux { lax_friedrichs };
        ConvectiveNumericalFlux conv_num_flux_type;

        enum DissipativeNumericalFlux { symm_internal_penalty };
        DissipativeNumericalFlux diss_num_flux_type;

        AllParameters();
        //FunctionParser<dim> initial_conditions;
        //BoundaryConditions  boundary_conditions[max_n_boundaries];
        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);

        //Parameters::Refinement::declare_parameters (prm);
        //Parameters::Flux::declare_parameters (prm);
        //Parameters::Output::declare_parameters (prm);
    };  
}

#endif

