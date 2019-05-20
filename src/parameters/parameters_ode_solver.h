#ifndef __PARAMETERS_ODE_SOLVER_H__
#define __PARAMETERS_ODE_SOLVER_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace Parameters
{
    using namespace dealii;
    class ODESolverParam
    {
    public:
        ODESolverParam ();
        enum ODESolverEnum { explicit_solver, implicit_solver };

        OutputEnum ode_output;
        ODESolverEnum ode_solver_type;

        unsigned int nonlinear_max_iterations;
        unsigned int print_iteration_modulo;
        double nonlinear_steady_residual_tolerance;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };
}
#endif
