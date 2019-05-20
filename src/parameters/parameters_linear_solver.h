#ifndef __PARAMETERS_LINEAR_SOLVER_H__
#define __PARAMETERS_LINEAR_SOLVER_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"
namespace Parameters
{
    using namespace dealii;

    class LinearSolverParam
    {
    public:
        LinearSolverParam ();
        enum LinearSolverEnum { direct, gmres };

        OutputEnum linear_solver_output;
        LinearSolverEnum linear_solver_type;

        // GMRES options
        double ilut_drop, ilut_rtol, ilut_atol;
        int ilut_fill;

        double linear_residual;
        int max_iterations;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };
}
#endif
