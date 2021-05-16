#ifndef __PARAMETERS_LINEAR_SOLVER_H__
#define __PARAMETERS_LINEAR_SOLVER_H__

#include <deal.II/base/parameter_handler.h>

#include "parameters.h"

namespace PHiLiP {
namespace Parameters {


/// Parameters related to the linear solver
class LinearSolverParam
{
public:
    LinearSolverParam (); ///< Constructor.
    /// Types of linear solvers available.
    enum LinearSolverEnum {
        direct, /// LU.
        gmres   /// GMRES.
    };

    /// Can either be verbose or quiet.
    /** Verbose will print the full dense matrix. Will not work for large matrices
     */
    OutputEnum linear_solver_output; ///< quiet or verbose.
    LinearSolverEnum linear_solver_type; ///< direct or gmres.

    // GMRES options
    double ilut_drop; ///< Threshold to drop terms close to zero.
    double ilut_rtol; ///< Multiplies diagonal by ilut_rtol for more diagonal dominance.
    double ilut_atol; ///< Add ilu_rtol to diagonal for more diagonal dominance.

    int ilut_fill; ///< ILU fill-in

    double linear_residual; ///< Tolerance for linear residual.
    int max_iterations; ///< Maximum number of linear iteration.
    int restart_number; ///< Number of iterations before restarting GMRES

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);
};

} // Parameters namespace
} // PHiLiP namespace
#endif
