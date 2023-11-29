#ifndef __REAL_GAS_CONSTANTS_H__
#define __REAL_GAS_CONSTANTS_H__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include "real_gas_constants.h"
#include "readspecies.h"

namespace PHiLiP {
namespace RealGasConstants {

/// Main parameter class that contains the various other sub-parameter classes.
class AllRealGasConstants
{
public:
    /// Constructor
    AllRealGasConstants();

    // TO DO in the future once its working -- make reactive var and noneq var separate objects; namespace for now is OK
    // /// Contains parameters for manufactured convergence study
    // ManufacturedConvergenceStudyParam manufactured_convergence_study_param;
    // /// Contains parameters for ODE solver
    // ODESolverParam ode_solver_param;

    /// Declare parameters that can be set as inputs and set up the default options
    /** This subroutine should call the sub-parameter classes static declare_parameters()
      * such that each sub-parameter class is responsible to declare their own parameters.
      */
    void read_species ();
protected:
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};  

} // Parameters namespace
} // PHiLiP namespace

#endif

