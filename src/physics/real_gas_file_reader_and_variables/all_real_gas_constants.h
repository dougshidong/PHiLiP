#ifndef __REAL_GAS_CONSTANTS_H__
#define __REAL_GAS_CONSTANTS_H__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include "readspecies.h"
// #include <string> // for strings

namespace PHiLiP {
namespace RealGasConstants {

/// Main parameter class that contains the various other sub-parameter classes.
class AllRealGasConstants
{
public:
    /// Constructor
    AllRealGasConstants();

    /// Reads and allocates all real gas constants and variables from input files
    void read_species ();
protected:
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};  

} // Parameters namespace
} // PHiLiP namespace

#endif

