#ifndef __CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_H__
#define __CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class ConvectionDiffusionPeriodic: public TestsBase
{
public:
	ConvectionDiffusionPeriodic() = delete;
	ConvectionDiffusionPeriodic(const Parameters::AllParameters *const parameters_input);
    int run_test () const override;
private:
    //const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
    const MPI_Comm mpi_communicator;
    dealii::ConditionalOStream pcout;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
