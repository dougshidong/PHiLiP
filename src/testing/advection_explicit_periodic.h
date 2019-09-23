#ifndef __ADVECTION_EXPLICIT_PERIODIC_H__
#define __ADVECTION_EXPLICIT_PERIODIC_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class AdvectionPeriodic: public TestsBase
{
public:
	AdvectionPeriodic() = delete;
	AdvectionPeriodic(const Parameters::AllParameters *const parameters_input);
    int run_test () const override;
private:
    //const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
    const MPI_Comm mpi_communicator;
    dealii::ConditionalOStream pcout;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
