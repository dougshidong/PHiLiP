#ifndef __DUALWEIGHTEDRESIDUALMESHADAPTATION_H__ 
#define __DUALWEIGHTEDRESIDUALMESHADAPTATION_H__

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "physics/physics.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Test to check the goal-oriented mesh adaptation locations for various manufactured solutions.
template <int dim, int nstate>
class DualWeightedResidualMeshAdaptation : public TestsBase
{
public:
    /// Constructor of DualWeightedResidualConvergence.
    DualWeightedResidualMeshAdaptation(const Parameters::AllParameters *const parameters_input,
                                       const dealii::ParameterHandler &parameter_handler_input);
    
    /// Parameter handler.
    const dealii::ParameterHandler &parameter_handler;

    /// Runs the test to check the location of refined cell after performing goal-oriented mesh adaptation.
    int run_test() const;
}; 

} // Tests namespace
} // PHiLiP namespace

#endif

