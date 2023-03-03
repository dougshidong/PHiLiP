#ifndef __ANISOTROPICMESHADAPTATIONCASES_H__ 
#define __ANISOTROPICMESHADAPTATIONCASES_H__ 

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Test to check anisotropic mesh adaptation.
template <int dim, int nstate>
class AnisotropicMeshAdaptationCases : public TestsBase
{
public:
    /// Constructor of DualWeightedResidualConvergence.
    AnisotropicMeshAdaptationCases(const Parameters::AllParameters *const parameters_input,
                                       const dealii::ParameterHandler &parameter_handler_input);
    
    /// Parameter handler.
    const dealii::ParameterHandler &parameter_handler;

    /// Runs the test related to anisotropic mesh adaptation.
    int run_test() const;
}; 

} // Tests namespace
} // PHiLiP namespace

#endif

