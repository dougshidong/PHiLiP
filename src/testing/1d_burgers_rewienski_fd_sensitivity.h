#ifndef __1D_BURGERS_REWIENSKI_FD_SENSITIVITY__
#define __1D_BURGERS_REWIENSKI_FD_SENSITIVITY__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers Rewienski sensitivity
template <int dim, int nstate>
class BurgersRewienskiSensitivity: public TestsBase
{
public:
    /// Constructor.
    BurgersRewienskiSensitivity(const Parameters::AllParameters *const parameters_input);

    /// Run test
    int run_test () const override;

    dealii::LinearAlgebra::distributed::Vector<double> run_solution(Parameters::AllParameters parameters_input) const;

    Parameters::AllParameters reinit_params(double rewienski_b) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif //PHILIP_1D_BURGERS_REWIENSKI_FD_SENSITIVITY_H
