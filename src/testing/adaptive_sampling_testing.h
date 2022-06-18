#ifndef __ADAPTIVE_SAMPLING_TESTING__
#define __ADAPTIVE_SAMPLING_TESTING__

#include "tests.h"
#include <fstream>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "pod_adaptive_sampling.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"
#include "functional/functional.h"
#include <Eigen/Dense>
#include "functional/lift_drag.hpp"

namespace PHiLiP {
namespace Tests {

using Eigen::RowVectorXd;
using Eigen::RowVector2d;
using Eigen::VectorXd;
using Eigen::MatrixXd;

/// Burgers Rewienski snapshot
template <int dim, int nstate>
class AdaptiveSamplingTesting: public TestsBase
{
public:
    /// Constructor.
    AdaptiveSamplingTesting(const Parameters::AllParameters *const parameters_input,
                            const dealii::ParameterHandler &parameter_handler_input);

    /// Run test
    int run_test () const override;

    Parameters::AllParameters reinitParams(RowVector2d parameter) const;

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    void outputErrors(int iteration) const;

    std::shared_ptr<Functional<dim,nstate,double>> functionalFactory(std::shared_ptr<DGBase<dim, double>> dg) const;
};


} // End of Tests namespace
} // End of PHiLiP namespace

#endif
