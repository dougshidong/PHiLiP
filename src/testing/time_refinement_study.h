#ifndef __TIME_REFINEMENT_STUDY_ADVECTION__
#define __TIME_REFINEMENT_STUDY_ADVECTION__

#include "tests.h"
#include <deal.II/base/convergence_table.h>
#include "dg/dg.h"

namespace PHiLiP {
namespace Tests {

/// Advection time refinement study 
template <int dim, int nstate>
class TimeRefinementStudy: public TestsBase
{
public:
    /// Constructor
    TimeRefinementStudy(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~TimeRefinementStudy() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
protected:
    /// Number of times to solve for convergence summary
    const int n_time_calculations = 4;

    /// Ratio to refine by
    const double refine_ratio = 0.5;

    /// Calculate L2 error at the final time in the passed parameters
    double calculate_L2_error_at_final_time_wrt_function(std::shared_ptr<DGBase<dim,double>> dg,const Parameters::AllParameters parameters) const;

    /// Reinitialize parameters while refining the timestep. Necessary because all_parameters is constant.
    Parameters::AllParameters reinit_params_and_refine_timestep(int refinement) const;

};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif