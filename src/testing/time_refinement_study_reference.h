#ifndef __TIME_REFINEMENT_STUDY_REFERENCE__
#define __TIME_REFINEMENT_STUDY_REFERENCE__

#include "tests.h"
#include <deal.II/base/convergence_table.h>
#include "dg/dg.h"

namespace PHiLiP {
namespace Tests {

/// Time refinement study which compares to a reference solution
template <int dim, int nstate>
class TimeRefinementStudyReference: public TestsBase
{
public:
    /// Constructor
    TimeRefinementStudyReference(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~TimeRefinementStudyReference() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
protected:
    /// Number of times to solve for convergence summary
    const int n_time_calculations;

    /// Ratio to refine by
    const double refine_ratio;

    /// Reinitialize parameters while refining the timestep. Necessary because all_parameters is constant.
    Parameters::AllParameters reinit_params_and_refine_timestep(int refinement) const;

    /// Reinitialize parameters and set initial_timestep according to reference solution and passed final time
    Parameters::AllParameters reinit_params_for_reference_solution(int number_of_timesteps, double final_time) const;

    dealii::LinearAlgebra::distributed::Vector<double> calculate_reference_solution(double final_time) const;
    
    /// Calculate L2 error at the final time in the passed parameters
    double calculate_L2_error_at_final_time_wrt_reference(
            std::shared_ptr<DGBase<dim,double>> dg,
            const Parameters::AllParameters parameters, 
            double final_time_actual,
            dealii::LinearAlgebra::distributed::Vector<double> reference_solution) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
