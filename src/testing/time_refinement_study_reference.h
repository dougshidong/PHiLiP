#ifndef __TIME_REFINEMENT_STUDY_REFERENCE__
#define __TIME_REFINEMENT_STUDY_REFERENCE__

#include <deal.II/base/convergence_table.h>

#include "dg/dg_base.hpp"
#include "general_refinement_study.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Time refinement study which compares to a reference solution
template <int dim, int nstate>
class TimeRefinementStudyReference: public GeneralRefinementStudy<dim,nstate>
{
public:
    /// Constructor
    TimeRefinementStudyReference(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Run test
    int run_test () const override;
protected:
    /// Reinitialize parameters and set initial_timestep according to reference solution and passed final time
    Parameters::AllParameters reinit_params_for_reference_solution(int number_of_timesteps, double final_time) const;

    dealii::LinearAlgebra::distributed::Vector<double> calculate_reference_solution(double final_time);
    
    /// Calculate L2 error at the final time in the passed parameters
    double calculate_L2_error_at_final_time_wrt_reference(
            std::shared_ptr<DGBase<dim,double>> dg,
            const Parameters::AllParameters parameters, 
            double final_time_actual
            ) const;
    
    /// Calculate the L2 error and return local testfail for the converged flowsolver.
    std::tuple<double,int> process_and_write_conv_tables(std::shared_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver, 
            const Parameters::AllParameters params, 
            double L2_error_old, 
            std::shared_ptr<dealii::ConvergenceTable> convergence_table,
            int refinement,
            const double expected_order) const override;
    
    // Hold reference solution using a small timestep size
    dealii::LinearAlgebra::distributed::Vector<double> reference_solution;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
