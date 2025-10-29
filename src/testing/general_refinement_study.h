#ifndef __TIME_REFINEMENT_STUDY__
#define __TIME_REFINEMENT_STUDY__

#include <deal.II/base/convergence_table.h>

#include "dg/dg_base.hpp"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// Advection time refinement study 
template <int dim, int nstate>
class GeneralRefinementStudy: public TestsBase
{
public:
    /// Type of refinement to run
    enum RefinementType { timestep, h }; // in the future, can also add p-refinement here

    /// Constructor
    GeneralRefinementStudy(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input,
            const RefinementType refinement_type_input
            );

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Run test
    int run_test () const override;
protected:
    /// Type of refinement to run
    const RefinementType refinement_type;
    
    /// Number of times to solve for convergence summary
    const int n_calculations;

    /// Ratio to refine by
    const double refine_ratio;

    /// Calculate Lp error at the final time in the passed parameters
    /// norm_p is used to indicate the error order -- e.g., norm_p=2 
    /// is L2 norm
    /// Negative norm_p is used to indicate L_infinity norm
    double calculate_Lp_error_at_final_time_wrt_function(std::shared_ptr<DGBase<dim,double>> dg,const Parameters::AllParameters parameters, double final_time, int norm_p) const;

    /// Reinitialize parameters while refining the timestep. Necessary because all_parameters is constant.
    Parameters::AllParameters reinit_params_and_refine(const Parameters::AllParameters *parameters_in, int refinement, const RefinementType how) const;

};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
