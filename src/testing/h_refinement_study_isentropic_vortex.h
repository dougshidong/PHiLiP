#ifndef __H_REFINEMENT_STUDY_ISENTROPIC_VORTEX__
#define __H_REFINEMENT_STUDY_ISENTROPIC_VORTEX__

#include <deal.II/base/convergence_table.h>

#include "dg/dg_base.hpp"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// h refinement test for the isentropic vortex advection test case. 
template <int dim, int nstate>
class HRefinementStudyIsentropicVortex: public TestsBase
{
public:
    /// Constructor
    HRefinementStudyIsentropicVortex(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~HRefinementStudyIsentropicVortex() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
protected:
    /// Number of times to solve for convergence summary
    const int n_calculations;

    /// Ratio to refine by
    const double refine_ratio;

    /// Calculate Lp error at the final time in the passed parameters
    /// norm_p is used to indicate the error order -- e.g., norm_p=2 
    /// is L2 norm
    /// Negative norm_p is used to indicate L_infinity norm
    void calculate_Lp_error_at_final_time_wrt_function(double &Lp_error_density, 
            double &Lp_error_pressure,
            std::shared_ptr<DGBase<dim,double>> dg,
            const Parameters::AllParameters parameters,
            double final_time, 
            int norm_p) const;

    /// Reinitialize parameters while refining the timestep. Necessary because all_parameters is constant.
    Parameters::AllParameters reinit_params_and_refine(int refinement) const;

};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
