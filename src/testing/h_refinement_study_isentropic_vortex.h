#ifndef __H_REFINEMENT_STUDY_ISENTROPIC_VORTEX__
#define __H_REFINEMENT_STUDY_ISENTROPIC_VORTEX__

#include <deal.II/base/convergence_table.h>

#include "general_refinement_study.h"
#include "dg/dg_base.hpp"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

/// h refinement test for the isentropic vortex advection test case. 
template <int dim, int nspecies, int nstate>
class HRefinementStudyIsentropicVortex: public GeneralRefinementStudy<dim,nspecies,nstate>
{
public:
    /// Constructor
    HRefinementStudyIsentropicVortex(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Run test
    int run_test () const override;
protected:

    /// Calculate Lp error at the final time in the passed parameters
    /// norm_p is used to indicate the error order -- e.g., norm_p=2 
    /// is L2 norm
    /// Negative norm_p is used to indicate L_infinity norm
    void calculate_Lp_error_at_final_time_wrt_function(double &Lp_error_density, 
            double &Lp_error_pressure,
            std::shared_ptr<DGBase<dim,nspecies,double>> dg,
            const Parameters::AllParameters parameters,
            double final_time, 
            int norm_p) const;

};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
