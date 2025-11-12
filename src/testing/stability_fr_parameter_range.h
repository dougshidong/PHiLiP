#ifndef __STABILITY_FR_PARAMETERS_RANGE_H__
#define __STABILITY_FR_PARAMETERS_RANGE_H__

#include "general_refinement_study.h"
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers' periodic unsteady test
template <int dim, int nspecies, int nstate>
class StabilityFRParametersRange: public GeneralRefinementStudy<dim,nspecies,nstate>
{
public:
    /// Constructor
    explicit StabilityFRParametersRange(const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Run test
    int run_test () const override;
    
    /// Reinitialize parameters for the c loop.
    Parameters::AllParameters reinit_params_c_value(const Parameters::AllParameters *parameters_in, const double c_value) const;

};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
