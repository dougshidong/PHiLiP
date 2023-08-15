#ifndef __BOUND_PRESERVING_LIMITER__
#define __BOUND_PRESERVING_LIMITER__

#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/dofs/dof_handler.h>
#include "dg.h"

namespace PHiLiP {

template<int dim, int nstate, typename real>
class BoundPreservingLimiter
    {
    public:
        /// Constructor
        BoundPreservingLimiter(const Parameters::AllParameters *const parameters_input);

        /// Destructor
        ~BoundPreservingLimiter();

        /// Pointer to parameters object
        const Parameters::AllParameters* const all_parameters;

        /// Initial global maximum of solution in domain.
        std::vector<real> global_max;
        /// Initial global minimum of solution in domain.
        std::vector<real> global_min;

        void get_global_max_and_min_of_solution();

        void apply_maximum_principle_limiter();
        void apply_positivity_preserving_limiter2010();
        void apply_positivity_preserving_limiter2011();
        void apply_tvb_limiter();

    }; // end of DGStrong class

} // PHiLiP namespace

#endif

