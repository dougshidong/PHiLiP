#ifndef __GRID_REFINEMENT_STUDY_H__
#define __GRID_REFINEMENT_STUDY_H__

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {

namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class GridRefinementStudy: public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    GridRefinementStudy() = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    GridRefinementStudy(
        const Parameters::AllParameters *const parameters_input);

    ~GridRefinementStudy() {}; ///< Destructor.

    int run_test() const;
};

} // Tests namespace

} // PHiLiP namespace

#endif // __GRID_REFINEMENT_STUDY_H__
