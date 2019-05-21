#ifndef __GRID_STUDY_H__
#define __GRID_STUDY_H__

#include "parameters/all_parameters.h"
namespace PHiLiP
{
    /// Manufactured grid convergence
    /** Currently the main function as all my test cases simply
     *  check for optimal convergence of the solution
     */
    template<int dim>
    int manufactured_grid_convergence (Parameters::AllParameters &parameters);
}
#endif
