#ifndef __PERIODIC_CUBE_FLOW_H__
#define __PERIODIC_CUBE_FLOW_H__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace Tests {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class PeriodicCubeFlow : public FlowSolverCaseBase<dim,nstate>
{
public:
    /// Constructor.
    PeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~PeriodicCubeFlow() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

protected:
    const int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_size; ///< Domain size (length in 1D, area in 2D, and volume in 3D)

    /// Display additional more specific flow case parameters
    virtual void display_additional_flow_case_specific_parameters() const override;

    /// Display grid parameters
    void display_grid_parameters() const;
};

} // Tests namespace
} // PHiLiP namespace
#endif
