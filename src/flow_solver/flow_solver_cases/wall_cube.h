#ifndef __WALL_CUBE_H__
#define __WALL_CUBE_H__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class WallCube : public FlowSolverCaseBase<dim,nstate>
{
public:
    /// Constructor.
    WallCube(const Parameters::AllParameters *const parameters_input);

    /// Destructor.
    ~WallCube() {};

    /// Function to generate the grid.
    std::shared_ptr<Triangulation> generate_grid() const override;

protected:
    /// Number of cells per direction for the grid.
    const int number_of_cells_per_direction;

    /// Domain left-boundary value for generating the grid.
    const double domain_left;

    /// Domain right-boundary value for generating the grid.
    const double domain_right;

    /// Domain size (length in 1D, area in 2D, and volume in 3D).
    const double domain_size;

    /// Display additional more specific flow case parameters.
    void display_additional_flow_case_specific_parameters() const override;

};

} // FlowSolver namespace
} // PHiLiP namespace
#endif