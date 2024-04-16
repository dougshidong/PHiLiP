#ifndef __DIPOLE_WALL_COLLISION_H__
#define __DIPOLE_WALL_COLLISION_H__

#include "periodic_turbulence.h"
#include "dg/dg.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class DipoleWallCollision : public PeriodicTurbulence<dim,nstate>
{
public:
    /// Constructor.
    DipoleWallCollision(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~DipoleWallCollision() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
