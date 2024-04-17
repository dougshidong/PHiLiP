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
    DipoleWallCollision(const Parameters::AllParameters *const parameters_input,
                        const bool is_oblique=false);

    /// Destructor
    ~DipoleWallCollision() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;
};

template <int dim, int nstate>
class DipoleWallCollision_Oblique : public DipoleWallCollision<dim,nstate>
{
public:
    /// Constructor.
    DipoleWallCollision_Oblique(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~DipoleWallCollision_Oblique() {};
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
