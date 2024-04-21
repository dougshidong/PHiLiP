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
                        const bool is_oblique_=false);

    /// Destructor
    ~DipoleWallCollision() {};

    const bool is_oblique; /// Flag to distinguish if oblique case
    const bool do_use_stretched_mesh; /// Flag to use stretched mesh

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

private:
    /// Function to generate the uniform grid
    std::shared_ptr<Triangulation> generate_grid_uniform() const;

    /// Function to generate the stretched grid
    std::shared_ptr<Triangulation> generate_grid_stretched() const;

    /// Returns a vector of mesh step sizes for a stretched mesh with clustering at the walls
    std::vector<double> get_mesh_step_size_stretched(
        const int number_of_cells_, 
        const double domain_length_) const;
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
