#ifndef __CUBE_FLOW_UNIFORM_GRID__
#define __CUBE_FLOW_UNIFORM_GRID__

#include "flow_solver_case_base.h"
#include "dg/dg_base.hpp"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class CubeFlow_UniformGrid : public FlowSolverCaseBase<dim, nstate>
{
public:
    explicit CubeFlow_UniformGrid(const Parameters::AllParameters *const parameters_input);

    /// Function to compute the adaptive time step
    virtual double get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Function to compute the initial adaptive time step
    virtual double get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) override;

    /// Updates the maximum local wave speed
    virtual void update_maximum_local_wave_speed(DGBase<dim, double> &dg);
 
protected:
    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

    /// Pointer to Physics object for computing things on the fly
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > pde_physics;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif