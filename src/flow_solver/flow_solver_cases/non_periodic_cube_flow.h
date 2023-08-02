#ifndef __NON_PERIODIC_CUBE_FLOW__
#define __NON_PERIODIC_CUBE_FLOW__

#include "flow_solver_case_base.h"
#include "dg/dg.h"
#include "physics/navier_stokes.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class NonPeriodicCubeFlow : public FlowSolverCaseBase<dim, nstate>
{
#if PHILIP_DIM==1
     using Triangulation = dealii::Triangulation<PHILIP_DIM>;
 #else
     using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
 #endif

 public:
     NonPeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);
     
     ~NonPeriodicCubeFlow() {};
 
     std::shared_ptr<Triangulation> generate_grid() const override;

     void display_additional_flow_case_specific_parameters() const override;

 protected:
    /// Function to compute the adaptive time step
    double get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Function to compute the initial adaptive time step
    double get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) override;

    /// Updates the maximum local wave speed
    void update_maximum_local_wave_speed(DGBase<dim, double> &dg);

    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

    /// Pointer to Navier-Stokes physics object for computing things on the fly
   // std::shared_ptr< Physics::NavierStokes<dim,dim+2,double> > navier_stokes_physics;
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > pde_physics;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif