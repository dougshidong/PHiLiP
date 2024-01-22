#ifndef __NON_PERIODIC_CUBE_FLOW__
#define __NON_PERIODIC_CUBE_FLOW__

#include "flow_solver_case_base.h"
#include "cube_flow_uniform_grid.h"
#include "dg/dg_base.hpp"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class NonPeriodicCubeFlow : public CubeFlow_UniformGrid<dim, nstate>
{
#if PHILIP_DIM==1
     using Triangulation = dealii::Triangulation<PHILIP_DIM>;
 #else
     using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
 #endif

 public:
     explicit NonPeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);
     
     std::shared_ptr<Triangulation> generate_grid() const override;

     void display_additional_flow_case_specific_parameters() const override;

 protected:
    /// Function to compute the adaptive time step
    using CubeFlow_UniformGrid<dim, nstate>::get_adaptive_time_step;
    // double get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Function to compute the initial adaptive time step
    using CubeFlow_UniformGrid<dim, nstate>::get_adaptive_time_step_initial;
    // double get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) override;

    /// Updates the maximum local wave speed
    using CubeFlow_UniformGrid<dim, nstate>::update_maximum_local_wave_speed;
    // void update_maximum_local_wave_speed(DGBase<dim, double> &dg);
 
 private:
    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

    /// Pointer to Physics object for computing things on the fly
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > pde_physics;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif