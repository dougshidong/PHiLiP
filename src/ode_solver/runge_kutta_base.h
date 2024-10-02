#ifndef __RUNGE_KUTTA_BASE__
#define __RUNGE_KUTTA_BASE__

#include "JFNK_solver/JFNK_solver.h"
#include "dg/dg_base.hpp"
#include "ode_solver_base.h"
#include "runge_kutta_methods/rk_tableau_base.h"
#include "relaxation_runge_kutta/empty_RRK_base.h"

namespace PHiLiP {
namespace ODE {

/// Runge-Kutta Base (explicit or implicit) derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RungeKuttaBase: public ODESolverBase <dim, real, MeshType>
{
public:
    RungeKuttaBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input); ///< Constructor.

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime);

    /// Function to allocate the ODE system
    void allocate_ode_system ();

    /// Function for Specific RK allocation
    virtual void allocate_runge_kutta_system () = 0; 

    /// Function to calculate stage
    virtual void calculate_stages (int i, real dt, const bool pseudotime) = 0;            

    /// Function to obtain stage
    virtual void obtain_stage (int i, real dt) = 0;                

    /// Function to sum stages and add to dg->solution
    virtual void sum_stages (real dt, const bool pseudotime) = 0;                  

    /// Function to apply limiter
    virtual void apply_limiter () = 0;               

    /// Function to adjust time step size
    virtual void adjust_time_step(real dt) = 0;             
protected:

    /// Stores functions related to relaxation Runge-Kutta (RRK).
    /// Functions are empty by default.
    std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> relaxation_runge_kutta;

    /// Implicit solver for diagonally-implicit RK methods, using Jacobian-free Newton-Krylov 
    /** This is initialized for any RK method, but solution-sized vectors are 
     *  only initialized if there is an implicit solve
     */
    JFNKSolver<dim,real,MeshType> solver;

    /// Storage for the derivative at each Runge-Kutta stage
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage;
    
    /// Indicator for zero diagonal elements; used to toggle implicit solve.
    std::vector<bool> butcher_tableau_aii_is_zero;
};
} // ODE namespace
} // PHiLiP namespace

#endif