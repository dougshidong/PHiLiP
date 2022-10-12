#ifndef __RUNGE_KUTTA_ODESOLVER__
#define __RUNGE_KUTTA_ODESOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"
#include "JFNK_solver/JFNK_solver.h"
#include "runge_kutta_methods/rk_tableau_base.h"

namespace PHiLiP {
namespace ODE {

/// Runge-Kutta ODE solver (explicit or implicit) derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RungeKuttaODESolver: public ODESolverBase <dim, real, MeshType>
{
public:
    RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input); ///< Constructor.

    /// Destructor
    ~RungeKuttaODESolver() {};

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime);

    /// Function to allocate the ODE system
    void allocate_ode_system ();

protected:
    /// Stores Butcher tableau a and b, which specify the RK method
    std::shared_ptr<RKTableauBase<dim,real,MeshType>> butcher_tableau;

    /// Implicit solver for diagonally-implicit RK methods, using Jacobian-free Newton-Krylov 
    /** This is initialized for any RK method, but solution-sized vectors are 
     *  only initialized if there is an implicit solve
     */
    JFNKSolver<dim,real,MeshType> solver;
    
    /// Storage for the derivative at each Runge-Kutta stage
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage;
    
    /// Modify timestep
    virtual void modify_time_step(real &dt); 

    /// Indicator for zero diagonal elements; used to toggle implicit solve.
    std::vector<bool> butcher_tableau_aii_is_zero;
};

} // ODE namespace
} // PHiLiP namespace

#endif
