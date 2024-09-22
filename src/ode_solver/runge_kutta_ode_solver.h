#ifndef __RUNGE_KUTTA_ODESOLVER__
#define __RUNGE_KUTTA_ODESOLVER__

#include "JFNK_solver/JFNK_solver.h"
#include "dg/dg_base.hpp"
#include "runge_kutta_base.h"
#include "runge_kutta_methods/rk_tableau_base.h"
#include "relaxation_runge_kutta/empty_RRK_base.h"

namespace PHiLiP {
namespace ODE {

/// Runge-Kutta ODE solver (explicit or implicit) derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RungeKuttaODESolver: public RungeKuttaBase <dim, real, n_rk_stages, MeshType>
{
public:
    RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input); ///< Constructor.

    void allocate_runge_kutta_system () override;

    void calculate_stages (int i, real dt, const bool pseudotime) override;

    void obtain_stage (int i, real dt) override;

    void sum_stages (const bool pseudotime) override;

    void apply_limiter () override;

    void adjust_time_step () override;

protected:

    /// Stores functions related to relaxation Runge-Kutta (RRK).
    /// Functions are empty by default.
    std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> relaxation_runge_kutta;

    /// Implicit solver for diagonally-implicit RK methods, using Jacobian-free Newton-Krylov 
    /** This is initialized for any RK method, but solution-sized vectors are 
     *  only initialized if there is an implicit solve
     */
    JFNKSolver<dim,real,MeshType> solver;
    
};

} // ODE namespace
} // PHiLiP namespace

#endif

