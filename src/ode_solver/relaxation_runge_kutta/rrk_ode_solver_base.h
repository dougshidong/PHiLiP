#ifndef __RRK_ODE_SOLVER_BASE_H__
#define __RRK_ODE_SOLVER_BASE_H__

#include "dg/dg_base.hpp"
#include "ode_solver/ode_solver_base.h"
//#include "runge_kutta_ode_solver.h"
#include "ode_solver/explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

/// Relaxation Runge-Kutta ODE solver base class derived from RungeKuttaODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RRKODESolverBase: public RungeKuttaODESolver <dim, real, n_rk_stages, MeshType>
{
public:
    /// Default constructor that will set the constants.
    RRKODESolverBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

    /// Destructor
    ~RRKODESolverBase() {};

    /// Relaxation Runge-Kutta parameter gamma^n
    /** See:  Ketcheson 2019, "Relaxation Runge--Kutta methods: Conservation and stability for inner-product norms"
     *       Ranocha 2020, "Relaxation Runge--Kutta Methods: Fully Discrete Explicit Entropy-Stable Schemes for the Compressible Euler and Navier--Stokes Equations"
     */
    real relaxation_parameter;

protected:

    /// Modify timestep based on relaxation
    void modify_time_step (real &dt) override;
    

    /// Compute relaxation parameter explicitly (i.e. if energy is the entropy variable)
    /// See Ketcheson 2019, Eq. 2.4
    /// See Ranocha 2020, Eq. 2.4
    virtual real compute_relaxation_parameter(real &dt) const = 0;

};

} // ODE namespace
} // PHiLiP namespace

#endif
