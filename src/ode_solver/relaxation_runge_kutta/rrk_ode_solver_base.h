#ifndef __RRK_ODE_SOLVER_BASE_H__
#define __RRK_ODE_SOLVER_BASE_H__

#include "dg/dg_base.hpp"
//#include "ode_solver/runge_kutta_ode_solver.h"
#include "ode_solver/explicit_ode_solver.h"
#include "ode_solver/relaxation_runge_kutta/runge_kutta_store_entropy.h"

namespace PHiLiP {
namespace ODE {

/// Relaxation Runge-Kutta ODE solver base class
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RRKODESolverBase: public RKNumEntropy<dim, real, MeshType>
{
public:
    /// Constructor
    explicit RRKODESolverBase(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

    /// Relaxation Runge-Kutta parameter gamma^n
    /** See:  Ketcheson 2019, "Relaxation Runge--Kutta methods: Conservation and stability for inner-product norms"
     *       Ranocha 2020, "Relaxation Runge--Kutta Methods: Fully Discrete Explicit Entropy-Stable Schemes for the Compressible Euler and Navier--Stokes Equations"
     */
    real relaxation_parameter = 1.0;

protected:

    /// Modify timestep based on relaxation
    real modify_time_step (const real dt,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg,
            std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
            dealii::LinearAlgebra::distributed::Vector<double> & solution_update) override;
    

    /// Compute relaxation parameter explicitly (i.e. if energy is the entropy variable)
    /// See Ketcheson 2019, Eq. 2.4
    /// See Ranocha 2020, Eq. 2.4
    virtual real compute_relaxation_parameter(const real dt,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg,
            std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
            dealii::LinearAlgebra::distributed::Vector<double> &/*solution_update*/
            ) = 0;

};

} // ODE namespace
} // PHiLiP namespace

#endif
