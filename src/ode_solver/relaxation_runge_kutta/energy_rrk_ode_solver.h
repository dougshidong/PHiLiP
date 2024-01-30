#ifndef __ENERGY_RRK_ODESOLVER_H__
#define __ENERGY_RRK_ODESOLVER_H__

#include "dg/dg_base.hpp"
#include "rrk_ode_solver_base.h"
#include "ode_solver/ode_solver_base.h"
//#include "runge_kutta_ode_solver.h"
#include "ode_solver/explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

/// Relaxation Runge-Kutta ODE solver, calculating the relaxation parameter as in Ketcheson 2019
/** "Relaxation Runge-Kutta Methods: Conservation and Stability for Inner-Product Norms" 
 */
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class AlgebraicRRKODESolver: public RRKODESolverBase<dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    explicit AlgebraicRRKODESolver(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

protected:

    /// Compute relaxation parameter explicitly (i.e. if energy is the entropy variable)
    /// See Ketcheson 2019, Eq. 2.4
    real compute_relaxation_parameter(const real dt,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg,
            std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
            dealii::LinearAlgebra::distributed::Vector<double> &/*solution_update*/
            ) override;
    
    /// Compute inner product according to the nodes being used
    real compute_inner_product(
            const dealii::LinearAlgebra::distributed::Vector<double> &stage_i,
            const dealii::LinearAlgebra::distributed::Vector<double> &stage_j,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg
            ) const;

};

} // ODE namespace
} // PHiLiP namespace

#endif
