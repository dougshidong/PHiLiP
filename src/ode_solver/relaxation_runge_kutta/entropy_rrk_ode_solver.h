#ifndef __ENTROPY_RRK_ODESOLVER_H_
#define __ENTROPY_RRK_ODESOLVER_H_

#include "dg/dg_base.hpp"
#include "rrk_ode_solver_base.h"

namespace PHiLiP {
namespace ODE {

/// Relaxation Runge-Kutta ODE solver, calculating the relaxation parameter as in Ranocha 2020
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EntropyRRKODESolver: public RRKODESolverBase<dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    EntropyRRKODESolver(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

protected:

    /// Compute relaxation parameter numerically (i.e. if energy is NOT the entropy variable)
    /// See Ranocha 2020, Eq. 2.4
    real compute_relaxation_parameter(real &dt) override;

    /// Compute the remainder of the root function Ranocha 2020 Eq. 2.4
    real compute_root_function(
            const double gamma,
            const dealii::LinearAlgebra::distributed::Vector<double> &u_n,
            const dealii::LinearAlgebra::distributed::Vector<double> &d,
            const double eta_n,
            const double e) const;

    /// Compute numerical entropy
    real compute_numerical_entropy(
            const dealii::LinearAlgebra::distributed::Vector<double> &u) const;
    
    /// Compute numerical entropy by integrating over quadrature points
    real compute_integrated_numerical_entropy(
            const dealii::LinearAlgebra::distributed::Vector<double> &u) const;
    
    /// Compute the estimated entropy change during a timestep
    real compute_entropy_change_estimate(real &dt,
            const bool use_M_norm_for_entropy_change_est = true) const;

    /// Storing cumulative entropy change for output 
    real FR_entropy_cumulative = 0;

};

} // ODE namespace
} // PHiLiP namespace

#endif
