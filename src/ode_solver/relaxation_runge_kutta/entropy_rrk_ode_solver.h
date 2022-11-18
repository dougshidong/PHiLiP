#ifndef __ENTROPY_RRK_ODESOLVER_H_
#define __ENTROPY_RRK_ODESOLVER_H_

#include "dg/dg.h"
#include "rrk_ode_solver_base.h"

namespace PHiLiP {
namespace ODE {

/// Relaxation Runge-Kutta ODE solver, calculating the relaxation parameter as in Ranocha 2020
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EntropyRRKODESolver: public RRKODESolverBase<dim, real, n_rk_stages, MeshType>
{
public:
    /// Default constructor that will set the constants.
    EntropyRRKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

    /// Destructor
    ~EntropyRRKODESolver() {};

protected:

    /// Storage for the solution at each Runge-Kutta stage
    /** Note that rk_stage is the time-derivative of the solution */
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage_solution;

    /// Update stored quantities at the current stage
    /** Stores solution at stage, rk_stage_solution */
    void compute_stored_quantities(const int istage) override;

    /// Compute relaxation parameter numerically (i.e. if energy is NOT the entropy variable)
    /// See Ranocha 2020, Eq. 2.4
    real compute_relaxation_parameter(real &dt) const override;

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
    real compute_entropy_change_estimate(real &dt) const;

    /// Return the entropy variables from a solution vector u
    dealii::LinearAlgebra::distributed::Vector<double> compute_entropy_vars(const dealii::LinearAlgebra::distributed::Vector<double> &u) const;

};

} // ODE namespace
} // PHiLiP namespace

#endif
