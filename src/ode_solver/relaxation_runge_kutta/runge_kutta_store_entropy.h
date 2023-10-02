#ifndef __RK_NUM_ENTROPY_H__
#define __RK_NUM_ENTROPY_H__

#include "dg/dg_base.hpp"
//#include "ode_solver/runge_kutta_ode_solver.h"
#include "ode_solver/explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

/// Relaxation Runge-Kutta ODE solver base class derived from RungeKuttaODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RKNumEntropy: public RungeKuttaODESolver <dim, real, n_rk_stages, MeshType>
{
public:
    /// Default constructor that will set the constants.
    RKNumEntropy(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

    /// Destructor
    ~RKNumEntropy() {};

    /// Calculate FR entropy adjustment
    /** FR_contribution = dt \sum_i=1^s b_i v^{(i)} K du^{(i)}/dt
     */
    double compute_FR_entropy_contribution() const override;
protected:

    /// Storage for the solution at each Runge-Kutta stage
    /** Note that rk_stage is the time-derivative of the solution */
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage_solution;

    /// Update stored quantities at the current stage
    /** Stores solution at stage, rk_stage_solution */
    void store_stage_solutions(const int istage) override;
    
    /// Return the entropy variables from a solution vector u
    dealii::LinearAlgebra::distributed::Vector<double> compute_entropy_vars(const dealii::LinearAlgebra::distributed::Vector<double> &u) const;

    // Euler physics pointer
    std::shared_ptr < Physics::Euler<dim, dim+2, double > > euler_physics;

};

} // ODE namespace
} // PHiLiP namespace

#endif
