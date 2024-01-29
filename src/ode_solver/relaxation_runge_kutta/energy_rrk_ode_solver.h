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
class EnergyRRKODESolver: public EmptyRRKBase<dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    EnergyRRKODESolver(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

protected:

    /// Number of RK stages
    const double n_rk_stages;

    /// Modify timestep based on relaxation
    double modify_time_step(const double dt) override; 

    /// Compute relaxation parameter explicitly (i.e. if energy is the entropy variable)
    /// See Ketcheson 2019, Eq. 2.4
    real compute_relaxation_parameter(real &dt);
    
    /// Compute inner product according to the nodes being used
    /** This is the same calculation as energy, but using the residual instead of solution
     */
    real compute_inner_product(
            const dealii::LinearAlgebra::distributed::Vector<double> &stage_i,
            const dealii::LinearAlgebra::distributed::Vector<double> &stage_j
            ) const;

public:

    /// Value of the relaxation parameter.
    real relaxation_parameter;


};

} // ODE namespace
} // PHiLiP namespace

#endif
