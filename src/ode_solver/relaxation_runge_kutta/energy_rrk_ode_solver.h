#ifndef __ENERGY_RRK_ODESOLVER_H__
#define __ENERGY_RRK_ODESOLVER_H__

#include "dg/dg_base.hpp"
#include "ode_solver_base.h"
//#include "runge_kutta_ode_solver.h"
#include "explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

/// Relaxation Runge-Kutta ODE solver, calculating the relaxation parameter as in Ketcheson 2019
/** "Relaxation Runge-Kutta Methods: Conservation and Stability for Inner-Product Norms" 
 */
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EnergyRRKODESolver: public RungeKuttaODESolver<dim, real, n_rk_stages, MeshType>
{
public:
    /// Default constructor that will set the constants.
    EnergyRRKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

    /// Destructor
    ~EnergyRRKODESolver() {};
    
    real relaxation_parameter;

protected:

    /// Modify timestep based on relaxation
    void modify_time_step (real &dt) override;

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

};

} // ODE namespace
} // PHiLiP namespace

#endif
