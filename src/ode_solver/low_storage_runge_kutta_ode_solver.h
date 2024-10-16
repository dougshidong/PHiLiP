#ifndef __LOW_STORAGE_RUNGE_KUTTA_ODESOLVER__
#define __LOW_STORAGE_RUNGE_KUTTA_ODESOLVER__

#include "JFNK_solver/JFNK_solver.h"
#include "dg/dg_base.hpp"
#include "runge_kutta_base.h"
#include "runge_kutta_methods/low_storage_rk_tableau_base.h"
#include "relaxation_runge_kutta/empty_RRK_base.h"

namespace PHiLiP {
namespace ODE {

/// Low-Storage Runge-Kutta three-register methods
/** see 
 *  Hedrik Ranocha, Lisandro Dalcin, Matteo Parsani, David Ketcheson. 
 *  "Optimized Runge-Kutta Methods with Automatic Step Size Control for Compressible Computational Fluid Dynamics" Communications on Applied Mathematics and Computation Volume 4 (2022): 1191-1228. 
 *  https://github.com/ranocha/Optimized-RK-CFD
 *  The correct coefficients for the [3S*+] method can be found here:
 *  https://github.com/SciML/OrdinaryDiffEq.jl/blob/e17f08ff3916dfc95aa436da037799b6ddbe4cca/lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_caches.jl */

/// Initial Starting Step Size
/** see
 * Hairer, E., Wanner, G., & Norsett, S. (1993). Solving ordinary differential equations 1. Nonstiff problems E. Hairer ; G. Wanner (2nd ed.). Springer. 
 * Page 169
*/

/// Low-Storage Runge-Kutta ODE solver derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class LowStorageRungeKuttaODESolver: public RungeKuttaBase <dim, real, n_rk_stages, MeshType>
{
public:
    LowStorageRungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<LowStorageRKTableauBase<dim,real,MeshType>> rk_tableau_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input); ///< Constructor.

    /// Function to evaluate automatic error adaptive time step
    double get_automatic_error_adaptive_step_size(real dt, const bool /*pseudotime*/);

    /// Function to evaluate automatic initial adaptive time step
    double get_automatic_initial_step_size(real dt, const bool /*pseudotime*/);

    /// Function to allocate the 
    void allocate_runge_kutta_system () override;

    /// Function to calculate stage
    void calculate_stage_solution (int istage, real dt, const bool pseudotime) override;

    /// Function to obtain stage
    void calculate_stage_derivative (int istage, real dt) override;

    /// Function to sum stages and add to dg->solution
    void sum_stages (real dt, const bool pseudotime) override;

    /// Function to apply limiter
    void apply_limiter () override;

    /// Function to adjust time step size
    real adjust_time_step(real dt) override;

    /// Function to prepare the LSRK for a step in time
    void prep_for_step_in_time();
protected:
    /// Stores Butcher tableau a and b, which specify the RK method
    std::shared_ptr<LowStorageRKTableauBase<dim,real,MeshType>> butcher_tableau;

    /// Storage for the derivative at each Runge-Kutta stage
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage;

    /// Storage of the solution for the first storage register
    dealii::LinearAlgebra::distributed::Vector<double> storage_register_1;

    /// Storage of the solution for the second storage register  
    dealii::LinearAlgebra::distributed::Vector<double> storage_register_2;

    /// Storage of the solution for the third storage register
    dealii::LinearAlgebra::distributed::Vector<double> storage_register_3;

    /// Storage of the solution for the fourth storage register
    dealii::LinearAlgebra::distributed::Vector<double> storage_register_4;

    /// Storage of the solution when calculating the right hand side
    dealii::LinearAlgebra::distributed::Vector<double> rhs;

    /// Storage for the weighted/relative error estimate
    real w;

    /// Size of all elements
    double global_size;

    /// Storage for the error estimate at step n-1, n, and n+1
    double epsilon[3];

    /// Storage for the absolute tolerance
    const double atol;

    /// Storage for the relative tolerance
    const double rtol;

    /// Storage for the order of the specified Runge-Kutta method
    const int rk_order;

    /// Storage for the algorithm to use
    const bool is_3Sstarplus;

    /// Storage for the number of delta values for a specified method
    const int num_delta;

    /// Storage for the first beta controller value
    const double beta1;

    /// Storage for the second beta controller value
    const double beta2;

    /// Storage for the third beta controller value
    const double beta3;

};

} // ODE namespace
} // PHiLiP namespace

#endif

