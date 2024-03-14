#ifndef __EMPTY_RRK_BASE_H__
#define __EMPTY_RRK_BASE_H__

#include "dg/dg_base.hpp"
#include "ode_solver/runge_kutta_methods/rk_tableau_base.h"

namespace PHiLiP {
namespace ODE {

/// Empty class stored by RK solvers which do not use RRK, and also do not need to calculate numerical entropy.
/// Functions in this class are empty or return a "dummy" value. 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EmptyRRKBase
{
public:
    /// Default constructor that will set the constants.
    explicit EmptyRRKBase(std::shared_ptr<RKTableauBase<dim,real,MeshType>> /*rk_tableau_input*/);

    /// Calculate FR entropy adjustment
    /** Empty here
     */
    virtual real compute_FR_entropy_contribution(const real /*dt*/,
            std::shared_ptr<DGBase<dim,real,MeshType>>/* dg*/,
            const std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &/*rk_stage*/,
            const bool /*compute_K_norm*/) const{
        return 0.0;
    };

    /// Update stored quantities at the current stage
    /** Does nothing here */
    virtual void store_stage_solutions(const int /*istage*/,
            const dealii::LinearAlgebra::distributed::Vector<double> /*rk_stage_i*/) {
        // Do not store anything
    };

    /// Return the modified time step according to the chosen RRK method.
    /** Returns the input dt here */
    virtual real modify_time_step(const real dt, 
            std::shared_ptr<DGBase<dim,real,MeshType>> /*dg*/,
            const std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &/*rk_stage*/,
            const dealii::LinearAlgebra::distributed::Vector<double> &/*solution_update*/
            ) {
        // Return unmodified dt
        return dt;
    };
    
public:
    
    /// Entropy FR correction at the current timestep
    /** Used in entropy-RRK ODE solver.
     * This is stored in dg such that both flow solver case and ode solver can access it. 
     * flow solver cases have no access to ode solver. */
    double FR_entropy_contribution_RRK_solver = 0;
    
    /// Entropy in the M norm
    /** Rather than M+K norm, which is relevant for stabililty in FR.
     * Used in entropy-RRK ODE solver.
     * This is stored in dg such that both flow solver case and ode solver can access it. 
     * flow solver cases have no access to ode solver. */
//    double entropy_M_norm_RRK_solver=0;

    /// Relaxation parameter
    /** Used in RRK ODE solver.
     * This is stored in dg such that both flow solver case and ode solver can access it. 
     * flow solver cases have no access to ode solver. */
    double relaxation_parameter_RRK_solver=1;

};

} // ODE namespace
} // PHiLiP namespace

#endif
