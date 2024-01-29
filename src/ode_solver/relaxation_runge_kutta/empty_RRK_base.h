#ifndef __EMPTY_RRK_BASE_H__
#define __EMPTY_RRK_BASE_H__

#include "dg/dg_base.hpp"
#include "ode_solver/runge_kutta_methods/rk_tableau_base.h"
//#include "ode_solver/runge_kutta_ode_solver.h"
//#include "ode_solver/explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

/// Empty class stored by RK solvers which do not use RRK, and also do not need to calculate numerical entropy.
/// Functions in this class are designed to use minimal CPU time.

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EmptyRRKBase
{
public:
    /// Default constructor that will set the constants.
    EmptyRRKBase(std::shared_ptr<RKTableauBase<dim,real,MeshType>> /*rk_tableau_input*/);

    /// Calculate FR entropy adjustment
    /** Empty here
     */
    virtual double compute_FR_entropy_contribution(const real /*dt*/,
            std::shared_ptr<DGBase<dim,double>>/* dg*/,
            std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &/*rk_stage*/,
            const bool /*compute_K_norm*/) const{
        return 0.0;
    };

    /// Update stored quantities at the current stage
    /** Does nothing here */
    virtual void store_stage_solutions(const int /*istage*/,
            const dealii::LinearAlgebra::distributed::Vector<double> /*rk_stage_i*/) {};

    virtual real modify_time_step(const real dt, 
            std::shared_ptr<DGBase<dim,double>> /*dg*/,
            std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &/*rk_stage*/,
            dealii::LinearAlgebra::distributed::Vector<double> &/*solution_update*/
            ) {
        return dt;
    }; // will likely also need to pass DG 
    
};

} // ODE namespace
} // PHiLiP namespace

#endif
