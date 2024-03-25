#ifndef __RK_NUM_ENTROPY_H__
#define __RK_NUM_ENTROPY_H__

#include "dg/dg_base.hpp"
#include "ode_solver/runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace ODE {

/// This class is to compute the FR correction to numerical entropy.
/// It is intended to be used in cases where we compare semi-discrete NSFR
/// with fully-discrete NSFR, such that numerical entropy is reported in a consistent fashion.
/// This class does not modify the behaviour of the ODE solver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RKNumEntropy: public EmptyRRKBase <dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    explicit RKNumEntropy(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input);

    /// Calculate FR entropy adjustment
    /** FR_contribution = dt \sum_i=1^s b_i v^{(i)} K du^{(i)}/dt
     */
    real compute_FR_entropy_contribution(const real dt,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg,
            const std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
            const bool compute_K_norm) const override;
    
    // "using" keyword to prevent compiler complaining
    using EmptyRRKBase<dim, real, MeshType>::compute_FR_entropy_contribution;
    using EmptyRRKBase<dim, real, MeshType>::update_relaxation_parameter;
    using EmptyRRKBase<dim, real, MeshType>::store_stage_solutions;
    

protected:

    /// Store pointer to RK tableau
    std::shared_ptr<RKTableauBase<dim,real,MeshType>> butcher_tableau;

    /// Number of RK stages
    const int n_rk_stages;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

    /// Storage for the solution at each Runge-Kutta stage
    /** Note that rk_stage is the time-derivative of the solution */
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage_solution;

    /// Update stored quantities at the current stage
    /** Stores solution at stage, rk_stage_solution */
    void store_stage_solutions(const int istage,
            const dealii::LinearAlgebra::distributed::Vector<double> rk_stage_i) override;
    
    /// Return the entropy variables from a solution vector u
    dealii::LinearAlgebra::distributed::Vector<double> compute_entropy_vars(
            const dealii::LinearAlgebra::distributed::Vector<double> &u,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg) const;

};

} // ODE namespace
} // PHiLiP namespace

#endif
