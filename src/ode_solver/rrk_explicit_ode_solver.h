#ifndef __RRK_EXPLICIT_ODESOLVER__
#define __RRK_EXPLICIT_ODESOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"
#include "explicit_ode_solver.h"

namespace PHiLiP {
namespace ODE {

/// Relaxation Runge-Kutta ODE solver derived from ExplicitODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RRKExplicitODESolver: public ExplicitODESolver <dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    RRKExplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input); ///< Constructor.

    /// Destructor
    ~RRKExplicitODESolver() {};

    /// Relaxation Runge-Kutta parameter gamma^n
    /** See:  Ketcheson 2019, "Relaxation Runge--Kutta methods: Conservation and stability for inner-product norms"
     *       Ranocha 2020, "Relaxation Runge--Kutta Methods: Fully Discrete Explicit Entropy-Stable Schemes for the Compressible Euler and Navier--Stokes Equations"
     */
    real relaxation_parameter;

protected:

    /// Compute relaxation parameter explicitly (i.e. if energy is the entropy variable)
    /// See Ketcheson 2019, Eq. 2.4
    real compute_relaxation_parameter_explicit() const;

    /// Modify timestep based on relaxation
    void modify_time_step (real &dt) override;

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
