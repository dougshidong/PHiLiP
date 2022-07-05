#ifndef __RRK_EXPLICIT_ODESOLVER__
#define __RRK_EXPLICIT_ODESOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"
#include "explicit_ode_solver.h"
//also include relevant physics

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

    /// Function to evaluate the solution update 
    /* Same as ExplicitODESolver except modifying timestep size
     * No implementation for pseudotime
     * Also choose explicit or implicit computation of gamma based on PDE
     */
    void step_in_time(real dt, const bool pseudotime) override;

    /// Relaxation Runge-Kutta parameter gamma^n
    /* See:  Ketcheson 2019, "Relaxation Runge--Kutta methods: Conservation and stability for inner-product norms"
     *       Ranocha 2020, "Relaxation Runge--Kutta Methods: Fully Discrete Explicit Entropy-Stable Schemes for the Compressible Euler and Navier--Stokes Equations"
     */
    real relaxation_parameter;

protected:

    /// Compute relaxation parameter explicitly (i.e. if energy is the entropy variable)
    // See Ketcheson 2019, Eq. 2.4
    real compute_relaxation_parameter_explicit();

    /// Compute gamma implicitly, using root-finding technique
    /* To be used if entropy variable is nonlinear
     * See Ranocha 2020, Eq. 2.4
     * Not currently implemented
     */
    // real compute_gamma_implicit();

    /// Compute inner product according to the nodes being used
    /* Goal is to point to physics, similar to compute_and_update_integrated_quantities() 
     * in periodic_turbulence.cpp
     */
    real compute_inner_product(
            dealii::LinearAlgebra::distributed::Vector<double> solution_or_stage_1,
            dealii::LinearAlgebra::distributed::Vector<double> solution_or_stage_2
            //std::unique_ptr<dealii::LinearAlgebra::distributed::Vector<double>> solution_or_stage_1,
            //std::unique_ptr<dealii::LinearAlgebra::distributed::Vector<double>> solution_or_stage_2
            );

    //Maybe also store the stage solutions (in which case allocate_ode_system would need override)



};

} // ODE namespace
} // PHiLiP namespace

#endif
