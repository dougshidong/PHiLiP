#ifndef __POD_PETROV_GALERKIN_ODE_SOLVER__
#define __POD_PETROV_GALERKIN_ODE_SOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"
#include "reduced_order/pod_basis_base.h"
#include <deal.II/lac/la_parallel_vector.h>

namespace PHiLiP {
namespace ODE {

/// POD-Petrov-Galerkin ODE solver derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PODPetrovGalerkinODESolver: public ODESolverBase<dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod); ///< Constructor.

    ///POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// Destructor
    ~PODPetrovGalerkinODESolver() {};

    /// Evaluate steady state solution.
    int steady_state () override;

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime) override;

    /// Function to allocate the ODE system
    void allocate_ode_system () override;

    /// Reduced solution update given by the ODE solver
    dealii::LinearAlgebra::distributed::Vector<double> reduced_solution_update;

    /// Reference solution for consistency
    dealii::LinearAlgebra::distributed::Vector<double> reference_solution;

    /// Reduced solution
    dealii::LinearAlgebra::distributed::Vector<double> reduced_solution;
};

} // ODE namespace
} // PHiLiP namespace

#endif