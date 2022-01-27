#ifndef __POD_PETROV_GALERKIN_ODE_SOLVER__
#define __POD_PETROV_GALERKIN_ODE_SOLVER__

#include "dg/dg.h"
#include "implicit_ode_solver.h"
#include "linear_solver/linear_solver.h"
#include "reduced_order/pod_basis.h"
#include <deal.II/lac/trilinos_sparsity_pattern.h>

namespace PHiLiP {
namespace ODE {

/// POD-Petrov-Galerkin ODE solver derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PODPetrovGalerkinODESolver: public ImplicitODESolver<dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod); ///< Constructor.

    ///POD
    std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod;

    /// Destructor
    ~PODPetrovGalerkinODESolver() {};

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime) override;

    /// Function to allocate the ODE system
    void allocate_ode_system () override;

    /// Reduced solution update given by the ODE solver
    std::unique_ptr<dealii::LinearAlgebra::distributed::Vector<double>> reduced_solution_update;

    /// Reduced rhs for linear solver
    std::unique_ptr<dealii::LinearAlgebra::distributed::Vector<double>> reduced_rhs;

    /// Psi = J * V
    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix> petrov_galerkin_basis;

    /// Reduced lhs for linear solver
    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix> reduced_lhs;

};

} // ODE namespace
} // PHiLiP namespace

#endif
