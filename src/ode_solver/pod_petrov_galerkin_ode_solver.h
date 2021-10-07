#ifndef __POD_PETROV_GALERKIN_ODE_SOLVER__
#define __POD_PETROV_GALERKIN_ODE_SOLVER__


#include "dg/dg.h"
#include "ode_solver_base.h"
#include "linear_solver/linear_solver.h"
#include "pod/proper_orthogonal_decomposition.h"
#include <deal.II/lac/trilinos_sparsity_pattern.h>

namespace PHiLiP {
namespace ODE {

/// POD-Galerkin ODE solver derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PODPetrovGalerkinODESolver: public ODESolverBase <dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD> pod); ///< Constructor.

    ///POD
    std::shared_ptr<ProperOrthogonalDecomposition::POD> pod;

    /// Destructor
    ~PODPetrovGalerkinODESolver() {};

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime);

    /// Function to allocate the ODE system
    void allocate_ode_system ();

    /// Reduced solution update given by the ODE solver
    dealii::LinearAlgebra::distributed::Vector<double> reduced_solution_update;

    /// Reduced rhs for linear solver
    dealii::LinearAlgebra::distributed::Vector<double> reduced_rhs;

    /// Psi = J * V
    dealii::TrilinosWrappers::SparseMatrix psi;

    /// Reduced lhs for linear solver
    dealii::TrilinosWrappers::SparseMatrix reduced_lhs;

};

} // ODE namespace
} // PHiLiP namespace

#endif
