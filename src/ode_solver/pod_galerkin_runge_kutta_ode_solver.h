#ifndef __POD_GALERKIN_RUNGE_KUTTA_ODESOLVER__
#define __POD_GALERKIN_RUNGE_KUTTA_ODESOLVER__

#include "JFNK_solver/JFNK_solver.h"
#include "dg/dg_base.hpp"
#include "runge_kutta_base.h"
#include "runge_kutta_methods/rk_tableau_base.h"
#include "relaxation_runge_kutta/empty_RRK_base.h"

namespace PHiLiP {
namespace ODE {

/// Runge-Kutta ODE solver (explicit or implicit) derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PODGalerkingRungeKuttaODESolver: public RungeKuttaBase <dim, real, n_rk_stages, MeshType>
{
public:
    PODGalerkingRungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
            std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod); ///< Constructor.

    void allocate_runge_kutta_system () override;

    void calculate_stages (int i, real dt, const bool pseudotime) override;

    void obtain_stage (int i, real dt) override;

    void sum_stages (const bool pseudotime) override;

    void apply_limiter () override;

    void adjust_time_step () override;

    /// Generate test basis
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(const Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &pod_basis);

    /// Generate reduced LHS
    std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(const Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &test_basis);

protected:

    /// Reduced Space sized Runge Kutta Stages
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> reduced_rk_stage;

    /// POD Basis
    Epetra_CrsMatrix epetra_pod_basis;

    /// System Matrix (Unsure if needed, do some testing)
    Epetra_CrsMatrix epetra_system_matrix; 
private:
    /// Function to multiply a dealii vector by an Epetra Matrix
    int multiply(Epetra_CrsMatrix &epetra_matrix, dealii::LinearAlgebra::distributed::Vector<double> &input_dealii_vector,
                 dealii::LinearAlgebra::distributed::Vector<double> &output_dealii_vector, dealii::IndexSet index_set, bool transpose);

    /// Function to convert a epetra_vector to dealii
    void epetra_to_dealii(Epetra_Vector &epetra_vector, dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector, dealii::IndexSet index_set);
};

} // ODE namespace
} // PHiLiP namespace

#endif

