#include "pod_galerkin_ode_solver.h"
#include <EpetraExt_MatrixMatrix.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
PODGalerkinODESolver<dim,real,MeshType>::PODGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
        : ReducedOrderODESolver<dim,real,MeshType>(dg_input, pod)
{}

template <int dim, typename real, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODGalerkinODESolver<dim,real,MeshType>::generate_test_basis(const Epetra_CrsMatrix &/*system_matrix*/, const Epetra_CrsMatrix &pod_basis)
{
    return std::make_shared<Epetra_CrsMatrix>(pod_basis);
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODGalerkinODESolver<dim,real,MeshType>::generate_reduced_lhs(const Epetra_CrsMatrix &system_matrix, Epetra_CrsMatrix &test_basis)
{
    Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
    Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, test_basis.RowMap(), test_basis.NumGlobalCols());
    EpetraExt::MatrixMatrix::Multiply(system_matrix, false, test_basis, false, epetra_reduced_lhs_tmp, true);
    EpetraExt::MatrixMatrix::Multiply(test_basis, true, epetra_reduced_lhs_tmp, false, epetra_reduced_lhs);

    return std::make_shared<Epetra_CrsMatrix>(epetra_reduced_lhs);
}


template class PODGalerkinODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class PODGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class PODGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
