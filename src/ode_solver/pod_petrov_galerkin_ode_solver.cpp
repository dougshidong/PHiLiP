#include "pod_petrov_galerkin_ode_solver.h"
#include <EpetraExt_MatrixMatrix.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
PODPetrovGalerkinODESolver<dim,real,MeshType>::PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
        : ReducedOrderODESolver<dim,real,MeshType>(dg_input, pod)
{}

template <int dim, typename real, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODPetrovGalerkinODESolver<dim,real,MeshType>::generate_test_basis(const Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis)
{
    Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
    Epetra_CrsMatrix petrov_galerkin_basis(Epetra_DataAccess::Copy, system_matrix_rowmap, pod_basis.NumGlobalCols());
    EpetraExt::MatrixMatrix::Multiply(system_matrix, false, pod_basis, false, petrov_galerkin_basis, true);

    return std::make_shared<Epetra_CrsMatrix>(petrov_galerkin_basis);
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODPetrovGalerkinODESolver<dim,real,MeshType>::generate_reduced_lhs(const Epetra_CrsMatrix &/*system_matrix*/, Epetra_CrsMatrix &test_basis)
{
    Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
    EpetraExt::MatrixMatrix::Multiply(test_basis, true, test_basis, false, epetra_reduced_lhs);

    return std::make_shared<Epetra_CrsMatrix>(epetra_reduced_lhs);
}


template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace//