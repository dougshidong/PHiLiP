#include "pod_basis_online.h"
#include <iostream>
#include <filesystem>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <eigen/Eigen/SVD>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template<int dim>
OnlinePOD<dim>::OnlinePOD(std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> _system_matrix)
        : basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , system_matrix(_system_matrix)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
{
}

template <int dim>
void OnlinePOD<dim>::addSnapshot(dealii::LinearAlgebra::distributed::Vector<double> snapshot) {
    pcout << "Adding new snapshot to snapshot matrix..." << std::endl;
    dealii::LinearAlgebra::ReadWriteVector<double> read_snapshot(snapshot.size());
    read_snapshot.import(snapshot, dealii::VectorOperation::values::insert);
    VectorXd eigen_snapshot(snapshot.size());
    for(unsigned int i = 0 ; i < snapshot.size() ; i++){
        eigen_snapshot(i) = read_snapshot(i);
    }
    snapshotMatrix.conservativeResize(snapshot.size(), snapshotMatrix.cols()+1);
    snapshotMatrix.col(snapshotMatrix.cols()-1) = eigen_snapshot;

    //Copy snapshot matrix to dealii Lapack matrix for easy printing to file
    dealiiSnapshotMatrix.reinit(snapshotMatrix.rows(), snapshotMatrix.cols());
    for (unsigned int m = 0; m < snapshotMatrix.rows(); m++) {
        for (unsigned int n = 0; n < snapshotMatrix.cols(); n++) {
            dealiiSnapshotMatrix.set(m, n, snapshotMatrix(m, n));
        }
    }
}

template <int dim>
void OnlinePOD<dim>::computeBasis() {
    pcout << "Computing POD basis..." << std::endl;

    VectorXd reference_state = snapshotMatrix.rowwise().mean();

    referenceState.reinit(reference_state.size());
    for(unsigned int i = 0 ; i < reference_state.size() ; i++){
        referenceState(i) = reference_state(i);
    }

    MatrixXd snapshotMatrixCentered = snapshotMatrix.colwise() - reference_state;

    Eigen::BDCSVD<MatrixXd, Eigen::DecompositionOptions::ComputeThinU> svd(snapshotMatrixCentered);
    MatrixXd pod_basis = svd.matrixU();

    const Epetra_CrsMatrix epetra_system_matrix = system_matrix->trilinos_matrix();
    Epetra_Map system_matrix_map = epetra_system_matrix.RowMap();
    Epetra_CrsMatrix epetra_basis(Epetra_DataAccess::Copy, system_matrix_map, pod_basis.cols());

    const int numMyElements = system_matrix_map.NumMyElements(); //Number of elements on the calling processor

    for (int localRow = 0; localRow < numMyElements; ++localRow){
        const int globalRow = system_matrix_map.GID(localRow);
        for(int n = 0 ; n < pod_basis.cols() ; n++){
            epetra_basis.InsertGlobalValues(globalRow, 1, &pod_basis(globalRow, n), &n);
        }
    }

    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map domain_map((int)pod_basis.cols(), 0, epetra_comm);

    epetra_basis.FillComplete(domain_map, system_matrix_map);

    basis->reinit(epetra_basis);

    pcout << "Done computing POD basis. Basis now has " << basis->n() << " columns." << std::endl;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> OnlinePOD<dim>::getPODBasis() {
    return basis;
}

template <int dim>
dealii::LinearAlgebra::ReadWriteVector<double> OnlinePOD<dim>::getReferenceState() {
    return referenceState;
}

template class OnlinePOD <PHILIP_DIM>;

}
}