#include "pod_basis_online.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template<int dim>
OnlinePOD<dim>::OnlinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    const bool compute_dRdW = true;
    dg_input->assemble_residual(compute_dRdW);
    massMatrix.reinit(dg_input->global_mass_matrix.m(), dg_input->global_mass_matrix.n());
    massMatrix.copy_from(dg_input->global_mass_matrix);
}

template <int dim>
void OnlinePOD<dim>::addSnapshot(dealii::LinearAlgebra::distributed::Vector<double> snapshot) {
    std::cout << "Adding new snapshot to POD basis..." << std::endl;
    snapshotVectors.push_back(snapshot);
}

template <int dim>
void OnlinePOD<dim>::computeBasis() {
    std::cout << "Computing POD basis..." << std::endl;

    dealii::LAPACKFullMatrix<double> snapshot_matrix(snapshotVectors[0].size(), snapshotVectors.size());

    for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
        for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
            snapshot_matrix(m,n) = snapshotVectors[n][m];
        }
    }

    std::ofstream out_file_snap("snapshot_matrix.txt");
    unsigned int precision1 = 16;
    snapshot_matrix.print_formatted(out_file_snap, precision1);

    std::cout << "Computing POD basis using the method of snapshots..." << std::endl;

    // Get mass weighted solution snapshots: massWeightedSolutionSnapshots = solutionSnapshots^T * massMatrix * solutionSnapshots
    dealii::LAPACKFullMatrix<double> tmp(snapshot_matrix.n(), snapshot_matrix.m());
    dealii::LAPACKFullMatrix<double> massWeightedSolutionSnapshots(snapshot_matrix.n(), snapshot_matrix.n());
    snapshot_matrix.Tmmult(tmp, massMatrix);
    tmp.mmult(massWeightedSolutionSnapshots, snapshot_matrix);
    // Compute SVD of mass weighted solution snapshots: massWeightedSolutionSnapshots = U * Sigma * V^T
    massWeightedSolutionSnapshots.compute_svd();

    // Get eigenvalues
    dealii::LAPACKFullMatrix<double> V = massWeightedSolutionSnapshots.get_svd_vt();
    dealii::LAPACKFullMatrix<double> eigenvectors_T = massWeightedSolutionSnapshots.get_svd_vt();
    dealii::LAPACKFullMatrix<double> eigenvectors(massWeightedSolutionSnapshots.get_svd_vt().n(), massWeightedSolutionSnapshots.get_svd_vt().m());
    eigenvectors_T.transpose(eigenvectors);

    //Form diagonal matrix of inverse singular values
    dealii::LAPACKFullMatrix<double> eigenvaluesSqrtInverse(snapshot_matrix.n(), snapshot_matrix.n());
    for (unsigned int idx = 0; idx < snapshot_matrix.n(); idx++) {
        eigenvaluesSqrtInverse(idx, idx) = 1 / std::sqrt(massWeightedSolutionSnapshots.singular_value(idx));
    }

    //Compute POD basis: fullBasis = solutionSnapshots * eigenvectors * simgularValuesInverse
    tmp.reinit(snapshot_matrix.n(), snapshot_matrix.n());
    eigenvectors.mmult(tmp, eigenvaluesSqrtInverse);

    dealii::LAPACKFullMatrix<double> fullBasis(snapshot_matrix.m(), snapshot_matrix.n());
    snapshot_matrix.mmult(fullBasis, tmp);

    std::ofstream out_file("POD_adaptation_basis.txt");
    unsigned int precision = 16;
    fullBasis.print_formatted(out_file, precision);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(snapshotVectors[0].size(), snapshotVectors.size(), snapshotVectors.size());

    for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
        for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
            basis_tmp.set(m, n, fullBasis(m,n));
        }
    }
    /*

    snapshot_matrix.compute_svd();
    dealii::LAPACKFullMatrix<double> svd_u = snapshot_matrix.get_svd_u();

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(snapshotVectors[0].size(), snapshotVectors.size(), snapshotVectors.size());
    std::cout << snapshotVectors[0].size() << " " << snapshotVectors.size() << std::endl;
    for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
        for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
            basis_tmp.set(m, n, svd_u(m,n));
        }
    }

    */

    basis_tmp.compress(dealii::VectorOperation::insert);

    basis->reinit(basis_tmp);
    basis->copy_from(basis_tmp);
    std::cout << "Done computing POD basis. Basis now has " << basis->n() << " columns." << std::endl;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> OnlinePOD<dim>::getPODBasis() {
    return basis;
}

template class OnlinePOD <PHILIP_DIM>;

}
}
