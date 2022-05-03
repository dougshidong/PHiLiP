#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include "pod_basis_online.h"
#include <deal.II/base/index_set.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template<int dim>
OnlinePOD<dim>::OnlinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , mass_matrix_sparsity(dg_input->global_mass_matrix.trilinos_sparsity_pattern())
        , dg(dg_input)
{
    const bool compute_dRdW = true;
    dg_input->assemble_residual(compute_dRdW);
    massMatrix.reinit(dg_input->global_mass_matrix.m(), dg_input->global_mass_matrix.n());
    massMatrix.copy_from(dg_input->global_mass_matrix);
}

template <int dim>
void OnlinePOD<dim>::addSnapshot(dealii::LinearAlgebra::distributed::Vector<double> snapshot) {
    std::cout << "Adding new snapshot to POD basis..." << std::endl;
    dealii::LinearAlgebra::ReadWriteVector<double> snapshot_vector(snapshot.size());
    snapshot_vector.import(snapshot, dealii::VectorOperation::values::insert);
    snapshotVectors.push_back(snapshot_vector);
}

template <int dim>
void OnlinePOD<dim>::computeBasis() {
    std::cout << "Computing POD basis..." << std::endl;

    dealii::LAPACKFullMatrix<double> snapshot_matrix(snapshotVectors[0].size(), snapshotVectors.size());

    std::cout << "lapack matrix generated" << std::endl;
    for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
        for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
            snapshot_matrix(m,n) = snapshotVectors[n][m];
        }
    }

    std::ofstream out_file_snap("snapshot_matrix.txt");
    unsigned int precision1 = 16;
    snapshot_matrix.print_formatted(out_file_snap, precision1);
    /*
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
    */

    snapshot_matrix.compute_svd();
    dealii::LAPACKFullMatrix<double> svd_u = snapshot_matrix.get_svd_u();

    /*
    dealii::SparsityPattern sparsity_pattern(svd_u.m(), snapshotVectors.size());
    dealii::DynamicSparsityPattern dynamic_sparsity_pattern(svd_u.m(), snapshotVectors.size());

    for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
        for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
            dynamic_sparsity_pattern.add(m, n);
        }
    }

    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    */
    /*
    std::cout << "here0" << std::endl;
    Epetra_CrsGraph epetra_graph(Epetra_DataAccess::Copy, mass_matrix_sparsity.DomainMap(), (int)snapshotVectors.size(), true);
    Epetra_CrsMatrix epetra_basis(Epetra_DataAccess::Copy, epetra_graph);
     */
    //Epetra_CrsMatrix epetra_basis(Epetra_DataAccess::Copy, epetra_mass_matrix->DomainMap(), (int)snapshotVectors.size(), true);

    //dealii::IndexSet indexSet = dg->global_mass_matrix.locally_owned_domain_indices();

    //dealii::TrilinosWrappers::SparseMatrix basis_sparse(indexSet);

    std::cout << "here1" << std::endl;
    /*
    dealii::TrilinosWrappers::SparsityPattern sparsity_pattern(svd_u.m(), snapshotVectors.size());

    for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
        for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
            sparsity_pattern.exists(m,n);
        }
    }


    dealii::IndexSet colIndexSet = sparsity_pattern.locally_owned_range_indices();
    */

    /*
    dealii::TrilinosWrappers::SparseMatrix basis_tmp(dg->global_mass_matrix.locally_owned_domain_indices(), basis_tmp0.locally_owned_range_indices(),
                                                     reinterpret_cast<MPI_Comm const>(&basis_tmp0));
    */

    /*

    std::cout << "here2" << std::endl;

    //basis->reinit(sparsity_pattern);

    for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
        for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
            basis_tmp.set(m, n, svd_u(m,n));
        }
    }
    */


    //dealii::IndexSet colIndexSet(snapshotVectors.size());
    //colIndexSet.add_range(0, snapshotVectors.size());
    //dealii::TrilinosWrappers::SparseMatrix basis_tmp(dg->global_mass_matrix.locally_owned_domain_indices(), colIndexSet);
    dealii::TrilinosWrappers::SparseMatrix basis_tmp(snapshotVectors[0].size(), snapshotVectors.size(), snapshotVectors.size());


    //basis->reinit(epetra_basis, false);
    std::cout << "here2" << std::endl;

    //basis->reinit(sparsity_pattern);

    for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
        for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
            basis_tmp.set(m, n, svd_u(m,n));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::add);

    basis->reinit(basis_tmp);
    basis->copy_from(basis_tmp);


    //auto petrov_galerkin_basis = std::make_unique<dealii::TrilinosWrappers::SparseMatrix>(basis_sparse.m(), basis_sparse.n(), basis_sparse.n());
    std::cout << "here2.5" << std::endl;
    //this->dg->global_mass_matrix.mmult(*petrov_galerkin_basis, basis_sparse); // petrov_galerkin_basis = system_matrix * pod_basis. Note, use transpose in subsequent multiplications



    std::cout << "here3" << std::endl;

    //basis->reinit(basis_sparse);
    //basis->copy_from(basis_sparse);


    //basis->compress(dealii::VectorOperation::insert);



    /*
    fullBasis.reinit(snapshot_matrix.m(), snapshot_matrix.n());

    for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
        for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
            fullBasis.set(m, n, svd_u(m,n));
        }
    }

    std::ofstream out_file("POD_adaptation_basis.txt");
    unsigned int precision = 16;
    fullBasis.print_formatted(out_file, precision);

    dealii::TrilinosWrappers::SparseMatrix basis_tmp(snapshotVectors[0].size(), snapshotVectors.size(), snapshotVectors.size());
    std::cout << snapshotVectors[0].size() << " " << snapshotVectors.size() << std::endl;
    for(unsigned int m = 0 ; m < snapshotVectors[0].size() ; m++){
        for(unsigned int n = 0 ; n < snapshotVectors.size() ; n++){
            basis_tmp.set(m, n, svd_u(m,n));
        }
    }

    basis_tmp.compress(dealii::VectorOperation::add);

    //basis->reinit(basis_tmp);
    basis->copy_from(basis_tmp);
    //basis->compress(dealii::VectorOperation::insert);
     */
    std::cout << "Done computing POD basis. Basis now has " << basis->n() << " columns." << std::endl;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> OnlinePOD<dim>::getPODBasis() {
    return basis;
}

template class OnlinePOD <PHILIP_DIM>;

}
}
