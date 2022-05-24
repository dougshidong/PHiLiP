#include "pod_basis_offline.h"
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/fe/mapping_q1_eulerian.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
OfflinePOD<dim>::OfflinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , dg(dg_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
{
    const bool compute_dRdW = true;
    dg_input->assemble_residual(compute_dRdW);

    pcout << "Searching files..." << std::endl;

    getPODBasisFromSnapshots();

    //buildPODBasis();
}

template <int dim>
bool OfflinePOD<dim>::getPODBasisFromSnapshots() {
    bool file_found = false;
    MatrixXd snapshotMatrix(0,0);
    std::string path = dg->all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"

    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

    for (const auto & entry : files_in_directory){
        if(std::string(entry.filename()).std::string::find("snapshot") != std::string::npos){
            pcout << "Processing " << entry << std::endl;
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                pcout << "Error opening file." << std::endl;
            }
            std::string line;
            int rows = 0;
            int cols = 0;
            //First loop set to count rows and columns
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                cols = 0;
                while (getline(stream, field,' ')){ //parse data values on each line
                    if (field.empty()){ //due to whitespace
                        continue;
                    } else {
                        cols++;
                    }
                }
                rows++;
            }

            snapshotMatrix.conservativeResize(rows-1, snapshotMatrix.cols()+cols); //Subtract 1 from row because of header row

            rows = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                cols = 0;
                if(rows != 0){
                    while (getline(stream, field,' ')) { //parse data values on each line
                        if (field.empty()) {
                            continue;
                        } else {
                            snapshotMatrix(rows-1, cols) = std::stod(field);
                            cols++;
                        }
                    }
                }
                rows++;
            }
            myfile.close();
        }
    }

    pcout << snapshotMatrix << std::endl;

    pcout << "Snapshot matrix generated." << std::endl;


    /* Reference for simple POD basis computation: Refer to Algorithm 1 in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
    */

    pcout << "Computing POD basis..." << std::endl;

    VectorXd reference_state = snapshotMatrix.rowwise().mean();

    referenceState.reinit(reference_state.size());
    for(unsigned int i = 0 ; i < reference_state.size() ; i++){
        referenceState(i) = reference_state(i);
    }

    snapshotMatrix = snapshotMatrix.colwise() - reference_state;

    Eigen::BDCSVD<MatrixXd> svd(snapshotMatrix, Eigen::DecompositionOptions::ComputeThinU);
    MatrixXd pod_basis = svd.matrixU();

    fullBasis.reinit(pod_basis.rows(), pod_basis.cols());

    for (unsigned int m = 0; m < pod_basis.rows(); m++) {
        for (unsigned int n = 0; n < pod_basis.cols(); n++) {
            fullBasis.set(m, n, pod_basis(m, n));
        }
    }

    std::ofstream out_file("POD_basis.txt");
    unsigned int precision = 16;
    fullBasis.print_formatted(out_file, precision);

    return file_found;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> OfflinePOD<dim>::getPODBasis() {
    return basis;
}

template <int dim>
dealii::LinearAlgebra::ReadWriteVector<double> OfflinePOD<dim>::getReferenceState() {
    return referenceState;
}

/*
template <int dim>
void OfflinePOD<dim>::buildPODBasis() {
    std::vector<int> row_index_set(fullBasis.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set),0);

    std::vector<int> column_index_set(fullBasis.n_cols());
    std::iota(std::begin(column_index_set), std::end(column_index_set),0);

    dealii::TrilinosWrappers::SparseMatrix pod_basis_tmp(fullBasis.n_rows(), fullBasis.n_cols(), fullBasis.n_cols());

    for(int i : row_index_set){
        for(int j : column_index_set){
            pod_basis_tmp.set(i, j, fullBasis(i, j));
        }
    }

    pod_basis_tmp.compress(dealii::VectorOperation::insert);

    fullPODBasis->reinit(pod_basis_tmp);
    fullPODBasis->copy_from(pod_basis_tmp);
}
*/
template class OfflinePOD <PHILIP_DIM>;

}
}
