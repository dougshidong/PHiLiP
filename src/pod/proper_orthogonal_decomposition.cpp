#include "proper_orthogonal_decomposition.h"
#include <deal.II/lac/vector_operation.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

POD::POD(int num_basis)
: num_basis(num_basis)
, mpi_communicator(MPI_COMM_WORLD)
, pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{}

dealii::LAPACKFullMatrix<double> POD::get_full_pod_basis() {
    std::vector<dealii::FullMatrix<double>> snapshotMatrixContainer;
    std::string path = "snapshot_generation"; //Search this directory for solutions_table.txt files
    for (const auto & entry : std::filesystem::recursive_directory_iterator(path)){ //Recursive seach
        if(entry.path().filename() == "solutions_table.txt"){
            pcout << "Processing " << entry.path() << std::endl;
            std::ifstream myfile(entry.path());
            if(!myfile)
            {
                pcout<<"Error opening file. Please ensure that solutions_table.txt is located in the current directory."<< std::endl;
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
                    }else{
                        cols++;
                    }
                }
                rows++;
            }

            dealii::FullMatrix<double> solutions_matrix(rows-1, cols); //Subtract 1 from row because of header row
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
                            solutions_matrix.set(rows - 1, cols, std::stod(field));
                            cols++;
                        }
                    }
                }
                rows++;
            }
            myfile.close();
            snapshotMatrixContainer.push_back(solutions_matrix);
        }
    }

    int numMat = snapshotMatrixContainer.size();
    int totalCols = 0;
    std::vector<int> j_offset;
    for(int i = 0; i < numMat; i++){
        totalCols = totalCols + snapshotMatrixContainer[i].n_cols();
        if (i == 0){
            j_offset.push_back(0);
        }else{
            j_offset.push_back(j_offset[i-1] + snapshotMatrixContainer[i-1].n_cols());
        }
    }

    //Convert to LAPACKFullMatrix to take the SVD
    dealii::LAPACKFullMatrix<double> snapshot_matrix(snapshotMatrixContainer[0].n_rows(), totalCols);

    for(int i = 0; i < numMat; i++){
        dealii::FullMatrix<double> snapshot_submatrix = snapshotMatrixContainer[i];
        snapshot_matrix.fill(snapshot_submatrix, 0, j_offset[i], 0, 0);
    }

    pcout << "Snapshot matrix generated." << std::endl;
    pcout << "Computing SVD." << std::endl;
    snapshot_matrix.compute_svd();

    svd_u = std::make_unique<dealii::LAPACKFullMatrix<double>>(snapshot_matrix.get_svd_u().n_rows(), snapshot_matrix.get_svd_u().n_cols());
    *svd_u = snapshot_matrix.get_svd_u();

    return *svd_u;
}


void POD::build_reduced_pod_basis() {
    std::vector<int> row_index_set(svd_u->n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set),0);

    std::vector<int> column_index_set(num_basis);
    std::iota(std::begin(column_index_set), std::end(column_index_set),0);

    dealii::TrilinosWrappers::SparseMatrix pod_basis_tmp(svd_u->n_rows(), num_basis, num_basis);

    for(int i : row_index_set){
        for(int j : column_index_set){
            pod_basis_tmp.set(i, j, svd_u->operator()(i,j));
        }
    }
    pod_basis_tmp.compress(dealii::VectorOperation::insert);

    pod_basis.copy_from(pod_basis_tmp);
}

}
}
