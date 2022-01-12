#include "pod_basis.h"
#include <deal.II/lac/vector_operation.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

POD::POD(int num_basis)
: num_basis(num_basis)
, mpi_communicator(MPI_COMM_WORLD)
, pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    getPODBasisFromSnapshots();
}

POD::POD()
        : num_basis(50)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{

    std::string path = "."; //Search current directory for files

    pcout << "Searching files..." << std::endl;
    bool file_found = false;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(path)){ //Recursive seach
        if(std::string(entry.path().filename()).std::string::find("full_POD_basis") != std::string::npos) {
            pcout << "Saved POD basis exists. Reading POD basis..." << std::endl;
            getSavedPODBasis();
            file_found = true;
            break;
        }else{
            if(std::string(entry.path().filename()).std::string::find("solutions_table") != std::string::npos){
                pcout << "Saved solution table exists. Reading solution..." << std::endl;
                getPODBasisFromSnapshots();
                file_found = true;
                break;
            }
        }
    }
    if (file_found == false){
        throw std::invalid_argument("Please ensure that there is a 'full_POD_basis.txt' or 'solutions_table.txt' file!");
    }
}

void POD::getPODBasisFromSnapshots() {
    std::vector<dealii::FullMatrix<double>> snapshotMatrixContainer;
    std::string path = "."; //Search current directory for files containing "solutions_table"
    for (const auto & entry : std::filesystem::recursive_directory_iterator(path)){ //Recursive seach
        if(std::string(entry.path().filename()).std::string::find("solutions_table") != std::string::npos){
            pcout << "Processing " << entry.path() << std::endl;
            std::ifstream myfile(entry.path());
            if(!myfile)
            {
                pcout << "Error opening solutions_table.txt."<< std::endl;
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
    /* Reference for POD basis computation: Refer to Algorithm 1 in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
     */
    pcout << "Snapshot matrix generated." << std::endl;
    pcout << "Computing SVD." << std::endl;
    snapshot_matrix.compute_svd();

    //fullPODBasis = std::make_unique<dealii::LAPACKFullMatrix<double>>(snapshot_matrix.get_svd_u().n_rows(), snapshot_matrix.get_svd_u().n_cols());
    fullPODBasis = snapshot_matrix.get_svd_u();

    num_basis = fullPODBasis.n_cols();
    saveFullPODBasisToFile();
}


void POD::saveFullPODBasisToFile() {
    //pcout << "Saving POD basis to file" << std::endl;
    std::ofstream out_file("full_POD_basis.txt");
    fullPODBasis.print_formatted(out_file);
    //dealii::TableHandler pod_basis;


}

void POD::getSavedPODBasis(){
    std::string path = "."; //Search current directory for files containing "solutions_table"
    for (const auto & entry : std::filesystem::recursive_directory_iterator(path)) { //Recursive seach
        if (std::string(entry.path().filename()).std::string::find("full_POD_basis") != std::string::npos) {
            pcout << "Processing " <<  std::endl;
            std::ifstream myfile(entry.path());
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

            pcout << cols << " " << rows;


            dealii::FullMatrix<double> pod_basis(rows, cols);
            rows = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build POD basis matrix

            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                cols = 0;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    } else {
                        pod_basis.set(rows, cols, std::stod(field));
                        cols++;
                    }
                }
                rows++;
            }
            myfile.close();
            pcout << "here";

            fullPODBasis.copy_from(pod_basis);
        }
    }
    pcout << "done" <<std::endl;
}


void POD::build_reduced_pod_basis() {
    std::vector<int> row_index_set(fullPODBasis.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set),0);

    std::vector<int> column_index_set(num_basis);
    std::iota(std::begin(column_index_set), std::end(column_index_set),0);

    dealii::TrilinosWrappers::SparseMatrix pod_basis_tmp(fullPODBasis.n_rows(), num_basis, num_basis);
    dealii::TrilinosWrappers::SparseMatrix pod_basis_transpose_tmp(num_basis, fullPODBasis.n_rows(), fullPODBasis.n_rows());

    for(int i : row_index_set){
        for(int j : column_index_set){
            pod_basis_tmp.set(i, j, fullPODBasis.operator()(i, j));
            pod_basis_transpose_tmp.set(j, i, fullPODBasis.operator()(i, j));
        }
    }

    pod_basis_tmp.compress(dealii::VectorOperation::insert);
    pod_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    pod_basis.copy_from(pod_basis_tmp);
    pod_basis_transpose.copy_from(pod_basis_transpose_tmp);

}

}
}
