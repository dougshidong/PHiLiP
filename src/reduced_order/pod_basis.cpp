#include <deal.II/lac/sparse_matrix.h>
#include "pod_basis.h"
#include <deal.II/fe/mapping_q1_eulerian.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
POD<dim>::POD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : fullPODBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , fullPODBasisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , dg(dg_input)
        , all_parameters(dg->all_parameters)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    pcout << "Searching files..." << std::endl;

    if(getSavedPODBasis() == false){ //will search for a saved basis
        if(getPODBasisFromSnapshots() == false){ //will search for saved snapshots
            throw std::invalid_argument("Please ensure that there is a 'full_POD_basis.txt' or 'solutions_table.txt' file!");
        }
    }
    saveFullPODBasisToFile();
    buildPODBasis();
}

template <int dim>
bool POD<dim>::getPODBasisFromSnapshots() {
    bool file_found = false;
    std::vector<dealii::FullMatrix<double>> snapshotMatrixContainer;
    std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
    for (const auto & entry : std::filesystem::directory_iterator(path)){
        if(std::string(entry.path().filename()).std::string::find("solutions_table") != std::string::npos){
            pcout << "Processing " << entry.path() << std::endl;
            file_found = true;
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
                    } else {
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

    if(all_parameters->reduced_order_param.method_of_snapshots) {
        /* Reference for POD basis computation using the method of snapshots:
        "Local improvements to reduced-order models using sensitivity analysis of the proper orthogonal decomposition"
        Alexander Hay, Jeffrey T. Borgaard, Dominique Pelletier
        J. Fluid Mech. (2009)
        */
        pcout << "Computing POD basis using the method of snapshots..." << std::endl;

        const bool compute_dRdW = true;
        this->dg->assemble_residual(compute_dRdW);

        dealii::LAPACKFullMatrix<double> system_matrix(dg->global_mass_matrix.m(), dg->global_mass_matrix.n());
        system_matrix.copy_from(dg->global_mass_matrix);

        dealii::LAPACKFullMatrix<double> tmp1(snapshot_matrix.n(), snapshot_matrix.m());
        dealii::LAPACKFullMatrix<double> A(snapshot_matrix.n(), snapshot_matrix.n());
        snapshot_matrix.Tmmult(tmp1, system_matrix);
        tmp1.mmult(A, snapshot_matrix);

        A.compute_svd();

        dealii::LAPACKFullMatrix<double> V = A.get_svd_vt();
        dealii::LAPACKFullMatrix<double> sigma(snapshot_matrix.n(), snapshot_matrix.n());

        //Form diagonal matrix of singular values
        for (unsigned int idx = 0; idx < snapshot_matrix.n(); idx++) {
            sigma(idx, idx) = 1 / A.singular_value(idx);
        }

        dealii::LAPACKFullMatrix<double> tmp2(snapshot_matrix.n(), snapshot_matrix.n());
        V.Tmmult(tmp2, sigma);

        fullPODBasisLAPACK.reinit(snapshot_matrix.m(), snapshot_matrix.n());
        snapshot_matrix.mmult(fullPODBasisLAPACK, tmp2);

        pcout << "POD basis computed using the method of snapshots" << std::endl;
    }
    else {
        /* Reference for simple POD basis computation: Refer to Algorithm 1 in the following reference:
        "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
        Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
        International Journal for Numerical Methods in Engineering, 2011
        */

        pcout << "Computing simple POD basis..." << std::endl;
        snapshot_matrix.compute_svd();
        fullPODBasisLAPACK = snapshot_matrix.get_svd_u(); //Note: this is the full U_svd, not the thin SVD. Columns beyond number of basis vectors are useless.

        pcout << "Simple POD basis computed." << std::endl;
    }
    return file_found;
}

template <int dim>
void POD<dim>::saveFullPODBasisToFile() {
    std::ofstream out_file("full_POD_basis.txt");
    unsigned int precision = 7;
    fullPODBasisLAPACK.print_formatted(out_file, precision);
}

template <int dim>
bool POD<dim>::getSavedPODBasis(){
    bool file_found = false;
    std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (std::string(entry.path().filename()).std::string::find("full_POD_basis") != std::string::npos) {
            pcout << "Processing " << entry.path() << std::endl;
            file_found = true;
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
                } else {
                    cols++;
                }
            }
            rows++;
            }

            dealii::FullMatrix<double> pod_basis_tmp(rows, cols);
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
                        pod_basis_tmp.set(rows, cols, std::stod(field));
                        cols++;
                    }
                }
                rows++;
            }
            myfile.close();
            fullPODBasisLAPACK.copy_from(pod_basis_tmp);
        }
    }
    return file_found;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> POD<dim>::getPODBasis(){
    return fullPODBasis;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> POD<dim>::getPODBasisTranspose(){
    return fullPODBasisTranspose;
}

template <int dim>
void POD<dim>::buildPODBasis() {
    std::vector<int> row_index_set(fullPODBasisLAPACK.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set),0);

    std::vector<int> column_index_set(fullPODBasisLAPACK.n_cols());
    std::iota(std::begin(column_index_set), std::end(column_index_set),0);

    dealii::TrilinosWrappers::SparseMatrix pod_basis_tmp(fullPODBasisLAPACK.n_rows(), fullPODBasisLAPACK.n_cols(), fullPODBasisLAPACK.n_cols());
    dealii::TrilinosWrappers::SparseMatrix pod_basis_transpose_tmp(fullPODBasisLAPACK.n_cols(), fullPODBasisLAPACK.n_rows(), fullPODBasisLAPACK.n_rows());

    for(int i : row_index_set){
        for(int j : column_index_set){
            pod_basis_tmp.set(i, j, fullPODBasisLAPACK(i, j));
            pod_basis_transpose_tmp.set(j, i, fullPODBasisLAPACK(i, j));
        }
    }

    pod_basis_tmp.compress(dealii::VectorOperation::insert);
    pod_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    fullPODBasis->reinit(pod_basis_tmp);
    fullPODBasis->copy_from(pod_basis_tmp);
    fullPODBasisTranspose->reinit(pod_basis_tmp);
    fullPODBasisTranspose->copy_from(pod_basis_transpose_tmp);
}

template class POD <PHILIP_DIM>;

}
}
