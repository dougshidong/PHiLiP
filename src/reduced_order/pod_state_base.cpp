#include <deal.II/lac/sparse_matrix.h>
#include "pod_state_base.h"
#include <deal.II/fe/mapping_q1_eulerian.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
    PODState<dim>::PODState(std::shared_ptr<DGBase<dim,double>> &dg_input)
    : fullPODBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
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
        saveFullPODBasisToFile();
    }
    saveFullPODBasisToFile();
    buildPODBasis();
}

template <int dim>
bool PODState<dim>::getPODBasisFromSnapshots() {
bool file_found = false;
std::vector<dealii::FullMatrix<double>> snapshotMatrixContainer;
std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"

std::vector<std::filesystem::path> files_in_directory;
std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

for (const auto & entry : files_in_directory){
    if(std::string(entry.filename()).std::string::find("solutions_table") != std::string::npos){
        pcout << "Processing " << entry << std::endl;
        file_found = true;
        std::ifstream myfile(entry);
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
solutionSnapshots.reinit(snapshotMatrixContainer[0].n_rows(), totalCols);

for(int i = 0; i < numMat; i++){
    dealii::FullMatrix<double> snapshot_submatrix = snapshotMatrixContainer[i];
    solutionSnapshots.fill(snapshot_submatrix, 0, j_offset[i], 0, 0);
}

pcout << "Snapshot matrix generated." << std::endl;

if(all_parameters->reduced_order_param.method_of_snapshots) {
    /* Reference for POD basis computation using the method of snapshots:
    "Local improvements to reduced-order models using sensitivity analysis of the proper orthogonal decomposition"
    Alexander Hay, Jeffrey T. Borgaard, Dominique Pelletier
    J. Fluid Mech. (2009)
    */
    pcout << "Computing POD basis using the method of snapshots..." << std::endl;

    // Get mass matrix
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    massMatrix.reinit(dg->global_mass_matrix.m(), dg->global_mass_matrix.n());
    massMatrix.copy_from(dg->global_mass_matrix);

    // Get mass weighted solution snapshots: massWeightedSolutionSnapshots = solutionSnapshots^T * massMatrix * solutionSnapshots
    dealii::LAPACKFullMatrix<double> tmp(solutionSnapshots.n(), solutionSnapshots.m());
    massWeightedSolutionSnapshots.reinit(solutionSnapshots.n(), solutionSnapshots.n());
    solutionSnapshots.Tmmult(tmp, massMatrix);
    tmp.mmult(massWeightedSolutionSnapshots, solutionSnapshots);

    /* IMPORTANT: This copy of massWeightedSolutionSnapshots is necessary due to an apparent bug in dealii. When
     * compute_svd() is called on the LAPACKFullMatrix, later trying to convert this LAPACKFullMatrix to a FullMatrix
     * (necessary in SensitivityPOD) will not give the right matrix. Keep a copy and convert the copy to a FullMatrix
    */
    massWeightedSolutionSnapshotsCopy = massWeightedSolutionSnapshots;

    // Compute SVD of mass weighted solution snapshots: massWeightedSolutionSnapshots = U * Sigma * V^T
    massWeightedSolutionSnapshots.compute_svd();

    // Get eigenvalues
    dealii::LAPACKFullMatrix<double> V = massWeightedSolutionSnapshots.get_svd_vt();
    dealii::LAPACKFullMatrix<double> eigenvectors_T = this->massWeightedSolutionSnapshots.get_svd_vt();
    eigenvectors.reinit(this->massWeightedSolutionSnapshots.get_svd_vt().n(), this->massWeightedSolutionSnapshots.get_svd_vt().m());
    eigenvectors_T.transpose(eigenvectors);

    //Form diagonal matrix of inverse singular values
    eigenvaluesSqrtInverse.reinit(solutionSnapshots.n(), solutionSnapshots.n());
    for (unsigned int idx = 0; idx < solutionSnapshots.n(); idx++) {
        eigenvaluesSqrtInverse(idx, idx) = 1 / std::sqrt(massWeightedSolutionSnapshots.singular_value(idx));
    }

    //Compute POD basis: fullBasis = solutionSnapshots * eigenvectors * simgularValuesInverse
    tmp.reinit(solutionSnapshots.n(), solutionSnapshots.n());
    eigenvectors.mmult(tmp, eigenvaluesSqrtInverse);
    fullBasis.reinit(solutionSnapshots.m(), solutionSnapshots.n());
    solutionSnapshots.mmult(fullBasis, tmp);

    pcout << fullBasis.n() << std::endl;

    pcout << "POD basis computed using the method of snapshots" << std::endl;
}
else {
    /* Reference for simple POD basis computation: Refer to Algorithm 1 in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
    */

    pcout << "Computing simple POD basis..." << std::endl;
    solutionSnapshots.compute_svd();
    fullBasis = solutionSnapshots.get_svd_u(); //Note: this is the full U_svd, not the thin SVD. Columns beyond number of basis vectors are useless.

    pcout << "Simple POD basis computed." << std::endl;
}
return file_found;
}

template <int dim>
void PODState<dim>::saveFullPODBasisToFile() {
std::ofstream out_file("full_POD_basis.txt");
unsigned int precision = 7;
fullBasis.print_formatted(out_file, precision);
}

template <int dim>
bool PODState<dim>::getSavedPODBasis(){
bool file_found = false;
std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
for (const auto & entry : std::filesystem::directory_iterator(path)) {
    if (std::string(entry.path().filename()).std::string::find("POD_adaptation_basis_7") != std::string::npos) {
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
        fullBasis.copy_from(pod_basis_tmp);
    }
}
return file_found;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> PODState<dim>::getPODBasis(){
return fullPODBasis;
}

template <int dim>
void PODState<dim>::buildPODBasis() {
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

template class PODState <PHILIP_DIM>;

}
}
