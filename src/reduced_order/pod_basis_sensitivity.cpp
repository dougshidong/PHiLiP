#include "pod_basis_sensitivity.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
SensitivityPOD<dim>::SensitivityPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : POD<dim>(dg_input)
        , sensitivityBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , sensitivityBasisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    this->pcout << "Searching files..." << std::endl;

    if(getSavedSensitivityPODBasis() == false){ //will search for a saved basis
        //If not saved basis, also need to compute POD basis from snapshots
        this->getPODBasisFromSnapshots();
        this->buildPODBasis();
        if(getSensitivityPODBasisFromSnapshots() == false){ //will search for saved snapshots
            throw std::invalid_argument("Please ensure that there is a 'full_POD_basis.txt' or 'solutions_table.txt' file!");
        }
        computeBasisSensitivity();
        buildSensitivityPODBasis();
        saveSensitivityPODBasisToFile();
    }
}


template <int dim>
void SensitivityPOD<dim>::saveSensitivityPODBasisToFile() {
    std::ofstream out_file("POD_sens_basis.txt");
    unsigned int precision = 7;
    fullBasisSensitivity.print_formatted(out_file, precision);
}

template <int dim>
bool SensitivityPOD<dim>::getSavedSensitivityPODBasis(){
    bool file_found = false;
    std::string path = this->all_parameters->reduced_order_param.path_to_search;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (std::string(entry.path().filename()).std::string::find("POD_sens_basis") != std::string::npos) {
            this->pcout << "Processing " << entry.path() << std::endl;
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
            fullBasisSensitivity.copy_from(pod_basis_tmp);
        }
    }
    return file_found;
}

template <int dim>
bool SensitivityPOD<dim>::getSensitivityPODBasisFromSnapshots() {
    bool file_found = false;
    std::vector<dealii::FullMatrix<double>> sensitivityMatrixContainer;
    std::string path = this->all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "sensitivity_table"
    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the ordinary basis

    for (const auto & entry : files_in_directory){
        if(std::string(entry.filename()).std::string::find("sensitivity") != std::string::npos){
            this->pcout << "Processing " << entry << std::endl;
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                this->pcout << "Error opening file."<< std::endl;
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

            dealii::FullMatrix<double> sensitivity(rows-1, cols); //Subtract 1 from row because of header row
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
                            sensitivity.set(rows - 1, cols, std::stod(field));
                            cols++;
                        }
                    }
                }
                rows++;
            }
            myfile.close();
            sensitivityMatrixContainer.push_back(sensitivity);
        }
    }

    int numMat = sensitivityMatrixContainer.size();
    int totalCols = 0;
    std::vector<int> j_offset;
    for(int i = 0; i < numMat; i++){
        totalCols = totalCols + sensitivityMatrixContainer[i].n_cols();
        if (i == 0){
            j_offset.push_back(0);
        }else{
            j_offset.push_back(j_offset[i-1] + sensitivityMatrixContainer[i-1].n_cols());
        }
    }

    sensitivitySnapshots.reinit(sensitivityMatrixContainer[0].n_rows(), totalCols);

    for(int i = 0; i < numMat; i++){
        dealii::FullMatrix<double> snapshot_submatrix = sensitivityMatrixContainer[i];
        sensitivitySnapshots.fill(snapshot_submatrix, 0, j_offset[i], 0, 0);
    }

    //Center data, do not use for now
    /*
    std::vector<double> rowSums(sensitivitySnapshots.n());
    for(unsigned int row = 0 ; row < sensitivitySnapshots.n(); row++){
        for(unsigned int col = 0 ; col < sensitivitySnapshots.m() ; col++){
            rowSums[row] = rowSums[row] + sensitivitySnapshots(row, col);
        }
    }

    for(unsigned int row = 0 ; row < sensitivitySnapshots.n(); row++){
        for(unsigned int col = 0 ; col < sensitivitySnapshots.m() ; col++){
            sensitivitySnapshots(row, col) = sensitivitySnapshots(row, col) - (rowSums[row]/sensitivitySnapshots.m());
        }
    }
     */

    this->pcout << "Sensitivity matrix generated." << std::endl;

    return file_found;
}

template <int dim>
void SensitivityPOD<dim>::computeBasisSensitivity() {
    /* Reference for POD basis sensitivity computation:
    "Local improvements to reduced-order models using sensitivity analysis of the proper orthogonal decomposition"
    Alexander Hay, Jeffrey T. Borgaard, Dominique Pelletier
    J. Fluid Mech. (2009)
    */
    this->pcout << "Computing basis sensitivities..." << std::endl;

    // Compute mass weighted sensitivity snapshots:
    /* massWeightedSensitivitySnapshots = sensitivitySnapshots^T * massMatrix * solutionSnapshots
     *      + solutionSnapshots^T * massMatrix * sensitivitySnapshots
    */
    massWeightedSensitivitySnapshots.reinit(this->massWeightedSolutionSnapshots.m(), this->massWeightedSolutionSnapshots.n());
    dealii::LAPACKFullMatrix<double> massWeightedSensitivitySnapshots_tmp(this->massWeightedSolutionSnapshots.m(), this->massWeightedSolutionSnapshots.n());
    dealii::LAPACKFullMatrix<double> tmp(this->solutionSnapshots.n(), this->solutionSnapshots.m());
    sensitivitySnapshots.Tmmult(tmp, this->massMatrix);
    tmp.mmult(massWeightedSensitivitySnapshots, this->solutionSnapshots);
    tmp.reinit(this->solutionSnapshots.n(), this->solutionSnapshots.m());
    this->solutionSnapshots.Tmmult(tmp, this->massMatrix);
    tmp.mmult(massWeightedSensitivitySnapshots_tmp, sensitivitySnapshots);
    massWeightedSensitivitySnapshots.add(1, massWeightedSensitivitySnapshots_tmp);

    // Initialize eigenvalue sensitivity matrix
    eigenvaluesSensitivity.reinit(this->eigenvectors.n(), this->eigenvectors.n());

    // Compute sensitivity of each eigenvector and eigenvalue
    dealii::LAPACKFullMatrix<double> eigenvectorsSensitivity(this->eigenvectors.m(), this->eigenvectors.n());

    //Compute only some sensitivities
    unsigned int numSensitivities = std::min((unsigned int)20, this->eigenvectors.n());

    for(unsigned int j = 0 ; j < numSensitivities; j++){ //For each column
        dealii::Vector<double> kEigenvectorSensitivity = computeModeSensitivity(j);
        for(unsigned int i = 0 ; i < kEigenvectorSensitivity.size(); i++){ //For each row
            eigenvectorsSensitivity(i, j) = kEigenvectorSensitivity(i);
            this->pcout << "Eigenvector sensitivity " << j << " " << kEigenvectorSensitivity(i) << std::endl;
        }
    }

    // Compute POD basis sensitivity
    /* fullBasisSensitivity = (sensitivitySnapshots * eigenvectors * eigenvaluesSqrtInverse)
     *      + (solutionSnapshots * eigenvectorsSensitivity * eigenvaluesSqrtInverse)
     *          - 1/2*(fullBasis * eigenvaluesSensitivity * eigenvaluesInverse)
    */
    fullBasisSensitivity.reinit(this->fullBasis.m(), this->fullBasis.n());
    dealii::LAPACKFullMatrix<double> basis_sensitivity_tmp1(this->fullBasis.m(), this->fullBasis.n());
    dealii::LAPACKFullMatrix<double> basis_sensitivity_tmp2(this->fullBasis.m(), this->fullBasis.n());
    dealii::LAPACKFullMatrix<double> eigenvaluesInverse(this->eigenvaluesSqrtInverse.m(), this->eigenvaluesSqrtInverse.n());
    tmp.reinit(this->fullBasis.m(), this->fullBasis.n());
    sensitivitySnapshots.mmult(tmp, this->eigenvectors);
    tmp.mmult(fullBasisSensitivity, this->eigenvaluesSqrtInverse);
    tmp.reinit(this->fullBasis.m(), this->fullBasis.n());
    this->solutionSnapshots.mmult(tmp, eigenvectorsSensitivity);
    tmp.mmult(basis_sensitivity_tmp1, this->eigenvaluesSqrtInverse);
    tmp.reinit(this->fullBasis.m(), this->fullBasis.n());
    this->fullBasis.mmult(tmp, eigenvaluesSensitivity);
    this->eigenvaluesSqrtInverse.mmult(eigenvaluesInverse, this->eigenvaluesSqrtInverse);
    tmp.mmult(basis_sensitivity_tmp2, eigenvaluesInverse);
    fullBasisSensitivity.add(1, basis_sensitivity_tmp1);
    fullBasisSensitivity.add(-0.5, basis_sensitivity_tmp2);

    std::ofstream out_file("basis_sensitivity.txt");
    unsigned int precision = 7;
    fullBasisSensitivity.print_formatted(out_file, precision);

    this->pcout << "Basis sensitivities computed." << std::endl;
}

template <int dim>
dealii::Vector<double> SensitivityPOD<dim>::computeModeSensitivity(int k) {
    this->pcout << "Computing mode sensitivity..." << std::endl;

    //Assemble LHS: (massWeightedSolutionSnapshotsCopy - diag(eigenvalue))
    dealii::FullMatrix<double> LHS(this->massWeightedSolutionSnapshots.m(), this->massWeightedSolutionSnapshots.n());
    LHS = this->massWeightedSolutionSnapshotsCopy; //USE MATRIX COPY MADE BEFORE TAKING SVD!!!
    LHS.diagadd(-1*this->massWeightedSolutionSnapshots.singular_value(k));

    //Get kth eigenvector
    dealii::Vector<double> kEigenvector(this->eigenvectors.n());
    for(unsigned int i = 0 ; i < this->eigenvectors.n(); i++){
        kEigenvector(i) = this->eigenvectors(i, k);
        this->pcout << "Eigenvector " << " " << kEigenvector(i) << std::endl;
    }

    //Get eigenvalue sensitivity: kEigenvalueSensitivity = kEigenvector^T * massWeightedSensitivitySnapshots * kEigenvector
    dealii::Vector<double> kEigenvalueSensitivity_tmp(this->eigenvectors.n());
    massWeightedSensitivitySnapshots.Tvmult(kEigenvalueSensitivity_tmp, kEigenvector);
    double kEigenvalueSensitivity = kEigenvalueSensitivity_tmp * kEigenvector;
    eigenvaluesSensitivity(k, k) = kEigenvalueSensitivity;

    this->pcout << "Eigenvalue" << k << " " << this->massWeightedSolutionSnapshots.singular_value(k) << std::endl;
    this->pcout << "Eigenvalue sensitivity " << k << " " << kEigenvalueSensitivity << std::endl;


    // Assemble RHS: -(massWeightedSensitivitySnapshots - diag(kEigenvalueSensitivity))*kEigenvector
    dealii::FullMatrix<double> RHS_tmp(massWeightedSensitivitySnapshots.m(), massWeightedSensitivitySnapshots.n());
    RHS_tmp = massWeightedSensitivitySnapshots;
    RHS_tmp.diagadd(-1 * kEigenvalueSensitivity);
    dealii::Vector<double> RHS(this->eigenvectors.n());
    RHS_tmp.vmult(RHS, kEigenvector);
    RHS*=-1;

    // Compute least squares solution
    dealii::Householder<double> householder (LHS);
    dealii::Vector<double> leastSquaresSolution(this->eigenvectors.n());
    householder.least_squares(leastSquaresSolution, RHS);

    // Compute eigenvector sensitivity by Gram Schmidt orthogonalization: kEigenvectorSensitivity = leastSquaresSolution - gamma*kEigenvector
    double gamma = leastSquaresSolution * kEigenvector;
    dealii::Vector<double> kEigenvectorSensitivity(this->eigenvectors.n());
    kEigenvectorSensitivity = leastSquaresSolution;
    kEigenvector *= gamma;
    kEigenvectorSensitivity -= kEigenvector;

    this->pcout << "Mode sensitivity computed." << std::endl;

    return kEigenvectorSensitivity;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SensitivityPOD<dim>::getPODBasis() {
    return sensitivityBasis;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SensitivityPOD<dim>::getPODBasisTranspose() {
    return sensitivityBasisTranspose;
}

template <int dim>
void SensitivityPOD<dim>::buildSensitivityPODBasis() {

    this->pcout << "Building sensitivity POD basis matrix..." << std::endl;

    std::vector<int> row_index_set(fullBasisSensitivity.n_rows());
    std::iota(std::begin(row_index_set), std::end(row_index_set),0);

    std::vector<int> column_index_set(fullBasisSensitivity.n_cols());
    std::iota(std::begin(column_index_set), std::end(column_index_set),0);

    dealii::TrilinosWrappers::SparseMatrix pod_basis_tmp(fullBasisSensitivity.n_rows(), fullBasisSensitivity.n_cols(), fullBasisSensitivity.n_cols());
    dealii::TrilinosWrappers::SparseMatrix pod_basis_transpose_tmp(fullBasisSensitivity.n_cols(), fullBasisSensitivity.n_rows(), fullBasisSensitivity.n_rows());

    for(int i : row_index_set){
        for(int j : column_index_set){
            pod_basis_tmp.set(i, j, fullBasisSensitivity(i, j));
            pod_basis_transpose_tmp.set(j, i, fullBasisSensitivity(i, j));
        }
    }

    pod_basis_tmp.compress(dealii::VectorOperation::insert);
    pod_basis_transpose_tmp.compress(dealii::VectorOperation::insert);

    sensitivityBasis->reinit(pod_basis_tmp);
    sensitivityBasis->copy_from(pod_basis_tmp);
    sensitivityBasisTranspose->reinit(pod_basis_tmp);
    sensitivityBasisTranspose->copy_from(pod_basis_transpose_tmp);
}

template class SensitivityPOD <PHILIP_DIM>;

}
}