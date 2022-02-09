#include "pod_basis_sensitivity.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
SensitivityPOD<dim>::SensitivityPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : POD<dim>(dg_input)
        , sensitivityBasis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , sensitivityBasisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    getSensitivityPODBasisFromSnapshots();
    computeBasisSensitivity();
    buildSensitivityPODBasis();
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

    this->pcout << "Sensitivity matrix generated." << std::endl;

    return file_found;
}

template <int dim>
void SensitivityPOD<dim>::computeBasisSensitivity() {

    this->pcout << "Computing basis sensitivities..." << std::endl;

    // Compute mass weighted sensitivity snapshots:
    /* massWeightedSensitivitySnapshots = sensitivitySnapshots^T * massMatrix * solutionSnapshots
     *      + solutionSnapshots^T * massMatrix * sensitivitySnapshots
    */
    massWeightedSensitivitySnapshots.reinit(this->massWeightedSolutionSnapshots.m(), this->massWeightedSolutionSnapshots.n());
    dealii::LAPACKFullMatrix<double> B_derivative_tmp(this->massWeightedSolutionSnapshots.m(), this->massWeightedSolutionSnapshots.n());
    dealii::LAPACKFullMatrix<double> tmp(this->solutionSnapshots.n(), this->solutionSnapshots.m());
    sensitivitySnapshots.Tmmult(tmp, this->massMatrix);
    tmp.mmult(massWeightedSensitivitySnapshots, this->solutionSnapshots);
    tmp.reinit(this->solutionSnapshots.n(), this->solutionSnapshots.m());
    this->solutionSnapshots.Tmmult(tmp, this->massMatrix);
    tmp.mmult(B_derivative_tmp, sensitivitySnapshots);
    massWeightedSensitivitySnapshots.add(1, B_derivative_tmp);

    // Initialize eigenvalue sensitivity matrix
    eigenvaluesSensitivity.reinit(this->eigenvectors.n(), this->eigenvectors.n());

    // Compute sensitivity of each eigenvector and eigenvalue
    dealii::LAPACKFullMatrix<double> eigenvectorsSensitivity(this->eigenvectors.m(), this->eigenvectors.n());
    for(unsigned int j = 0 ; j < this->eigenvectors.n() ; j++){ //For each column
        dealii::Vector<double> kEigenvectorSensitivity = computeModeSensitivity(j);
        for(unsigned int i = 0 ; i < kEigenvectorSensitivity.size(); i++){ //For each row
            eigenvectorsSensitivity(i, j) = kEigenvectorSensitivity(i);
        }
    }

    // Compute POD basis sensitivity
    /* fullBasisSensitivity = (sensitivitySnapshots * eigenvectors * simgularValuesInverse)
     *      + (solutionSnapshots * eigenvectorsSensitivity * simgularValuesInverse)
     *          - 1/2*(fullBasis * eigenvaluesSensitivity * singularValuesInverseSquared)
    */
    fullBasisSensitivity.reinit(this->fullBasis.m(), this->fullBasis.n());
    dealii::LAPACKFullMatrix<double> basis_sensitivity_tmp1(this->fullBasis.m(), this->fullBasis.n());
    dealii::LAPACKFullMatrix<double> basis_sensitivity_tmp2(this->fullBasis.m(), this->fullBasis.n());
    dealii::LAPACKFullMatrix<double> singularValuesInverseSquared(this->simgularValuesInverse.m(), this->simgularValuesInverse.n());
    tmp.reinit(this->fullBasis.m(), this->fullBasis.n());
    sensitivitySnapshots.mmult(tmp, this->eigenvectors);
    tmp.mmult(fullBasisSensitivity, this->simgularValuesInverse);
    tmp.reinit(this->fullBasis.m(), this->fullBasis.n());
    this->solutionSnapshots.mmult(tmp, eigenvectorsSensitivity);
    tmp.mmult(basis_sensitivity_tmp1, this->simgularValuesInverse);
    tmp.reinit(this->fullBasis.m(), this->fullBasis.n());
    this->fullBasis.mmult(tmp, eigenvaluesSensitivity);
    this->simgularValuesInverse.mmult(singularValuesInverseSquared, this->simgularValuesInverse);
    tmp.mmult(basis_sensitivity_tmp2, singularValuesInverseSquared);
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

    // Assemble LHS: (massWeightedSolutionSnapshots - diag(singular_value(k)^2))
    dealii::FullMatrix<double> LHS(this->massWeightedSolutionSnapshots.m(), this->massWeightedSolutionSnapshots.n());
    LHS = this->massWeightedSolutionSnapshots;
    LHS.diagadd(-1*pow(this->massWeightedSolutionSnapshots.singular_value(k), 2));

    //Get kth eigenvector
    dealii::Vector<double> kEigenvector(this->eigenvectors.n());
    for(unsigned int i = 0 ; i < this->eigenvectors.n(); i++){
        kEigenvector(i) = this->eigenvectors(i, k);
    }

    //Get eigenvalue sensitivity: kEigenvalueSensitivity = kEigenvector^T * massWeightedSensitivitySnapshots * kEigenvector
    dealii::Vector<double> kEigenvalueSensitivity_tmp(this->eigenvectors.n());
    massWeightedSensitivitySnapshots.Tvmult(kEigenvalueSensitivity_tmp, kEigenvector);
    double kEigenvalueSensitivity = kEigenvalueSensitivity_tmp * kEigenvector;
    eigenvaluesSensitivity(k, k) = kEigenvalueSensitivity;

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

    // Compute eigenvector sensitivity: kEigenvectorSensitivity = leastSquaresSolution - gamma*kEigenvector
    double gamma = -(leastSquaresSolution * kEigenvector);
    dealii::Vector<double> kEigenvectorSensitivity(this->eigenvectors.n());
    kEigenvectorSensitivity.add(1, leastSquaresSolution, gamma*=-1, kEigenvector); //Vk_derivative += 1*leastSquaresSolution + -1*gamma*Vk.

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