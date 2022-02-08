#include "pod_basis_sensitivity.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
SensitivityPOD<dim>::SensitivityPOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : POD<dim>(dg_input)
        , basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , basisTranspose(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
    getSensitivityPODBasisFromSnapshots();
    computeBasisSensitivity();
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
        if(std::string(entry.filename()).std::string::find("sensitivity_table") != std::string::npos){
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

    sensitivity_matrix.reinit(sensitivityMatrixContainer[0].n_rows(), totalCols);

    for(int i = 0; i < numMat; i++){
        dealii::FullMatrix<double> snapshot_submatrix = sensitivityMatrixContainer[i];
        sensitivity_matrix.fill(snapshot_submatrix, 0, j_offset[i], 0, 0);
    }

    this->pcout << "Sensitivity matrix generated." << std::endl;

    return file_found;
}

template <int dim>
void SensitivityPOD<dim>::computeBasisSensitivity() {

    //Construct B derivative
    B_derivative.reinit(this->B.m(), this->B.n());
    dealii::LAPACKFullMatrix<double> B_derivative_tmp(this->B.m(), this->B.n());
    dealii::LAPACKFullMatrix<double> tmp(this->snapshot_matrix.n(), this->snapshot_matrix.m());
    sensitivity_matrix.Tmmult(tmp, this->system_matrix);
    tmp.mmult(B_derivative, this->snapshot_matrix);
    tmp.reinit(this->snapshot_matrix.n(), this->snapshot_matrix.m());
    this->snapshot_matrix.Tmmult(tmp, this->system_matrix);
    tmp.mmult(B_derivative_tmp, sensitivity_matrix);
    B_derivative.add(1, B_derivative_tmp);

    //Get V
    dealii::LAPACKFullMatrix<double> Vt = this->B.get_svd_vt();
    V.reinit(this->B.get_svd_vt().n(), this->B.get_svd_vt().m());
    Vt.transpose(V);

    lambda_sensitivity.reinit(V.n(), V.n());

    dealii::LAPACKFullMatrix<double> V_derivative(V.m(), V.n());
    for(unsigned int j = 0 ; j < this->V.n() ; j++){
        this->pcout << j << std::endl;
        dealii::Vector<double> Vk_derivative = computeModeSensitivity(j);
        for(unsigned int i = 0 ; i < Vk_derivative.size(); i++){
            V_derivative(i,j) = Vk_derivative(i);
        }
    }

    basis_sensitivity.reinit(this->fullPODBasisLAPACK.m(), this->fullPODBasisLAPACK.n());
    dealii::LAPACKFullMatrix<double> basis_sensitivity_tmp1(this->fullPODBasisLAPACK.m(), this->fullPODBasisLAPACK.n());
    dealii::LAPACKFullMatrix<double> basis_sensitivity_tmp2(this->fullPODBasisLAPACK.m(), this->fullPODBasisLAPACK.n());

    tmp.reinit(this->fullPODBasisLAPACK.m(), this->fullPODBasisLAPACK.n());

    sensitivity_matrix.mmult(tmp, V);
    tmp.mmult(basis_sensitivity, this->sigma_inverse);

    tmp.reinit(this->fullPODBasisLAPACK.m(), this->fullPODBasisLAPACK.n());

    this->snapshot_matrix.mmult(tmp, V_derivative);
    tmp.mmult(basis_sensitivity_tmp1, this->sigma_inverse);

    tmp.reinit(this->fullPODBasisLAPACK.m(), this->fullPODBasisLAPACK.n());

    this->fullPODBasisLAPACK.mmult(tmp, lambda_sensitivity);
    dealii::LAPACKFullMatrix<double> sigma_inverse_squared(this->sigma_inverse.m(), this->sigma_inverse.n());
    this->sigma_inverse.mmult(sigma_inverse_squared, this->sigma_inverse);
    tmp.mmult(basis_sensitivity_tmp2, sigma_inverse_squared);

    basis_sensitivity.add(1, basis_sensitivity_tmp1);
    basis_sensitivity.add(-0.5, basis_sensitivity_tmp2);

    std::ofstream out_file("basis_sensitivity.txt");
    unsigned int precision = 7;
    basis_sensitivity.print_formatted(out_file, precision);

}

template <int dim>
dealii::Vector<double> SensitivityPOD<dim>::computeModeSensitivity(int k) {
    dealii::FullMatrix<double> LHS(this->B.m(), this->B.n());
    LHS = this->B;
    LHS.diagadd(-1*pow(this->B.singular_value(k), 2));

    //Get kth vector of V
    dealii::Vector<double> Vk(V.n());
    for(unsigned int i = 0 ; i < V.n(); i++){
        Vk[i] = V(i,k);
    }

    //Get lambda derivative
    dealii::Vector<double> lambda_tmp(V.n());
    B_derivative.Tvmult(lambda_tmp, Vk);
    double lambda_derivative = lambda_tmp*Vk;

    //Add to sigma derivative
    lambda_sensitivity(k,k) = lambda_derivative;

    dealii::FullMatrix<double> RHS_tmp(B_derivative.m(), B_derivative.n());
    RHS_tmp = B_derivative;
    RHS_tmp.diagadd(-1*lambda_derivative);
    dealii::Vector<double> RHS(V.n());
    RHS_tmp.vmult(RHS, Vk);
    RHS*=-1;

    dealii::Householder<double> householder (LHS);
    dealii::Vector<double> Sk(V.n());
    householder.least_squares(Sk, RHS);

    double gamma = -(Sk*Vk);

    dealii::Vector<double> Vk_derivative(V.n());
    Vk_derivative.add(1, Sk, gamma*=-1, Vk); //Vk_derivative += 1*Sk+gamma*Vk.

    std::ofstream out_file("Vk_derivative_test.txt");
    unsigned int precision = 7;
    Vk_derivative.print(out_file, precision);

    return Vk_derivative;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SensitivityPOD<dim>::getPODBasis() {
    return basis;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> SensitivityPOD<dim>::getPODBasisTranspose() {
    return basisTranspose;
}

template <int dim>
void SensitivityPOD<dim>::addPODBasisColumns(const std::vector<unsigned int> /*addColumns*/) {

}

template class SensitivityPOD <PHILIP_DIM>;

}
}