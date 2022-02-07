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
    for (const auto & entry : std::filesystem::directory_iterator(path)){
        if(std::string(entry.path().filename()).std::string::find("sensitivity_table") != std::string::npos){
            this->pcout << "Processing " << entry.path() << std::endl;
            file_found = true;
            std::ifstream myfile(entry.path());
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
    dealii::LAPACKFullMatrix<double> tmp1(this->snapshot_matrix.n(), this->snapshot_matrix.m());
    dealii::LAPACKFullMatrix<double> tmp2(this->snapshot_matrix.n(), this->snapshot_matrix.m());
    sensitivity_matrix.Tmmult(tmp1, this->system_matrix);
    tmp1.mmult(B_derivative, this->snapshot_matrix);
    this->snapshot_matrix.Tmmult(tmp2, this->system_matrix);
    tmp2.mmult(B_derivative_tmp, sensitivity_matrix);
    B_derivative.add(1, B_derivative_tmp);

    //Get V
    //this->B.compute_svd();
    dealii::LAPACKFullMatrix<double> Vt = this->B.get_svd_vt();
    V.reinit(this->B.get_svd_vt().n(), this->B.get_svd_vt().m());
    V.transpose(Vt);

    for(unsigned int i = 0 ; i < this->snapshot_matrix.n() ; i++){
        this->pcout << i << std::endl;
        computeModeSensitivity(i);
        this->pcout << std::setprecision(15) << V(i,5) << std::endl;

    }
}

template <int dim>
void SensitivityPOD<dim>::computeModeSensitivity(int k) {
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

    for(unsigned int i = 0 ; i < V.n(); i++){
        this->pcout << std::setprecision(15) << Vk[i] << std::endl;
    }

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