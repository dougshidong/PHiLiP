#include "low_storage_rk_tableau_base.h"


namespace PHiLiP {
namespace ODE {
    
    //NOTE TO SELF: Remove these functions before PR!
void print_table(const dealii::Table<2,double> tab, const int nrow, const int ncol) {
    
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    for (int irow = 0; irow < nrow; ++irow){
        for (int icol = 0; icol < ncol; ++icol){
            pcout << tab[irow][icol] << " ";
        }
        pcout << std::endl;
    }
    pcout<<std::endl;

}
void print_table(const dealii::Table<1,double> tab, const int nrow) {
    
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    for (int irow = 0; irow < nrow; ++irow){
        pcout << tab[irow] << " ";
        pcout << std::endl;
    }
    pcout<<std::endl;

}

template <int dim, typename real, typename MeshType> 
LowStorageRKTableauBase<dim,real, MeshType> :: LowStorageRKTableauBase (const int n_rk_stages_input, const int num_delta_input,
        const std::string rk_method_string_input)
    : RKTableauBase<dim,real,MeshType>(n_rk_stages_input,rk_method_string_input)
      , num_delta(num_delta_input)
{
    this->butcher_tableau_gamma.reinit(this->n_rk_stages+1,3);
    this->butcher_tableau_beta.reinit(this->n_rk_stages+1);
    this->butcher_tableau_delta.reinit(this->num_delta);
    this->butcher_tableau_b_hat.reinit(this->n_rk_stages+1);
}

template <int dim, typename real, typename MeshType> 
void LowStorageRKTableauBase<dim,real, MeshType> :: set_tableau ()
{
    set_gamma();
    set_beta();
    set_delta();
    set_b_hat();
    this->pcout << "Set standard values" << std::endl;
    set_a_and_b();
    this->pcout << "Assigned RK method: " << this->rk_method_string << std::endl;
}

template <int dim, typename real, typename MeshType> 
void LowStorageRKTableauBase<dim,real, MeshType> :: set_a_and_b ()
{
    // Symbols, equation numbers & sections herein correspond to Ketcheson 2010 paper.

    // Solve for beta and alpha per Section 4.3
    dealii::FullMatrix<double> beta(this->n_rk_stages+1,this->n_rk_stages);
    beta.reinit(this->n_rk_stages+1,this->n_rk_stages);

    dealii::FullMatrix<double> alpha(this->n_rk_stages+1,this->n_rk_stages);
    alpha.reinit(this->n_rk_stages+1, this->n_rk_stages);
   
    for (int i = 1; i < this->n_rk_stages+1; ++i){
        // "beta" as defined in the Ranocha papers coincides with beta(i,i-1) in Ketcheson paper
        beta[i][i-1] = butcher_tableau_beta[i];
        
        // first eq of sec 4.3
        if (i < this->n_rk_stages){
        beta[i+1][i-1] = -1.0 * this->get_gamma(i+1,1) / this->get_gamma(i,1) * beta[i][i-1];
        // second eq of sec 4.3
        alpha[i+1][0] = this->get_gamma(i+1,2) - this->get_gamma(i+1,1) / this->get_gamma(i,1) * this->get_gamma(i,2);
        //third eq of sec 4.3
        alpha[i+1][i-1] = -1.0 * this->get_gamma(i+1,1) / this->get_gamma(i,1) * this->get_gamma(i,0);
        //fourth equation of sec 4.3
        alpha[i+1][i] = 1.0 - alpha[i+1][i-1] - alpha[i+1][0];
        }

    
    }
    
    // eq 9 a
    dealii::FullMatrix<double> identity_m_alpha0(this->n_rk_stages,this->n_rk_stages);
    for (int irow = 0; irow < this->n_rk_stages;++irow){
        for (int icol = 0; icol < this->n_rk_stages;++icol) {
            if (irow == icol) identity_m_alpha0[irow][icol] = 1.0;
            identity_m_alpha0[irow][icol] = identity_m_alpha0[irow][icol] - alpha[irow][icol];
        }
    }
    
    /// invert identity_m_alpha0
    identity_m_alpha0.gauss_jordan();

    // assign beta_0
    dealii::FullMatrix<double> beta_0(this->n_rk_stages,this->n_rk_stages);
    for (int irow = 0; irow < this->n_rk_stages;++irow){
        for (int icol = 0; icol < this->n_rk_stages;++icol) {
            beta_0[irow][icol] = beta[irow][icol];
        }
    }

    dealii::FullMatrix<double> A(this->n_rk_stages,this->n_rk_stages);
    // eq 9a:A = inv(I-alpha_0) * (beta_0)
    identity_m_alpha0.mmult(A,beta_0);

    this->pcout<< "A" << std::endl;
    print_table(A,this->n_rk_stages,this->n_rk_stages);

    // 9 b
    dealii::Vector<double> beta_1(this->n_rk_stages);
    dealii::Vector<double> alpha_1(this->n_rk_stages);
    for (int i = 0; i < this->n_rk_stages; ++i){
        beta_1[i] = beta[this->n_rk_stages][i];
        alpha_1[i] = alpha[this->n_rk_stages][i];
    }

    dealii::Vector<double> b_vec(this->n_rk_stages);
    // b<-- alpha_1*A)
    A.Tvmult(b_vec,alpha_1); // need to transpose all terms!
    // eq 9 b: b = beta_1 + A^T*alpha_1
    b_vec+=beta_1;
    
    // fill the appropriate table with b
    // butcher_tableau_b
    this->butcher_tableau_b.reinit(this->n_rk_stages);
    double sum_b=0.0;
    for (int i = 0; i < this->n_rk_stages; ++i){
        this->butcher_tableau_b[i] = b_vec[i];
        sum_b+=b_vec[i];
    }

    this->butcher_tableau_a.reinit(this->n_rk_stages,this->n_rk_stages);
    for (int i = 0; i < this->n_rk_stages; ++i){
        for (int j = 0; j < this->n_rk_stages; ++j){
            this->butcher_tableau_a[i][j] = A[i][j];
        }
    }

    // Check that sum(b) = 1
    if (abs(sum_b-1.0) > 1E-8){
        this->pcout << "WARNING: Butcher b vector does not sum to 1 !" << std::endl;
    }

    print_table(this->butcher_tableau_b,this->n_rk_stages);

}

template <int dim, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,real, MeshType> :: get_gamma (const int i, const int j) const
{
    return butcher_tableau_gamma[i][j];
}

template <int dim, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,real, MeshType> :: get_beta (const int i) const
{
    return butcher_tableau_beta[i];
}

template <int dim, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,real, MeshType> :: get_delta (const int i) const
{
    return butcher_tableau_delta[i];
}

template <int dim, typename real, typename MeshType> 
double LowStorageRKTableauBase<dim,real, MeshType> :: get_b_hat (const int i) const
{
    return butcher_tableau_b_hat[i];
}

template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class LowStorageRKTableauBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
