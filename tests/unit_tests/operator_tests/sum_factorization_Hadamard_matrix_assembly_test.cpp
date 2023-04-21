#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <bits/stdc++.h>

//#include <ctime>
#include <time.h>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_fe_field.h> 

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "operators/operators.h"

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix_ez.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

const double TOLERANCE = 1E-6;
using namespace std;


int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    all_parameters_new.nstate = nstate;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    bool different = false;
    bool different_mass = false;
    const unsigned int poly_max = 16;
    const unsigned int poly_min = 2;
    std::array<clock_t,poly_max> time_diff_sparse;
    std::array<clock_t,poly_max> time_diff_original;
    for(unsigned int poly_degree=poly_min; poly_degree<poly_max; poly_degree++){

        PHiLiP::OPERATOR::basis_functions<dim,2*dim> basis(nstate,poly_degree, 1);
        dealii::QGauss<1> quad1D (poly_degree+1);
        const dealii::FE_DGQArbitraryNodes<1> fe_dg(quad1D);
        const dealii::FESystem<1,1> fe_system(fe_dg, 1);
        basis.build_1D_volume_operator(fe_system,quad1D);
        basis.build_1D_gradient_operator(fe_system,quad1D);

        const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);
        const unsigned int n_dofs_1D = nstate * (poly_degree+1);
        const unsigned int n_quad_pts_1D = quad1D.size();
        const unsigned int n_quad_pts = pow(n_quad_pts_1D, dim);

        for(unsigned int ielement=0; ielement<10; ielement++){//do several loops as if there were elements

            std::vector<real> sol_hat(n_dofs);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8))) );
            }
            std::vector<double> weights(n_quad_pts_1D);
            dealii::FullMatrix<double> weights_mat(n_quad_pts_1D);
            for(unsigned int idof=0; idof<n_quad_pts_1D; idof++){
                weights[idof] = sqrt( 1e-4 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(1.0-1e-4))) );
                weights_mat[idof][idof] = weights[idof];
            }
            
            dealii::FullMatrix<real> sol_hat_mat(n_dofs);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                    sol_hat_mat[idof][idof2] = sol_hat[idof] * sol_hat[idof2];
                }
            }
             

            //Direction 0
            clock_t tfirst;
            tfirst = clock();
            //get the rows and columns 
            const unsigned int row_size = pow(n_dofs_1D,dim+1);
            const unsigned int col_size = pow(n_dofs_1D,dim+1);
            std::vector<std::array<unsigned int,dim>> Hadamard_rows_sparsity(row_size);
            std::vector<std::array<unsigned int,dim>> Hadamard_columns_sparsity(col_size);
            //extract the dof pairs that give non-zero entries for each direction
            basis.sum_factorized_Hadamard_sparsity_pattern(n_dofs_1D, n_dofs_1D, Hadamard_rows_sparsity, Hadamard_columns_sparsity);

            const unsigned int n_dofs_dim = pow(n_dofs_1D,dim);//should equal n_quad_pts
            std::array<dealii::FullMatrix<real>,dim> sol_hat_sparse;
            for(int idim=0; idim<dim; idim++){
                sol_hat_sparse[idim].reinit(n_dofs_dim, n_dofs_1D);
                //build the n^d x n flux matrix for Hadamard product
                for(unsigned int index=0, counter=0; index<row_size; index++, counter++){
                    if(counter == n_dofs_1D)
                        counter = 0;
                    //create the sparse matrix as if a 2pt flux
                    sol_hat_sparse[idim][Hadamard_rows_sparsity[index][idim]][counter] = sol_hat[Hadamard_rows_sparsity[index][idim]]
                                                                                       * sol_hat[Hadamard_columns_sparsity[index][idim]];
                }
            }
            std::array<dealii::FullMatrix<real>,dim> basis_sparse;
            for(int idim=0; idim<dim; idim++){
                basis_sparse[idim].reinit(n_dofs_dim, n_dofs_1D);
            }
            basis.sum_factorized_Hadamard_basis_assembly(n_dofs_1D, n_dofs_1D, 
                                                         Hadamard_rows_sparsity, Hadamard_columns_sparsity,
                                                         basis.oneD_grad_operator, 
                                                         weights,
                                                         basis_sparse);

            std::array<dealii::FullMatrix<real>,dim> sol_1D;//solution of A*u with sum-factorization
            for(int idim=0; idim<dim; idim++){
                sol_1D[idim].reinit(n_dofs_dim, n_dofs_1D);
                basis.Hadamard_product(basis_sparse[idim], sol_hat_sparse[idim],sol_1D[idim]);
            }
            dealii::Vector<real> ones_1D(n_dofs_1D);
            std::array<dealii::Vector<real>,dim> sol_1D_sum;//solution of A*u with sum-factorization
            for(int idim=0; idim<dim; idim++){
                sol_1D_sum[idim].reinit(n_dofs_dim);
                sol_1D[idim].vmult(sol_1D_sum[idim],ones_1D);
            }


            if(ielement==0)
                time_diff_sparse[poly_degree] = clock() - tfirst;
            else
                time_diff_sparse[poly_degree] += clock() - tfirst;
            //compute A*u using sum-factorization
            time_t tsum;
            tsum = clock();
            std::array<dealii::FullMatrix<real>,dim> basis_dim;
            std::array<dealii::FullMatrix<real>,dim> sol_dim;//solution of A*u normally
            for(int idim=0; idim<dim; idim++){
                basis_dim[idim].reinit(n_quad_pts,n_quad_pts);
                sol_dim[idim].reinit(n_quad_pts,n_quad_pts);
                if(idim==0){
                   // basis_dim[idim] = basis.tensor_product(basis.oneD_grad_operator, basis.oneD_vol_operator, basis.oneD_vol_operator);
                    basis_dim[idim] = basis.tensor_product(basis.oneD_grad_operator, weights_mat, weights_mat);
                }
                if(idim==1){
                    basis_dim[idim] = basis.tensor_product(weights_mat, basis.oneD_grad_operator, weights_mat);
                }
                if(idim==2){
                    basis_dim[idim] = basis.tensor_product(weights_mat, weights_mat, basis.oneD_grad_operator);
                }

                for(unsigned int idof=0; idof< n_quad_pts; idof++){
                    for(unsigned int idof2=0; idof2< n_quad_pts; idof2++){
                        sol_dim[idim][idof][idof2] = sol_hat_mat[idof][idof2]
                                                   * basis_dim[idim][idof][idof2];
                    }
                }
            }
            dealii::Vector<real> ones_dim(n_quad_pts);
            std::array<dealii::Vector<real>,dim> sol_dim_sum;//solution of A*u with sum-factorization
            for(int idim=0; idim<dim; idim++){
                sol_dim_sum[idim].reinit(n_quad_pts);
                sol_dim[idim].vmult(sol_dim_sum[idim],ones_dim);
            }
             
             
            if(ielement==0)
                time_diff_original[poly_degree] = clock() - tsum;
            else
                time_diff_original[poly_degree] += clock() - tsum;
             
            for(int idim=0; idim<dim; idim++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    if(std::abs(sol_dim_sum[idim][iquad] - sol_1D_sum[idim][iquad])>1e-11){
                        pcout<<"sol dim "<<sol_dim_sum[idim][iquad]<<" sol 1D "<<sol_1D_sum[idim][iquad]<<std::endl;
                        different = true;
                    }
                }
            }

            for(int idim=0; idim<dim; idim++){
                for(unsigned int index=0, counter=0; index<row_size; index++, counter++){
                    if(counter == n_dofs_1D)
                        counter = 0;
                    if(abs(sol_dim[idim][Hadamard_rows_sparsity[index][idim]][Hadamard_columns_sparsity[index][idim]]
                        - sol_1D[idim][Hadamard_rows_sparsity[index][idim]][counter]) > 1e-11){
                        pcout<<"sol dim "<<sol_dim[idim][Hadamard_rows_sparsity[index][idim]][Hadamard_columns_sparsity[index][idim]]<<" sol 1D "<<sol_1D[idim][Hadamard_rows_sparsity[index][idim]][counter]<<std::endl;
                        different = true;
                    }
                }
            }

            for(int idim=0; idim<dim; idim++){
                for(unsigned int index=0, counter=0; index<row_size; index++, counter++){
                    if(counter == n_dofs_1D)
                        counter = 0;
                    if(abs(basis_dim[idim][Hadamard_rows_sparsity[index][idim]][Hadamard_columns_sparsity[index][idim]]
                        - basis_sparse[idim][Hadamard_rows_sparsity[index][idim]][counter]) > 1e-11){
                        pcout<<"basis dim "<<basis_dim[idim][Hadamard_rows_sparsity[index][idim]][Hadamard_columns_sparsity[index][idim]]<<" basis sparse "<<basis_sparse[idim][Hadamard_rows_sparsity[index][idim]][counter]<<std::endl;
                        different = true;
                    }
                }
            }

        }//end of element loop
    }//end of poly_degree loop

    double first_slope = std::log(((double)time_diff_sparse[poly_max-1]/CLOCKS_PER_SEC) / ((double)time_diff_sparse[poly_max-2]/CLOCKS_PER_SEC))
                        / std::log((double)((poly_max-1.0)/(poly_max-2.0)));//technically over (p+1) since (p+1)^dim basis functions 
    double sum_slope = std::log(((double)time_diff_original[poly_max-1]/CLOCKS_PER_SEC) /( (double)time_diff_original[poly_max-2]/CLOCKS_PER_SEC))
                        / std::log((double)((poly_max-1.0)/(poly_max-2.0))); 

    const double sum_factorized_slope_mpi = (dealii::Utilities::MPI::max(first_slope, MPI_COMM_WORLD));
    const double original_slope_mpi = (dealii::Utilities::MPI::max(sum_slope, MPI_COMM_WORLD));

    double avg_slope1 = 0.0;
    pcout<<"Times for operation A*u"<<std::endl;
    pcout<<"Normal operation A*u  | Slope |  "<<"Sum factorization | Slope "<<std::endl;
    for(unsigned int i=poly_min+1; i<poly_max; i++){
        pcout<<(double)time_diff_sparse[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_diff_sparse[i]/CLOCKS_PER_SEC) / ((double)time_diff_sparse[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<" "<<
        (double)time_diff_original[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_diff_original[i]/CLOCKS_PER_SEC) /( (double)time_diff_original[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<
        std::endl;
        if(i>poly_max-4){
            avg_slope1 += std::log(((double)time_diff_sparse[i]/CLOCKS_PER_SEC) /( (double)time_diff_sparse[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)));
        }
    }
    avg_slope1 /= (3.0);

    pcout<<" sum-factorization slope "<<sum_factorized_slope_mpi<<" regular original slope "<<original_slope_mpi<<std::endl;

    pcout<<"average slope 1 "<<avg_slope1<<std::endl;


    if(different==true){
        pcout<<"Sum factorization not recover same vector for A*u."<<std::endl;
        return 1;
    }
    if(different_mass==true){
        pcout<<"Sum factorization not recover same vector Mass*u."<<std::endl;
        return 1;
    }
    if(avg_slope1 > dim+1.6){
        pcout<<"Sum factorization not give correct comp cost slope."<<std::endl;
        pcout<<"average slope 1 "<<avg_slope1<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main

