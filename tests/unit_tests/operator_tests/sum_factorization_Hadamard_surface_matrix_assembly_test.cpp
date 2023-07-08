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
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    bool different = false;
    bool different_mass = false;
    const unsigned int poly_max = 16;
    const unsigned int poly_min = 2;
    std::array<clock_t,poly_max> time_diff_sparse;
    std::array<clock_t,poly_max> time_diff_original;
    for(unsigned int poly_degree=poly_min; poly_degree<poly_max; poly_degree++){

        PHiLiP::OPERATOR::basis_functions<dim,2*dim> basis(1,poly_degree, 1);
        PHiLiP::OPERATOR::local_mass<dim,2*dim> mass(1, poly_degree, 1);
        dealii::QGauss<1> quad1D (poly_degree+1);
        dealii::QGauss<0> quad1D_surf (poly_degree+1);
        const dealii::FE_DGQArbitraryNodes<1> fe_dg(quad1D);
        const dealii::FESystem<1,1> fe_system(fe_dg, 1);
        basis.build_1D_surface_operator(fe_system,quad1D_surf);
        mass.build_1D_volume_operator(fe_system,quad1D);
        basis.build_1D_volume_operator(fe_system,quad1D);
        basis.build_1D_gradient_operator(fe_system,quad1D);
        const std::vector<real> &weights = quad1D.get_weights();

        const unsigned int n_quad_pts_1D = quad1D.size();
        const unsigned int n_quad_pts = pow(n_quad_pts_1D, dim);
        const unsigned int n_face_quad_pts = pow(n_quad_pts_1D, dim-1);
        const int n_faces = 2 * dim;
        const unsigned int n_elements = (dim ==3 ) ? 50 : 10000;

        for(unsigned int ielement=0; ielement<n_elements; ielement++){//do several loops as if there were elements
            for(int iface=0; iface<n_faces; iface++){

                const int iface_1D = iface % 2;
                const int dim_not_zero = iface / 2;

                std::vector<real> sol_hat(n_quad_pts);
                for(unsigned int idof=0; idof<n_quad_pts; idof++){
                    sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8))) );
                }
                
                dealii::FullMatrix<real> sol_hat_mat(n_face_quad_pts, n_quad_pts);
                for(unsigned int idof=0; idof<n_face_quad_pts; idof++){
                    for(unsigned int idof2=0; idof2<n_quad_pts; idof2++){
                        sol_hat_mat[idof][idof2] = sol_hat[idof] * sol_hat[idof2];
                    }
                }
                 
                 
                //Direction 0
                clock_t tfirst;
                tfirst = clock();
                //For surface, size only need to be n_face_quad by n
                //get the rows and columns 
                const unsigned int row_size = n_face_quad_pts * n_quad_pts_1D;
                const unsigned int col_size = n_face_quad_pts * n_quad_pts_1D;
                std::vector<unsigned int> Hadamard_rows_sparsity(row_size);
                std::vector<unsigned int> Hadamard_columns_sparsity(col_size);
                //extract the dof pairs that give non-zero entries for each direction
                basis.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D, Hadamard_rows_sparsity, Hadamard_columns_sparsity, dim_not_zero);
                 
                dealii::FullMatrix<real> sol_hat_sparse(n_face_quad_pts, n_quad_pts_1D);
                //build the n^d x n flux matrix for Hadamard product
                for(unsigned int index=0, counter=0; index<row_size; index++, counter++){
                    if(counter == n_quad_pts_1D)
                        counter = 0;
                    //create the sparse matrix as if a 2pt flux
                    sol_hat_sparse[Hadamard_rows_sparsity[index]][counter] = sol_hat[Hadamard_rows_sparsity[index]]
                                                                                       * sol_hat[Hadamard_columns_sparsity[index]];
                }

                dealii::FullMatrix<real> basis_sparse(n_face_quad_pts, n_quad_pts_1D);
                basis.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D, 
                                                                     Hadamard_rows_sparsity, Hadamard_columns_sparsity,
                                                                     basis.oneD_surf_operator[iface_1D], 
                                                                     weights,
                                                                     basis_sparse,
                                                                     dim_not_zero);
                 
                dealii::FullMatrix<real> sol_1D(n_face_quad_pts, n_quad_pts_1D);//solution of A*u with sum-factorization
                basis.Hadamard_product(basis_sparse, sol_hat_sparse,sol_1D);

                std::vector<real> sol_1D_sum_surf(n_face_quad_pts);//solution of A*u with sum-factorization
                std::vector<real> sol_1D_sum_vol(n_quad_pts);//solution of A*u with sum-factorization
                for(unsigned int iface_quad=0; iface_quad<n_face_quad_pts; iface_quad++){
                    for(unsigned int iquad=0; iquad<n_quad_pts_1D; iquad++){
                        sol_1D_sum_surf[iface_quad] -= sol_1D[iface_quad][iquad];
                        sol_1D_sum_vol[Hadamard_columns_sparsity[iface_quad * n_quad_pts_1D + iquad]] += sol_1D[iface_quad][iquad];
                    }
                }
                 
                 
                if(ielement==0)
                    time_diff_sparse[poly_degree] = clock() - tfirst;
                else
                    time_diff_sparse[poly_degree] += clock() - tfirst;
                //compute A*u using sum-factorization
                time_t tsum;
                tsum = clock();
                dealii::FullMatrix<real> surf_basis_dim(n_face_quad_pts,n_quad_pts);
                if(dim_not_zero==0){
                    surf_basis_dim = basis.tensor_product(basis.oneD_surf_operator[iface_1D], mass.oneD_vol_operator, mass.oneD_vol_operator);
                }
                if(dim_not_zero==1){
                    surf_basis_dim = basis.tensor_product(mass.oneD_vol_operator, basis.oneD_surf_operator[iface_1D], mass.oneD_vol_operator);
                }
                if(dim_not_zero==2){
                    surf_basis_dim = basis.tensor_product(mass.oneD_vol_operator, mass.oneD_vol_operator, basis.oneD_surf_operator[iface_1D]);
                }
                
                dealii::FullMatrix<real> surf_sol_dim(n_face_quad_pts,n_quad_pts);//solution of A*u normally
                for(unsigned int idof=0; idof< n_face_quad_pts; idof++){
                    for(unsigned int idof2=0; idof2< n_quad_pts; idof2++){
                        surf_sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                                   * surf_basis_dim[idof][idof2];
                    }
                }
                 
                std::vector<real> surf_sol_dim_surf(n_face_quad_pts);//solution of A*u normally
                std::vector<real> surf_sol_dim_vol(n_quad_pts);//solution of A*u normally
                for(unsigned int iface_quad=0; iface_quad<n_face_quad_pts; iface_quad++){
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        surf_sol_dim_surf[iface_quad] -= surf_sol_dim[iface_quad][iquad];
                        surf_sol_dim_vol[iquad] += surf_sol_dim[iface_quad][iquad];
                    }
                }
                 
                if(ielement==0)
                    time_diff_original[poly_degree] = clock() - tsum;
                else
                    time_diff_original[poly_degree] += clock() - tsum;
                 
                for(unsigned int index=0, counter=0; index<row_size; index++, counter++){
                    if(counter == n_quad_pts_1D)
                        counter = 0;
                    if(abs(sol_hat_mat[Hadamard_rows_sparsity[index]][Hadamard_columns_sparsity[index]]
                        - sol_hat_sparse[Hadamard_rows_sparsity[index]][counter]) > 1e-11){
                        pcout<<"sparsity pattern is wrong"<<std::endl;
                        pcout<<"sol hat "<<sol_hat_mat[Hadamard_rows_sparsity[index]][Hadamard_columns_sparsity[index]]<<" sol sparse "<<sol_hat_sparse[Hadamard_rows_sparsity[index]][counter]<<std::endl;
                        different = true;
                    }
                }
                for(unsigned int index=0, counter=0; index<row_size; index++, counter++){
                    if(counter == n_quad_pts_1D)
                        counter = 0;
                    if(abs(surf_sol_dim[Hadamard_rows_sparsity[index]][Hadamard_columns_sparsity[index]]
                        - sol_1D[Hadamard_rows_sparsity[index]][counter]) > 1e-11){
                        pcout<<"index "<<index<<" row "<<Hadamard_rows_sparsity[index]<<" column "<<Hadamard_columns_sparsity[index]<<" counter "<<counter<<std::endl;
                        pcout<<"direction "<<dim_not_zero<<std::endl;
                        pcout<<"sol dim "<<surf_sol_dim[Hadamard_rows_sparsity[index]][Hadamard_columns_sparsity[index]]<<" sol 1D "<<sol_1D[Hadamard_rows_sparsity[index]][counter]<<std::endl;
                        different = true;
                    }
                }
                 
                for(unsigned int index=0, counter=0; index<row_size; index++, counter++){
                    if(counter == n_quad_pts_1D)
                        counter = 0;
                    if(abs(surf_basis_dim[Hadamard_rows_sparsity[index]][Hadamard_columns_sparsity[index]]
                        - basis_sparse[Hadamard_rows_sparsity[index]][counter]) > 1e-11){
                        pcout<<"index "<<index<<" row "<<Hadamard_rows_sparsity[index]<<" column "<<Hadamard_columns_sparsity[index]<<" counter "<<counter<<std::endl;
                        pcout<<"direction "<<dim_not_zero<<std::endl;
                        pcout<<"basis dim "<<surf_basis_dim[Hadamard_rows_sparsity[index]][Hadamard_columns_sparsity[index]]<<" basis sparse "<<basis_sparse[Hadamard_rows_sparsity[index]][counter]<<std::endl;
                        different = true;
                    }
                }
                for(unsigned int i=0; i<n_face_quad_pts; i++){
                    if(abs(surf_sol_dim_surf[i] - sol_1D_sum_surf[i])>1e-11){
                        pcout<<"the surface sum wrong for "<<surf_sol_dim_surf[i]<<" vs sum factorized "<<sol_1D_sum_surf[i]<<std::endl;
                        different = true;
                    }
                }
                for(unsigned int i=0; i<n_quad_pts; i++){
                    if(abs(surf_sol_dim_vol[i] - sol_1D_sum_vol[i])>1e-11){
                        pcout<<"the vol sum wrong for "<<surf_sol_dim_vol[i]<<" vs sum factorized "<<sol_1D_sum_vol[i]<<std::endl;
                        different = true;
                    }
                }


            }//end of face loop
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
    if(avg_slope1 > dim+0.6){
        pcout<<"Sum factorization not give correct comp cost slope. For surface, converge at n^d as compared to original conerge at n^{2d-1}"<<std::endl;
        pcout<<"average slope 1 "<<avg_slope1<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main

