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
    std::array<clock_t,poly_max> time_diff;
    std::array<clock_t,poly_max> time_diff_sum;
    std::array<clock_t,poly_max> time_deriv_sum_cons;
    std::array<clock_t,poly_max> time_diff_dir2;
    std::array<clock_t,poly_max> time_diff_sum_dir2;
    std::array<clock_t,poly_max> time_diff_dir3;
    std::array<clock_t,poly_max> time_diff_sum_dir3;
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

        std::vector<double> ones(n_quad_pts_1D, 1.0);//to be used as the weights

        for(unsigned int ielement=0; ielement<10; ielement++){//do several loops as if there were elements

            std::vector<real> sol_hat(n_dofs);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8))) );
            }
            
            dealii::FullMatrix<real> sol_hat_mat(n_dofs);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                    sol_hat_mat[idof][idof2] = sol_hat[idof] * sol_hat[idof2];
                }
            }
             
            dealii::FullMatrix<real> sol_dim(n_quad_pts);//solution of A*u normally
            dealii::FullMatrix<real> sol_1D(n_quad_pts);//solution of A*u with sum-factorization
            dealii::FullMatrix<real> basis_dim(n_quad_pts);//solution of A*u with sum-factorization
            basis_dim = basis.tensor_product(basis.oneD_grad_operator, basis.oneD_vol_operator, basis.oneD_vol_operator);
             
            //Compute A*u normally
            clock_t tfirst;
            tfirst = clock();
            if(dim==2){
                for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D; idof++){
                    for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D; idof2++){
                    sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                         * basis_dim[idof][idof2];
                    }
                }
            }
            std::cout<<std::endl<<std::endl;
            if(dim==3){
                for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
                    for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
                    sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                         * basis_dim[idof][idof2];
                    }
                }
            }
            if(ielement==0)
                time_diff[poly_degree] = clock() - tfirst;
            else
                time_diff[poly_degree] += clock() - tfirst;
             
            //compute A*u using sum-factorization
            time_t tsum;
            tsum = clock();
            basis.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, basis.oneD_grad_operator, ones, 0);
             
            if(ielement==0)
                time_diff_sum[poly_degree] = clock() - tsum;
            else
                time_diff_sum[poly_degree] += clock() - tsum;
             
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                    if(std::abs(sol_dim[iquad][iquad2] - sol_1D[iquad][iquad2])>1e-11){
                        pcout<<"sol dim "<<sol_dim[iquad][iquad2]<<" sol 1D "<<sol_1D[iquad][iquad2]<<std::endl;
                        different = true;
                    }
                }
            }
             
            //do sum factorization derivative
            std::vector<real> sol_deriv_time(n_quad_pts);//solution of A*u with sum-factorization
            time_t tderiv_sum_cons;
            tderiv_sum_cons = clock();
            basis.matrix_vector_mult(sol_hat, sol_deriv_time, 
                                     basis.oneD_vol_operator,
                                     basis.oneD_vol_operator,
                                     basis.oneD_grad_operator);
             
            if(ielement==0)
                time_deriv_sum_cons[poly_degree] = clock() - tderiv_sum_cons;
            else
                time_deriv_sum_cons[poly_degree] += clock() - tderiv_sum_cons;
             
            //check the other 2 directions match
            basis_dim = basis.tensor_product(basis.oneD_vol_operator, basis.oneD_grad_operator, basis.oneD_vol_operator);
             
            clock_t tfirst_dir2;
            tfirst_dir2 = clock();
            if(dim==2){
                for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D; idof++){
                    for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D; idof2++){
                    sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                         * basis_dim[idof][idof2];
                    }
                }
            }
            std::cout<<std::endl<<std::endl;
            if(dim==3){
                for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
                    for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
                    sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                         * basis_dim[idof][idof2];
                    }
                }
            }
            if(ielement==0)
                time_diff_dir2[poly_degree] = clock() - tfirst_dir2;
            else
                time_diff_dir2[poly_degree] += clock() - tfirst_dir2;
             
            sol_1D.reinit(n_quad_pts, n_quad_pts);
             
            time_t tsum_dir2;
            tsum_dir2 = clock();
            basis.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, basis.oneD_grad_operator, ones, 1);
            if(ielement==0)
                time_diff_sum_dir2[poly_degree] = clock() - tsum_dir2;
            else
                time_diff_sum_dir2[poly_degree] += clock() - tsum_dir2;
             
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                    if(std::abs(sol_dim[iquad][iquad2] - sol_1D[iquad][iquad2])>1e-11){
                        pcout<<"DIR 2 sol dim "<<sol_dim[iquad][iquad2]<<" sol 1D "<<sol_1D[iquad][iquad2]<<std::endl;
                        different = true;
                    }
                }
            }
             
            if(dim ==3){
                //dir 3
                basis_dim = basis.tensor_product(basis.oneD_vol_operator, basis.oneD_vol_operator, basis.oneD_grad_operator);
                clock_t tfirst_dir3;
                tfirst_dir3 = clock();
                if(dim==2){
                    for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D; idof++){
                        for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D; idof2++){
                        sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                             * basis_dim[idof][idof2];
                        }
                    }
                }
                std::cout<<std::endl<<std::endl;
                if(dim==3){
                    for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
                        for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
                        sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                             * basis_dim[idof][idof2];
                        }
                    }
                }
                std::cout<<std::endl<<std::endl;
                if(ielement==0)
                    time_diff_dir3[poly_degree] = clock() - tfirst_dir3;
                else
                    time_diff_dir3[poly_degree] += clock() - tfirst_dir3;
                 
                sol_1D.reinit(n_quad_pts, n_quad_pts);
                 
                time_t tsum_dir3;
                tsum_dir3 = clock();
                basis.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, basis.oneD_grad_operator, ones, 2);
                if(ielement==0)
                    time_diff_sum_dir3[poly_degree] = clock() - tsum_dir3;
                else
                    time_diff_sum_dir3[poly_degree] += clock() - tsum_dir3;
                 
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                        if(std::abs(sol_dim[iquad][iquad2] - sol_1D[iquad][iquad2])>1e-11){
                            different = true;
                        }
                    }
                }
            }

        }//end of element loop
    }//end of poly_degree loop

    double first_slope = std::log(((double)time_diff[poly_max-1]/CLOCKS_PER_SEC) / ((double)time_diff[poly_max-2]/CLOCKS_PER_SEC))
                        / std::log((double)((poly_max-1.0)/(poly_max-2.0)));//technically over (p+1) since (p+1)^dim basis functions 
    double sum_slope = std::log(((double)time_diff_sum[poly_max-1]/CLOCKS_PER_SEC) /( (double)time_diff_sum[poly_max-2]/CLOCKS_PER_SEC))
                        / std::log((double)((poly_max-1.0)/(poly_max-2.0))); 

    const double first_slope_mpi = (dealii::Utilities::MPI::max(first_slope, MPI_COMM_WORLD));
    const double sum_slope_mpi = (dealii::Utilities::MPI::max(sum_slope, MPI_COMM_WORLD));

    double avg_slope1 = 0.0;
    pcout<<"Times for operation A*u"<<std::endl;
    pcout<<"Normal operation A*u  | Slope |  "<<"Sum factorization | Slope "<<std::endl;
    for(unsigned int i=poly_min+1; i<poly_max; i++){
        pcout<<(double)time_diff[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_diff[i]/CLOCKS_PER_SEC) / ((double)time_diff[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<" "<<
        (double)time_diff_sum[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_diff_sum[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<
        std::endl;
        if(i>poly_max-4){
            avg_slope1 += std::log(((double)time_diff_sum[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)));
        }
    }
    avg_slope1 /= (3.0);

    pcout<<" regular slope "<<first_slope_mpi<<" sum-factorization slope "<<sum_slope_mpi<<std::endl;

    double avg_slope2 = 0.0;
    pcout<<"Times for operation A*u in Direction y"<<std::endl;
    pcout<<"Normal operation A*u  | Slope |  "<<"Sum factorization | Slope "<<std::endl;
    for(unsigned int i=poly_min+1; i<poly_max; i++){
        pcout<<(double)time_diff_dir2[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_diff_dir2[i]/CLOCKS_PER_SEC) / ((double)time_diff_dir2[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<" "<<
        (double)time_diff_sum_dir2[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_diff_sum_dir2[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum_dir2[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<
        std::endl;
        if(i>poly_max-4){
            avg_slope2 += std::log(((double)time_diff_sum_dir2[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum_dir2[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)));
        }
    }
    avg_slope2 /= (3.0);

    double avg_slope3 = 0.0;
    if(dim==3){
        pcout<<"Times for operation A*u in Direction z"<<std::endl;
        pcout<<"Normal operation A*u  | Slope |  "<<"Sum factorization | Slope "<<std::endl;
        for(unsigned int i=poly_min+1; i<poly_max; i++){
            pcout<<(double)time_diff_dir3[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_diff_dir3[i]/CLOCKS_PER_SEC) / ((double)time_diff_dir3[i-1]/CLOCKS_PER_SEC))
                            / std::log((double)((i)/(i-1.0)))<<" "<<
            (double)time_diff_sum_dir3[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_diff_sum_dir3[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum_dir3[i-1]/CLOCKS_PER_SEC))
                            / std::log((double)((i)/(i-1.0)))<<
            std::endl;
            if(i>poly_max-4){
                avg_slope3 += std::log(((double)time_diff_sum_dir3[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum_dir3[i-1]/CLOCKS_PER_SEC))
                            / std::log((double)((i)/(i-1.0)));
            }
        }
            avg_slope3 /= (3.0);

    }
    pcout<<"average slope 1 "<<avg_slope1<<" average slope 2 "<<avg_slope2<<" average slope 3 "<<avg_slope3<<std::endl;

    //output sum factorization derivative slope
    pcout<<"Sum factorization Direct Conservative A*u  | Slope "<<std::endl;
    for(unsigned int i=poly_min+1; i<poly_max; i++){
        pcout<<(double)time_deriv_sum_cons[i]/CLOCKS_PER_SEC<<" "<<std::log(((double)time_deriv_sum_cons[i]/CLOCKS_PER_SEC) / ((double)time_deriv_sum_cons[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<std::endl;
    }


    if(different==true){
        pcout<<"Sum factorization not recover same vector for A*u."<<std::endl;
        return 1;
    }
    if(different_mass==true){
        pcout<<"Sum factorization not recover same vector Mass*u."<<std::endl;
        return 1;
    }
    if(avg_slope1 > dim+1.8 || avg_slope2 > dim+1.8 ||avg_slope3 > dim+1.8){
        //check if because of random number generator, take one more value for average since should converge by this order.
        avg_slope1 = 0.0;
        avg_slope2 = 0.0;
        avg_slope3 = 0.0;
        for(unsigned int i=poly_max-5; i<poly_max; i++){
                avg_slope1 += std::log(((double)time_diff_sum[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum[i-1]/CLOCKS_PER_SEC))
                           // / std::log((double)((i)/(i-1.0)));
                            / std::log((double)((i+1.0)/(i)));
                avg_slope2 += std::log(((double)time_diff_sum_dir2[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum_dir2[i-1]/CLOCKS_PER_SEC))
                           // / std::log((double)((i)/(i-1.0)));
                            / std::log((double)((i+1.0)/(i)));
                if constexpr(dim==3){
                    avg_slope3 += std::log(((double)time_diff_sum_dir3[i]/CLOCKS_PER_SEC) /( (double)time_diff_sum_dir3[i-1]/CLOCKS_PER_SEC))
                             //   / std::log((double)((i)/(i-1.0)));
                                / std::log((double)((i+1.0)/(i)));
                }
        }
        avg_slope1 /= 4.0;
        avg_slope2 /= 4.0;
        if constexpr(dim==3){
            avg_slope3 /= 4.0;
        }

        if(avg_slope1 > dim+1.8 || avg_slope2 > dim+1.8 ||avg_slope3 > dim+1.8){
            pcout<<"Sum factorization not give correct comp cost slope."<<std::endl;
            pcout<<"average slope 1 "<<avg_slope1<<" average slope 2 "<<avg_slope2<<" average slope 3 "<<avg_slope3<<std::endl;
            return 1;
        }
    }
    else{
        return 0;
    }

}//end of main

