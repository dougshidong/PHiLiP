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

template <typename real>
void build_mass_matrix( const unsigned int n_dofs_1D, 
                        const unsigned int dimension,
                        const dealii::FullMatrix<real> &mass_1D,
                        dealii::FullMatrix<real> &mass)
{
    if(dimension==2){
        for(unsigned int jdof=0; jdof<n_dofs_1D; jdof++){
            for(unsigned int kdof=0; kdof<n_dofs_1D; kdof++){
                for(unsigned int ndof=0; ndof<n_dofs_1D; ndof++){
                    for(unsigned int odof=0; odof<n_dofs_1D; odof++){
                        const unsigned int index_row = n_dofs_1D*jdof + kdof;
                        const unsigned int index_col = n_dofs_1D*ndof + odof;
                        mass[index_row][index_col] = mass_1D[jdof][ndof] * mass_1D[kdof][odof];
                    }
                }
            }
        }
    }
    if(dimension==3){
        for(unsigned int idof=0; idof<n_dofs_1D; idof++){
            for(unsigned int jdof=0; jdof<n_dofs_1D; jdof++){
                for(unsigned int kdof=0; kdof<n_dofs_1D; kdof++){
                    for(unsigned int mdof=0; mdof<n_dofs_1D; mdof++){
                        for(unsigned int ndof=0; ndof<n_dofs_1D; ndof++){
                            for(unsigned int odof=0; odof<n_dofs_1D; odof++){
                                const unsigned int index_row = pow(n_dofs_1D,2)*idof + n_dofs_1D*jdof + kdof;
                                const unsigned int index_col = pow(n_dofs_1D,2)*mdof + n_dofs_1D*ndof + odof;
                                mass[index_row][index_col] = mass_1D[idof][mdof] * mass_1D[jdof][ndof] * mass_1D[kdof][odof];
                            }
                        }
                    }
                }
            }
        }
    }
}

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
//    all_parameters_new.use_collocated_nodes = true;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    bool different = false;
    bool different_mass = false;
    const unsigned int poly_max = 18;
    const unsigned int poly_min = 2;
    std::array<clock_t,poly_max> time_diff;
    std::array<clock_t,poly_max> time_diff_sum;
    std::array<clock_t,poly_max> time_diff_mass;
    std::array<clock_t,poly_max> time_diff_mass_sum;
    for(unsigned int poly_degree=poly_min; poly_degree<poly_max; poly_degree++){

        PHiLiP::OPERATOR::local_mass<dim,2*dim> mass_matrix(nstate, poly_degree, 1);
        PHiLiP::OPERATOR::basis_functions<dim,2*dim> basis(nstate, poly_degree, 1);
        dealii::QGauss<1> quad1D (poly_degree+1);
        const dealii::FE_DGQ<1> fe_dg(poly_degree);
        const dealii::FESystem<1,1> fe_system(fe_dg, nstate);
        mass_matrix.build_1D_volume_operator(fe_system,quad1D);
        basis.build_1D_volume_operator(fe_system,quad1D);

        const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);
        const unsigned int n_dofs_1D = nstate * (poly_degree+1);
        const unsigned int n_quad_pts_1D = quad1D.size();
        const unsigned int n_quad_pts = pow(n_quad_pts_1D, dim);

        for(unsigned int ielement=0; ielement<6; ielement++){//do several loops as if there were elements

        std::vector<real> sol_hat(n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            sol_hat[idof] = 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8)));
        }
        std::vector<real> sol_dim(n_quad_pts);//solution of A*u normally
        std::vector<real> sol_1D(n_quad_pts);//solution of A*u with sum-factorization
        std::vector<real> sol_mass(n_dofs);//solution of M*u normally
        std::vector<real> sol_mass_sum(n_dofs);//solution of M*u with sum-factorization

        //Compute A*u normally
        clock_t tfirst;
        tfirst = clock();
        if(dim==2){
            for(unsigned int jquad=0; jquad<n_quad_pts_1D; jquad++){
                for(unsigned int kquad=0; kquad<n_quad_pts_1D; kquad++){
                    const unsigned int quad_index = jquad*n_quad_pts_1D + kquad;
                    sol_dim[quad_index] = 0.0;
                    for(unsigned int jdof=0; jdof<n_dofs_1D; jdof++){
                        for(unsigned int kdof=0; kdof<n_dofs_1D; kdof++){
                            const unsigned int dof_index = jdof*n_dofs_1D + kdof;
                            sol_dim[quad_index] += sol_hat[dof_index]
                                                 * basis.oneD_vol_operator[jquad][jdof]
                                                 * basis.oneD_vol_operator[kquad][kdof];
                        }
                    }
                }
            }
        }
        if(dim==3){
            for(unsigned int iquad=0; iquad<n_quad_pts_1D; iquad++){
                for(unsigned int jquad=0; jquad<n_quad_pts_1D; jquad++){
                    for(unsigned int kquad=0; kquad<n_quad_pts_1D; kquad++){
                        const unsigned int quad_index = iquad * pow(n_quad_pts_1D,2) + jquad*n_quad_pts_1D + kquad;
                        sol_dim[quad_index] = 0.0;
                        for(unsigned int idof=0; idof<n_dofs_1D; idof++){
                            for(unsigned int jdof=0; jdof<n_dofs_1D; jdof++){
                                for(unsigned int kdof=0; kdof<n_dofs_1D; kdof++){
                                    const unsigned int dof_index = idof * pow(n_dofs_1D,2) + jdof*n_dofs_1D + kdof;
                                    sol_dim[quad_index] += sol_hat[dof_index]
                                                         * basis.oneD_vol_operator[iquad][idof]
                                                         * basis.oneD_vol_operator[jquad][jdof]
                                                         * basis.oneD_vol_operator[kquad][kdof];
                                }
                            }
                        }
                    }
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
        basis.matrix_vector_mult(sol_hat, sol_1D, basis.oneD_vol_operator, basis.oneD_vol_operator, basis.oneD_vol_operator);

        if(ielement==0)
            time_diff_sum[poly_degree] = clock() - tsum;
        else
            time_diff_sum[poly_degree] += clock() - tsum;

        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        //    pcout<<"first "<<sol_dim[iquad]<<" second "<<sol_1D[iquad]<<std::endl;
           if(std::abs(sol_dim[iquad] - sol_1D[iquad])>1e-11)
               different = true;
        }


        //Mass matrix check now. So we explicitly compute Mass_matrix*u by 1) building the 3D mass matrix and doing M*u normally, 
        //and 2) using M*u=\chi^T* W *\chi *u, we do sum-factorization in 2 steps: a) y = \chi*u, then b) \chi^T * W * y 
        //First way
        //build 3D mass matrix
        time_t tmass;
        tmass = clock();
        dealii::FullMatrix<real> mass(n_dofs);
        mass = mass_matrix.tensor_product_state(nstate,
                                                mass_matrix.oneD_vol_operator,
                                                mass_matrix.oneD_vol_operator,mass_matrix.oneD_vol_operator);
        //compute M*u
        for(unsigned int idof=0; idof<n_dofs; idof++){
            sol_mass[idof] = 0.0;
            for(unsigned int jdof=0; jdof<n_dofs; jdof++){
                sol_mass[idof] += mass[idof][jdof] * sol_hat[jdof];
            }
        }
        if(ielement==0)
            time_diff_mass[poly_degree] = clock() - tmass;
        else
            time_diff_mass[poly_degree] += clock() - tmass;
        
        //second way
        time_t tmass_sum;
        tmass_sum = clock();
        const std::vector<real> &quad_weights_1D = quad1D.get_weights(); 
        std::vector<real> quad_weights(n_quad_pts);
        //get 2D or 3D quad weights from 1D
        if(dim==2){
            for(unsigned int iquad=0; iquad<n_quad_pts_1D; iquad++){
                for(unsigned int jquad=0; jquad<n_quad_pts_1D; jquad++){
                    quad_weights[iquad*n_quad_pts_1D + jquad] = quad_weights_1D[iquad] * quad_weights_1D[jquad];
                }
            }
        }
        if(dim==3){
            for(unsigned int iquad=0; iquad<n_quad_pts_1D; iquad++){
                for(unsigned int jquad=0; jquad<n_quad_pts_1D; jquad++){
                    for(unsigned int kquad=0; kquad<n_quad_pts_1D; kquad++){
                        const unsigned int index_quad = iquad*pow(n_quad_pts_1D,2) + jquad*n_quad_pts_1D + kquad;
                        quad_weights[index_quad] = quad_weights_1D[iquad] * quad_weights_1D[jquad] * quad_weights_1D[kquad];
                    }
                }
            }
        }
        std::vector<real> interm_step(n_quad_pts);
        //matrix-vect oper
        basis.matrix_vector_mult(sol_hat, interm_step, basis.oneD_vol_operator, basis.oneD_vol_operator, basis.oneD_vol_operator);
        //inner prod
        basis.inner_product(interm_step, quad_weights, sol_mass_sum, basis.oneD_vol_operator, basis.oneD_vol_operator, basis.oneD_vol_operator);
        if(ielement==0)
            time_diff_mass_sum[poly_degree] = clock() - tmass_sum;
        else
            time_diff_mass_sum[poly_degree] += clock() - tmass_sum;

        for(unsigned int iquad=0; iquad<n_dofs; iquad++){
        //    pcout<<"first "<<sol_mass[iquad]<<" second "<<sol_mass_sum[iquad]<<std::endl;
           if(std::abs(sol_mass[iquad] - sol_mass_sum[iquad])>1e-11)
               different_mass = true;
        }

        }//end of element loop
    }//end of poly_degree loop

    double first_slope = std::log(((float)time_diff[poly_max-1]/CLOCKS_PER_SEC) / ((float)time_diff[poly_max-2]/CLOCKS_PER_SEC))
                        / std::log((double)((poly_max)/(poly_max-1.0)));//technically over (p+1) since (p+1)^dim basis functions 
                       // / ((double)((poly_max)/(poly_max-1.0)));//technically over (p+1) since (p+1)^dim basis functions 
    double sum_slope = std::log(((float)time_diff_sum[poly_max-1]/CLOCKS_PER_SEC) /( (float)time_diff_sum[poly_max-2]/CLOCKS_PER_SEC))
                        / std::log((double)((poly_max)/(poly_max-1.0))); 
                        /// ((double)((poly_max)/(poly_max-1.0))); 
    double first_slope_mass = std::log(((float)time_diff_mass[poly_max-1]/CLOCKS_PER_SEC) / ((float)time_diff_mass[poly_max-2]/CLOCKS_PER_SEC))
                        / std::log((double)((poly_max)/(poly_max-1.0)));//technically over (p+1) since (p+1)^dim basis functions 
                       // / ((double)((poly_max)/(poly_max-1.0)));//technically over (p+1) since (p+1)^dim basis functions 
    double sum_slope_mass = std::log(((float)time_diff_mass_sum[poly_max-1]/CLOCKS_PER_SEC) /( (float)time_diff_mass_sum[poly_max-2]/CLOCKS_PER_SEC))
                        / std::log((double)((poly_max)/(poly_max-1.0))); 
                        /// ((double)((poly_max)/(poly_max-1.0))); 

    const double first_slope_mpi = (dealii::Utilities::MPI::max(first_slope, MPI_COMM_WORLD));
    const double sum_slope_mpi = (dealii::Utilities::MPI::max(sum_slope, MPI_COMM_WORLD));
    const double first_slope_mpi_mass = (dealii::Utilities::MPI::max(first_slope_mass, MPI_COMM_WORLD));
    const double sum_slope_mpi_mass = (dealii::Utilities::MPI::max(sum_slope_mass, MPI_COMM_WORLD));

    pcout<<"Times for operation A*u"<<std::endl;
    pcout<<"Normal operation A*u  | Slope |  "<<"Sum factorization | Slope "<<std::endl;
    double avg_slope = 0.0;
    for(unsigned int i=poly_min+1; i<poly_max; i++){
        pcout<<(float)time_diff[i]/CLOCKS_PER_SEC<<" "<<std::log(((float)time_diff[i]/CLOCKS_PER_SEC) / ((float)time_diff[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<" "<<
        (float)time_diff_sum[i]/CLOCKS_PER_SEC<<" "<<std::log(((float)time_diff_sum[i]/CLOCKS_PER_SEC) /( (float)time_diff_sum[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)))<<
        std::endl;
        if(i>poly_max-4){
            avg_slope += std::log(((float)time_diff_sum[i]/CLOCKS_PER_SEC) /( (float)time_diff_sum[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)));
        }
    }
    avg_slope /= (3);

    pcout<<" regular slope "<<first_slope_mpi<<" sum-factorization slope "<<sum_slope_mpi<<std::endl;
    pcout<<"sum-factorization average slope "<<avg_slope<<std::endl;

    pcout<<"Times for operation M*u"<<std::endl;
    pcout<<"Normal operation M*u  |  "<<"Sum factorization "<<std::endl;
    double avg_slope1 = 0.0;
    for(unsigned int i=poly_min; i<poly_max; i++){
        pcout<<(float)time_diff_mass[i]/CLOCKS_PER_SEC<<" "<<(float)time_diff_mass_sum[i]/CLOCKS_PER_SEC<<std::endl;
        if(i>poly_max-4){
            avg_slope1 += std::log(((float)time_diff_mass_sum[i]/CLOCKS_PER_SEC) /( (float)time_diff_mass_sum[i-1]/CLOCKS_PER_SEC))
                        / std::log((double)((i)/(i-1.0)));
        }
    }
    avg_slope1 /= (3);

    pcout<<" regular slope "<<first_slope_mpi_mass<<" sum-factorization slope "<<sum_slope_mpi_mass<<std::endl;

    pcout<<"sum-factorization mass average slope "<<avg_slope1<<std::endl;

    if(different==true){
        pcout<<"Sum factorization not recover same vector for A*u."<<std::endl;
        return 1;
    }
    if(different_mass==true){
        pcout<<"Sum factorization not recover same vector Mass*u."<<std::endl;
        return 1;
    }
    if(avg_slope > dim+1.1){
        pcout<<"Sum factorization not give correct comp cost slope."<<std::endl;
        return 1;
    }
    if(avg_slope1 > dim+1.1){
        pcout<<"Sum factorization not give correct comp cost slope."<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main

