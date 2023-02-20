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
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    bool different = false;
    const unsigned int poly_max = 6;
    const unsigned int poly_min = 2;
    for(unsigned int poly_degree=poly_min; poly_degree<poly_max; poly_degree++){

        PHiLiP::OPERATOR::local_basis_stiffness<dim,2*dim> stiffness(1, poly_degree, 1);
        PHiLiP::OPERATOR::local_mass<dim,2*dim> mass(1, poly_degree, 1);
        dealii::QGauss<1> quad1D (poly_degree+1);
        const dealii::FE_DGQArbitraryNodes<1> fe_dg(quad1D);
        const dealii::FESystem<1,1> fe_system(fe_dg, 1);
        stiffness.build_1D_volume_operator(fe_system,quad1D);
        mass.build_1D_volume_operator(fe_system,quad1D);
        const std::vector<real> &quad_weights_1D = quad1D.get_weights();

        const unsigned int n_dofs = pow(poly_degree+1,dim);
        const unsigned int n_quad_pts_1D = quad1D.size();
        const unsigned int n_quad_pts = pow(n_quad_pts_1D, dim);

        assert(n_quad_pts==n_dofs);//since checking for flux basis

        //build dim-sized stiffness operator
        dealii::FullMatrix<real> stiffness_dim(n_quad_pts);//solution of A*u with sum-factorization
        stiffness_dim = stiffness.tensor_product(stiffness.oneD_vol_operator, mass.oneD_vol_operator, mass.oneD_vol_operator);

        //random solution vector
        std::vector<real> sol_hat(n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8))) );
        }
        
        //random solution matrix
        dealii::FullMatrix<real> sol_hat_mat(n_quad_pts);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                sol_hat_mat[idof][idof2] = sol_hat[idof] * sol_hat[idof2];
            }
        }
        //Hadamard product of dim-sized matrices
        dealii::FullMatrix<real> sol_dim(n_quad_pts);//solution of A*u normally
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                     * stiffness_dim[idof][idof2];
            }
        }


        dealii::FullMatrix<real> sol_1D(n_quad_pts);//solution of A*u normally
        //use a sum-factorization "type" algorithm with matrix structure for Hadamard product
        stiffness.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, stiffness.oneD_vol_operator, quad_weights_1D, 0);

        //compare that got same answer
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for(unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                if(std::abs(sol_dim[iquad][iquad2] - sol_1D[iquad][iquad2])>1e-11){
                    pcout<<"Dir 0 sol dim "<<sol_dim[iquad][iquad2]<<" sol 1D "<<sol_1D[iquad][iquad2]<<std::endl;
                    different = true;
                }
            }
        }

        if constexpr(dim>1){
            //dir = 1 (y-direction)
            stiffness_dim = stiffness.tensor_product(mass.oneD_vol_operator, stiffness.oneD_vol_operator, mass.oneD_vol_operator);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                    sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                         * stiffness_dim[idof][idof2];
                }
            }
             
            //use a sum-factorization "type" algorithm with matrix structure for Hadamard product
            sol_1D.reinit(n_quad_pts, n_quad_pts);
            stiffness.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, stiffness.oneD_vol_operator, quad_weights_1D, 1);
             
            //compare that got same answer
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                    if(std::abs(sol_dim[iquad][iquad2] - sol_1D[iquad][iquad2])>1e-11){
                        pcout<<"Dir 1 sol dim "<<sol_dim[iquad][iquad2]<<" sol 1D "<<sol_1D[iquad][iquad2]<<std::endl;
                        different = true;
                    }
                }
            }
            if constexpr(dim==3){
                //dir = 2 (z-direction)
                stiffness_dim = stiffness.tensor_product(mass.oneD_vol_operator, mass.oneD_vol_operator, stiffness.oneD_vol_operator);
                for(unsigned int idof=0; idof<n_dofs; idof++){
                    for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                        sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                                             * stiffness_dim[idof][idof2];
                    }
                }
                 
                //use a sum-factorization "type" algorithm with matrix structure for Hadamard product
                sol_1D.reinit(n_quad_pts, n_quad_pts);
                stiffness.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, stiffness.oneD_vol_operator, quad_weights_1D, 2);
                 
                //compare that got same answer
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                        if(std::abs(sol_dim[iquad][iquad2] - sol_1D[iquad][iquad2])>1e-11){
                            pcout<<"Dir 2 sol dim "<<sol_dim[iquad][iquad2]<<" sol 1D "<<sol_1D[iquad][iquad2]<<std::endl;
                            different = true;
                        }
                    }
                }
            }
        }
    }//end of poly_degree loop

    if(different==true){
        pcout<<"Hadamard product with diagonal weight not recover same vector for A*u."<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main

