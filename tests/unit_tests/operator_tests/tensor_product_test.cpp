#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <ctime>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

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

    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    all_parameters_new.flux_reconstruction_type = FR_enum::cHU;

    bool equiv = true;
    bool sum_fact = true;
    for(unsigned int poly_degree=2; poly_degree<4; poly_degree++){

        const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);
        dealii::QGauss<dim> vol_quad_dim (poly_degree+1);
        const dealii::FE_DGQ<dim> fe_dim(poly_degree);
        const dealii::FESystem<dim,dim> fe_system_dim(fe_dim, nstate);
        
        dealii::QGauss<1> quad_1D (poly_degree+1);
        const dealii::FE_DGQ<1> fe(poly_degree);
        const dealii::FESystem<1,1> fe_system(fe, nstate);
        PHiLiP::OPERATOR::basis_functions<dim,2*dim,real> basis_1D(nstate, poly_degree, 1);
        basis_1D.build_1D_volume_operator(fe, quad_1D);
        basis_1D.build_1D_gradient_operator(fe, quad_1D);
        dealii::FullMatrix<double> basis_dim(n_dofs);
        basis_dim = basis_1D.tensor_product(basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator);

        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int iquad=0; iquad<n_dofs; iquad++){
                dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
                if(fe_system_dim.shape_grad_component(idof,qpoint,0)[0] != basis_dim[iquad][idof])
                    equiv = false;
            } 
        } 
        if(dim >= 2){
            basis_dim = basis_1D.tensor_product(basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator,basis_1D.oneD_vol_operator);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                for(unsigned int iquad=0; iquad<n_dofs; iquad++){
                    dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
                    if(fe_system_dim.shape_grad_component(idof,qpoint,0)[1] != basis_dim[iquad][idof])
                        equiv = false;
                } 
            } 
        }
        if(dim >= 3){
            basis_dim = basis_1D.tensor_product(basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                for(unsigned int iquad=0; iquad<n_dofs; iquad++){
                    dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
                    if(fe_system_dim.shape_grad_component(idof,qpoint,0)[2] != basis_dim[iquad][idof])
                        equiv = false;
                } 
            } 
        }

        std::vector<double> sol_hat(n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
           // sol_hat[idof] = 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8)));
            sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8))) );
        }
        std::vector<double> sol_dim(n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            sol_dim[idof] = 0.0;
            for(unsigned int iquad=0; iquad<n_dofs; iquad++){
                sol_dim[idof] += basis_dim[idof][iquad] * sol_hat[iquad];
            }
        }
        std::vector<double> sol_sum_fact(n_dofs);
        if(dim==1)
            basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator, basis_1D.oneD_vol_operator);
        if(dim==2)
            basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator);
        if(dim==3)
            basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_vol_operator, basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator);
        

        for(unsigned int idof=0; idof<n_dofs; idof++){
            if(std::abs(sol_dim[idof] - sol_sum_fact[idof])>1e-12){
                sum_fact = false;
                pcout<<"sum fact wrong "<<sol_dim[idof]<<" "<<sol_sum_fact[idof]<<std::endl;
            }
        }


    }//end of poly_degree loop

    if( equiv == false){
        pcout<<" Tensor product not recover original !"<<std::endl;
        return 1;
    }
    if(sum_fact == false){
        pcout<<" sum fcatorization not recover A*u"<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main

