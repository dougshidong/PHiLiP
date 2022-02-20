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
#include "dg/dg.h"
#include "dg/dg_factory.hpp"

const double TOLERANCE = 1E-6;
using namespace std;



int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 2;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    all_parameters_new.nstate = nstate;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    all_parameters_new.flux_reconstruction_type = FR_enum::cHU;
   // all_parameters_new.overintegration = 2;
   // const unsigned int overint= all_parameters_new.overintegration;
   // all_parameters_new.use_collocated_nodes = true;

   // double skew_sym = 0.0;
    double M_K_HU =0.0;
    double max_dp1 = 0.0;
    double deriv3_dif = 0.0;
    double deriv4_dif = 0.0;
    for(unsigned int poly_degree=2; poly_degree<6; poly_degree++){
    double left = 0.0;
    double right = 1.0;
    const bool colorize = true;
    const unsigned int igrid= 2;



    //Generate a standard grid

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
        dealii::GridGenerator::hyper_cube (*grid, left, right, colorize);
        grid->refine_global(igrid);

        OPERATOR::OperatorBase<dim,real> operators(&all_parameters_new, nstate, poly_degree, poly_degree, poly_degree); 

        const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);
       // dealii::QGaussLobatto<dim> vol_quad_GLL (poly_degree+1+overint);
        dealii::QGaussLobatto<dim> vol_quad_GLL (poly_degree+1);
        const std::vector<real> &quad_weights = vol_quad_GLL.get_weights ();
#if 0
pcout<<"GLL"<<std::endl;
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                if(idof==idof2)
                    pcout<<quad_weights[idof]<<" ";
                else
                    pcout<<0<<" ";
            }
    pcout<<std::endl;
        }
pcout<<"M+K"<<std::endl;
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                if(std::abs(operators.local_mass[poly_degree][idof][idof2] +
                            operators.local_K_operator[poly_degree][idof][idof2] )<1e-12)
                    pcout<<0<<" ";
                else
                    pcout<<operators.local_mass[poly_degree][idof][idof2] +
                            operators.local_K_operator[poly_degree][idof][idof2]<<" ";
            }
    pcout<<std::endl;
        }
#endif


        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                if(idof == idof2){
                    const unsigned int ishape = operators.fe_collection_basis[poly_degree].system_to_component_index(idof).second;
                    if(std::abs(quad_weights[ishape] - operators.local_mass[poly_degree][idof][idof2] -
                            operators.local_K_operator[poly_degree][idof][idof2]) > 1e-12)
                        M_K_HU = std::abs(quad_weights[idof] - operators.local_mass[poly_degree][idof][idof2] +
                            operators.local_K_operator[poly_degree][idof][idof2]);
                }
                else{
                    if(std::abs(operators.local_mass[poly_degree][idof][idof2] +
                            operators.local_K_operator[poly_degree][idof][idof2]) > 1e-12)
                        if(std::abs(operators.local_mass[poly_degree][idof][idof2] +
                            operators.local_K_operator[poly_degree][idof][idof2]) > M_K_HU)
                            M_K_HU = std::abs(operators.local_mass[poly_degree][idof][idof2] +
                            operators.local_K_operator[poly_degree][idof][idof2]); 
                }
            }
        }
#if 0
pcout<<"MASS"<<std::endl;
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
    pcout<<operators.local_mass[poly_degree][idof][idof2]<<" ";
            }
    pcout<<" "<<std::endl;
        }

pcout<<"basis of row "<<operators.basis_at_vol_cubature[poly_degree].m()<< " "<<operators.basis_at_vol_cubature[poly_degree].n()<<std::endl;

        for(unsigned int idof=0; idof<n_quad_pts; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
    pcout<<operators.basis_at_vol_cubature[poly_degree][idof][idof2]<<" ";
            }
    pcout<<" "<<std::endl;
        }
pcout<<"local basis stiff"<<std::endl;
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
    pcout<<operators.local_basis_stiffness[poly_degree][0][idof][idof2]<<" ";
            }
    pcout<<" "<<std::endl;
        }
pcout<<"flux stiff"<<std::endl;
pcout<<"flux stiff row "<<operators.local_flux_basis_stiffness[poly_degree][0].m()<< " "<<operators.local_flux_basis_stiffness[poly_degree][0].n()<<std::endl;
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<operators.local_flux_basis_stiffness[poly_degree][0].n(); idof2++){
    pcout<<operators.local_flux_basis_stiffness[poly_degree][0][idof][idof2]<<" ";
            }
    pcout<<" "<<std::endl;
        }
#endif
        
#if 0
    
        dealii::FullMatrix<real> KD(n_dofs); 
        for(int idim=0; idim<dim; idim++){
            operators.local_K_operator[poly_degree].mmult(KD, operators.modal_basis_differential_operator[poly_degree][idim]); 
            dealii::Vector<real> random_u(n_dofs);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                random_u[idof] = (double) (rand() % 100);
            }
            dealii::Vector<real> KDu(n_dofs);
            KD.vmult(KDu, random_u);
            double uKDu = random_u * KDu;
            if(std::abs(uKDu)>skew_sym)
                skew_sym = std::abs(uKDu);
    pcout<<"SKEW SYM "<<skew_sym<<std::endl;
        }
#endif

        dealii::FullMatrix<real> Dp(n_dofs);
        const unsigned int n_quad_pts = operators.volume_quadrature_collection[poly_degree].size();
        dealii::FullMatrix<real> Dp1(n_quad_pts, n_dofs);
        dealii::FullMatrix<real> Dp2(n_quad_pts, n_dofs);
        for(int idim=0; idim<dim; idim++){
            operators.derivative_p[poly_degree][idim].mmult(Dp,operators.modal_basis_differential_operator[poly_degree][idim]);
            operators.basis_at_vol_cubature[poly_degree].mmult(Dp1, Dp);
            operators.basis_at_vol_cubature[poly_degree].mmult(Dp2, operators.derivative_p[poly_degree][idim]);
        if(poly_degree == 3){
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
               const dealii::Point<dim> qpoint  = operators.volume_quadrature_collection[poly_degree].point(iquad);
                const int istate = operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
                dealii::Tensor<3, dim, real> deriv_3 = operators.fe_collection_basis[poly_degree].shape_3rd_derivative_component(idof,qpoint, istate); 
                if( std::abs(Dp2[iquad][idof]-deriv_3[idim][idim][idim])> deriv3_dif)
                    deriv3_dif = std::abs(Dp2[iquad][idof]-deriv_3[idim][idim][idim]);
            }
        }
        }
        if(poly_degree == 4){
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
               const dealii::Point<dim> qpoint  = operators.volume_quadrature_collection[poly_degree].point(iquad);
                const int istate = operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
                dealii::Tensor<4, dim, real> deriv_4 = operators.fe_collection_basis[poly_degree].shape_4th_derivative_component(idof,qpoint, istate); 
                if( std::abs(Dp2[iquad][idof]-deriv_4[idim][idim][idim][idim])> deriv4_dif)
                    deriv4_dif = std::abs(Dp2[iquad][idof]-deriv_4[idim][idim][idim][idim]);
            }
        }
        }
        for(unsigned int idof=0; idof<n_quad_pts; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                if(std::abs(Dp1[idof][idof2])>max_dp1)
                    max_dp1 = Dp1[idof][idof2];
            }
        }
        }

    }//end of poly_degree loop

   //     pcout<<" max p+1 derivative "<<max_dp1<<std::endl;
   //   pcout<<" deriv 3 dif "<<deriv3_dif<<std::endl;
   //   pcout<<" deriv 4 dif "<<deriv4_dif<<std::endl;
   // pcout<<"MGLL "<<M_K_HU<<std::endl;
    if( max_dp1 >1e-7){
        pcout<<" One of the pth order derivatives is wrong !"<<std::endl;
        return 1;
    }
    if( deriv3_dif >1e-11){
        pcout<<" 3rd order derivatives is wrong !"<<std::endl;
        return 1;
    }
    if( deriv4_dif >1e-9){
        pcout<<" 4th order derivatives is wrong !"<<std::endl;
        return 1;
    }
    if( M_K_HU >1e-15){
        pcout<<" KHU does not recover Collocated GLL M+K mass matrix with exact integration !"<<std::endl;
        return 1;
    }
#if 0
    if( skew_sym > 1e-11){
        printf(" KD is not skew symmetric !\n");
        return 1;
    }
#endif
    else{
        return 0;
    }

}//end of main
