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
#include <deal.II/base/convergence_table.h>

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
#include "operators/operators_new.h"
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
    double max_dif_int_parts = 0.0;
    double max_dif_surf_int = 0.0;
    double max_dif_surf_int_FR = 0.0;
    double max_dif_int_parts_dim = 0.0;
    for(unsigned int poly_degree=2; poly_degree<6; poly_degree++){

        const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);
        const unsigned int n_dofs_1D = nstate * (poly_degree+1);

        //build stiffnes 1D
        PHiLiP::OPERATOR::local_basis_stiffness<dim,2*dim> stiffness(nstate, poly_degree, 1);
        dealii::QGauss<1> quad1D (poly_degree+1);
        const dealii::FE_DGQ<1> fe_dg(poly_degree);
        const dealii::FESystem<1,1> fe_system(fe_dg, nstate);
        stiffness.build_1D_volume_operator(fe_system,quad1D);

        //volume integration by parts
        dealii::FullMatrix<real> vol_int_parts(n_dofs_1D);
        vol_int_parts.add(1.0, stiffness.oneD_vol_operator);
        vol_int_parts.Tadd(1.0, stiffness.oneD_vol_operator);

        //compute surface integral
        dealii::FullMatrix<real> face_int_parts(n_dofs_1D);
        dealii::QGauss<0> face_quad1D (poly_degree+1);
        PHiLiP::OPERATOR::face_integral_basis<dim,2*dim> face_int(nstate, poly_degree, 1);
        face_int.build_1D_surface_operator(fe_system, face_quad1D);
        PHiLiP::OPERATOR::basis_functions<dim,2*dim> face_basis(nstate, poly_degree, 1);
        face_basis.build_1D_surface_operator(fe_system, face_quad1D);

        const unsigned int n_face_quad_pts = face_quad1D.size();
        for(unsigned int iface=0; iface< dealii::GeometryInfo<1>::faces_per_cell; iface++){
            const dealii::Tensor<1,1,real> unit_normal = dealii::GeometryInfo<1>::unit_normal_vector[iface];
            for(unsigned int itest=0; itest<n_dofs_1D; itest++){
                const unsigned int istate_test = fe_system.system_to_component_index(itest).first;
                for(unsigned int idof=0; idof<n_dofs_1D; idof++){
                    const unsigned int istate_dof = fe_system.system_to_component_index(idof).first;
                    double value= 0.0;
                    for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
                        value +=    face_int.oneD_surf_operator[iface][iquad][itest]
                                *   unit_normal[0]
                                *   face_basis.oneD_surf_operator[iface][iquad][idof];
                    }
                    if(istate_test == istate_dof){
                        face_int_parts[itest][idof] += value;
                    }
                }
            }
        }


        for(unsigned int idof=0; idof<n_dofs_1D; idof++){
            for(unsigned int idof2=0; idof2<n_dofs_1D; idof2++){
                if(std::abs(face_int_parts[idof][idof2] - vol_int_parts[idof][idof2])>max_dif_int_parts)
                    max_dif_int_parts = std::abs(face_int_parts[idof][idof2] - vol_int_parts[idof][idof2]);
            }
        }

        PHiLiP::OPERATOR::lifting_operator<dim,2*dim> lifting(nstate, poly_degree, 1);
        lifting.build_1D_volume_operator(fe_system, quad1D);
        lifting.build_1D_surface_operator(fe_system, face_quad1D);
        std::array<dealii::FullMatrix<real>,2> surface_int_from_lift;
        for(unsigned int iface=0; iface<2; iface++){
            surface_int_from_lift[iface].reinit(n_dofs_1D, n_face_quad_pts);
            lifting.oneD_vol_operator.mmult(surface_int_from_lift[iface], lifting.oneD_surf_operator[iface]); 
        }
        for(unsigned int iface=0; iface<2; iface++){
            for(unsigned int idof=0; idof<n_dofs_1D; idof++){
                for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
                    if(std::abs(face_int.oneD_surf_operator[iface][iquad][idof] - surface_int_from_lift[iface][idof][iquad])>max_dif_surf_int)
                        max_dif_surf_int = std::abs(face_int.oneD_surf_operator[iface][iquad][idof] - surface_int_from_lift[iface][idof][iquad]);
                }
            }
        }
        PHiLiP::OPERATOR::lifting_operator_FR<dim,2*dim> lifting_FR(nstate, poly_degree, 1, FR_enum::cPlus);
        lifting_FR.build_1D_volume_operator(fe_system, quad1D);
        lifting_FR.build_1D_surface_operator(fe_system, face_quad1D);
        std::array<dealii::FullMatrix<real>,2> surface_int_from_lift_FR;
        for(unsigned int iface=0; iface<2; iface++){
            surface_int_from_lift_FR[iface].reinit(n_dofs_1D, n_face_quad_pts);
            lifting_FR.oneD_vol_operator.mmult(surface_int_from_lift_FR[iface], lifting_FR.oneD_surf_operator[iface]); 
        }
        for(unsigned int iface=0; iface<2; iface++){
            for(unsigned int idof=0; idof<n_dofs_1D; idof++){
                for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
                    if(std::abs(face_int.oneD_surf_operator[iface][iquad][idof] - surface_int_from_lift_FR[iface][idof][iquad])>max_dif_surf_int)
                        max_dif_surf_int_FR = std::abs(face_int.oneD_surf_operator[iface][iquad][idof] - surface_int_from_lift_FR[iface][idof][iquad]);
                }
            }
        }
        
        dealii::FullMatrix<real> face_int_parts_dim(n_dofs);
        face_int_parts_dim = face_int.tensor_product_state(nstate, face_int_parts, lifting.oneD_vol_operator, lifting.oneD_vol_operator); 
        dealii::FullMatrix<real> vol_int_parts_dim(n_dofs);
        dealii::FullMatrix<real> stiffness_dim(n_dofs);
        stiffness_dim = stiffness.tensor_product_state(nstate, stiffness.oneD_vol_operator, lifting.oneD_vol_operator, lifting.oneD_vol_operator);
        vol_int_parts_dim.add(1.0, stiffness_dim);
        vol_int_parts_dim.Tadd(1.0, stiffness_dim);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                if(std::abs(face_int_parts_dim[idof][idof2] - vol_int_parts_dim[idof][idof2])>max_dif_int_parts_dim)
                    max_dif_int_parts_dim = std::abs(face_int_parts_dim[idof][idof2] - vol_int_parts_dim[idof][idof2]);
            }
        }

    }//end of poly_degree loop

    const double max_dif_int_parts_mpi= (dealii::Utilities::MPI::max(max_dif_int_parts, MPI_COMM_WORLD));
    const double max_dif_int_parts_dim_mpi= (dealii::Utilities::MPI::max(max_dif_int_parts_dim, MPI_COMM_WORLD));
    const double max_dif_surf_int_mpi= (dealii::Utilities::MPI::max(max_dif_surf_int, MPI_COMM_WORLD));
    const double max_dif_surf_int_mpi_FR= (dealii::Utilities::MPI::max(max_dif_surf_int_FR, MPI_COMM_WORLD));
    if( max_dif_int_parts_mpi >1e-11){
        pcout<<" Surface operator not satisfy integration by parts !"<<std::endl;
        return 1;
    }
    if( max_dif_int_parts_dim_mpi >1e-11){
        pcout<<" Surface operator tensor product not satisfy integration by parts !"<<std::endl;
        return 1;
    }
    else if( max_dif_surf_int_mpi >1e-11){
        pcout<<" Surface lifting operator not correct !"<<std::endl;
        return 1;
    }
    else if( max_dif_surf_int_mpi_FR >1e-11){
        pcout<<" Surface lifting FR operator not correct !"<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main

