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

#include <deal.II/fe/fe_bernstein.h>

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
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    all_parameters_new.nstate = nstate;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    all_parameters_new.flux_reconstruction_type = FR_enum::cHU;
   // all_parameters_new.use_collocated_nodes=true;
    all_parameters_new.overintegration = 2;

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
    for(unsigned int poly_degree=2; poly_degree<6; poly_degree++){

        const unsigned int n_dofs_1D = nstate * (poly_degree + 1);
//        const unsigned int n_dofs    = nstate * pow(poly_degree + 1, dim);
        dealii::QGauss<1> quad1D (poly_degree+1);
        const dealii::FE_DGQ<1> fe_dg(poly_degree);
        const dealii::FESystem<1,1> fe_system(fe_dg, nstate);
        const dealii::FE_DGQArbitraryNodes<1> fe_dg_flux(quad1D);
        const dealii::FESystem<1,1> fe_system_flux(fe_dg_flux, nstate);
        PHiLiP::OPERATOR::vol_integral_gradient_basis<dim,2*dim> vol_int_grad_basis(nstate, poly_degree, 1);
        vol_int_grad_basis.build_1D_gradient_operator(fe_system, quad1D);
        PHiLiP::OPERATOR::local_flux_basis_stiffness<dim,nstate,2*dim> flux_stiffness(poly_degree, 1);
        flux_stiffness.build_1D_gradient_state_operator(fe_system_flux, quad1D);
        flux_stiffness.build_1D_volume_state_operator(fe_system, quad1D);

       // PHiLiP::OPERATOR::basis_functions_state<dim,nstate,2*dim> flux_basis_quad(poly_degree, 1);
        PHiLiP::OPERATOR::flux_basis_functions_state<dim,nstate,2*dim> flux_basis_quad(poly_degree, 1);
        flux_basis_quad.build_1D_volume_state_operator(fe_system_flux, quad1D);

        dealii::FullMatrix<real> vol_int_parts(n_dofs_1D);
        vol_int_grad_basis.oneD_grad_operator.Tmmult(vol_int_parts, flux_basis_quad.oneD_vol_state_operator[0]);
        vol_int_parts.add(1.0, flux_stiffness.oneD_vol_state_operator[0]);

        //compute surface integral
        dealii::QGauss<0> face_quad1D (poly_degree+1);
        dealii::FullMatrix<real> surf_int_parts(n_dofs_1D);
        flux_basis_quad.build_1D_surface_state_operator(fe_system_flux, face_quad1D);
        PHiLiP::OPERATOR::face_integral_basis<dim,2*dim> surf_int_basis(nstate, poly_degree, 1);
        surf_int_basis.build_1D_surface_operator(fe_system, face_quad1D);
        for(unsigned int iface=0; iface< dealii::GeometryInfo<1>::faces_per_cell; iface++){
            const dealii::Tensor<1,1,real> unit_normal_1D = dealii::GeometryInfo<1>::unit_normal_vector[iface];
            dealii::FullMatrix<real> surf_int_face(n_dofs_1D);
            surf_int_basis.oneD_surf_operator[iface].Tmmult(surf_int_face, flux_basis_quad.oneD_surf_state_operator[0][iface]);
            surf_int_parts.add(unit_normal_1D[0], surf_int_face);
        }
        //check difference between the two for integration-by-parts
        for(unsigned int idof=0; idof<n_dofs_1D; idof++){
            for(unsigned int idof2=0; idof2<n_dofs_1D; idof2++){
                if(std::abs(surf_int_parts[idof][idof2] - vol_int_parts[idof][idof2])>max_dif_int_parts)
                    max_dif_int_parts = std::abs(surf_int_parts[idof][idof2] - vol_int_parts[idof][idof2]);
            }
        }
        
    }//end of poly_degree loop

    const double max_dif_int_parts_mpi= (dealii::Utilities::MPI::max(max_dif_int_parts, MPI_COMM_WORLD));
    if( max_dif_int_parts_mpi >1e-12){
        pcout<<" Surface operator not satisfy integration by parts !"<<max_dif_int_parts_mpi<<std::endl;
       // printf(" One of the pth order deirvatives is wrong !\n");
        return 1;
    }
    else{
        return 0;
    }

}//end of main

