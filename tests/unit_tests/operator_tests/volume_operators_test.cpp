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

        const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);
        // dealii::QGaussLobatto<dim> vol_quad_GLL (poly_degree+1+overint);
        dealii::QGaussLobatto<dim> vol_quad_GLL (poly_degree+1);
        const std::vector<real> &quad_weights = vol_quad_GLL.get_weights ();
        const dealii::FE_DGQ<dim> fe_dim(poly_degree);
        const dealii::FESystem<dim,dim> fe_system_dim(fe_dim, nstate);

        PHiLiP::OPERATOR::local_mass<dim,2*dim> mass_matrix(nstate, poly_degree, 1);
        dealii::QGauss<1> quad1D (poly_degree+1);
        const dealii::FE_DGQ<1> fe_dg(poly_degree);
        const dealii::FESystem<1,1> fe_system(fe_dg, nstate);
        mass_matrix.build_1D_volume_operator(fe_system,quad1D);
        dealii::FullMatrix<real> mass_dim(n_dofs);
        mass_dim = mass_matrix.tensor_product_state(nstate,
                                                    mass_matrix.oneD_vol_operator,
                                                    mass_matrix.oneD_vol_operator,mass_matrix.oneD_vol_operator);

        PHiLiP::OPERATOR::local_Flux_Reconstruction_operator<dim,2*dim> local_FR(nstate, poly_degree, 1, FR_enum::cHU);
        local_FR.build_1D_volume_operator(fe_system,quad1D);
        dealii::FullMatrix<real> FR_dim(n_dofs);
        FR_dim = local_FR.build_dim_Flux_Reconstruction_operator(mass_matrix.oneD_vol_operator, nstate, n_dofs);

        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                double sum = mass_dim[idof][idof2] + FR_dim[idof][idof2]; 
                if(idof == idof2){
                    const unsigned int ishape = fe_system_dim.system_to_component_index(idof).second;
                    if(std::abs(quad_weights[ishape] - sum) > 1e-12)
                        M_K_HU = std::abs(quad_weights[idof] - sum);
                }
                else{
                    if(std::abs(sum) > 1e-12)
                        if(std::abs(sum) > M_K_HU)
                            M_K_HU = std::abs(sum); 
                }
            }
        }
    }//end of poly_degree loop

    if(M_K_HU > 1e-15){
        pcout<<" KHU does not recover Collocated GLL M+K mass matrix with exact integration !"<<std::endl;
        return 1;
    }
    else{
        return 0;
    }
}//end of main
