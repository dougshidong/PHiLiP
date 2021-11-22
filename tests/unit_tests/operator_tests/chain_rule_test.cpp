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
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    all_parameters_new.nstate = nstate;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    all_parameters_new.flux_reconstruction_type = FR_enum::cHU;
//    all_parameters_new.overintegration = 2;
   // const unsigned int overint= all_parameters_new.overintegration;
   // all_parameters_new.use_collocated_nodes = true;

    double chain_rule =0.0;
    for(unsigned int poly_degree=6; poly_degree<7; poly_degree++){
    double left = 0.0;
    double right = 1.0;
    const bool colorize = true;
    const unsigned int igrid= 2;
    unsigned int q_degree = poly_degree + 2;



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

      //  OPERATOR::OperatorBase<dim,real> operators_p_q(&all_parameters_new, nstate, poly_degree + q_degree, poly_degree + q_degree, 1); 
        all_parameters_new.overintegration = q_degree;
        OPERATOR::OperatorBase<dim,real> operators_p(&all_parameters_new, nstate, poly_degree, poly_degree, 1); 
        all_parameters_new.overintegration = poly_degree;
        OPERATOR::OperatorBase<dim,real> operators_q(&all_parameters_new, nstate, q_degree, q_degree, 1); 

        const unsigned int n_dofs_p = pow(poly_degree+1, dim);
        const unsigned int n_dofs_q = pow(q_degree+1, dim);
        const unsigned int n_dofs_p_q = pow(all_parameters_new.overintegration+q_degree+1, dim);
        dealii::Vector<double> u_node(n_dofs_p);
        dealii::Vector<double> v_node(n_dofs_q);
        dealii::QGaussLobatto<dim> GLL_p (poly_degree+1);
        dealii::QGaussLobatto<dim> GLL_q (q_degree+1);
        for(unsigned int idof=0; idof<n_dofs_p; idof++){
           // const dealii::Point<dim> qpoint = (fe_values_vol.quadrature_point(iquad));
            const dealii::Point<dim> qpoint = GLL_p.point(idof);
            const double pi = atan(1)*4.0;
            u_node[idof] = pow(std::sin(qpoint[0]*pi)+1.0,5);
        }
        for(unsigned int idof=0; idof<n_dofs_q; idof++){
           // const dealii::Point<dim> qpoint = (fe_values_vol.quadrature_point(iquad));
            const dealii::Point<dim> qpoint = GLL_q.point(idof);
            v_node[idof] = 2.0*pow(std::sin(qpoint[0]),3.0);
        }

        dealii::Vector<double> u_at_p_q_nodes(n_dofs_p_q);
        dealii::Vector<double> v_at_p_q_nodes(n_dofs_p_q);
        operators_p.basis_at_vol_cubature[poly_degree].vmult(u_at_p_q_nodes, u_node);
        operators_q.basis_at_vol_cubature[q_degree].vmult(v_at_p_q_nodes, v_node);
        dealii::Vector<double> Du_at_p_q_nodes(n_dofs_p_q);
        dealii::Vector<double> Dv_at_p_q_nodes(n_dofs_p_q);
        operators_p.gradient_flux_basis[poly_degree][0][0].vmult(Du_at_p_q_nodes, u_at_p_q_nodes);
        operators_q.gradient_flux_basis[q_degree][0][0].vmult(Dv_at_p_q_nodes, v_at_p_q_nodes);
        dealii::Vector<double> u_v_at_p_q_nodes(n_dofs_p_q);
        dealii::Vector<double> Du_v_at_p_q_nodes(n_dofs_p_q);
        dealii::Vector<double> u_Dv_at_p_q_nodes(n_dofs_p_q);
        for(unsigned int idof=0; idof<n_dofs_p_q; idof++){
            u_v_at_p_q_nodes[idof] = u_at_p_q_nodes[idof] * v_at_p_q_nodes[idof];
            Du_v_at_p_q_nodes[idof] = Du_at_p_q_nodes[idof] * v_at_p_q_nodes[idof];
            u_Dv_at_p_q_nodes[idof] = u_at_p_q_nodes[idof] * Dv_at_p_q_nodes[idof];
        }
        dealii::Vector<double> D_u_v_at_p_q_nodes(n_dofs_p_q);
        operators_p.gradient_flux_basis[poly_degree][0][0].vmult(D_u_v_at_p_q_nodes, u_v_at_p_q_nodes);
    
    pcout<<" ABS difference"<<std::endl;
    for(unsigned int idof=0; idof<n_dofs_p_q; idof++){
        pcout<<std::abs(D_u_v_at_p_q_nodes[idof]-(Du_v_at_p_q_nodes[idof]+u_Dv_at_p_q_nodes[idof]))<<std::endl;
        if(std::abs(D_u_v_at_p_q_nodes[idof]-(Du_v_at_p_q_nodes[idof]+u_Dv_at_p_q_nodes[idof]))>chain_rule)
            chain_rule = std::abs(D_u_v_at_p_q_nodes[idof]-(Du_v_at_p_q_nodes[idof]+u_Dv_at_p_q_nodes[idof]));
    }
    pcout<<" Derivative of (u times v)"<<std::endl;
    for(unsigned int idof=0; idof<n_dofs_p_q; idof++){
        pcout<<D_u_v_at_p_q_nodes[idof]<<std::endl;
    }
    pcout<<" u*Dv + v*Du"<<std::endl;
    for(unsigned int idof=0; idof<n_dofs_p_q; idof++){
        pcout<<Du_v_at_p_q_nodes[idof]+u_Dv_at_p_q_nodes[idof]<<std::endl;
    }
    pcout<<" v*Du"<<std::endl;
    for(unsigned int idof=0; idof<n_dofs_p_q; idof++){
        pcout<<Du_v_at_p_q_nodes[idof]<<std::endl;
    }
    pcout<<" u*Dv"<<std::endl;
    for(unsigned int idof=0; idof<n_dofs_p_q; idof++){
        pcout<<u_Dv_at_p_q_nodes[idof]<<std::endl;
    }



    }//end of poly_degree loop

    if( chain_rule >1e-11){
        pcout<<" chain rule not satisfied !"<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main
