#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

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

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include "dg/dg_factory.hpp"
#include "operators/operators.h"
#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"

const double TOLERANCE = 1E-6;
using namespace std;
//namespace PHiLiP {

int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);

    //Generate a standard grid
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    bool mapping_supp_match = true;
    const unsigned int n_refinements = 1;
    for(unsigned int poly_degree = 5; poly_degree<6; poly_degree++){

        // set the warped grid
        const unsigned int grid_degree = poly_degree;
        PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);

        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        dg->allocate_system ();

        const dealii::FESystem<dim> &fe_metric = (dg->high_order_grid->get_current_fe_system());
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell; 
        const unsigned int n_grid_nodes = n_metric_dofs / dim;
        auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
        for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
            if (!current_cell->is_locally_owned()) continue;
        
            std::vector<dealii::types::global_dof_index> current_metric_dofs_indices(n_metric_dofs);
            metric_cell->get_dof_indices (current_metric_dofs_indices);
            std::array<std::vector<real>,dim> mapping_support_points;
            std::array<std::vector<real>,dim> mapping_support_points_new;
            for(int idim=0; idim<dim; idim++){
                mapping_support_points[idim].resize(n_grid_nodes);
                mapping_support_points_new[idim].resize(n_grid_nodes);
            }
            for (unsigned int igrid_node = 0; igrid_node< n_grid_nodes; ++igrid_node) {
                for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
                    const real val = (dg->high_order_grid->volume_nodes[current_metric_dofs_indices[idof]]);
                    const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
                    mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,dg->high_order_grid->dim_grid_nodes.point(igrid_node),istate); 
                }
            }
            //get index renumbering from FE_Q->FE_DGQ
            const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(poly_degree);
            for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
                const unsigned int idim = fe_metric.system_to_component_index(idof).first; 
                const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
                const unsigned int igrid_node = index_renumbering[ishape];
                const real val = (dg->high_order_grid->volume_nodes[current_metric_dofs_indices[idof]]);
                mapping_support_points_new[idim][igrid_node] = val; 
            }
            for(int idim =0; idim<dim; idim++){
                for (unsigned int igrid_node = 0; igrid_node< n_grid_nodes; ++igrid_node) {
                    if(abs(mapping_support_points_new[idim][igrid_node]-mapping_support_points[idim][igrid_node])>1e-12){
                        std::cout<<"original "<<mapping_support_points[idim][igrid_node]<<" new "<<mapping_support_points_new[idim][igrid_node]<<std::endl;
                        std::cout<<"for idim "<<idim<<" and node "<<igrid_node<<std::endl;
                        mapping_supp_match=false;
                    }
                }
            }
        }//end of cell loop

    }//end poly degree loop

    if(!mapping_supp_match){
        pcout<<"The mapping support nodes not computed correctly."<<std::endl;
        return 1;
    }
    else{
        pcout<<" Mapping support points reduced cost extraction works perfectly.\n"<<std::endl;
        return 0;
    }
}

//}//end PHiLiP namespace

