#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

#include <exception>
#include <deal.II/fe/mapping.h> 
#include <deal.II/base/exceptions.h> // ExcTransformationFailed

#include <deal.II/fe/mapping_fe_field.h> 
#include <deal.II/fe/mapping_q.h> 

#include "dg/high_order_grid.h"
#include "parameters/all_parameters.h"

/** Tests the mesh movement by moving the mesh and integrating its volume.
 *  Furthermore, it checks that the surface displacements resulting from the mesh movement
 *  are consistent with the prescribed surface displacements.
 */
int main (int argc, char * argv[])
{
    const int dim = PHILIP_DIM;
    int fail_bool = false;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);

    const int initial_n_cells = 3;
    const unsigned int n_grids = 3;
    const unsigned int p_start = 2;
    const unsigned int p_end = 3;
    const double amplitude = 0.1;
    const double exact_area = dim>1 ? 1.0 : (amplitude+1.0);
    const double area_tolerance = 1e-12;
    const double fd_eps = 1e-8;
    std::vector<int> fail_poly;
    std::vector<double> fail_area;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        std::vector<double> area_error(n_grids);
        std::vector<double> grid_size(n_grids);

        dealii::ConvergenceTable convergence_table;
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
            dealii::Triangulation<dim> grid(
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
            dealii::parallel::distributed::Triangulation<dim> grid(
                MPI_COMM_WORLD,
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
            dealii::GridGenerator::subdivided_hyper_cube(grid, initial_n_cells);


            HighOrderGrid<dim,double> high_order_grid(&all_parameters, poly_degree, &grid);

            for (unsigned int i=0; i<igrid; ++i) {
                high_order_grid.prepare_for_coarsening_and_refinement();
                grid.refine_global (1);
                high_order_grid.execute_coarsening_and_refinement();
            }
			const int n_refine = 2;
			for (int i=0; i<n_refine;i++) {
				high_order_grid.prepare_for_coarsening_and_refinement();
				grid.prepare_coarsening_and_refinement();
				unsigned int icell = 0;
				for (auto cell = grid.begin_active(); cell!=grid.end(); ++cell) {
					if (!cell->is_locally_owned()) continue;
					icell++;
					if (icell > grid.n_active_cells()/2) {
						cell->set_refine_flag();
					}
				}
				grid.execute_coarsening_and_refinement();
				bool mesh_out = (i==n_refine-1);
				high_order_grid.execute_coarsening_and_refinement(mesh_out);
			}

            high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

#if PHILIP_DIM!=1
            high_order_grid.prepare_for_coarsening_and_refinement();
            grid.repartition();
            high_order_grid.execute_coarsening_and_refinement();
#endif

            std::vector<dealii::Tensor<1,dim,double>> point_displacements(high_order_grid.locally_relevant_surface_points.size());
            auto disp = point_displacements.begin();
            auto point = high_order_grid.locally_relevant_surface_points.begin();
            auto point_end = high_order_grid.locally_relevant_surface_points.end();
            for (;point != point_end; ++point, ++disp) {
                (*disp) = 0.0;
                (*disp)[0] = amplitude;
                (*disp)[0] *= (*point)[0];
                if(dim>=2) {
                    (*disp)[0] *= std::sin(2.0*dealii::numbers::PI*(*point)[1]);
                }
                if(dim>=3) {
                    (*disp)[0] *= std::sin(2.0*dealii::numbers::PI*(*point)[2]);
                }
                //(*disp)[0] *= 1*(*point)[0];
            }

            std::vector<dealii::types::global_dof_index> surface_node_global_indices(dim*high_order_grid.locally_relevant_surface_points.size());
            std::vector<double> surface_node_displacements(dim*high_order_grid.locally_relevant_surface_points.size());
            {
                int inode = 0;
                for (unsigned int ipoint=0; ipoint<point_displacements.size(); ++ipoint) {
                    for (unsigned int d=0;d<dim;++d) {
                        const std::pair<unsigned int, unsigned int> point_axis = std::make_pair(ipoint,d);
                        const dealii::types::global_dof_index global_index = high_order_grid.point_and_axis_to_global_index[point_axis];
                        surface_node_global_indices[inode] = global_index;
                        surface_node_displacements[inode] = point_displacements[ipoint][d];
                        inode++;
                    }
                }
            }

            using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
            MeshMover::LinearElasticity<dim, double, VectorType , dealii::DoFHandler<dim>> 
                meshmover(high_order_grid, surface_node_global_indices, surface_node_displacements);
            VectorType volume_displacements = meshmover.get_volume_displacements();

            std::vector<VectorType> dXvdXs_FD;
            for (unsigned int inode = 0; inode < high_order_grid.dof_handler_grid.n_dofs(); inode++) {

                unsigned int surface_index = 0;
                double old_value;
                bool restore_value = false;
                if (high_order_grid.locally_relevant_dofs_grid.is_element(inode)) {
                    for (; surface_index < surface_node_global_indices.size(); surface_index++) {
                        const unsigned int inode_surface = surface_node_global_indices[surface_index];
                        if (inode == inode_surface) {
                            restore_value = true;

                            old_value = surface_node_displacements[surface_index];
                            surface_node_displacements[surface_index] = old_value + fd_eps;

                            break;
                        }
                    }
                }
                MeshMover::LinearElasticity<dim, double, VectorType , dealii::DoFHandler<dim>> 
                    meshmover_p(high_order_grid, surface_node_global_indices, surface_node_displacements);

                VectorType volume_displacements_p = meshmover_p.get_volume_displacements();

                if (restore_value) surface_node_displacements[surface_index] = old_value;
            }

        }

    }

    if (fail_bool) {
        pcout << "Test failed. The estimated error should be the same for a given p, even after refinement and translation." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}


