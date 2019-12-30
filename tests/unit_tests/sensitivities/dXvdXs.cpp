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

template<int dim>
dealii::Point<dim> initial_deformation(dealii::Point<dim> point) {
    const double amplitude = 0.1;
    dealii::Tensor<1,dim,double> disp;
    disp[0] = amplitude;
    disp[0] *= point[0];
    if(dim>=2) {
        disp[0] *= std::sin(2.0*dealii::numbers::PI*point[1]);
    }
    if(dim>=3) {
        disp[0] *= std::sin(2.0*dealii::numbers::PI*point[2]);
    }
    return point + disp;
}
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
    const int n_mpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); (void) n_mpi;
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);

    const double amplitude = 0.1;
    const int initial_n_cells = 3;
    const unsigned int n_grids = 3;
    const unsigned int p_start = 2;
    const unsigned int p_end = 3;
    const double fd_eps = 1e-0;
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
                //(*disp)[0] *= 0;
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
            std::function<dealii::Point<dim>(dealii::Point<dim>)> transformation = initial_deformation<dim>;
            VectorType surface_node_displacements_vector = high_order_grid.transform_surface_nodes(transformation);
            surface_node_displacements_vector -= high_order_grid.surface_nodes;
            surface_node_displacements_vector.update_ghost_values();

            // Test surface indices match
            for (int impi=0;impi<n_mpi;++impi) {
                if (impi == mpi_rank) {
                    std::vector<int> v1, v2;
                    for (const auto &i: surface_node_global_indices) {
                        v1.push_back(i);
                    }

                    const dealii::IndexSet owned = high_order_grid.surface_indices.locally_owned_elements();
                    const dealii::IndexSet ghosted = high_order_grid.surface_indices.get_partitioner()->ghost_indices();
                    for (unsigned int i=0; i<high_order_grid.surface_indices.size(); ++i) {
                        if(owned.is_element(i) || ghosted.is_element(i)) {
                            v2.push_back(high_order_grid.surface_indices[i]);
                        }
                    }
                    std::sort(v1.begin(),v1.end());
                    std::sort(v2.begin(),v2.end());
                    assert(v1 == v2);
                }
            }
            // Test surface indices match
            for (int impi=0;impi<n_mpi;++impi) {
                if (impi == mpi_rank) {
                    std::vector<double> v1, v2;
                    std::cout << "List " << std::endl;
                    for (const auto &i: surface_node_displacements) {
                        std::cout << i << std::endl;
                        v1.push_back(i);
                    }

                    const dealii::IndexSet owned = surface_node_displacements_vector.locally_owned_elements();
                    const dealii::IndexSet ghosted = surface_node_displacements_vector.get_partitioner()->ghost_indices();
                    std::cout << "Vector " << std::endl;
                    for (unsigned int i=0; i<surface_node_displacements_vector.size(); ++i) {
                        if(owned.is_element(i) || ghosted.is_element(i)) {
                            v2.push_back(surface_node_displacements_vector[i]);
                            std::cout << surface_node_displacements_vector[i] << std::endl;
                        }
                    }
                    std::sort(v1.begin(),v1.end());
                    std::sort(v2.begin(),v2.end());
                    assert(v1.size() == v2.size());
                    for (unsigned int i=0; i<v1.size(); ++i) {
                        //double denom = 1e-15;
                        //denom = std::max(std::abs(v1[i]), denom);
                        //denom = std::max(std::abs(v2[i]), denom);
                        //std::cout<< "asdasd" <<  ((v1[i] - v2[i])/denom) << std::endl;
                        assert(std::abs(v1[i] - v2[i]) < 1e-12);
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            MeshMover::LinearElasticity<dim, double, VectorType , dealii::DoFHandler<dim>> 
                meshmover(high_order_grid, surface_node_displacements_vector);
            VectorType volume_displacements = meshmover.get_volume_displacements();

            // Analytical dXvdXs
            meshmover.evaluate_dXvdXs();
            // Start finite difference
            std::vector<VectorType> dXvdXs_FD;
            for (unsigned int isurface = 0; isurface < surface_node_displacements_vector.size(); isurface++) {

                bool iown = surface_node_displacements_vector.locally_owned_elements().is_element(isurface);
                double old_value = 0;
                unsigned int corresponding_volume_dof = 999999999;
                if (iown) {
                    corresponding_volume_dof = high_order_grid.surface_indices[isurface];
                    std::cout << "Performing finite difference for node: " << high_order_grid.surface_indices[isurface] << std::endl;
                    old_value = surface_node_displacements_vector[isurface];
                    surface_node_displacements_vector[isurface] = old_value + fd_eps;
                }
                surface_node_displacements_vector.update_ghost_values();

                MeshMover::LinearElasticity<dim, double, VectorType , dealii::DoFHandler<dim>> 
                    meshmover_p(high_order_grid, surface_node_displacements_vector);

                VectorType volume_displacements_p = meshmover_p.get_volume_displacements();

                high_order_grid.nodes += volume_displacements_p;
                high_order_grid.nodes.update_ghost_values();
                high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);
                high_order_grid.nodes -= volume_displacements_p;
                high_order_grid.nodes.update_ghost_values();
                high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);


                bool central_fd = false;
                VectorType volume_displacements_n;
                double denom = fd_eps;
                if (central_fd) {
                    if (iown) surface_node_displacements_vector[isurface] = old_value - fd_eps;
                    surface_node_displacements_vector.update_ghost_values();

                    MeshMover::LinearElasticity<dim, double, VectorType , dealii::DoFHandler<dim>> 
                        meshmover_n(high_order_grid, surface_node_displacements_vector);

                    volume_displacements_n = meshmover_n.get_volume_displacements();
                    denom *= 2.0;
                } else {
                    volume_displacements_n = volume_displacements;
                }

                volume_displacements_p.add(-1.0, volume_displacements_n);
                volume_displacements_p /= denom;

                dXvdXs_FD.push_back(volume_displacements_p);
                pcout << "Finite difference: " << std::endl;
                dXvdXs_FD[isurface].print(std::cout);
                pcout << "Analytical difference: " << std::endl;
                meshmover.dXvdXs[isurface].print(std::cout);

                if (iown) {
                    std::cout << "dXvdXs_FD: " << dXvdXs_FD[isurface][corresponding_volume_dof]
                            << " meshmover.dXvdXs " << meshmover.dXvdXs[isurface][corresponding_volume_dof]
                            << " should be 1" << std::endl;
                }

                dXvdXs_FD[isurface].add(-1.0,meshmover.dXvdXs[isurface]);

                const double l2_error = dXvdXs_FD[isurface].l2_norm();
                pcout << "*********************************" << std::endl;
                pcout << "L2-norm of difference: " << l2_error << std::endl;
                pcout << "*********************************" << std::endl;

                if (l2_error > 1e-4) {
                    pcout << "Nodes: " << std::endl;
                    high_order_grid.nodes.print(std::cout);
                    pcout << "Error vector: " << std::endl;
                    dXvdXs_FD[isurface].print(std::cout,3,true,false);
                    std::abort();
                }

                if (iown) surface_node_displacements_vector[isurface] = old_value;
                surface_node_displacements_vector.update_ghost_values();
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


