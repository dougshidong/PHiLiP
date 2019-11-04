#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

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

// template<int dim>
// void output_grid(const std::string prefix, const unsigned int poly_degree, PHiLiP::HighOrderGrid<dim,double> &high_order_grid)
// {
//     const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
//     // Output for Paraview visualization
//     dealii::DataOut<dim> data_out;
//     data_out.attach_dof_handler(high_order_grid.dof_handler_grid);
//     std::vector<std::string> solution_names;
//     for (int d=0;d<dim;++d) {
//         if (d==0) solution_names.push_back("x");
//         if (d==1) solution_names.push_back("y");
//         if (d==2) solution_names.push_back("z");
//     }
//     data_out.add_data_vector(high_order_grid.nodes, solution_names);
// 
//     auto mapping = (high_order_grid.mapping_fe_field);
//     data_out.build_patches(*mapping, poly_degree, dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells);
//     std::string filename = prefix + dealii::Utilities::int_to_string(dim, 1) +"D-";
//     filename += "Degree" + dealii::Utilities::int_to_string(poly_degree, 2) + ".";
//     filename += dealii::Utilities::int_to_string(mpi_rank, 4);
//     filename += ".vtu";
//     std::ofstream output(filename);
//     data_out.write_vtu(output);
// 
//     if (mpi_rank == 0) {
//         std::vector<std::string> filenames;
//         for (unsigned int mpi_rank = 0; mpi_rank < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++mpi_rank) {
//             std::string fn = prefix + dealii::Utilities::int_to_string(dim, 1) +"D-";
//             fn += "Degree" + dealii::Utilities::int_to_string(poly_degree, 2) + ".";
//             fn += dealii::Utilities::int_to_string(mpi_rank, 4);
//             fn += ".vtu";
//             filenames.push_back(fn);
//         }
//         std::string master_fn = prefix + dealii::Utilities::int_to_string(dim, 1) +"D-";
//         master_fn += "Degree" + dealii::Utilities::int_to_string(poly_degree, 2) + ".";
//         master_fn += ".pvtu";
//         std::ofstream master_output(master_fn);
//         data_out.write_pvtu_record(master_output, filenames);
//     }
// }

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

    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    const int poly_degree = 4;
    const unsigned int n_grids = 3;
    //const std::vector<int> n_1d_cells = {2,4,8,16};

    //const unsigned int n_cells_circle = n_1d_cells[0];
    //const unsigned int n_cells_radial = 3*n_cells_circle;

    dealii::Point<dim> center; // Constructor initializes Point at the origin
    const double inner_radius = 1, outer_radius = inner_radius*10;

    // Generate the original grid and assign a manifold to it
    dealii::parallel::distributed::Triangulation<dim> grid(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    const int n_cells = 0;
    const bool colorize = true;
    dealii::GridGenerator::quarter_hyper_shell<dim>(grid, center, inner_radius, outer_radius, n_cells, colorize);
    // Set a spherical manifold
    // Works but we will usually not have a volume manifold that we can provide.
    //grid.set_all_manifold_ids(0);
    //grid.set_manifold(0, dealii::SphericalManifold<dim>(center));

    // Set a spherical manifold on the boundary and a TransfiniteInterpolationManifold in the domain
    // This is more realistic with what we will be doing since we will usually provide and boundary parametrization
    // but have no idea of the volume parametrization. The initial curving of the TransfiniteInterpolationManifold
    // will ensure that no cells initially have a negative Jacobians. This curvature will be transfered to a polynomial
    // representation using MappingFEField
    grid.set_all_manifold_ids(1);
    grid.set_all_manifold_ids_on_boundary(0);
    grid.set_manifold(0, dealii::SphericalManifold<dim>(center));
    dealii::TransfiniteInterpolationManifold<dim> transfinite_interpolation;
    transfinite_interpolation.initialize(grid);
    grid.set_manifold(1, transfinite_interpolation);


    HighOrderGrid<dim,double> high_order_grid(&all_parameters, poly_degree, &grid);
    //std::shared_ptr < dealii::MappingFEField<dim,dim,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> > mapping = (high_order_grid.mapping_fe_field);
    grid.reset_all_manifolds();

    dealii::ConvergenceTable convergence_table;
    std::vector<double> grid_size(n_grids);
    std::vector<double> volume_error(n_grids);

    for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

        high_order_grid.prepare_for_coarsening_and_refinement();

        grid.refine_global (1);
        //int icell = 0;
        //for (auto cell = grid.begin_active(); cell!=grid.end(); ++cell) {
        //    if (!cell->is_locally_owned()) continue;
        //    icell++;
        //    if (icell < 3) {
        //        cell->set_refine_flag();
        //    } else if (icell%5 == 0) {
        //        cell->set_refine_flag();
        //    } else if (icell%3 == 0) {
        //        cell->set_coarsen_flag();
        //    }
        //}
        //grid.execute_coarsening_and_refinement();

        high_order_grid.execute_coarsening_and_refinement();

        high_order_grid.prepare_for_coarsening_and_refinement();
        grid.repartition();
        high_order_grid.execute_coarsening_and_refinement(true);
    }

    //output_grid("before", poly_degree, high_order_grid);
    // Deform the y = 0 face
    for (auto indices = high_order_grid.locally_owned_surface_nodes_indices.begin(); indices!=high_order_grid.locally_owned_surface_nodes_indices.end(); ++indices) {
    }
    

    //const unsigned int n_dofs = high_order_grid.dof_handler_grid.n_dofs();
    //const unsigned int n_global_active_cells = grid.n_global_active_cells();


    if (fail_bool) {
        pcout << "Test failed. The estimated error should be the same for a given p, even after refinement and translation." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}

