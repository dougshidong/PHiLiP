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

    // Generate the original grid and assign a manifold to it
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

    const int n_cells = 2;
    dealii::GridGenerator::subdivided_hyper_cube(grid, n_cells);

    HighOrderGrid<dim,double> high_order_grid(&all_parameters, poly_degree, &grid);

    std::vector<double> grid_size(n_grids);
    std::vector<double> volume_error(n_grids);

    for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

        high_order_grid.prepare_for_coarsening_and_refinement();
        grid.refine_global (1);
        high_order_grid.execute_coarsening_and_refinement();

#if PHILIP_DIM!=1
        high_order_grid.prepare_for_coarsening_and_refinement();
        grid.repartition();
        high_order_grid.execute_coarsening_and_refinement(true);
#endif

        for (auto node1 = high_order_grid.all_surface_nodes.begin(); node1 != high_order_grid.all_surface_nodes.end(); node1+=dim) {
            std::array<double, dim> point;
            for (int d=0;d<dim;++d) {
                point[d] = *(node1+d);
            }
            //double displacement = amplitude * std::sin(2.0*dealii::numbers::PI*
        }
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

