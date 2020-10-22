#include <deal.II/grid/grid_out.h>
#include "mesh/gmsh_reader.hpp"

int main (int argc, char * argv[])
{
    const int dim = PHILIP_DIM;
    int fail_bool = false;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;

    std::string filename = std::to_string(dim) + "D_square.msh";

    std::shared_ptr< HighOrderGrid<dim, double> > high_order_grid = read_gmsh <dim, dim> (filename);


    dealii::GridOut gridout;
    gridout.write_mesh_per_processor_as_vtu(*(high_order_grid->triangulation), "tria");

    high_order_grid->output_results_vtk(0);


    if (fail_bool) {
        pcout << "Test failed. The estimated error should be the same for a given p, even after refinement and translation." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}


