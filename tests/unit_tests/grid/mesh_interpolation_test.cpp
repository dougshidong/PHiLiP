#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/data_out.h> 
#include <filesystem>                
#include <iostream>
#include <fstream>

#include "dg/dg_factory.hpp"
#include "mesh/mesh_adaptation/meshes_interpolation.h"
#include "physics/manufactured_solution.h"
#include "parameters/all_parameters.h"


/// Tests mesh interpolation by interpolating a manufactured solution from one mesh to another
int main(int argc, char* argv[])
{
    const int dim = PHILIP_DIM;
    const int nstate = 1; 
    int fail_bool = false;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank == 0);
    const unsigned int n_procs = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    using namespace PHiLiP;

    // Test parameters
    const unsigned int poly_degree = 3;
    const double tolerance = 1e-3; // Tolerance for L2 error

    // File paths for source and target mesh
    std::string source_mesh_file = "../../integration_tests_control_files/grid_refinement/msh_in/output_ss15_fb_p1_2.msh";
    std::string target_mesh_file = "../../integration_tests_control_files/grid_refinement/msh_out/input_ss15_fb_p2_3.msh";

    // Create source mesh from file
    pcout << "Loading source mesh from: " << source_mesh_file << std::endl;

    // Create parallel distributed triangulation
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> source_grid = std::make_shared<Triangulation>(MPI_COMM_WORLD);

    std::ifstream source_file(source_mesh_file);
    if (!source_file.good()) {
        pcout << "Error: Could not open source mesh file " << source_mesh_file << "\n";
        return 1;
    }

    dealii::GridIn<dim> source_grid_in;
    source_grid_in.attach_triangulation(*source_grid);
    source_grid_in.read_msh(source_file);

    // Create source DG
    pcout << "Creating source DG with polynomial degree " << poly_degree << std::endl;
    Parameters::AllParameters params;
    params.use_weak_form = true;
    params.pde_type = Parameters::AllParameters::convection_diffusion;
    params.conv_num_flux_type = Parameters::AllParameters::ConvectiveNumericalFlux::lax_friedrichs;
    params.diss_num_flux_type = Parameters::AllParameters::DissipativeNumericalFlux::symm_internal_penalty;
    
    auto source_dg = DGFactory<dim, double, Triangulation>::create_discontinuous_galerkin(
        &params, poly_degree, poly_degree, 1, source_grid);
    source_dg->allocate_system();

    // Create and interpolate manufactured solution
    pcout << "Interpolating manufactured solution onto source mesh..." << "\n";
    auto manufactured_solution = std::make_shared<ManufacturedSolutionSShock<dim, double>>();
    // Vector for the solution
    dealii::LinearAlgebra::distributed::Vector<double> source_solution;
    // Distribute dofs
    source_solution.reinit(source_dg->locally_owned_dofs, MPI_COMM_WORLD);

    dealii::VectorTools::interpolate(
        source_dg->dof_handler,
        *manufactured_solution,
        source_solution);

    source_dg->solution = source_solution;
    source_dg->solution.update_ghost_values();

    pcout << "Source mesh: " << source_grid->n_active_cells() << " cells, "
        << source_dg->dof_handler.n_dofs() << " DoFs" << "\n";

    // Check if target mesh exists
    std::ifstream target_check(target_mesh_file);
    if (!target_check.good()) {
        pcout << "Error: Could not open target mesh file " << target_mesh_file << "\n";
        return 1;
    }

    // Perform interpolation using MeshInterpolation class
    pcout << "\nPerforming mesh interpolation..." << "\n";
    MeshInterpolation<dim, nstate, Triangulation> interpolator(pcout.get_stream());

    // This now returns the target_dg with interpolated solution!
    auto target_dg = interpolator.perform_mesh_interpolation(
        source_dg,
        params,
        poly_degree,
        target_mesh_file);

    pcout << "\nTarget mesh: " << target_dg->triangulation->n_active_cells() << " cells, "
        << target_dg->dof_handler.n_dofs() << " DoFs" << std::endl;

    // Outputting the source solution
    pcout << "Outputting source solution to interpolation_output/ " << std::endl;
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(source_dg->dof_handler); //attach triangulation
        data_out.add_data_vector(source_dg->solution, "manufactured_solution"); //add solution
        data_out.build_patches(); //Converts the discrete FE solution into a piecewise©\polynomial patch

        const std::string output_directory = "interpolation_output";
        std::filesystem::create_directory(output_directory);
        const std::string base_filename = output_directory + "/source_solution";
        const std::string filename = base_filename + "." + dealii::Utilities::int_to_string(mpi_rank, 4) + ".vtu";
        std::ofstream output(filename);
        data_out.write_vtu(output);

        if (mpi_rank == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i = 0; i < n_procs; ++i)
            {
                filenames.push_back("source_solution." + dealii::Utilities::int_to_string(i, 4) + ".vtu");
            }
            const std::string master_filename = output_directory + "/source_solution.pvtu";
            std::ofstream master_output(master_filename);
            data_out.write_pvtu_record(master_output, filenames);
        }
    }

    // Outputting the target solution
    pcout << "Outputting target solution to interpolation_output/ " << std::endl;
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(target_dg->dof_handler);
        data_out.add_data_vector(target_dg->solution, "interpolated_solution");
        data_out.build_patches();

        const std::string output_directory = "interpolation_output";
        std::filesystem::create_directory(output_directory);
        const std::string base_filename = output_directory + "/target_solution";
        const std::string filename = base_filename + "." + dealii::Utilities::int_to_string(mpi_rank, 4) + ".vtu";
        std::ofstream output(filename);
        data_out.write_vtu(output);

        if (mpi_rank == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i = 0; i < n_procs; ++i)
            {
                filenames.push_back("target_solution." + dealii::Utilities::int_to_string(i, 4) + ".vtu");
            }
            const std::string master_filename = output_directory + "/target_solution.pvtu";
            std::ofstream master_output(master_filename);
            data_out.write_pvtu_record(master_output, filenames);
        }
    }

    // Compute L2 error (integrate_difference)
    pcout << "\nComputing L2 error using integrate_difference " << std::endl;

    dealii::Vector<double> difference_per_cell(target_dg->triangulation->n_active_cells());
    dealii::VectorTools::integrate_difference(
        target_dg->dof_handler,
        target_dg->solution,
        *manufactured_solution,
        difference_per_cell,
        dealii::QGauss<dim>(poly_degree + 2),
        dealii::VectorTools::L2_norm);

    const double l2_error = dealii::VectorTools::compute_global_error(
        *target_dg->triangulation,
        difference_per_cell,
        dealii::VectorTools::L2_norm);

    //const double l2_error = std::sqrt(difference_per_cell.norm_sqr());

    // compute L2 norm of exact solution for relative error
    dealii::Vector<double> exact_per_cell(target_dg->triangulation->n_active_cells());
    dealii::LinearAlgebra::distributed::Vector<double> zero_vector;
    zero_vector.reinit(target_dg->locally_owned_dofs, MPI_COMM_WORLD);
    zero_vector = 0.0;

    dealii::VectorTools::integrate_difference(
        target_dg->dof_handler,
        zero_vector,
        *manufactured_solution,
        exact_per_cell,
        dealii::QGauss<dim>(poly_degree + 2),
        dealii::VectorTools::L2_norm);

    const double l2_exact = dealii::VectorTools::compute_global_error(
        *target_dg->triangulation,
        exact_per_cell,
        dealii::VectorTools::L2_norm);

    const double relative_l2_error = l2_error / l2_exact;

    pcout << "L2 Error: " << l2_error << std::endl;
    pcout << "L2 Exact: " << l2_exact << std::endl;
    pcout << "Relative L2 Error: " << relative_l2_error << std::endl;

    // Check if error is within tolerance
    if (relative_l2_error > tolerance) {
        pcout << "\nTest failed. Relative L2 error " << relative_l2_error
            << " exceeds tolerance " << tolerance << std::endl;
        fail_bool = true;
    }
    else {
        pcout << "\n Relative L2 error within tolerance." << std::endl;
        fail_bool = false;
    }

    return fail_bool;

}