#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/vector_tools.h>

#include "mesh/grids/gaussian_bump.h"

#include "testing/tests.h"

#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"


namespace PHiLiP {
namespace Tests {

/// Performs grid convergence for various polynomial degrees.
template <int dim, int nstate>
class EulerBumpResidualAssembly: public TestsBase
{
public:
    EulerBumpResidualAssembly () = delete;
    EulerBumpResidualAssembly(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
    {};

    int run_test () const;
};

template<int dim, int nstate>
int EulerBumpResidualAssembly<dim,nstate>::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(dim != 1, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int n_grids_input       = manu_grid_conv_param.number_of_grids;

    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);

    std::vector<dealii::ConvergenceTable> convergence_table_vector_residual_assembly;
    std::vector<dealii::ConvergenceTable> convergence_table_vector_dRdW_vmult;
    std::vector<dealii::ConvergenceTable> convergence_table_vector_dRdX_vmult;
    std::vector<dealii::ConvergenceTable> convergence_table_vector_dRdW;

    // p0 tends to require a finer grid to reach asymptotic region
    unsigned int n_grids = n_grids_input;

    std::vector<double> grid_size(n_grids);

    const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);


    std::vector<unsigned int> n_subdivisions(dim);
    n_subdivisions[1] = n_1d_cells[0]; // y-direction
    n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction
    if (dim == 3) n_subdivisions[2] = n_subdivisions[1];

    for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

        dealii::ConvergenceTable convergence_table_residual_assembly;
        dealii::ConvergenceTable convergence_table_dRdW_vmult;
        dealii::ConvergenceTable convergence_table_dRdX_vmult;
        dealii::ConvergenceTable convergence_table_dRdW;

        for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

            using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                MPI_COMM_WORLD,
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));

            const double channel_length = 3.0;
            const double channel_height = 0.8;
            Grids::gaussian_bump<dim>(*grid, n_subdivisions, channel_length, channel_height);
            grid->refine_global(igrid);

            const double solution_degree = poly_degree;
            const double grid_degree = solution_degree+1;
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, grid);

            // Initialize coarse grid solution with free-stream
            dg->allocate_system ();
            dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
            dg->output_results_vtk(igrid*1000+poly_degree);

            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            double timing_residual;
            { 
                double timing_start = MPI_Wtime();
                for (int i = 0; i < 10; ++i) {
                    dg->assemble_residual ();
                }
                double timing_end = MPI_Wtime();
                timing_residual = timing_end - timing_start;
                pcout << "It took " << timing_residual << " seconds to run." << std::endl;

                // Convergence table
                double dx = 1.0/pow(n_dofs,(1.0/dim));
                //dx = dealii::GridTools::maximal_cell_diameter(*grid);
                grid_size[igrid] = dx;

                convergence_table_residual_assembly.add_value("p", poly_degree);
                convergence_table_residual_assembly.add_value("cells", n_global_active_cells);
                convergence_table_residual_assembly.add_value("DoFs", n_dofs);
                convergence_table_residual_assembly.add_value("dx", dx);
                convergence_table_residual_assembly.add_value("Timing", timing_residual);
            }
            { 
                const bool compute_dRdW = true;
                double timing_start = MPI_Wtime();
                dg->assemble_residual (compute_dRdW);
                double timing_end = MPI_Wtime();
                const int n_nonzero_elements = dg->system_matrix.n_nonzero_elements();
                pcout << "Number of nonzeros in dRdW: " << n_nonzero_elements << std::endl;
                double timing = timing_end - timing_start;
                pcout << "It took " << 10*timing << " seconds to run." << std::endl;

                // Convergence table
                double dx = 1.0/pow(n_dofs,(1.0/dim));
                //dx = dealii::GridTools::maximal_cell_diameter(*grid);
                grid_size[igrid] = dx;

                convergence_table_dRdW.add_value("p", poly_degree);
                convergence_table_dRdW.add_value("cells", n_global_active_cells);
                convergence_table_dRdW.add_value("DoFs", n_dofs);
                convergence_table_dRdW.add_value("dx", dx);
                convergence_table_dRdW.add_value("Timing", timing);
                convergence_table_dRdW.add_value("Relative Timing", timing/timing_residual);
            }
            { 
                double timing_start = MPI_Wtime();
                for (int i = 0; i < 10; ++i) {
                    dg->system_matrix.vmult(dg->right_hand_side, dg->solution);
                }
                double timing_end = MPI_Wtime();
                double timing = timing_end - timing_start;
                pcout << "It took " << timing << " seconds to run." << std::endl;

                // Convergence table
                double dx = 1.0/pow(n_dofs,(1.0/dim));
                //dx = dealii::GridTools::maximal_cell_diameter(*grid);
                grid_size[igrid] = dx;

                convergence_table_dRdW_vmult.add_value("p", poly_degree);
                convergence_table_dRdW_vmult.add_value("cells", n_global_active_cells);
                convergence_table_dRdW_vmult.add_value("DoFs", n_dofs);
                convergence_table_dRdW_vmult.add_value("dx", dx);
                convergence_table_dRdW_vmult.add_value("Timing", timing);
                convergence_table_dRdW_vmult.add_value("Relative Timing", timing/timing_residual);
            }
            { 
                const bool compute_dRdW = false;
                const bool compute_dRdX = true;
                int n_nonzero_elements = dg->dRdXv.n_nonzero_elements();
                pcout << "Number of nonzeros in dRdX: " << n_nonzero_elements << std::endl;
                dg->assemble_residual (compute_dRdW, compute_dRdX);
                n_nonzero_elements = dg->dRdXv.n_nonzero_elements();
                pcout << "Number of nonzeros in dRdX: " << n_nonzero_elements << std::endl;
                double timing_start = MPI_Wtime();
                for (int i = 0; i < 10; ++i) {
                    dg->dRdXv.vmult(dg->right_hand_side, dg->high_order_grid.volume_nodes);
                }
                double timing_end = MPI_Wtime();
                double timing = timing_end - timing_start;
                pcout << "It took " << timing << " seconds to run." << std::endl;

                // Convergence table
                double dx = 1.0/pow(n_dofs,(1.0/dim));
                //dx = dealii::GridTools::maximal_cell_diameter(*grid);
                grid_size[igrid] = dx;

                convergence_table_dRdX_vmult.add_value("p", poly_degree);
                convergence_table_dRdX_vmult.add_value("cells", n_global_active_cells);
                convergence_table_dRdX_vmult.add_value("DoFs", n_dofs);
                convergence_table_dRdX_vmult.add_value("dx", dx);
                convergence_table_dRdX_vmult.add_value("Timing", timing);
                convergence_table_dRdX_vmult.add_value("Relative Timing", timing/timing_residual);
            }

        }
        convergence_table_residual_assembly.evaluate_convergence_rates("Timing", "p", dealii::ConvergenceTable::reduction_rate, dim);
        convergence_table_residual_assembly.set_scientific("dx", true);
        convergence_table_residual_assembly.set_scientific("Timing", true);
        if (pcout.is_active()) convergence_table_residual_assembly.write_text(pcout.get_stream());
        convergence_table_vector_residual_assembly.push_back(convergence_table_residual_assembly);

        convergence_table_dRdW_vmult.evaluate_convergence_rates("Timing", "p", dealii::ConvergenceTable::reduction_rate, dim);
        convergence_table_dRdW_vmult.set_scientific("dx", true);
        convergence_table_dRdW_vmult.set_scientific("Timing", true);
        if (pcout.is_active()) convergence_table_dRdW_vmult.write_text(pcout.get_stream());
        convergence_table_vector_dRdW_vmult.push_back(convergence_table_dRdW_vmult);

        convergence_table_dRdW.evaluate_convergence_rates("Timing", "p", dealii::ConvergenceTable::reduction_rate, dim);
        convergence_table_dRdW.set_scientific("dx", true);
        convergence_table_dRdW.set_scientific("Timing", true);
        if (pcout.is_active()) convergence_table_dRdW.write_text(pcout.get_stream());
        convergence_table_vector_dRdW.push_back(convergence_table_dRdW);

        convergence_table_dRdX_vmult.evaluate_convergence_rates("Timing", "p", dealii::ConvergenceTable::reduction_rate, dim);
        convergence_table_dRdX_vmult.set_scientific("dx", true);
        convergence_table_dRdX_vmult.set_scientific("Timing", true);
        if (pcout.is_active()) convergence_table_dRdX_vmult.write_text(pcout.get_stream());
        convergence_table_vector_dRdX_vmult.push_back(convergence_table_dRdX_vmult);
    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Residual assembly timing summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector_residual_assembly.begin(); conv!=convergence_table_vector_residual_assembly.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " dRdW vmult timing summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector_dRdW_vmult.begin(); conv!=convergence_table_vector_dRdW_vmult.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " dRdX vmult timing summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector_dRdX_vmult.begin(); conv!=convergence_table_vector_dRdX_vmult.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " dRdW timing summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector_dRdW.begin(); conv!=convergence_table_vector_dRdW.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

    const int no_error = 0;
    return no_error;
}



#if PHILIP_DIM==2
    template class EulerBumpResidualAssembly <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

int main (int argc, char *argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int n_mpi = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (n_mpi==1 || mpi_rank==0) {
		dealii::deallog.depth_console(99);
	} else {
		dealii::deallog.depth_console(0);
	}

    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    pcout << "Starting program with " << n_mpi << " processors..." << std::endl;
    if ((PHILIP_DIM==1) && !(n_mpi==1)) {
        std::cout << "********************************************************" << std::endl;
        std::cout << "Can't use mpirun -np X, where X>1, for 1D." << std::endl
                  << "Currently using " << n_mpi << " processors." << std::endl
                  << "Aborting..." << std::endl;
        std::cout << "********************************************************" << std::endl;
        std::abort();
    }
    int test_error = 1;
    try
    {
        // Declare possible inputs
        dealii::ParameterHandler parameter_handler;
        PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
        PHiLiP::Parameters::parse_command_line (argc, argv, parameter_handler);

        // Read inputs from parameter file and set those values in AllParameters object
        PHiLiP::Parameters::AllParameters all_parameters;
        pcout << "Reading input..." << std::endl;
        all_parameters.parse_parameters (parameter_handler);

        AssertDimension(all_parameters.dimension, PHILIP_DIM);

        const int dim = PHILIP_DIM;
        const int nstate = dim+2;
        PHiLiP::Tests::EulerBumpResidualAssembly<dim,nstate> test(&all_parameters);
        test_error = test.run_test();

        pcout << "Finished test with test error code: " << test_error << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl
                  << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl
                  << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    //std::cout << "MPI process " << mpi_rank+1 << " out of " << n_mpi << "reached end of program." << std::endl;
    pcout << "End of program" << std::endl;
    return test_error;
}


