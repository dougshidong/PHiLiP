#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include "euler_gaussian_bump.h"
#include "mesh/grids/gaussian_bump.h"

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
EulerBumpResidualAssembly<dim,nstate>::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(dim == 2, dealii::ExcDimensionMismatch(dim, param.dimension));
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
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // p0 tends to require a finer grid to reach asymptotic region
        unsigned int n_grids = n_grids_input;
        if (poly_degree <= 1) n_grids = n_grids_input;

        std::vector<double> grid_size(n_grids);

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

        std::vector<unsigned int> n_subdivisions(dim);
        n_subdivisions[1] = n_1d_cells[0]; // y-direction
        n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction

        // const double channel_length = 3.0;
        // const double channel_height = 0.8;
        // Grids::gaussian_bump(*grid, n_subdivisions, channel_length, channel_height);

        // const double solution_degree = poly_degree;
        // const double grid_degree = 3;
        // // Create DG object
        // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, &grid);

        // // Initialize coarse grid solution with free-stream
        // dg->allocate_system ();
        // dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

        // // Create ODE solver and ramp up the solution from p0
        // std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        // ode_solver->initialize_steady_polynomial_ramping (poly_degree);

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {


            //if (igrid!=0) {
            //    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
            //    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim>> solution_transfer(dg->dof_handler);
            //    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
            //    dg->high_order_grid.prepare_for_coarsening_and_refinement();
            //    grid->refine_global (1);
            //    dg->high_order_grid.execute_coarsening_and_refinement(true);
            //    dg->allocate_system ();
            //    dg->solution.zero_out_ghosts();
            //    solution_transfer.interpolate(dg->solution);
            //    dg->solution.update_ghost_values();
            //}

            using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                MPI_COMM_WORLD,
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));

            const double channel_length = 3.0;
            const double channel_height = 0.8;
            Grids::gaussian_bump(*grid, n_subdivisions, channel_length, channel_height);
            grid->refine_global(igrid);

            const double solution_degree = poly_degree;
            const double grid_degree = solution_degree+1;
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, grid);

            // Initialize coarse grid solution with free-stream
            dg->allocate_system ();
            dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Create ODE solver and ramp up the solution from p0
            std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
            ode_solver->initialize_steady_polynomial_ramping (poly_degree);


            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

            const double entropy_inf = euler_physics_double.entropy_inf;

            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));


            // Convergence table
            double dx = 1.0/pow(n_dofs,(1.0/dim));
            //dx = dealii::GridTools::maximal_cell_diameter(*grid);
            grid_size[igrid] = dx;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("L2_entropy_error", l2error_mpi_sum);
        }
        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates("L2_entropy_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("L2_entropy_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;

        const double last_slope = log(entropy_error[n_grids-1]/entropy_error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        //double before_last_slope = last_slope;
        //if ( n_grids > 2 ) {
        //    before_last_slope = log(entropy_error[n_grids-2]/entropy_error[n_grids-3])
        //                        / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        //}
        //const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_avg = last_slope;
        const double slope_diff = slope_avg-expected_slope;

        double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);
        if(poly_degree == 0) slope_deficit_tolerance *= 2; // Otherwise, grid sizes need to be much bigger for p=0

        if (slope_diff < slope_deficit_tolerance) {
            pcout << std::endl
                 << "Convergence order not achieved. Average last 2 slopes of "
                 << slope_avg << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
            // p=0 just requires too many meshes to get into the asymptotic region.
            if(poly_degree!=0) fail_conv_poly.push_back(poly_degree);
            if(poly_degree!=0) fail_conv_slop.push_back(slope_avg);
        }

    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }
    int n_fail_poly = fail_conv_poly.size();
    if (n_fail_poly > 0) {
        for (int ifail=0; ifail < n_fail_poly; ++ifail) {
            const double expected_slope = fail_conv_poly[ifail]+1;
            const double slope_deficit_tolerance = -0.1;
            pcout << std::endl
                 << "Convergence order not achieved for polynomial p = "
                 << fail_conv_poly[ifail]
                 << ". Slope of "
                 << fail_conv_slop[ifail] << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
        }
    }
    return n_fail_poly;
}


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
        EulerBumpResidualAssembly<dim,nstate> test();
        test.run_test();

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


#if PHILIP_DIM==2
    template class EulerBumpResidualAssembly <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

