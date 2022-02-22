#include <fstream>

#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_tools.h>

#include "ode_solver/ode_solver_factory.h"
#include "dg/dg_factory.hpp"
#include "parameters/parameters.h"
#include "physics/physics_factory.h"

using PDEType  = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

const double TOLERANCE = 1E-4;
const double EPS = 1E-4;

/** This test checks that dRdX evaluated using automatic differentiation
 *  matches with the results obtained using finite-difference.
 */
template<int dim, int nstate>
int test (
    const unsigned int poly_degree,
    const std::shared_ptr<Triangulation> grid,
    const PHiLiP::Parameters::AllParameters &all_parameters)
{
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    using namespace PHiLiP;
    // Assemble Jacobian
    std::shared_ptr < DGBase<PHILIP_DIM, double> > dg = DGFactory<PHILIP_DIM,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, grid);

    const int n_refine = 1;
    for (int i=0; i<n_refine;i++) {
        dg->high_order_grid->prepare_for_coarsening_and_refinement();
        grid->prepare_coarsening_and_refinement();
        unsigned int icell = 0;
        for (auto cell = grid->begin_active(); cell!=grid->end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            icell++;
            if (icell < grid->n_active_cells()/2) {
                cell->set_refine_flag();
            }
        }
        grid->execute_coarsening_and_refinement();
        bool mesh_out = (i==n_refine-1);
        dg->high_order_grid->execute_coarsening_and_refinement(mesh_out);
    }
    dg->allocate_system ();

    pcout << "Poly degree " << poly_degree << " ncells " << grid->n_global_active_cells() << " ndofs: " << dg->dof_handler.n_dofs() << std::endl;

    // Initialize solution with something
    using solutionVector = dealii::LinearAlgebra::distributed::Vector<double>;

    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    solutionVector solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
    dg->solution = solution_no_ghost;
    for (auto it = dg->solution.begin(); it != dg->solution.end(); ++it) {
        // Interpolating the exact manufactured solution caused some problems at the boundary conditions.
        // The manufactured solution is exactly equal to the manufactured_solution_function at the boundary,
        // therefore, the finite difference will change whether the flow is incoming or outgoing.
        // As a result, we would be differentiating at a non-differentiable point.
        // Hence, we fix this issue by taking the second derivative at a non-exact solution.
        //(*it) += 1.0;
    }
    dg->solution.update_ghost_values();

    // Solving the flow to make sure that we're not at the point of non-differentiality between elements.
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->steady_state();

    // Set dual to 1.0 so that every 2nd derivative of the residual is accounted for.
    for (auto it = dg->dual.begin(); it != dg->dual.end(); ++it) {
        (*it) = 1.0;
    }
    dg->dual.update_ghost_values();


    dealii::TrilinosWrappers::SparseMatrix d2RdWdW_fd;
    dealii::SparsityPattern sparsity_pattern = dg->get_d2RdWdW_sparsity_pattern();

    const dealii::IndexSet &row_parallel_partitioning = dg->locally_owned_dofs;
    const dealii::IndexSet &col_parallel_partitioning = dg->locally_owned_dofs;
    d2RdWdW_fd.reinit(row_parallel_partitioning, col_parallel_partitioning, sparsity_pattern, MPI_COMM_WORLD);

    pcout << "Evaluating AD..." << std::endl;
    double timing_start = MPI_Wtime();
    int n = 5;
    for (int i=0; i < n; ++i) {
        std::cout << i << " out of " << n << std::endl;
        *(dg->solution.begin()) += 1e-7;
        dg->assemble_residual(false, false, true);
    }
    double timing_end = MPI_Wtime();
    std::cout << "The process took " << timing_end - timing_start << " seconds to run." << std::endl;

    return 0;
}

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;
    const int dim = PHILIP_DIM;
    int error = 0;
    //int success_bool = true;

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);

    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    all_parameters.ode_solver_param.initial_time_step = 1e+2;
    //all_parameters.ode_solver_param.time_step_factor_residual_exp = 4.0;
    all_parameters.ode_solver_param.time_step_factor_residual = 25;
    //all_parameters.ode_solver_param.output_solution_every_x_steps = 1;
    all_parameters.ode_solver_param.time_step_factor_residual_exp = 4.0;
    all_parameters.ode_solver_param.nonlinear_max_iterations = 2;
    //all_parameters.linear_solver_param.linear_solver_type = PHiLiP::Parameters::LinearSolverParam::LinearSolverEnum::direct;
    std::vector<PDEType> pde_type {
         //  PDEType::diffusion
         //, PDEType::advection
         //, PDEType::convection_diffusion
         //, PDEType::advection_vector
          PDEType::euler
        , PDEType::navier_stokes 
    };
    std::vector<std::string> pde_name {
        " PDEType::euler ",
        " PDEType::navier_stokes "
        // " PDEType::diffusion "
        //, " PDEType::advection "
        //, " PDEType::convection_diffusion "
        //, " PDEType::advection_vector "
        //, " PDEType::euler "
    };

    int ipde = -1;
    for (auto pde = pde_type.begin(); pde != pde_type.end() || error == 1; pde++) {
        ipde++;
        for (unsigned int poly_degree=4; poly_degree<5; ++poly_degree) {
            for (unsigned int igrid=3; igrid<4; ++igrid) {
                pcout << "Using " << pde_name[ipde] << std::endl;
                all_parameters.pde_type = *pde;
                // Generate grids
                std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
#if PHILIP_DIM!=1
                    MPI_COMM_WORLD,
#endif
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::smoothing_on_refinement |
                        dealii::Triangulation<dim>::smoothing_on_coarsening));

                dealii::GridGenerator::subdivided_hyper_cube(*grid, igrid);

                const double random_factor = 0.2;
                const bool keep_boundary = false;
                if (random_factor > 0.0) dealii::GridTools::distort_random (random_factor, *grid, keep_boundary);
                for (auto &cell : grid->active_cell_iterators()) {
                    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                        if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
                    }
                }

                if ((*pde==PDEType::euler) || (*pde==PDEType::navier_stokes)) {
                    error = test<dim,dim+2>(poly_degree, grid, all_parameters);
                } else if (*pde==PDEType::burgers_inviscid) {
                    error = test<dim,dim>(poly_degree, grid, all_parameters);
                } else if (*pde==PDEType::advection_vector) {
                    error = test<dim,2>(poly_degree, grid, all_parameters);
                } else {
                    error = test<dim,1>(poly_degree, grid, all_parameters);
                }
                if (error) return error;
            }
        }
    }

    return error;
}




