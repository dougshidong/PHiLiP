#include <fstream>

#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_tools.h>

#include "dg/dg_factory.hpp"
#include "parameters/parameters.h"
#include "physics/physics_factory.h"
#include "numerical_flux/convective_numerical_flux.hpp"

using PDEType  = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;
using ConvType = PHiLiP::Parameters::AllParameters::ConvectiveNumericalFlux;
using DissType = PHiLiP::Parameters::AllParameters::DissipativeNumericalFlux;

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

const double TOLERANCE = 1E-5;

/** This test checks that dRdW evaluated using automatic differentiation
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
        (*it) += 1.0;
    }
    dg->solution.update_ghost_values();


    dealii::TrilinosWrappers::SparseMatrix dRdW_fd;
    dealii::SparsityPattern sparsity_pattern = dg->get_dRdW_sparsity_pattern ();

    const dealii::IndexSet &row_parallel_partitioning = dg->locally_owned_dofs;
    const dealii::IndexSet &col_parallel_partitioning = dg->locally_owned_dofs;
    dRdW_fd.reinit(row_parallel_partitioning, col_parallel_partitioning, sparsity_pattern, MPI_COMM_WORLD);

    const double eps = 1e-5;

    pcout << "Evaluating AD..." << std::endl;
    dg->assemble_residual(true, false, false);

    pcout << "Evaluating FD..." << std::endl;
    const unsigned int n_dofs = dg->dof_handler.n_dofs();
    for (unsigned int idof = 0; idof < n_dofs; ++idof) {
        if (idof % 100 == 0) pcout << "idof " << idof+1 << " out of " << n_dofs << std::endl;
        double old_dof = -99999;
        // Positive perturbation
        if (dg->locally_owned_dofs.is_element(idof) ) {
            old_dof = dg->solution[idof];
            dg->solution(idof) = old_dof+eps;
        }
        dg->assemble_residual(false, false, false);
        solutionVector perturbed_residual_p = dg->right_hand_side;

        // Negative perturbation
        if (dg->locally_owned_dofs.is_element(idof) ) {
            dg->solution(idof) = old_dof-eps;
        }
        dg->assemble_residual(false, false, false);
        solutionVector perturbed_residual_m = dg->right_hand_side;

        // Finite-difference
        perturbed_residual_p -= perturbed_residual_m;
        perturbed_residual_p /= (2.0*eps);

        // Reset node
        if (dg->locally_owned_dofs.is_element(idof) ) {
            dg->solution(idof) = old_dof;
        }

        // Set
        for (unsigned int iresidual = 0; iresidual < dg->dof_handler.n_dofs(); ++iresidual) {
            if (dg->locally_owned_dofs.is_element(iresidual) ) {
                const double drdx_entry = perturbed_residual_p[iresidual];
                if (std::abs(drdx_entry) >= 1e-12) {
                    dRdW_fd.add(iresidual,idof,drdx_entry);
                }
            }
        }
    }
    dRdW_fd.compress(dealii::VectorOperation::add);

    dRdW_fd.add(-1.0,dg->system_matrix);

    const double diff_lone_norm = dRdW_fd.l1_norm();
    const double diff_linf_norm = dRdW_fd.linfty_norm();
    pcout << "(dRdW_FD - dRdW_AD) L1-norm = " << diff_lone_norm << std::endl;
    pcout << "(dRdW_FD - dRdW_AD) Linf-norm = " << diff_linf_norm << std::endl;

    if (diff_lone_norm > TOLERANCE) 
    {
        const unsigned int n_digits = 5;
        const unsigned int n_spacing = 7+n_digits;
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        dealii::FullMatrix<double> fullA(dRdW_fd.m(),dRdW_fd.n());
        fullA.copy_from(dRdW_fd);
        pcout<<"Dense matrix from FD-AD:"<<std::endl;

        std::string path = "./FD_minus_AD_matrix.dat";
        std::ofstream outfile (path,std::ofstream::out);

        if (pcout.is_active()) fullA.print_formatted(outfile, n_digits, true, n_spacing, "0", 1., 0.);

        return 1;
    }

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
    std::vector<PDEType> pde_type {
        PDEType::diffusion
        , PDEType::advection
        // , PDEType::convection_diffusion
        , PDEType::advection_vector
        , PDEType::euler
        , PDEType::navier_stokes
    };
    std::vector<std::string> pde_name {
         " PDEType::diffusion "
        , " PDEType::advection "
        // , " PDEType::convection_diffusion "
        , " PDEType::advection_vector "
        , " PDEType::euler "
        , " PDEType::navier_stokes "
    };

    int ipde = -1;
    for (auto pde = pde_type.begin(); pde != pde_type.end() || error == 1; pde++) {
        ipde++;
        for (unsigned int poly_degree=1; poly_degree<3; ++poly_degree) {
            for (unsigned int igrid=2; igrid<4; ++igrid) {
                pcout << "Using " << pde_name[ipde] << std::endl;
                all_parameters.pde_type = *pde;
                all_parameters.diss_num_flux_type = Parameters::AllParameters::DissipativeNumericalFlux::bassi_rebay_2;
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


