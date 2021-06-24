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

using PDEType  = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

const double TOLERANCE = 1E-5;
const double EPS = 1e-6;

/** This test checks that dRdX evaluated using automatic differentiation
 *  matches with the results obtained using finite-difference.
 */
template<int dim, int nstate>
int test (
    const unsigned int poly_degree,
    std::shared_ptr<Triangulation> grid,
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
    dg->high_order_grid->ensure_conforming_mesh();
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


    dealii::TrilinosWrappers::SparseMatrix dRdXv_fd;
    dealii::SparsityPattern sparsity_pattern = dg->get_dRdX_sparsity_pattern ();

    const dealii::IndexSet &row_parallel_partitioning = dg->locally_owned_dofs;
    const dealii::IndexSet &col_parallel_partitioning = dg->high_order_grid->locally_owned_dofs_grid;
    dRdXv_fd.reinit(row_parallel_partitioning, col_parallel_partitioning, sparsity_pattern, MPI_COMM_WORLD);

    std::shared_ptr<PHiLiP::HighOrderGrid<dim,double>> high_order_grid = dg->high_order_grid;

    using nodeVector = dealii::LinearAlgebra::distributed::Vector<double>;
    nodeVector old_volume_nodes = high_order_grid->volume_nodes;
    old_volume_nodes.update_ghost_values();

    dealii::AffineConstraints<double> hanging_node_constraints;
    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(high_order_grid->dof_handler_grid, hanging_node_constraints);
    hanging_node_constraints.close();

    pcout << "Evaluating AD..." << std::endl;
    dg->assemble_residual(false, true, false);

    pcout << "Evaluating FD..." << std::endl;
    for (unsigned int inode = 0; inode<high_order_grid->dof_handler_grid.n_dofs(); ++inode) {
        if (inode % 100 == 0) pcout << "inode " << inode+1 << " out of " << high_order_grid->dof_handler_grid.n_dofs() << std::endl;
        double old_node = -99999;
        // Positive perturbation
        if (high_order_grid->locally_relevant_dofs_grid.is_element(inode) ) {
            old_node = high_order_grid->volume_nodes[inode];
            high_order_grid->volume_nodes(inode) = old_node+EPS;
        }
        // This should be uncommented once we fix:
        // https://github.com/dougshidong/PHiLiP/issues/48#issue-771199898
        //high_order_grid->ensure_conforming_mesh();
        //hanging_node_constraints.distribute(high_order_grid->volume_nodes);
        //high_order_grid->volume_nodes.update_ghost_values();

        dg->assemble_residual(false, false, false);
        solutionVector perturbed_residual_p = dg->right_hand_side;

        //high_order_grid->volume_nodes = old_volume_nodes;
        //high_order_grid->volume_nodes.update_ghost_values();

        // Negative perturbation
        if (high_order_grid->locally_relevant_dofs_grid.is_element(inode) ) {
            high_order_grid->volume_nodes(inode) = old_node-EPS;
        }
        // This should be uncommented once we fix:
        // https://github.com/dougshidong/PHiLiP/issues/48#issue-771199898
        //high_order_grid->ensure_conforming_mesh();
        //hanging_node_constraints.distribute(high_order_grid->volume_nodes);
        //high_order_grid->volume_nodes.update_ghost_values();

        dg->assemble_residual(false, false, false);
        solutionVector perturbed_residual_m = dg->right_hand_side;

        //high_order_grid->volume_nodes = old_volume_nodes;
        //high_order_grid->volume_nodes.update_ghost_values();

        // Finite difference
        perturbed_residual_p -= perturbed_residual_m;
        perturbed_residual_p /= (2.0*EPS);

        // Reset node
        if (high_order_grid->locally_relevant_dofs_grid.is_element(inode) ) {
            high_order_grid->volume_nodes(inode) = old_node;
        }

        // Set
        for (unsigned int iresidual = 0; iresidual < dg->dof_handler.n_dofs(); ++iresidual) {
            if (dg->locally_owned_dofs.is_element(iresidual) ) {
                const double drdx_entry = perturbed_residual_p[iresidual];
                if (std::abs(drdx_entry) >= 1e-12) {
                    std::cout << iresidual << " " << inode << std::endl;
                    dRdXv_fd.add(iresidual,inode,drdx_entry);
                }
            }
        }
    }
    dRdXv_fd.compress(dealii::VectorOperation::add);

    pcout << "(dRdX_FD frob_norm) " << dRdXv_fd.frobenius_norm();
    pcout << "(dRdX_AD frob_norm) " << dg->dRdXv.frobenius_norm() << std::endl;
    dRdXv_fd.add(-1.0,dg->dRdXv);

    const double diff_lone_norm = dRdXv_fd.l1_norm();
    const double diff_linf_norm = dRdXv_fd.linfty_norm();
    pcout << "(dRdX_FD - dRdX_AD) L1-norm = " << diff_lone_norm << std::endl;
    pcout << "(dRdX_FD - dRdX_AD) Linf-norm = " << diff_linf_norm << std::endl;

    if (diff_lone_norm > TOLERANCE) 
    {
        const unsigned int n_digits = 5;
        const unsigned int n_spacing = 7+n_digits;
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        dealii::FullMatrix<double> fullA(dRdXv_fd.m(),dRdXv_fd.n());
        fullA.copy_from(dRdXv_fd);
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


