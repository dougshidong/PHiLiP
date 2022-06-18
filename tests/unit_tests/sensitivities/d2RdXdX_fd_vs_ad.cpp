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

const double TOLERANCE = 1E-3;
const double EPS = 1E-4;

//    const unsigned int inode, const bool inode_relevant,
//    const unsigned int jnode, const bool jnode_relevant,
//    const int dipert,
//    const int djpert,
//    const double EPS,
//    const PHiLiP::DGBase<PHILIP_DIM, double> &dg,
//    )
//{
//    double old_inode, old_jnode;
//    std::shared_ptr<PHiLiP::HighOrderGrid<dim,double>> high_order_grid = dg->high_order_grid;
//
//    if (inode_relevant) {
//        old_inode = high_order_grid->volume_nodes(inode);
//        high_order_grid->volume_nodes(inode) = old_inode+dipert*EPS;
//    }
//    if (jnode_relevant) {
//        old_jnode = high_order_grid->volume_nodes(jnode);
//        if (inode == jnode) {
//            high_order_grid->volume_nodes(jnode) += j*EPS;
//        } else {
//            high_order_grid->volume_nodes(jnode) = old_jnode+djpert*EPS;
//        }
//    }
//    dg->assemble_residual(false, false, false);
//    perturbed_dual_dot_residual[ij] = dg->right_hand_side * dg->dual;
//
//    if (inode_relevant) {
//        high_order_grid->volume_nodes(inode) = old_inode;
//    }
//    if (jnode_relevant) {
//        high_order_grid->volume_nodes(jnode) = old_jnode;
//    }
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
    dg->allocate_system ();

    pcout << "Poly degree " << poly_degree << " ncells " << grid->n_global_active_cells() << " ndofs: " << dg->dof_handler.n_dofs() << std::endl;

    // Initialize solution with something
    using solutionVector = dealii::LinearAlgebra::distributed::Vector<double>;

    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    solutionVector solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
    dg->solution = solution_no_ghost;
    //for (auto it = dg->solution.begin(); it != dg->solution.end(); ++it) {
    //    // Interpolating the exact manufactured solution caused some problems at the boundary conditions.
    //    // The manufactured solution is exactly equal to the manufactured_solution_function at the boundary,
    //    // therefore, the finite difference will change whether the flow is incoming or outgoing.
    //    // As a result, we would be differentiating at a non-differentiable point.
    //    // Hence, we fix this issue by taking the second derivative at a non-exact solution.
    //    (*it) *= 1.1;
    //}
    //dg->solution.update_ghost_values();

    // Solving the flow to make sure that we're not at the point of non-differentiality between elements.
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->steady_state();

    // Set dual to 1.0 so that every 2nd derivative of the residual is accounted for.
    for (auto it = dg->dual.begin(); it != dg->dual.end(); ++it) {
        (*it) = 1.0;
    }
    dg->dual.update_ghost_values();

    pcout << "Evaluating AD..." << std::endl;
    //dg->assemble_residual(false, false, true);
    dg->assemble_residual(false, false, true);


    dealii::TrilinosWrappers::SparseMatrix d2RdXdX_fd;
    dealii::SparsityPattern sparsity_pattern = dg->get_d2RdXdX_sparsity_pattern();

    const dealii::IndexSet &row_parallel_partitioning = dg->high_order_grid->locally_owned_dofs_grid;
    //const dealii::IndexSet &col_parallel_partitioning = dg->high_order_grid->locally_relevant_dofs_grid;
    const dealii::IndexSet &col_parallel_partitioning = dg->high_order_grid->locally_owned_dofs_grid;
    d2RdXdX_fd.reinit(row_parallel_partitioning, col_parallel_partitioning, sparsity_pattern, MPI_COMM_WORLD);

    std::shared_ptr<PHiLiP::HighOrderGrid<dim,double>> high_order_grid = dg->high_order_grid;

    using nodeVector = dealii::LinearAlgebra::distributed::Vector<double>;
    nodeVector old_volume_nodes = high_order_grid->volume_nodes;
    old_volume_nodes.update_ghost_values();

    pcout << "Evaluating FD..." << std::endl;
    for (unsigned int inode = 0; inode<high_order_grid->dof_handler_grid.n_dofs(); ++inode) {

        if (inode % 1 == 0) pcout << "inode " << inode+1 << " out of " << high_order_grid->dof_handler_grid.n_dofs() << std::endl;

        for (unsigned int jnode = inode; jnode<high_order_grid->dof_handler_grid.n_dofs(); ++jnode) {

            const bool local_isnonzero = sparsity_pattern.exists(inode,jnode);
            bool global_isnonzero;
            MPI_Allreduce(&local_isnonzero, &global_isnonzero, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            if (!global_isnonzero) continue;

            const bool inode_relevant = high_order_grid->locally_relevant_dofs_grid.is_element(inode);
            const bool jnode_relevant = high_order_grid->locally_relevant_dofs_grid.is_element(jnode);
            double old_inode = -99999;
            double old_jnode = -99999;

            if (inode_relevant) {
                old_inode = high_order_grid->volume_nodes[inode];
            }
            if (jnode_relevant) {
                old_jnode = high_order_grid->volume_nodes[jnode];
            }
            std::array<std::array<int, 2>, 25> pert;
            for (int i=-2; i<3; ++i) {
                for (int j=-2; j<3; ++j) {
                    int ij = (i+2)*5 + (j+2);
                    pert[ij][0] = i;
                    pert[ij][1] = j;
                }
            }

            std::array<double, 25> perturbed_dual_dot_residual;

            for (int i=-2; i<3; ++i) {
                for (int j=-2; j<3; ++j) {
                    int ij = (i+2)*5 + (j+2);

                    if (inode_relevant) {
                        high_order_grid->volume_nodes(inode) = old_inode+i*EPS;
                    }
                    if (jnode_relevant) {
                        if (inode == jnode) {
                            high_order_grid->volume_nodes(jnode) += j*EPS;
                        } else {
                            high_order_grid->volume_nodes(jnode) = old_jnode+j*EPS;
                        }
                    }
                    dg->assemble_residual(false, false, false);
                    perturbed_dual_dot_residual[ij] = dg->right_hand_side * dg->dual;

                    if (inode_relevant) {
                        high_order_grid->volume_nodes(inode) = old_inode;
                    }
                    if (jnode_relevant) {
                        high_order_grid->volume_nodes(jnode) = old_jnode;
                    }
                }
            }

            // http://www.holoborodko.com/pavel/2014/11/04/computing-mixed-derivatives-by-finite-differences/
            double fd_entry = 0.0;
            int i,j, ij;

            // http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/#comment-5289
            fd_entry = 0.0;

            double term[4] = { 0.0,0.0,0.0,0.0 };
            i =  1; j = -2 ; ij = (i+2)*5 + (j+2);
            term[0] += perturbed_dual_dot_residual[ij];
            i =  2; j = -1 ; ij = (i+2)*5 + (j+2);
            term[0] += perturbed_dual_dot_residual[ij];
            i = -2; j =  1 ; ij = (i+2)*5 + (j+2);
            term[0] += perturbed_dual_dot_residual[ij];
            i = -1; j =  2 ; ij = (i+2)*5 + (j+2);
            term[0] += perturbed_dual_dot_residual[ij];

            term[0] *= -63.0;

            i = -1; j = -2 ; ij = (i+2)*5 + (j+2);
            term[1] += perturbed_dual_dot_residual[ij];
            i = -2; j = -1 ; ij = (i+2)*5 + (j+2);
            term[1] += perturbed_dual_dot_residual[ij];
            i =  2; j =  1 ; ij = (i+2)*5 + (j+2);
            term[1] += perturbed_dual_dot_residual[ij];
            i =  1; j =  2 ; ij = (i+2)*5 + (j+2);
            term[1] += perturbed_dual_dot_residual[ij];

            term[1] *= 63.0;

            i =  2; j = -2 ; ij = (i+2)*5 + (j+2);
            term[2] += perturbed_dual_dot_residual[ij];
            i = -2; j =  2 ; ij = (i+2)*5 + (j+2);
            term[2] += perturbed_dual_dot_residual[ij];
            i = -2; j = -2 ; ij = (i+2)*5 + (j+2);
            term[2] -= perturbed_dual_dot_residual[ij];
            i =  2; j =  2 ; ij = (i+2)*5 + (j+2);
            term[2] -= perturbed_dual_dot_residual[ij];

            term[2] *= 44.0;

            i = -1; j = -1 ; ij = (i+2)*5 + (j+2);
            term[3] += perturbed_dual_dot_residual[ij];
            i =  1; j =  1 ; ij = (i+2)*5 + (j+2);
            term[3] += perturbed_dual_dot_residual[ij];
            i =  1; j = -1 ; ij = (i+2)*5 + (j+2);
            term[3] -= perturbed_dual_dot_residual[ij];
            i = -1; j =  1 ; ij = (i+2)*5 + (j+2);
            term[3] -= perturbed_dual_dot_residual[ij];

            term[3] *= 74.0;

            fd_entry = term[0] + term[1] + term[2] + term[3];
            fd_entry /= (600.0*EPS*EPS);

            // Reset node
            if (inode_relevant) {
                high_order_grid->volume_nodes(inode) = old_inode;
            }
            if (jnode_relevant) {
                high_order_grid->volume_nodes(jnode) = old_jnode;
            }

            // Set
            if (dg->high_order_grid->locally_owned_dofs_grid.is_element(inode) ) {
                if (std::abs(fd_entry) >= 1e-12) {
                    d2RdXdX_fd.add(inode,jnode,fd_entry);
                }
            }
            if (inode != jnode && dg->high_order_grid->locally_owned_dofs_grid.is_element(jnode) ) {
                if (std::abs(fd_entry) >= 1e-12) {
                    d2RdXdX_fd.add(jnode,inode,fd_entry);
                }
            }
        }
    }
    d2RdXdX_fd.compress(dealii::VectorOperation::add);

    dg->assemble_residual(false, false, true);
    {
        const unsigned int n_digits = 5;
        const unsigned int n_spacing = 7+n_digits;
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        dealii::FullMatrix<double> fullA(d2RdXdX_fd.m(),d2RdXdX_fd.n());
        fullA.copy_from(d2RdXdX_fd);
        pcout<<"Dense matrix from FD-AD:"<<std::endl;

        std::string path = "./FD_matrix.dat";
        std::ofstream outfile (path,std::ofstream::out);

        if (pcout.is_active()) fullA.print_formatted(outfile, n_digits, true, n_spacing, "0", 1., 0.);

    }
    {
        const unsigned int n_digits = 5;
        const unsigned int n_spacing = 7+n_digits;
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        dealii::FullMatrix<double> fullA(dg->d2RdXdX.m(),dg->d2RdXdX.n());
        fullA.copy_from(dg->d2RdXdX);
        pcout<<"Dense matrix from FD-AD:"<<std::endl;

        std::string path = "./AD_matrix.dat";
        std::ofstream outfile (path,std::ofstream::out);

        if (pcout.is_active()) fullA.print_formatted(outfile, n_digits, true, n_spacing, "0", 1., 0.);
    }

    const double ad_frob_norm = dg->d2RdXdX.frobenius_norm();
    const double fd_frob_norm = d2RdXdX_fd.frobenius_norm();
    const double frob_norm = std::max(ad_frob_norm, fd_frob_norm);

    pcout << "FD-norm = " << d2RdXdX_fd.frobenius_norm() << std::endl;
    pcout << "AD-norm = " << dg->d2RdXdX.frobenius_norm() << std::endl;
    d2RdXdX_fd.add(-1.0,dg->d2RdXdX);

    const double diff_lone_norm = d2RdXdX_fd.l1_norm() / frob_norm;
    const double diff_linf_norm = d2RdXdX_fd.linfty_norm() / frob_norm;
    pcout << "(dRdX_FD - dRdX_AD) L1-norm = " << diff_lone_norm << std::endl;
    pcout << "(dRdX_FD - dRdX_AD) Linf-norm = " << diff_linf_norm << std::endl;

    //if (diff_lone_norm > TOLERANCE) 
    {
        const unsigned int n_digits = 5;
        const unsigned int n_spacing = 7+n_digits;
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        dealii::FullMatrix<double> fullA(d2RdXdX_fd.m(),d2RdXdX_fd.n());
        fullA.copy_from(d2RdXdX_fd);
        pcout<<"Dense matrix from FD-AD:"<<std::endl;

        std::string path = "./FD_minus_AD_matrix.dat";
        std::ofstream outfile (path,std::ofstream::out);

        if (pcout.is_active()) fullA.print_formatted(outfile, n_digits, true, n_spacing, "0", 1., 0.);

    }
    if (diff_lone_norm > TOLERANCE) return 1;

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
         //, PDEType::convection_diffusion
         //, PDEType::advection_vector
         , PDEType::euler
         , PDEType::navier_stokes
    };
    std::vector<std::string> pde_name {
         " PDEType::diffusion "
        , " PDEType::advection "
        //, " PDEType::convection_diffusion "
        //, " PDEType::advection_vector "
        , " PDEType::euler "
        , " PDEType::navier_stokes "
    };

    int ipde = -1;
    for (auto pde = pde_type.begin(); pde != pde_type.end() || error == 1; pde++) {
        ipde++;
        for (unsigned int poly_degree=0; poly_degree<3; ++poly_degree) {
            for (unsigned int igrid=2; igrid<3; ++igrid) {
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

                if (*pde==PDEType::euler || *pde==PDEType::navier_stokes) {
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


