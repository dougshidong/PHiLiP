#include <fstream>

#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_tools.h>

#include "dg/dg.h"
#include "parameters/parameters.h"
#include "physics/physics_factory.h"
#include "numerical_flux/numerical_flux.h"

using PDEType  = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;
using ConvType = PHiLiP::Parameters::AllParameters::ConvectiveNumericalFlux;
using DissType = PHiLiP::Parameters::AllParameters::DissipativeNumericalFlux;

const double TOLERANCE = 1E-12;

template<int dim, int nstate>
int test (
    const unsigned int poly_degree,
#if PHILIP_DIM==1
    dealii::Triangulation<dim> &grid,
#else
    dealii::parallel::distributed::Triangulation<dim> &grid,
#endif
    const PHiLiP::Parameters::AllParameters &all_parameters)
{
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    using namespace PHiLiP;
    // Assemble Jacobian
    std::shared_ptr < DGBase<PHILIP_DIM, double> > dg = DGFactory<PHILIP_DIM,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, &grid);

    const int n_refine = 1;
    for (int i=0; i<n_refine;i++) {
        dg->high_order_grid.prepare_for_coarsening_and_refinement();
        grid.prepare_coarsening_and_refinement();
        unsigned int icell = 0;
        for (auto cell = grid.begin_active(); cell!=grid.end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            icell++;
            if (icell < grid.n_active_cells()/2) {
                cell->set_refine_flag();
            }
        }
        grid.execute_coarsening_and_refinement();
        bool mesh_out = (i==n_refine-1);
        dg->high_order_grid.execute_coarsening_and_refinement(mesh_out);
    }
    dg->allocate_system ();

    std::cout << "Poly degree " << poly_degree << " ncells " << grid.n_active_cells() << " ndofs: " << dg->dof_handler.n_dofs() << std::endl;

    // Initialize solution with something
    using solutionVector = dealii::LinearAlgebra::distributed::Vector<double>;

    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    solutionVector solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
    dg->solution = solution_no_ghost;


    dealii::TrilinosWrappers::SparseMatrix dRdXv_fd;
    dealii::SparsityPattern sparsity_pattern = dg->get_dRdX_sparsity_pattern ();

    const dealii::IndexSet &row_parallel_partitioning = dg->locally_owned_dofs;
    const dealii::IndexSet &col_parallel_partitioning = dg->high_order_grid.locally_owned_dofs_grid;
    dRdXv_fd.reinit(row_parallel_partitioning, col_parallel_partitioning, sparsity_pattern, MPI_COMM_WORLD);

    const double eps = 1e-6;
    PHiLiP::HighOrderGrid<dim,double> &high_order_grid = dg->high_order_grid;

    using nodeVector = dealii::LinearAlgebra::distributed::Vector<double>;
    nodeVector old_nodes = high_order_grid.nodes;
    old_nodes.update_ghost_values();

    dealii::AffineConstraints<double> hanging_node_constraints;
    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(high_order_grid.dof_handler_grid, hanging_node_constraints);
    hanging_node_constraints.close();

    pcout << "Evaluating AD..." << std::endl;
    dg->assemble_residual(false, true);

    pcout << "Evaluating FD..." << std::endl;
    for (unsigned int inode = 0; inode<high_order_grid.dof_handler_grid.n_dofs(); ++inode) {
        if (inode % 100 == 0) pcout << "inode " << inode+1 << " out of " << high_order_grid.dof_handler_grid.n_dofs() << std::endl;
        double old_node = -99999;
        // Positive perturbation
        if (high_order_grid.locally_relevant_dofs_grid.is_element(inode) ) {
            old_node = high_order_grid.nodes[inode];
            high_order_grid.nodes(inode) = old_node+eps;
        }
        //hanging_node_constraints.distribute(high_order_grid.nodes);
        //high_order_grid.nodes.update_ghost_values();

        dg->assemble_residual(false, false);
        solutionVector perturbed_residual_p = dg->right_hand_side;

        //std::cout << "perturb nodes " << std::endl;  high_order_grid.nodes.print(std::cout, 5);
        //high_order_grid.nodes = old_nodes;
        //high_order_grid.nodes.update_ghost_values();
        //std::cout << "oldnodes " << std::endl; high_order_grid.nodes.print(std::cout, 5);

        // Negative perturbation
        if (high_order_grid.locally_relevant_dofs_grid.is_element(inode) ) {
            high_order_grid.nodes(inode) = old_node-eps;
        }
        //hanging_node_constraints.distribute(high_order_grid.nodes);
        //high_order_grid.nodes.update_ghost_values();

        dg->assemble_residual(false, false);
        solutionVector perturbed_residual_m = dg->right_hand_side;

        //std::cout << "perturb nodes " << std::endl; high_order_grid.nodes.print(std::cout, 5);
        //high_order_grid.nodes = old_nodes;
        //high_order_grid.nodes.update_ghost_values();
        //std::cout << "old nodes " << std::endl; high_order_grid.nodes.print(std::cout, 5);

        // Finite difference
        //std::cout << "perturb residual p " << std::endl; perturbed_residual_p.print(std::cout, 5);
        //std::cout << "perturb residual n " << std::endl; perturbed_residual_m.print(std::cout, 5);

        perturbed_residual_p -= perturbed_residual_m;
        //std::cout << "perturb residual diff " << std::endl; perturbed_residual_p.print(std::cout, 5);
        perturbed_residual_p /= (2.0*eps);
        //std::cout << "fd residual " << std::endl; perturbed_residual_p.print(std::cout, 5);

        // Reset node
        if (high_order_grid.locally_relevant_dofs_grid.is_element(inode) ) {
            high_order_grid.nodes(inode) = old_node;
        }

        // Set
        for (unsigned int iresidual = 0; iresidual < dg->dof_handler.n_dofs(); ++iresidual) {
            if (dg->locally_owned_dofs.is_element(iresidual) ) {
                const double drdx_entry = perturbed_residual_p[iresidual];
                if (std::abs(drdx_entry) >= 1e-12) {
                    dRdXv_fd.add(iresidual,inode,drdx_entry);
                }
            }
        }
    }
    dRdXv_fd.compress(dealii::VectorOperation::add);

    // {
    //     const unsigned int n_digits = 5;
    //     const unsigned int n_spacing = 7+n_digits;
    //     dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    //     dealii::FullMatrix<double> fullA(dRdXv_fd.m(),dRdXv_fd.n());
    //     fullA.copy_from(dRdXv_fd);
    //     pcout<<"Dense matrix from FD:"<<std::endl;
    //     if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), n_digits, true, n_spacing, "0", 1., 0.);
    // }

    // {
    //     const unsigned int n_digits = 5;
    //     const unsigned int n_spacing = 7+n_digits;
    //     dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    //     dealii::FullMatrix<double> fullA(dRdXv_fd.m(),dRdXv_fd.n());
    //     fullA.copy_from(dg->dRdXv);
    //     pcout<<"Dense matrix from AD:"<<std::endl;
    //     if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), n_digits, true, n_spacing, "0", 1., 0.);
    // }

    dRdXv_fd.add(-1.0,dg->dRdXv);

    const double diff_lone_norm = dRdXv_fd.l1_norm();
    const double diff_linf_norm = dRdXv_fd.linfty_norm();
    pcout << "(dRdX_FD - dRdX_AD) L1-norm = " << diff_lone_norm << std::endl;
    pcout << "(dRdX_FD - dRdX_AD) Linf-norm = " << diff_linf_norm << std::endl;

    if (diff_lone_norm > 1e-5) 
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
        // , PDEType::euler
    };
    std::vector<std::string> pde_name {
         " PDEType::diffusion "
        , " PDEType::advection "
        // , " PDEType::convection_diffusion "
        , " PDEType::advection_vector "
        // , " PDEType::euler "
    };

    int ipde = -1;
    for (auto pde = pde_type.begin(); pde != pde_type.end() || error == 1; pde++) {
        ipde++;
        for (unsigned int poly_degree=1; poly_degree<3; ++poly_degree) {
            for (unsigned int igrid=2; igrid<4; ++igrid) {
                std::cout << "Using " << pde_name[ipde] << std::endl;
                all_parameters.pde_type = *pde;
                // Generate grids
#if PHILIP_DIM==1
                dealii::Triangulation<dim> grid(
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::MeshSmoothing::smoothing_on_refinement |
                        dealii::Triangulation<dim>::MeshSmoothing::smoothing_on_coarsening));
#else
                dealii::parallel::distributed::Triangulation<dim> grid(MPI_COMM_WORLD,
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::MeshSmoothing::smoothing_on_refinement |
                        dealii::Triangulation<dim>::MeshSmoothing::smoothing_on_coarsening));
#endif

                dealii::GridGenerator::subdivided_hyper_cube(grid, igrid);

                const double random_factor = 0.2;
                const bool keep_boundary = false;
                if (random_factor > 0.0) dealii::GridTools::distort_random (random_factor, grid, keep_boundary);
                for (auto &cell : grid.active_cell_iterators()) {
                    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                        if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
                    }
                }

                if (*pde==PDEType::euler) {
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


