#include <Epetra_RowMatrixTransposer.h>

#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

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

const double TOLERANCE = 1E-12;

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
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, grid);
    dg->allocate_system ();
    const int n_refine = 2;
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
    pcout << "Poly degree " << poly_degree << " ncells " << grid->n_active_cells() << " ndofs: " << dg->dof_handler.n_dofs() << std::endl;
    dg->allocate_system ();

    // Initialize solution with something
    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(*(dg->high_order_grid->mapping_fe_field), dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
    dg->solution = solution_no_ghost;

    bool compute_dRdW, compute_dRdX, compute_d2R;

    pcout << "Evaluating RHS only..." << std::endl;
    compute_dRdW = false; compute_dRdX = false, compute_d2R = false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dealii::LinearAlgebra::distributed::Vector<double> rhs_only(dg->right_hand_side);
    // pcout << "*******************************************************************************" << std::endl;


    pcout << "Evaluating RHS with d2R..." << std::endl;
    compute_dRdW = false; compute_dRdX = false, compute_d2R = true;
    dealii::LinearAlgebra::distributed::Vector<double> dummy_dual(dg->right_hand_side);
    dg->set_dual(dummy_dual);
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dealii::LinearAlgebra::distributed::Vector<double> rhs_d2R(dg->right_hand_side);
    // pcout << "*******************************************************************************" << std::endl;

    dealii::TrilinosWrappers::SparseMatrix d2RdWdW_transpose;
    {
        Epetra_CrsMatrix *transpose_CrsMatrix;
        Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&dg->d2RdWdW.trilinos_matrix()));
        epmt.CreateTranspose(false, transpose_CrsMatrix);
        d2RdWdW_transpose.reinit(*transpose_CrsMatrix);
        d2RdWdW_transpose.add(-1.0,dg->d2RdWdW);
    }

    dealii::TrilinosWrappers::SparseMatrix d2RdXdX_transpose;
    {
        Epetra_CrsMatrix *transpose_CrsMatrix;
        Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&dg->d2RdXdX.trilinos_matrix()));
        epmt.CreateTranspose(false, transpose_CrsMatrix);
        d2RdXdX_transpose.reinit(*transpose_CrsMatrix);
        d2RdXdX_transpose.add(-1.0,dg->d2RdXdX);
    }

    // {
    //     dealii::FullMatrix<double> fullA(dg->d2RdWdW.m());
    //     fullA.copy_from(dg->d2RdWdW);
    //     pcout<<"d2RdWdW:"<<std::endl;
    //     if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);
    // }

    // {
    //     dealii::FullMatrix<double> fullA(dg->d2RdXdX.m());
    //     fullA.copy_from(dg->d2RdXdX);
    //     pcout<<"d2RdXdX:"<<std::endl;
    //     if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);
    // }

    pcout << "dg->d2RdWdW.frobenius_norm()  " << dg->d2RdWdW.frobenius_norm() << std::endl;
    pcout << "dg->d2RdXdX.frobenius_norm()  " << dg->d2RdXdX.frobenius_norm() << std::endl;

    const double d2RdWdW_norm = dg->d2RdWdW.frobenius_norm();
    const double d2RdWdW_abs_diff = d2RdWdW_transpose.frobenius_norm();
    const double d2RdWdW_rel_diff = d2RdWdW_abs_diff / d2RdWdW_norm;

    const double d2RdXdX_norm = dg->d2RdXdX.frobenius_norm();
    const double d2RdXdX_abs_diff = d2RdXdX_transpose.frobenius_norm();
    const double d2RdXdX_rel_diff = d2RdXdX_abs_diff / d2RdXdX_norm;

    const double tol = 1e-11;
    pcout << "Error: "
                    << " d2RdWdW_abs_diff: " << d2RdWdW_abs_diff
                    << " d2RdWdW_rel_diff: " << d2RdWdW_rel_diff
                    << std::endl
                    << " d2RdXdX_abs_diff: " << d2RdXdX_abs_diff
                    << " d2RdXdX_rel_diff: " << d2RdXdX_rel_diff
                    << std::endl;
    if (d2RdWdW_abs_diff > tol && d2RdWdW_rel_diff > tol) return 1;
    if (d2RdXdX_abs_diff > tol && d2RdXdX_rel_diff > tol) return 1;

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
        PDEType::diffusion,
        PDEType::advection,
        PDEType::convection_diffusion,
        PDEType::advection_vector,
        PDEType::euler,
        PDEType::navier_stokes
    };
    std::vector<std::string> pde_name {
        " PDEType::diffusion "
        , " PDEType::advection "
        , " PDEType::convection_diffusion "
        , " PDEType::advection_vector "
        , " PDEType::euler "
        , " PDEType::navier_stokes "
    };

    int ipde = -1;
    for (auto pde = pde_type.begin(); pde != pde_type.end() && error == 0; pde++) {
        ipde++;
        for (unsigned int poly_degree=1; poly_degree<3 && error == 0; ++poly_degree) {
            for (unsigned int igrid=2; igrid<4 && error == 0; ++igrid) {
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

                const double random_factor = 0.3;
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
            }
        }
    }

    if (error != 0) pcout << "Found a non-symmetric Hessian." << std::endl;

    return error;
}


