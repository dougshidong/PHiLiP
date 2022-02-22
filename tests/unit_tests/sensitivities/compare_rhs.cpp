#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

#include "dg/dg_factory.hpp"
#include "mesh/grids/gaussian_bump.h"
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
    std::shared_ptr < DGBase<PHILIP_DIM, double> > dg = DGFactory<PHILIP_DIM,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, grid);
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
            //else if (icell%2 == 0) {
            //    cell->set_refine_flag();
            //} else if (icell%3 == 0) {
            //    //cell->set_coarsen_flag();
            //}
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

    pcout << "Evaluating RHS with dRdW..." << std::endl;
    dg->right_hand_side *= 0.0;
    compute_dRdW = true; compute_dRdX = false, compute_d2R = false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dealii::LinearAlgebra::distributed::Vector<double> rhs_dRdW(dg->right_hand_side);
    // pcout << "*******************************************************************************" << std::endl;

    pcout << "Evaluating RHS with dRdX..." << std::endl;
    dg->right_hand_side *= 0.0;
    compute_dRdW = false; compute_dRdX = true, compute_d2R = false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dealii::LinearAlgebra::distributed::Vector<double> rhs_dRdX(dg->right_hand_side);
    // pcout << "*******************************************************************************" << std::endl;

    pcout << "Evaluating RHS with d2R..." << std::endl;
    dg->right_hand_side *= 0.0;
    compute_dRdW = false; compute_dRdX = false, compute_d2R = true;
    dealii::LinearAlgebra::distributed::Vector<double> dummy_dual(dg->right_hand_side);
    dg->set_dual(dummy_dual);
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dealii::LinearAlgebra::distributed::Vector<double> rhs_d2R(dg->right_hand_side);
    // pcout << "*******************************************************************************" << std::endl;

    const double norm_rhs_only = rhs_only.l2_norm();

    rhs_dRdW -= rhs_only;
    const double dRdW_vs_rhs_rel_diff1 = rhs_dRdW.l2_norm() / norm_rhs_only;
    rhs_dRdX -= rhs_only;
    const double dRdX_vs_rhs_rel_diff2 = rhs_dRdX.l2_norm() / norm_rhs_only;
    rhs_d2R -= rhs_only;
    const double d2R_vs_rhs_rel_diff2 = rhs_d2R.l2_norm() / norm_rhs_only;

    const double tol = 1e-11;
    pcout << "Error: dRdW_vs_rhs_rel_diff1: " << dRdW_vs_rhs_rel_diff1
                    << " dRdX_vs_rhs_rel_diff2: " << dRdX_vs_rhs_rel_diff2
                    << " d2R_vs_rhs_rel_diff2: " << d2R_vs_rhs_rel_diff2
                    << std::endl;
    if (dRdW_vs_rhs_rel_diff1 > tol) {
        return 1;
    } if (dRdX_vs_rhs_rel_diff2 > tol) {
        return 1;
    } if (d2R_vs_rhs_rel_diff2 > tol) {
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
#if PHILIP_DIM==1
                std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
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
#else
                std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                    MPI_COMM_WORLD,
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
#endif

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

    return error;
}


