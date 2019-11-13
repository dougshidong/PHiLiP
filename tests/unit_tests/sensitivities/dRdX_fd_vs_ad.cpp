#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

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
    using namespace PHiLiP;
    // Assemble Jacobian
    std::shared_ptr < DGBase<PHILIP_DIM, double> > dg = DGFactory<PHILIP_DIM,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, &grid);
    dg->allocate_system ();

    // Initialize solution with something
    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, physics_double->manufactured_solution_function, solution_no_ghost);
    dg->solution = solution_no_ghost;

    const bool compute_dRdW = false, compute_dRdX = true;
    dg->assemble_residual(compute_dRdW, compute_dRdX);

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
        PDEType::advection,
        PDEType::diffusion,
        PDEType::convection_diffusion,
        PDEType::advection_vector,
        PDEType::euler
    };

    for (auto pde = pde_type.begin(); pde != pde_type.end() || error == 1; pde++) {
        for (unsigned int poly_degree=1; poly_degree<3; ++poly_degree) {
            for (unsigned int igrid=2; igrid<5; ++igrid) {
                all_parameters.pde_type = *pde;
                // Generate grids
#if PHILIP_DIM==1
                dealii::Triangulation<dim> grid;
#else
                dealii::parallel::distributed::Triangulation<dim> grid(MPI_COMM_WORLD);
#endif
                dealii::GridGenerator::subdivided_hyper_cube(grid, igrid);

                const double random_factor = 0.2;
                const bool keep_boundary = true;
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
            }
        }
    }

    return error;
}


