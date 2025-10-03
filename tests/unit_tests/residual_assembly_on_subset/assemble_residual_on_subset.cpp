#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "dg/dg_factory.hpp"
#include "mesh/grids/gaussian_bump.h"
#include "parameters/parameters.h"
#include "physics/physics_factory.h"
#include "numerical_flux/convective_numerical_flux.hpp"
#include "parameters/parameters_ode_solver.h"

using PDEType   = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;
using ConvType  = PHiLiP::Parameters::AllParameters::ConvectiveNumericalFlux;
using DissType  = PHiLiP::Parameters::AllParameters::DissipativeNumericalFlux;
using ModelType = PHiLiP::Parameters::AllParameters::ModelType;

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
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
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
    pcout << "Poly degree " << poly_degree << " ncells " << grid->n_active_cells() << " ndofs: " << dg->dof_handler.n_dofs() << std::endl << std::flush;
    dg->allocate_system ();

    // ############################# hereafter code differs from ../sensitivities/compare_rhs.cpp
    // Choose locations on which to evaluate the residual
    dealii::LinearAlgebra::distributed::Vector<int> locations_to_evaluate_rhs;
    locations_to_evaluate_rhs.reinit(dg->triangulation->n_active_cells());
    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, dg->fe_collection, dg->volume_quadrature_collection,
                                dealii::update_quadrature_points);

    // Cell loop for assigning the locations to evaluate RHS.
    for (auto soln_cell = dg->dof_handler.begin_active(); soln_cell != dg->dof_handler.end(); ++soln_cell) {
        // First, check whether the cell is known to the current processor, either local or ghost.
        if (soln_cell->is_locally_owned() || soln_cell->is_ghost()) {
            // Get FEValues for the current cell
            const int i_fele = soln_cell->active_fe_index();
            const int i_quad = i_fele;
            const int i_mapp = 0;
            fe_values_collection.reinit (soln_cell, i_quad, i_mapp, i_fele);
            const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();

        
            // Next, check a condition which identifies cells belonging to a group.
            //////////////////////////////////////////////////////////////////////
            // This test is aiming to test on an arbitrary partition.
            // Thus, somewhat arbitrarily partitioning based on (x+y<0.5) for the first point of the cell.
            const double point_x = fe_values.quadrature_point(0)[0];
#if PHILIP_DIM==2
            const double point_y = fe_values.quadrature_point(0)[1];
#endif
            if (point_x
#if PHILIP_DIM==2
                    +point_y
#endif
                    < 0.5){
                locations_to_evaluate_rhs(soln_cell->active_cell_index())=1;
            }
            /////////////////////////////////////////////////////////////////////
        }
    }

    // set the group ID to 10 (arbitrary choice of int)
    dg->set_list_of_cell_group_IDs(locations_to_evaluate_rhs, 10); 
    pcout << std::endl << "Assigned group ID." << std::endl;


    // Initialize solution with something
    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(*(dg->high_order_grid->mapping_fe_field), dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
    dg->solution = solution_no_ghost;

    dg->output_results_vtk(0,0);

    bool compute_dRdW, compute_dRdX, compute_d2R;

    pcout << "Evaluating RHS only on group ID = 10..." << std::endl;
    const int active_group_ID = 10;
    compute_dRdW = false; compute_dRdX = false, compute_d2R = false;
    const double CFL = 0.0; 
    // pass group ID of 10
    dg->right_hand_side=0.0;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, CFL, active_group_ID);
    dealii::LinearAlgebra::distributed::Vector<double> rhs_partitioned(dg->right_hand_side);
    rhs_partitioned = dg->right_hand_side;

    pcout << "Evauating RHS on entire mesh for comparison..." << std::endl;
    //Reset all cells to group 0
    dg->set_list_of_cell_group_IDs(locations_to_evaluate_rhs,0);
    dg->right_hand_side=0.0;
    // Assemble without passing a group, which defaults to group 0
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, CFL);
    dealii::LinearAlgebra::distributed::Vector<double> rhs_NOT_partitioned(dg->right_hand_side);
    rhs_NOT_partitioned = dg->right_hand_side;

    int testfail = 0; // assume pass
    int nonzero_errors = 0;
    int diff_errors=0;
    // Check that rhs_partitioned is zero where the residual was not assembled.
    for (auto soln_cell = dg->dof_handler.begin_active(); soln_cell != dg->dof_handler.end(); ++soln_cell) {
        if (!soln_cell->is_locally_owned()){                                                                                                                                                           
            continue;
        }    
        const unsigned int n_dofs_per_cell = soln_cell->get_fe().n_dofs_per_cell();
        std::vector<dealii::types::global_dof_index> current_dofs_indices(n_dofs_per_cell);
        soln_cell->get_dof_indices(current_dofs_indices);
        std::cout <<  "cell active index: " << soln_cell->active_cell_index()<< " eval?  " <<  locations_to_evaluate_rhs(soln_cell->active_cell_index()) << " MPI: " << mpi_rank << " ";
        for (unsigned int idof = 0; idof < n_dofs_per_cell; ++idof){
                std::cout << rhs_partitioned(current_dofs_indices[idof]) << " ";
                // Check if there are nonzero values on non-active cells
                if (locations_to_evaluate_rhs(soln_cell->active_cell_index())==0 && abs(rhs_partitioned(current_dofs_indices[idof])) > 1E-14) {
                    testfail = 1;
                    std::cout << "Nonzero here! ";
                    nonzero_errors++;
                }
                // Check if values match an unpartitioned assembly on active cells
                if (locations_to_evaluate_rhs(soln_cell->active_cell_index())==1 // cell is in the active group
                        && abs(rhs_partitioned(current_dofs_indices[idof])-rhs_NOT_partitioned(current_dofs_indices[idof])) > 1E-13) // there is a difference between partitioned and unpartitioned RHS
                {
                    testfail = 1;
                    std::cout << "Diff here! correct:" << rhs_NOT_partitioned(current_dofs_indices[idof]) << "... ";
                    diff_errors++;
                }
        }
        std::cout << "Testfail: " <<  testfail << std::endl;
    }

    std::cout << std::endl;
    testfail = dealii::Utilities::MPI::sum(testfail, MPI_COMM_WORLD);
    nonzero_errors = dealii::Utilities::MPI::sum(nonzero_errors, MPI_COMM_WORLD);
    diff_errors = dealii::Utilities::MPI::sum(diff_errors, MPI_COMM_WORLD);
    if (testfail) pcout << "FAILING with " << nonzero_errors << " incorrect nonzeros and " << diff_errors << " differences with ref soln." << std::endl;
    // Uncomment to output vtk
    //dg->output_results_vtk(1, 0.0);
    return testfail ;
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
#if PHILIP_DIM==3
        , PDEType::physics_model
#endif
    };
    std::vector<std::string> pde_name {
        " PDEType::diffusion "
        , " PDEType::advection "
        , " PDEType::convection_diffusion "
        , " PDEType::advection_vector "
        , " PDEType::euler "
        , " PDEType::navier_stokes "
#if PHILIP_DIM==3
        , " PDEType::physics_model "
#endif
    };
#if PHILIP_DIM==3
    ModelType model = ModelType::large_eddy_simulation
#endif

    int ipde = -1;
    for (auto pde = pde_type.begin(); pde != pde_type.end() && error == 0; pde++) {
        ipde++;
        for (unsigned int poly_degree=1; poly_degree<3 && error == 0; ++poly_degree) {
            for (unsigned int igrid=2; igrid<4 && error == 0; ++igrid) {
                pcout << "Using " << pde_name[ipde] << std::endl;
                all_parameters.pde_type = *pde;
                all_parameters.use_weak_form = false;
                all_parameters.use_inverse_mass_on_the_fly = true;
                using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
                all_parameters.ode_solver_param.ode_solver_type = ODEEnum::runge_kutta_solver; // Set as runge_kutta to supress an abort.
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

                if ((*pde==PDEType::euler) || (*pde==PDEType::navier_stokes)
#if PHILIP_DIM==3
         || ((*pde==PDEType::physics_model) && (model==ModelType::large_eddy_simulation))
#endif
                    ) {
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


