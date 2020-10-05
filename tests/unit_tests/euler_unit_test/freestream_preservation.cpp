#include <random>
#include <set>
#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/numerics/vector_tools.h> // interpolate initial conditions

#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/grids/curved_periodic_grid.hpp"

#include "physics/euler.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver.h"

const bool NACAGRID = false;
const double PERT_SIZE = NACAGRID ? 0.0 : 0.0e-3;
const double FINAL_TIME = 5.0;
const int POLY_DEGREE = 1;
//const int GRID_DEGREE = POLY_DEGREE+1;
const int GRID_DEGREE = 4;
const int OVERINTEGRATION = 0;
const unsigned int NX_CELL = 2;
const unsigned int NY_CELL = 3;
const unsigned int NZ_CELL = 4;

double random_pert(double lower, double upper)
{
    double f = (double)rand() / RAND_MAX;
    return lower + f * (upper - lower);
}

template<int dim>
int test()
{
    //srand (1.123456789);
    //srand (0.123456789);
    srand (1.0);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    int test_error = 0;
    using namespace PHiLiP;
    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    parameter_handler.set("pde_type", "euler");
    parameter_handler.set("conv_num_flux", "lax_friedrichs");
    parameter_handler.set("dimension", (long int)dim);
    parameter_handler.set("overintegration", (long int) OVERINTEGRATION);
    parameter_handler.set("use_collocated_nodes", false);
    parameter_handler.enter_subsection("euler");
    parameter_handler.set("mach_infinity", 0.3);
    parameter_handler.set("angle_of_attack", 36.0);
    //parameter_handler.set("side_slip_angle", -13.0);
    parameter_handler.set("side_slip_angle", 0.0);
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("ODE solver");
    parameter_handler.set("ode_solver_type", "explicit");
    parameter_handler.set("nonlinear_max_iterations", (long int) 1);
    //double U_plus_c = 1.0 + 1/0.3;
    // double time_step = (1.0/NX_CELL) / U_plus_c;
    // time_step = std::min((1.0/NY_CELL) / U_plus_c, time_step);
    // if(dim == 3) time_step = std::min((1.0/NZ_CELL) / U_plus_c, time_step);
    // time_step = 0.1 * time_step;
    double time_step = 1e-8;
    parameter_handler.set("initial_time_step", time_step);
    parameter_handler.leave_subsection();

    Parameters::AllParameters param;
    param.parse_parameters (parameter_handler);

    param.euler_param.parse_parameters (parameter_handler);

    std::vector<unsigned int> n_subdivisions(dim);
    n_subdivisions[0] = NX_CELL;
    n_subdivisions[1] = NY_CELL;
    if (dim == 3) n_subdivisions[2] = NZ_CELL;

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    Grids::curved_periodic_sine_grid<dim>(*grid, n_subdivisions);

    grid->clear();
    dealii::Point< dim > inner_center, outer_center;
    for (int d=0; d<dim; ++d) {
        inner_center[d] = 0.5;
        outer_center[d] = 0.0;
    }
    const double 	inner_radius = 1.0;
    const double 	outer_radius = 3.0;
    const unsigned int n_cells = 0;
    dealii::GridGenerator::eccentric_hyper_shell<dim>( *grid, inner_center, outer_center, inner_radius, outer_radius, n_cells);
    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 1) {
                    cell->face(face)->set_boundary_id (1004); // farfield
                } else {
                    cell->face(face)->set_boundary_id (1004); // farfield
                }
            }
        }
    }
    grid->refine_global();
    if (NACAGRID == true) {
        if constexpr (dim == 2) {
            grid->clear();
            dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
            airfoil_data.airfoil_type = "NACA";
            airfoil_data.naca_id      = "0012";
            airfoil_data.airfoil_length = 1.0;
            airfoil_data.height         = 150.0; // Farfield radius.
            airfoil_data.length_b2      = 150.0;
            airfoil_data.incline_factor = 0.0;
            airfoil_data.bias_factor    = 4.5;
            airfoil_data.refinements    = 0;
            airfoil_data.n_subdivision_x_0 = 60;
            airfoil_data.n_subdivision_x_1 = 60;
            airfoil_data.n_subdivision_x_2 = 60;
            airfoil_data.n_subdivision_y = 40;

            airfoil_data.n_subdivision_x_0 = 40;
            airfoil_data.n_subdivision_x_1 = 30;
            airfoil_data.n_subdivision_x_2 = 40;
            airfoil_data.n_subdivision_y = 20;

            airfoil_data.n_subdivision_x_0 = 15;
            airfoil_data.n_subdivision_x_1 = 15;
            airfoil_data.n_subdivision_x_2 = 15;
            airfoil_data.n_subdivision_y = 15;

            airfoil_data.n_subdivision_x_0 = 50;
            airfoil_data.n_subdivision_x_1 = 50;
            airfoil_data.n_subdivision_x_2 = 50;
            airfoil_data.n_subdivision_y = 50;

            airfoil_data.airfoil_sampling_factor = 1000000; // default 2

            // Set boundary type and design type
            for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
                for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
                    if (cell->face(face)->at_boundary()) {
                        unsigned int current_id = cell->face(face)->boundary_id();
                        if (current_id == 0 || current_id == 1) {
                            cell->face(face)->set_boundary_id (1004); // farfield
                        } else {
                            cell->face(face)->set_boundary_id (1004); // farfield
                        }
                    }
                }
            }

            n_subdivisions[0] = airfoil_data.n_subdivision_x_0 + airfoil_data.n_subdivision_x_1 + airfoil_data.n_subdivision_x_2;
            n_subdivisions[1] = airfoil_data.n_subdivision_y;
            //Grids::naca_airfoil(*grid, airfoil_data.naca_id, n_subdivisions, airfoil_data.height);
            Grids::naca_airfoil(*grid, airfoil_data);
        }
    }

    pcout << "Number of cells: " << grid->n_active_cells() << std::endl;

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, POLY_DEGREE, GRID_DEGREE, grid);
    dg->allocate_system ();

    const dealii::DoFHandler<dim> &DH_grid = dg->high_order_grid.dof_handler_grid;
    const dealii::FESystem<dim,dim> &fe_grid = DH_grid.get_fe();
    dealii::IndexSet locally_owned_dofs_grid = DH_grid.locally_owned_dofs();
    const unsigned int dofs_per_cell = fe_grid.dofs_per_cell;
    const unsigned int dofs_per_face = fe_grid.dofs_per_face;

    std::vector<dealii::types::global_dof_index> dof_indices(fe_grid.dofs_per_cell);

    for (auto cell = DH_grid.begin_active(); cell != DH_grid.end(); ++cell) {

        if (!cell->is_locally_owned()) continue;

        cell->get_dof_indices(dof_indices);

        // Store boundary face dofs.
        std::set<dealii::types::global_dof_index> boundary_face_dofs;
        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
            if (cell->face(iface)->at_boundary()) {
                for (unsigned int idof_face=0; idof_face<dofs_per_face; ++idof_face) {
                    unsigned int idof_cell = fe_grid.face_to_cell_index(idof_face, iface);
                    boundary_face_dofs.insert(idof_cell);
                }
            }
        }

        for (unsigned int idof=0; idof<dofs_per_cell; ++idof) {
            const bool is_not_boundary_dof = (boundary_face_dofs.find(idof) == boundary_face_dofs.end());
            if (is_not_boundary_dof) {
                const dealii::types::global_dof_index global_idof_index = dof_indices[idof];
                double pert = random_pert(-PERT_SIZE, PERT_SIZE);
                if (dim == 3) pert /= 3;
                dg->high_order_grid.volume_nodes[global_idof_index] += pert;
            }
        }

    }
    dg->high_order_grid.volume_nodes.update_ghost_values();
    dg->high_order_grid.update_mapping_fe_field();
    dg->high_order_grid.output_results_vtk(9999);

    // Initialize coarse grid solution with free-stream
    Physics::Euler<dim,dim+2,double> euler_physics_double = Physics::Euler<dim, dim+2, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,dim+2> initial_conditions(euler_physics_double);
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

    const auto initial_constant_solution = dg->solution;

    dg->high_order_grid.output_results_vtk(9998);

    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->allocate_ode_system();

    //dealii::FullMatrix<double> fullA(dg->global_inverse_mass_matrix.m());
    //fullA.copy_from(dg->global_inverse_mass_matrix);
    //pcout<<"Dense matrix:"<<std::endl;
    //if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);

    //int n_steps = FINAL_TIME / time_step;
    int n_steps = 1;
    pcout << "Time step: " << time_step << std::endl;
    for (int i=0; i < n_steps; ++i) {
        ode_solver->current_iteration = i;
        pcout << " ********************************************************** "
              << std::endl
              << " Iteration: " << ode_solver->current_iteration + 1
              << " out of: " << n_steps
              << std::endl;
        
        dg->assemble_residual();
        dg->output_results_vtk(i);
        const bool pseudotime = true;
        ode_solver->step_in_time(time_step, pseudotime);
        auto diff = initial_constant_solution;
        diff -= dg->solution;
        double diff_norm = diff.l2_norm();
        double residual_norm = dg->get_residual_l2norm();

        pcout << "Residual norm: " << residual_norm << std::endl;
        pcout << "Solution change norm: " << diff_norm << std::endl;

        if (diff_norm > 1e-13 || residual_norm > 1e-13) {
            std::cout << "Freestream flow is not preserved" << std::endl;
            dg->output_results_vtk(9999);
            return 1;
        }
    }

    dg->output_results_vtk(9999);

    return test_error;
}


int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int test_error = false;
    try {
         test_error += test<PHILIP_DIM>();
    }
    catch (std::exception &exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }
    catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }

    return test_error;
}


