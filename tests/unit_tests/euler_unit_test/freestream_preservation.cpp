#include <fenv.h> // catch nan
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
#include "mesh/grids/wavy_periodic_grid.hpp"

#include "physics/initial_conditions/initial_condition.h"
#include "physics/euler.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"

using namespace PHiLiP;

const bool NONCONFORMING = false;//true;
enum GridType { eccentric_hyper_shell, abe2015_wavy, naca0012 };
const GridType GRID_TYPE = eccentric_hyper_shell;
const double PERT_SIZE = 1.0e-5;
const int POLY_DEGREE_START = 0;
const int POLY_DEGREE_END = 4;
const int GRID_DEGREE_START = 1;
const int GRID_DEGREE_END = 5;
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
void perturb_high_order_grid ( std::shared_ptr < DGBase<dim, double> > dg, const double perturbation_size )
{
    const dealii::DoFHandler<dim> &DH_grid = dg->high_order_grid->dof_handler_grid;
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
                double pert = random_pert(-perturbation_size, perturbation_size);
                if (dim == 3) pert /= 3;
                dg->high_order_grid->volume_nodes[global_idof_index] += pert;
            }
        }

    }
    dg->high_order_grid->ensure_conforming_mesh();
}

template<int dim>
void create_curved_grid (std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> grid, const GridType grid_type) {

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;

    if (grid_type == GridType::eccentric_hyper_shell) {
        dealii::Point< dim > inner_center, outer_center;
        for (int d=0; d<dim; ++d) {
            inner_center[d] = 0.5;
            outer_center[d] = 0.0;
        }
        const double 	inner_radius = 1.0;
        const double 	outer_radius = 3.0;
        const unsigned int n_cells = 0;
        dealii::GridGenerator::eccentric_hyper_shell<dim>( *grid, inner_center, outer_center, inner_radius, outer_radius, n_cells);
        for (typename Triangulation::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    cell->face(face)->set_boundary_id (1004); // riemann
                    //cell->face(face)->set_boundary_id (1005); // farfield
                }
            }
        }
        grid->refine_global();
    }


    if (grid_type == GridType::naca0012) {
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

            airfoil_data.n_subdivision_x_0 = 15;
            airfoil_data.n_subdivision_x_1 = 15;
            airfoil_data.n_subdivision_x_2 = 15;
            airfoil_data.n_subdivision_y = 15;

            airfoil_data.airfoil_sampling_factor = 10000;

            std::vector<unsigned int> n_subdivisions(dim);
            n_subdivisions[0] = airfoil_data.n_subdivision_x_0 + airfoil_data.n_subdivision_x_1 + airfoil_data.n_subdivision_x_2;
            n_subdivisions[1] = airfoil_data.n_subdivision_y;
            Grids::naca_airfoil(*grid, airfoil_data);
            for (typename Triangulation::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
                for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                    if (cell->face(face)->at_boundary()) {
                        cell->face(face)->set_boundary_id (1004); // riemann
                        //cell->face(face)->set_boundary_id (1005); // farfield
                    }
                }
            }
        } else {
            create_curved_grid (grid, GridType::eccentric_hyper_shell);
        }
    }
    if (grid_type == GridType::abe2015_wavy) {
        grid->clear();
        std::vector<unsigned int> n_subdivisions(dim);
        for (int d=0; d<dim; ++d) {
            n_subdivisions[d] = 7;
        }
        Grids::wavy_grid_Abe_2015<dim>(*grid, n_subdivisions);
        // Has periodic BC. Should not replace BC conditions.
    }

    if (NONCONFORMING) {
        const int n_refine = 2;
        for (int i=0; i<n_refine;i++) {
            grid->prepare_coarsening_and_refinement();
            unsigned int icell = 0;
            for (auto cell = grid->begin_active(); cell!=grid->end(); ++cell) {
                if (!cell->is_locally_owned()) continue;
                icell++;
                if (icell < grid->n_active_cells()/7) {
                    cell->set_refine_flag();
                }
                if (icell == 1 && dim == 3) {
                    cell->set_refine_flag();
                }
            }
            grid->execute_coarsening_and_refinement();
        }
    }
}

template<int dim>
int test()
{
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
    parameter_handler.set("use_collocated_nodes", false);
    parameter_handler.enter_subsection("euler");
    parameter_handler.set("mach_infinity", 0.3);
    parameter_handler.set("angle_of_attack", 36.0);
    parameter_handler.set("side_slip_angle", 0.0);
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("ODE solver");
    parameter_handler.set("ode_solver_type", "explicit");
    parameter_handler.set("nonlinear_max_iterations", (long int) 1);
    double time_step = 1e-8;
    parameter_handler.set("initial_time_step", time_step);
    parameter_handler.leave_subsection();

    Parameters::AllParameters param;
    param.parse_parameters (parameter_handler);

    param.euler_param.parse_parameters (parameter_handler);

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    create_curved_grid (grid, GRID_TYPE);

    pcout << "Number of cells: " << grid->n_active_cells() << std::endl;

    std::vector< std::vector<double> > residuals(10, std::vector<double>(10,1));

    for (int POLY_DEGREE = POLY_DEGREE_START; POLY_DEGREE <= POLY_DEGREE_END; POLY_DEGREE++) {
        for (int GRID_DEGREE = GRID_DEGREE_START; GRID_DEGREE <= GRID_DEGREE_END; GRID_DEGREE++) {

            std::cout << " POLY_DEGREE : " << POLY_DEGREE
                      << " OVERINTEGRATION : " << OVERINTEGRATION
                      << " GRID_DEGREE : " << GRID_DEGREE << std::endl;

            std::cout << " Integration strength : " << 2*POLY_DEGREE+1 + OVERINTEGRATION
                      << " INTEGRAND : " << POLY_DEGREE-1+2*(GRID_DEGREE-1) << std::endl;

            parameter_handler.set("overintegration", (long int) OVERINTEGRATION);
            //parameter_handler.set("overintegration", (long int) std::max(0,(POLY_DEGREE+2*GRID_DEGREE) - (2*POLY_DEGREE-1))+10);

            // Update param with new overintegration parameter.
            param.parse_parameters (parameter_handler);

            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, POLY_DEGREE, GRID_DEGREE, grid);
            dg->allocate_system ();
            
            perturb_high_order_grid (dg, PERT_SIZE);
            //dg->high_order_grid->output_results_vtk(9999);

            // Initialize coarse grid solution with free-stream
            Physics::Euler<dim,dim+2,double> euler_physics_double = Physics::Euler<dim, dim+2, double>(
                        param.euler_param.ref_length,
                        param.euler_param.gamma_gas,
                        param.euler_param.mach_inf,
                        param.euler_param.angle_of_attack,
                        param.euler_param.side_slip_angle);
            Physics::FreeStreamInitialConditions<dim,dim+2> initial_conditions(euler_physics_double);
            dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

            dg->assemble_residual();
            double residual_norm = dg->get_residual_l2norm();
            residuals[POLY_DEGREE][GRID_DEGREE] = residual_norm;
            pcout << "Residual norm: " << residual_norm << std::endl;
            if (residual_norm > 1e-13) {
                std::cout << "Freestream flow is not preserved" << std::endl;
                //dg->output_results_vtk(9999);
                //return 1;
            }

            dg->output_results_vtk(9000+GRID_DEGREE);

        }
    }

    std::cout << std::setprecision(6) << std::scientific;
    std::cout << "S/G ";
    for (int GRID_DEGREE = GRID_DEGREE_START; GRID_DEGREE <= GRID_DEGREE_END; GRID_DEGREE++) {
        std::cout << GRID_DEGREE << "            ";
    }
    std::cout << std::endl;
    for (int POLY_DEGREE = POLY_DEGREE_START; POLY_DEGREE <= POLY_DEGREE_END; POLY_DEGREE++) {
        std::cout << POLY_DEGREE << "  ";
        for (int GRID_DEGREE = GRID_DEGREE_START; GRID_DEGREE <= GRID_DEGREE_END; GRID_DEGREE++) {
            std::cout << " " << residuals[POLY_DEGREE][GRID_DEGREE];
            if (GRID_DEGREE <= POLY_DEGREE+2 && residuals[POLY_DEGREE][GRID_DEGREE] > 1e-14) {
                test_error += 1;
            }
        }
        std::cout << std::endl;
    }

    if (test_error) {
        std::cout << "Freestream preservation is not satisfied for grids of order p_g <= p+2" << std::endl;
    }

    return test_error;
}


int main (int argc, char * argv[])
{
#if !defined(__APPLE__)
    feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
#endif
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


