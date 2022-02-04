#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include "physics/initial_conditions/initial_condition.h"
#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"

#include "mesh/grids/naca_airfoil_grid.hpp"
#include "euler_naca0012.hpp"

#include "functional/lift_drag.hpp"


#include "mesh/gmsh_reader.hpp"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerNACA0012<dim,nstate>::EulerNACA0012(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
int EulerNACA0012<dim,nstate>
::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(dim == 2, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int n_grids_input       = manu_grid_conv_param.number_of_grids;

    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // p0 tends to require a finer grid to reach asymptotic region
        unsigned int n_grids = n_grids_input;
        if (poly_degree <= 1) n_grids = n_grids_input;

        std::vector<double> entropy_error(n_grids);
        std::vector<double> grid_size(n_grids);

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

        std::vector<unsigned int> n_subdivisions(dim);
        n_subdivisions[1] = n_1d_cells[0]; // y-direction
        n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction

        // const double channel_length = 3.0;
        // const double channel_height = 0.8;
        // Grids::gaussian_bump(*grid, n_subdivisions, channel_length, channel_height);

        // const double solution_degree = poly_degree;
        // const double grid_degree = 3;
        // // Create DG object
        // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, &grid);

        // // Initialize coarse grid solution with free-stream
        // dg->allocate_system ();
        // dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

        // // Create ODE solver and ramp up the solution from p0
        // std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        // ode_solver->initialize_steady_polynomial_ramping (poly_degree);

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {


            //if (igrid!=0) {
            //    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
            //    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim>> solution_transfer(dg->dof_handler);
            //    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
            //    dg->high_order_grid->prepare_for_coarsening_and_refinement();
            //    grid->refine_global (1);
            //    dg->high_order_grid->execute_coarsening_and_refinement(true);
            //    dg->allocate_system ();
            //    dg->solution.zero_out_ghosts();
            //    solution_transfer.interpolate(dg->solution);
            //    dg->solution.update_ghost_values();
            //}

            using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                MPI_COMM_WORLD,
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));

            dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
            airfoil_data.airfoil_type = "NACA";
            airfoil_data.naca_id      = "0012";
            airfoil_data.airfoil_length = 1.0;
            airfoil_data.height         = 150.0; // Farfield radius.
            airfoil_data.length_b2      = 150.0;
            airfoil_data.incline_factor = 0.0;
            airfoil_data.bias_factor    = 5.0;
            airfoil_data.refinements    = 0;
            airfoil_data.n_subdivision_x_0 = 60;
            airfoil_data.n_subdivision_x_1 = 60;
            airfoil_data.n_subdivision_x_2 = 60;
            airfoil_data.n_subdivision_y = 40;
            //airfoil_data.n_subdivision_x_0 = 18;
            //airfoil_data.n_subdivision_x_1 = 9;
            //airfoil_data.n_subdivision_x_2 = 9;
            //airfoil_data.n_subdivision_y = 12;

            //airfoil_data.n_subdivision_x_0 = 16;
            //airfoil_data.n_subdivision_x_1 = 16;
            //airfoil_data.n_subdivision_x_2 = 16;
            //airfoil_data.n_subdivision_y = 16;

            airfoil_data.n_subdivision_x_0 = 4;
            airfoil_data.n_subdivision_x_1 = 4;
            airfoil_data.n_subdivision_x_2 = 4;
            airfoil_data.n_subdivision_y = 4;

            airfoil_data.airfoil_sampling_factor = 100000; // default 2

            // dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
            // // Assign a manifold to have curved geometry
            // unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
            // grid->reset_all_manifolds();
            // grid->set_all_manifold_ids(manifold_id);
            // // // Set Flat manifold on the domain, but not on the boundary.
            // grid->set_manifold(manifold_id, dealii::FlatManifold<2>());

            // // Set boundary type and design type
            // for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
            //     for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            //         if (cell->face(face)->at_boundary()) {
            //             unsigned int current_id = cell->face(face)->boundary_id();
            //             if (current_id == 0 || current_id == 1) {
            //                 cell->face(face)->set_boundary_id (1004); // farfield
            //             } else {
            //                 cell->face(face)->set_boundary_id (1001); // Outflow with supersonic or back_pressure
            //             }
            //         }
            //     }
            // }

            n_subdivisions[0] = airfoil_data.n_subdivision_x_0 + airfoil_data.n_subdivision_x_1 + airfoil_data.n_subdivision_x_2;
            n_subdivisions[1] = airfoil_data.n_subdivision_y;
            //Grids::naca_airfoil(*grid, airfoil_data.naca_id, n_subdivisions, airfoil_data.height);
            Grids::naca_airfoil(*grid, airfoil_data);

            grid->refine_global();

            const unsigned int solution_degree = poly_degree;
            const unsigned int grid_degree = 2;//solution_degree+1;
            //const unsigned int grid_degree = 1;
            //const unsigned int grid_degree = 2;//solution_degree+1;
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, solution_degree, solution_degree, grid_degree, grid);

            //std::shared_ptr<HighOrderGrid<dim,double>> joukowski_mesh = read_gmsh <dim, dim> ("joukowski_R1_Q"+std::to_string(grid_degree)+".msh");
            //std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("new_msh41.msh");
            //std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012_hopw.msh");
            std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012.msh");
            dg->set_high_order_grid(naca0012_mesh);
            for (unsigned int i=0; i<igrid; ++i) {
                dg->high_order_grid->refine_global();
            }
            //dg->high_order_grid->refine_global();

            // Initialize coarse grid solution with free-stream
            dg->allocate_system ();
            dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

            const unsigned int n_global_active_cells = dg->triangulation->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Create ODE solver and ramp up the solution from p0
            std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
            ode_solver->initialize_steady_polynomial_ramping (poly_degree);
            //ode_solver->steady_state();
            LiftDragFunctional<dim,dim+2,double> lift_functional( dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::lift );
            double lift = lift_functional.evaluate_functional();

            LiftDragFunctional<dim,dim+2,double> drag_functional( dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::drag );
            double drag = drag_functional.evaluate_functional();

            std::cout << " Resulting lift : " << lift << std::endl;
            std::cout << " Resulting drag : " << drag << std::endl;

        }

    }
    int n_fail_poly = 0;
    return n_fail_poly;
}


#if PHILIP_DIM==2
    template class EulerNACA0012 <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


