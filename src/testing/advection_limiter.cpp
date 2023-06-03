#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/convergence_table.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "advection_limiter.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"

#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"

#include<fenv.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <fstream>

namespace PHiLiP {
    namespace Tests {
        template <int dim, int nstate>
        AdvectionLimiter<dim, nstate>::AdvectionLimiter(const PHiLiP::Parameters::AllParameters* const parameters_input)
            :
            TestsBase::TestsBase(parameters_input)
        {}


        template <int dim, int nstate>
        int AdvectionLimiter<dim, nstate>::run_test() const
        {
            pcout << " Running Advection limiter test. " << std::endl;
            pcout << dim << "    " << nstate << std::endl;
            PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;
            double left = -1;
            double right = 1;
            const unsigned int n_grids = 5;
            unsigned int poly_degree = 2;
            const unsigned int igrid_start = 4;
            //const unsigned int grid_degree = 1;
            const unsigned int grid_degree = poly_degree;

            for (unsigned int igrid = igrid_start; igrid < n_grids; igrid++) {

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
                using Triangulation = dealii::Triangulation<dim>;
                std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::smoothing_on_refinement |
                        dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
                using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
                std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                    MPI_COMM_WORLD,
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::smoothing_on_refinement |
                        dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
                //straight grid setup
                dealii::GridGenerator::hyper_cube(*grid, left, right, true);
#if PHILIP_DIM==1
                std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
                dealii::GridTools::collect_periodic_faces(*grid, 0, 1, 0, matched_pairs);
                grid->add_periodicity(matched_pairs);
#else
                std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
                dealii::GridTools::collect_periodic_faces(*grid, 0, 1, 0, matched_pairs);
                if (dim >= 2) dealii::GridTools::collect_periodic_faces(*grid, 2, 3, 1, matched_pairs);
                if (dim >= 3) dealii::GridTools::collect_periodic_faces(*grid, 4, 5, 2, matched_pairs);
                grid->add_periodicity(matched_pairs);
#endif
                grid->refine_global(igrid);
                pcout << "Grid generated and refined" << std::endl;


                const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
                pcout << "n_global_active_cells2:  " << n_global_active_cells2 << std::endl;
                double delta_x = (right - left) / n_global_active_cells2;
                all_parameters_new.ode_solver_param.initial_time_step = (1.0 / 3.0) * pow(delta_x, 2.0);
                //allocate dg
                std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim, double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
                pcout << "dg created" << std::endl;
                dg->allocate_system();

                //initialize IC
                pcout << "Setting up Initial Condition" << std::endl;
                //Create initial condition function
                std::shared_ptr< InitialConditionFunction<dim, nstate, double> > initial_condition_function =
                    InitialConditionFactory<dim, nstate, double>::create_InitialConditionFunction(&all_parameters_new);
                SetInitialCondition<dim, nstate, double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

                //get the global max and min of initial condition.
                dg->get_global_max_and_min_of_solution();
                // Create ODE solver using the factory and providing the DG object
                std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);


                double finalTime = 2.0;
                ode_solver->advance_solution_time(finalTime);

            }//end of grid loop
            //want to add some condition to check for if(){return 1} else{return 0}
            return 0; //if got to here means passed the test, otherwise would've failed earlier
        }


        template class AdvectionLimiter <PHILIP_DIM, 1>;

    } //Tests namespace
} //PHiLiP namespace