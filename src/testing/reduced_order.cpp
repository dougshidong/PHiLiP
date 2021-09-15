#include <fstream>

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

#include "reduced_order.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver.h"


namespace PHiLiP {
    namespace Tests {

        template <int dim, int nstate>
        ReducedOrder<dim, nstate>::ReducedOrder(const PHiLiP::Parameters::AllParameters *const parameters_input)
                : TestsBase::TestsBase(parameters_input)
        {}


        template <int dim, int nstate>
        int ReducedOrder<dim, nstate>::run_test() const
        {
            pcout << " Running Burgers energy stability. " << std::endl;
            using Triangulation = dealii::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>();

            const Parameters::AllParameters param = *(TestsBase::all_parameters);

            double left = 0.0;
            double right = 2.0;
            const bool colorize = true;
            int n_refinements = 5;
            unsigned int poly_degree = 3;
            dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);

            std::vector<dealii::GridTools::PeriodicFacePair<typename Triangulation::cell_iterator> > matched_pairs;
            dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
            grid->add_periodicity(matched_pairs);


            grid->refine_global(n_refinements);
            pcout << "Grid generated and refined" << std::endl;

            std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
            pcout << "dg created" <<std::endl;
            dg->allocate_system ();

            pcout << "Implement initial conditions" << std::endl;
            dealii::FunctionParser<1> initial_condition;
            std::string variables = "x";
            std::map<std::string,double> constants;
            constants["pi"] = dealii::numbers::PI;
            std::string expression = "sin(pi*(x)) + 0.01";
            initial_condition.initialize(variables,
                                         expression,
                                         constants);
            dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
            // Create ODE solver using the factory and providing the DG object
            std::shared_ptr<PHiLiP::ODE::ODESolver<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

            double finalTime = 30.;
            double dt = 0.5;

            for (int i = 0; i < std::ceil(finalTime/dt); ++ i) {
                ode_solver->advance_solution_time(dt);
                dg->output_results_vtk(i);
            }

            //TEST NEW PARAMETER
            pcout << "***TESTING NEW ROM PARAMETER***";
            pcout << param.rom_param.mach_number;

            return 0; //need to change
        }
#if PHILIP_DIM==1
        template class ReducedOrder<PHILIP_DIM,PHILIP_DIM>;
#endif
    } // Tests namespace
} // PHiLiP namespace
