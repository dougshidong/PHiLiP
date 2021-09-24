#include <fstream>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "burgers_rewienski_test.h"
#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver.h"


namespace PHiLiP {
    namespace Tests {

        template <int dim, int nstate>
        BurgersRewienskiTest<dim, nstate>::BurgersRewienskiTest(const PHiLiP::Parameters::AllParameters *const parameters_input)
                : TestsBase::TestsBase(parameters_input)
        {}

        template <int dim, int nstate>
        int BurgersRewienskiTest<dim, nstate>::run_test() const
        {
            const Parameters::AllParameters param = *(TestsBase::all_parameters);

            pcout << "Running Burgers Rewienski with parameter a: "
                  << param.reduced_order_param.rewienski_a
                  << " and parameter b: "
                  << param.reduced_order_param.rewienski_b
                  << std::endl;

            using Triangulation = dealii::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>();

            double left = param.grid_refinement_study_param.grid_left;
            double right = param.grid_refinement_study_param.grid_right;
            const bool colorize = true;
            int n_refinements = param.grid_refinement_study_param.num_refinements;
            unsigned int poly_degree = param.grid_refinement_study_param.poly_degree;
            dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);

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
            std::string expression = "1";
            initial_condition.initialize(variables, expression, constants);
            dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);

            // Create ODE solver using the factory and providing the DG object
            std::shared_ptr<PHiLiP::ODE::ODESolver<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

            pcout << "Dimension: " << dim
                  << "\t Polynomial degree p: " << poly_degree
                  << std::endl
                  << ". Number of active cells: " << grid->n_global_active_cells()
                  << ". Number of degrees of freedom: " << dg->dof_handler.n_dofs()
                  << std::endl;

            double finalTime = param.reduced_order_param.final_time;
            ode_solver->advance_solution_time(finalTime);

            return 0; //need to change
        }
#if PHILIP_DIM==1
        template class BurgersRewienskiTest<PHILIP_DIM,PHILIP_DIM>;
#endif
    } // Tests namespace
} // PHiLiP namespace
