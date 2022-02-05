#include "1d_burgers_rewienski_fd_sensitivity.h"

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

#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersRewienskiSensitivity<dim, nstate>::BurgersRewienskiSensitivity(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int BurgersRewienskiSensitivity<dim, nstate>::run_test() const
{
    double h = 1E-06;

    Parameters::AllParameters params1 = reinit_params(0.02);
    Parameters::AllParameters params2 = reinit_params(0.02+h);
    dealii::LinearAlgebra::distributed::Vector<double> solution1 = run_solution(params1);
    dealii::LinearAlgebra::distributed::Vector<double> solution2 = run_solution(params2);
    dealii::LinearAlgebra::distributed::Vector<double> sensitivity_dWdb(solution1.size());

    dealii::TableHandler solutions_table;
    for(unsigned int i = 0 ; i < solution1.size(); i++){
        sensitivity_dWdb[i] = (solution2[i] - solution1[i])/h;
        pcout << (solution2[i] - solution1[i])/h <<std::endl;
        solutions_table.add_value("Sensitivity:", sensitivity_dWdb[i]);
    }

    solutions_table.set_precision("Sensitivity:", 16);
    std::ofstream out_file("sensitivity_fd.txt");
    solutions_table.write_text(out_file);

    return 0;
}

template <int dim, int nstate>
Parameters::AllParameters BurgersRewienskiSensitivity<dim, nstate>::reinit_params(double rewienski_b) const {
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    PHiLiP::Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    all_parameters.reduced_order_param.rewienski_b = rewienski_b;
    all_parameters.reduced_order_param.rewienski_a = 3.0;
    all_parameters.grid_refinement_study_param.num_refinements = 8;
    all_parameters.grid_refinement_study_param.poly_degree = 0;
    all_parameters.grid_refinement_study_param.grid_left = 0.0;
    all_parameters.grid_refinement_study_param.grid_right = 100.0;
    all_parameters.dimension = 1;
    all_parameters.pde_type = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation::burgers_rewienski;
    all_parameters.manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term = true;
    all_parameters.ode_solver_param.initial_time_step = 0.1;
    return all_parameters;
}

template <int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> BurgersRewienskiSensitivity<dim, nstate>::run_solution(Parameters::AllParameters parameters_input) const {
    const Parameters::AllParameters *param = &parameters_input;

    pcout << "Running Burgers Rewienski with parameter a: "
          << param->reduced_order_param.rewienski_a
          << " and parameter b: "
          << param->reduced_order_param.rewienski_b
          << std::endl;

    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>();

    double left = param->grid_refinement_study_param.grid_left;
    double right = param->grid_refinement_study_param.grid_right;
    const bool colorize = true;
    int n_refinements = param->grid_refinement_study_param.num_refinements;
    unsigned int poly_degree = param->grid_refinement_study_param.poly_degree;
    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);

    grid->refine_global(n_refinements);
    pcout << "Grid generated and refined" << std::endl;

    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(param, poly_degree, grid);
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
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

    pcout << "Dimension: " << dim
          << "\t Polynomial degree p: " << poly_degree
          << std::endl
          << ". Number of active cells: " << grid->n_global_active_cells()
          << ". Number of degrees of freedom: " << dg->dof_handler.n_dofs()
          << std::endl;

    ode_solver->steady_state();

    return dg->solution;
}
#if PHILIP_DIM==1
        template class BurgersRewienskiSensitivity<PHILIP_DIM,PHILIP_DIM>;
#endif
    } // Tests namespace
} // PHiLiP namespace