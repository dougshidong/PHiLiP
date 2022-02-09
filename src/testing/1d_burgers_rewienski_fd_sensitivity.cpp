#include "1d_burgers_rewienski_fd_sensitivity.h"
#include <fstream>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersRewienskiSensitivity<dim, nstate>::BurgersRewienskiSensitivity(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int BurgersRewienskiSensitivity<dim, nstate>::run_test() const
{
    dealii::TableHandler sensitivity_table;
    double h = 1E-06;
    Parameters::AllParameters params1 = reinit_params(this->all_parameters->reduced_order_param.rewienski_b);
    Parameters::AllParameters params2 = reinit_params(this->all_parameters->reduced_order_param.rewienski_b + h);
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_1 = initialize_ode_solver(params1);
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_2 = initialize_ode_solver(params2);

    ode_solver_1->allocate_ode_system();
    ode_solver_2->allocate_ode_system();

    if(this->all_parameters->flow_solver_param.steady_state == true){
        ode_solver_1->steady_state();
        ode_solver_2->steady_state();

        dealii::LinearAlgebra::distributed::Vector<double> solution1 = ode_solver_1->dg->solution;
        dealii::LinearAlgebra::distributed::Vector<double> solution2 = ode_solver_2->dg->solution;
        dealii::LinearAlgebra::distributed::Vector<double> sensitivity_dWdb(solution1.size());

        dealii::TableHandler solutions_table;
        for(unsigned int i = 0 ; i < solution1.size(); i++){
            sensitivity_dWdb[i] = (solution2[i] - solution1[i])/h;
            pcout << (solution2[i] - solution1[i])/h <<std::endl;
            solutions_table.add_value("Sensitivity:", sensitivity_dWdb[i]);
        }

        solutions_table.set_precision("Sensitivity:", 16);
        std::ofstream out_file("steady_state_sensitivity_fd.txt");
        solutions_table.write_text(out_file);
    }
    else{

        while(ode_solver_1->current_time < this->all_parameters->flow_solver_param.final_time) {
            ode_solver_1->step_in_time(this->all_parameters->ode_solver_param.initial_time_step, false);
            ode_solver_2->step_in_time(this->all_parameters->ode_solver_param.initial_time_step, false);

            dealii::LinearAlgebra::distributed::Vector<double> solution1 = ode_solver_1->dg->solution;
            dealii::LinearAlgebra::distributed::Vector<double> solution2 = ode_solver_2->dg->solution;
            dealii::LinearAlgebra::distributed::Vector<double> sensitivity_dWdb(solution1.size());

            for(unsigned int i = 0 ; i < solution1.size(); i++){
                sensitivity_dWdb[i] = (solution2[i] - solution1[i])/h;
                sensitivity_table.add_value(std::to_string(ode_solver_1->current_time), sensitivity_dWdb[i]);
                sensitivity_table.set_precision(std::to_string(ode_solver_1->current_time), 16);
            }
        }

        std::ofstream out_file("unsteady_sensitivity_fd.txt");
        sensitivity_table.write_text(out_file);
    }

    return 0;
}

template <int dim, int nstate>
Parameters::AllParameters BurgersRewienskiSensitivity<dim, nstate>::reinit_params(double rewienski_b) const {
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    PHiLiP::Parameters::AllParameters parameters;
    parameters.parse_parameters (parameter_handler);
    parameters.reduced_order_param.rewienski_b = rewienski_b;
    parameters.reduced_order_param.rewienski_a = this->all_parameters->reduced_order_param.rewienski_a;
    parameters.grid_refinement_study_param.num_refinements = this->all_parameters->grid_refinement_study_param.num_refinements;
    parameters.grid_refinement_study_param.poly_degree = this->all_parameters->grid_refinement_study_param.poly_degree;
    parameters.grid_refinement_study_param.grid_left = this->all_parameters->grid_refinement_study_param.grid_left;
    parameters.grid_refinement_study_param.grid_right = this->all_parameters->grid_refinement_study_param.grid_right;
    parameters.dimension = this->all_parameters->dimension;
    parameters.pde_type = this->all_parameters->pde_type;
    parameters.manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term = this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term;
    parameters.ode_solver_param.initial_time_step = this->all_parameters->ode_solver_param.initial_time_step;
    return parameters;
}

template <int dim, int nstate>
std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> BurgersRewienskiSensitivity<dim, nstate>::initialize_ode_solver(Parameters::AllParameters parameters_input) const {
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

    return ode_solver;
}

#if PHILIP_DIM==1
        template class BurgersRewienskiSensitivity<PHILIP_DIM,PHILIP_DIM>;
#endif
    } // Tests namespace
} // PHiLiP namespace