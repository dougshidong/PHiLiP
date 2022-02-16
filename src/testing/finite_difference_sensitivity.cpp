#include "finite_difference_sensitivity.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
FiniteDifferenceSensitivity<dim, nstate>::FiniteDifferenceSensitivity(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int FiniteDifferenceSensitivity<dim, nstate>::run_test() const
{
    double h = 1E-06;
    Parameters::AllParameters params_1 = reinit_params(0);
    Parameters::AllParameters params_2 = reinit_params(h);

    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_1 = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_1);
    std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_2 = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params_2);

    dealii::TableHandler sensitivity_table;
    dealii::TableHandler solutions_table;

    // Set up Flow Solver
    flow_solver_1->setup_test();
    flow_solver_2->setup_test();

    flow_solver_1->ode_solver->allocate_ode_system();
    flow_solver_2->ode_solver->allocate_ode_system();

    dealii::LinearAlgebra::distributed::Vector<double> solution1 = flow_solver_1->dg->solution;
    dealii::LinearAlgebra::distributed::Vector<double> solution2 = flow_solver_2->dg->solution;
    dealii::LinearAlgebra::distributed::Vector<double> sensitivity_dWdParam(solution1.size());


    if(this->all_parameters->flow_solver_param.steady_state == true){
        flow_solver_1->ode_solver->steady_state();
        flow_solver_2->ode_solver->steady_state();

        solution1 = flow_solver_1->dg->solution;
        solution2 = flow_solver_2->dg->solution;
        sensitivity_dWdParam(solution1.size());

        for(unsigned int i = 0 ; i < solution1.size(); i++){
            sensitivity_dWdParam[i] = (solution2[i] - solution1[i]) / h;
            sensitivity_table.add_value("Sensitivity:", sensitivity_dWdParam[i]);
            solutions_table.add_value("Solution:", solution1[i]);
        }

        std::ofstream sensitivity_out("steady_state_sensitivity_snapshots.txt");
        std::ofstream solutions_out("steady_state_solution_snapshots.txt");
        sensitivity_table.set_precision("Sensitivity:", 16);
        sensitivity_table.set_precision("Solution:", 16);
        sensitivity_table.write_text(sensitivity_out);
        solutions_table.write_text(solutions_out);
    }
    else{

        for(unsigned int i = 0 ; i < solution1.size(); i++){
            sensitivity_dWdParam[i] = (solution2[i] - solution1[i]) / h;
            sensitivity_table.add_value(std::to_string(flow_solver_1->ode_solver->current_time), sensitivity_dWdParam[i]);
            sensitivity_table.set_precision(std::to_string(flow_solver_1->ode_solver->current_time), 16);
            solutions_table.add_value(std::to_string(flow_solver_1->ode_solver->current_time), solution1[i]);
            solutions_table.set_precision(std::to_string(flow_solver_1->ode_solver->current_time), 16);
        }

        while(flow_solver_1->ode_solver->current_time < this->all_parameters->flow_solver_param.final_time) {
            flow_solver_1->ode_solver->step_in_time(this->all_parameters->ode_solver_param.initial_time_step, false);
            flow_solver_2->ode_solver->step_in_time(this->all_parameters->ode_solver_param.initial_time_step, false);

            solution1 = flow_solver_1->dg->solution;
            solution2 = flow_solver_2->dg->solution;
            sensitivity_dWdParam(solution1.size());

            for(unsigned int i = 0 ; i < solution1.size(); i++){
                sensitivity_dWdParam[i] = (solution2[i] - solution1[i]) / h;
                sensitivity_table.add_value(std::to_string(flow_solver_1->ode_solver->current_time), sensitivity_dWdParam[i]);
                sensitivity_table.set_precision(std::to_string(flow_solver_1->ode_solver->current_time), 16);
                solutions_table.add_value(std::to_string(flow_solver_1->ode_solver->current_time), solution1[i]);
                solutions_table.set_precision(std::to_string(flow_solver_1->ode_solver->current_time), 16);
            }
        }

        std::ofstream sensitivity_out("unsteady_sensitivity_snapshots.txt");
        std::ofstream solutions_out("unsteady_solution_snapshots.txt");
        sensitivity_table.write_text(sensitivity_out);
        solutions_table.write_text(solutions_out);
    }

    return 0;
}

template <int dim, int nstate>
Parameters::AllParameters FiniteDifferenceSensitivity<dim, nstate>::reinit_params(double pertubation) const{
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    PHiLiP::Parameters::AllParameters parameters;
    parameters.parse_parameters(parameter_handler);

    // Copy all parameters
    parameters.manufactured_convergence_study_param = this->all_parameters->manufactured_convergence_study_param;
    parameters.ode_solver_param = this->all_parameters->ode_solver_param;
    parameters.linear_solver_param = this->all_parameters->linear_solver_param;
    parameters.euler_param = this->all_parameters->euler_param;
    parameters.navier_stokes_param = this->all_parameters->navier_stokes_param;
    parameters.reduced_order_param= this->all_parameters->reduced_order_param;
    parameters.burgers_param = this->all_parameters->burgers_param;
    parameters.grid_refinement_study_param = this->all_parameters->grid_refinement_study_param;
    parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
    parameters.flow_solver_param = this->all_parameters->flow_solver_param;
    parameters.mesh_adaptation_param = this->all_parameters->mesh_adaptation_param;
    parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
    parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
    parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
    parameters.dimension = this->all_parameters->dimension;
    parameters.pde_type = this->all_parameters->pde_type;
    parameters.use_weak_form = this->all_parameters->use_weak_form;
    parameters.use_collocated_nodes = this->all_parameters->use_collocated_nodes;

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        parameters.burgers_param.rewienski_b = parameters.burgers_param.rewienski_b + pertubation;
    }
    else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot){
        parameters.burgers_param.diffusion_coefficient = parameters.burgers_param.diffusion_coefficient + pertubation;
    }
    else{
        std::cout << "Invalid flow case. You probably forgot to add it to the list of flow cases in finite_difference_sensitivity.cpp" << std::endl;
        std::abort();
    }

    return parameters;
}

#if PHILIP_DIM==1
        template class FiniteDifferenceSensitivity<PHILIP_DIM,PHILIP_DIM>;
#endif
    } // Tests namespace
} // PHiLiP namespace