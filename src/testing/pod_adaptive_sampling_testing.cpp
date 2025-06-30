#include "pod_adaptive_sampling_testing.h"

#include <deal.II/base/numbers.h>

#include <eigen/Eigen/Dense>
#include <fstream>

#include "dg/dg_base.hpp"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "functional/functional.h"
#include "ode_solver/ode_solver_factory.h"
#include "parameters/all_parameters.h"
#include "reduced_order/pod_basis_offline.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
AdaptiveSamplingTesting<dim, nstate>::AdaptiveSamplingTesting(const PHiLiP::Parameters::AllParameters *const parameters_input,
                                                              const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int AdaptiveSamplingTesting<dim, nstate>::run_test() const
{

    RowVectorXd params_1 {{4.0000000000000000,
                           2.0000000000000000
                          }};
    //INPUT AS RADIANS IF ANGLE OF ATTACK
    RowVectorXd params_2 {{0.0325000000000000,
                           0.0100000000000000
                          }};

    std::cout << params_1 << std::endl;
    std::cout << params_2 << std::endl;

    std::shared_ptr<dealii::TableHandler> data_table = std::make_shared<dealii::TableHandler>();

    for(int i=0 ; i < params_1.size() ; i++){
        std::cout << "Index: " << i << std::endl;
        std::cout << "Parameter 1: " << params_1(i) << std::endl;
        std::cout << "Parameter 2: " << params_2(i) << std::endl;

        RowVector2d parameter = {params_1(i), params_2(i)};
        Parameters::AllParameters params = reinit_params(parameter);

        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_implicit = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);

        auto functional_implicit = FunctionalFactory<dim,nstate,double>::create_Functional(params.functional_param, flow_solver_implicit->dg);

        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);
        auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
        std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_standard = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver->dg);
        flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver->dg, pod_standard);
        flow_solver->ode_solver->allocate_ode_system();
        auto functional = FunctionalFactory<dim,nstate,double>::create_Functional(params.functional_param, flow_solver->dg);


        flow_solver->ode_solver->steady_state();
        flow_solver_implicit->run();

        dealii::LinearAlgebra::distributed::Vector<double> standard_solution(flow_solver->dg->solution);
        dealii::LinearAlgebra::distributed::Vector<double> implicit_solution(flow_solver_implicit->dg->solution);

        double standard_error = ((standard_solution-=implicit_solution).l2_norm()/implicit_solution.l2_norm());
        double standard_func_error = functional->evaluate_functional(false,false) - functional_implicit->evaluate_functional(false,false);

        pcout << "State error: " << std::setprecision(15) << standard_error << std::setprecision(6) << std::endl;
        pcout << "Functional error: " << std::setprecision(15) << standard_func_error << std::setprecision(6) << std::endl;

        data_table->add_value("Parameter 1", parameter(0));
        data_table->add_value("Parameter 2", parameter(1));
        data_table->add_value("FOM func", functional_implicit->evaluate_functional(false,false));
        data_table->add_value("ROM func", functional->evaluate_functional(false,false));
        data_table->add_value("Func error", standard_func_error);
        data_table->add_value("State error", standard_error);

        data_table->set_precision("Parameter 1", 16);
        data_table->set_precision("Parameter 2", 16);
        data_table->set_precision("FOM func", 16);
        data_table->set_precision("ROM func", 16);
        data_table->set_precision("Func error", 16);
        data_table->set_precision("State error", 16);

        std::ofstream data_table_file("adaptation_data_table.txt");
        data_table->write_text(data_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }
    return 0;
}

template <int dim, int nstate>
Parameters::AllParameters AdaptiveSamplingTesting<dim, nstate>::reinit_params(RowVector2d parameter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        parameters.burgers_param.rewienski_a = parameter(0);
        parameters.burgers_param.rewienski_b = parameter(1);
    }
    else if (flow_type == FlowCaseEnum::naca0012){
        //const double pi = atan(1.0) * 4.0;
        parameters.euler_param.mach_inf = parameter(0);
        parameters.euler_param.angle_of_attack = parameter(1); //Already in radians
    }
    else{
        this->pcout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return parameters;
}

#if PHILIP_DIM==1
        template class AdaptiveSamplingTesting<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class AdaptiveSamplingTesting<PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

