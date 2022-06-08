#include "adaptive_sampling_testing.h"
#include <fstream>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "reduced_order_pod_adaptation.h"
#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/pod_adaptation.h"
#include "reduced_order/pod_sensitivity_base.h"
#include "reduced_order/pod_basis_sensitivity_types.h"
#include "flow_solver.h"



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
    /*
    RowVectorXd params_1 {{2.00,
                           10.00,
                          8.000,
                          8.000,
                          6.000,
                          6.000,
                          4.000,
                          4.000,
                          2.000,
                          9.219,
                          7.219,
                          9.219
                          }};
    RowVectorXd params_2 {{0.01,
                           0.055,
                          0.078,
                          0.033,
                          0.100,
                          0.010,
                          0.033,
                          0.078,
                          0.055,
                          0.043,
                          0.065,
                          0.088
                          }};
    */


    RowVectorXd params_1 {{0.9000000000000000, 0.8000000000000000, 0.7000000000000000
                          }};
    RowVectorXd params_2 {{0.0000000000000000, 0.0349065850398866, 0.0698131700797732
                          }};

    std::cout << params_1 << std::endl;
    std::cout << params_2 << std::endl;

    std::shared_ptr<dealii::TableHandler> data_table = std::make_shared<dealii::TableHandler>();

    for(int i=0 ; i < params_1.size() ; i++){
        std::cout << "Index: " << i << std::endl;
        std::cout << "Parameter 1: " << params_1(i) << std::endl;
        std::cout << "Parameter 2: " << params_2(i) << std::endl;

        RowVector2d parameter = {params_1(i), params_2(i)};
        Parameters::AllParameters params = reinitParams(parameter);

        std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_implicit = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params, parameter_handler);
        auto functional_implicit = functionalFactory(flow_solver_implicit->dg);

        std::unique_ptr<FlowSolver<dim,nstate>> flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params, parameter_handler);
        auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
        std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_standard = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver->dg);
        flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver->dg, pod_standard);
        flow_solver->ode_solver->allocate_ode_system();
        auto functional = functionalFactory(flow_solver->dg);

        flow_solver_implicit->ode_solver->steady_state();
        flow_solver->ode_solver->steady_state();

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
Parameters::AllParameters AdaptiveSamplingTesting<dim, nstate>::reinitParams(RowVector2d parameter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        parameters.burgers_param.rewienski_a = parameter(0);
        parameters.burgers_param.rewienski_b = parameter(1);
    }
    else if (flow_type == FlowCaseEnum::naca0012){
        const double pi = atan(1.0) * 4.0;
        parameters.euler_param.mach_inf = parameter(0);
        parameters.euler_param.angle_of_attack = parameter(1)*pi/180; //Convert to radians
    }
    else{
        this->pcout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return parameters;
}

template <int dim, int nstate>
std::shared_ptr<Functional<dim,nstate,double>> AdaptiveSamplingTesting<dim, nstate>::functionalFactory(std::shared_ptr<DGBase<dim, double>> dg) const
{
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if constexpr (dim==1 && nstate==dim){
            std::shared_ptr< DGBaseState<dim,nstate,double>> dg_state = std::dynamic_pointer_cast< DGBaseState<dim,nstate,double>>(dg);
            return std::make_shared<BurgersRewienskiFunctional<dim,nstate,double>>(dg,dg_state->pde_physics_fad_fad,true,false);
        }
    }
    else if (flow_type == FlowCaseEnum::naca0012){
        if constexpr (dim==2 && nstate==dim+2){
            return std::make_shared<LiftDragFunctional<dim,nstate,double>>(dg, LiftDragFunctional<dim,nstate,double>::Functional_types::lift);
        }
    }
    else{
        this->pcout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return nullptr;
}


template class AdaptiveSamplingTesting<PHILIP_DIM, PHILIP_DIM>;
template class AdaptiveSamplingTesting<PHILIP_DIM, PHILIP_DIM+2>;
} // Tests namespace
} // PHiLiP namespace

