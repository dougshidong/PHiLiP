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
AdaptiveSamplingTesting<dim, nstate>::AdaptiveSamplingTesting(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int AdaptiveSamplingTesting<dim, nstate>::run_test() const
{
    //Generate points to test:

    RowVectorXd snap_a {{2,2,10,10,6,4,8,3,7,5}};
    RowVectorXd snap_b {{0.01,0.1,0.1,0.01,0.0397,0.0703,0.0199,0.0496,0.0802,0.0298}};

    RowVectorXd rom_a {{6.667, 2.33333,9.333,3,     4.3333,4,      5.6667,   5	,5	,7,	6.3333,	8.333,	6.3333,	3.33333}};
    RowVectorXd rom_b {{0.0133,0.0532,0.0433,0.0733,0.0835,	0.0499,0.0634	,0.0199,0.0466,0.0466,	0.0934	,0.0667,0.0298,0.0298}};

    RowVectorXd A = VectorXd::LinSpaced(7,2,10).replicate(snap_a.size(),1).transpose();
    MatrixXd b = VectorXd::LinSpaced(7, 0.01, 0.1).replicate(1,snap_b.size());
    b.transposeInPlace();
    VectorXd B_col(Eigen::Map<VectorXd>(b.data(), b.cols()*b.rows()));
    RowVectorXd B = B_col.transpose();

    RowVectorXd params_a(A.size() + snap_a.size() + rom_a.size());
    params_a << snap_a, rom_a, A;
    std::cout << params_a << std::endl;

    RowVectorXd params_b(B.size() + snap_b.size() + rom_b.size());
    params_b << snap_b, rom_b, B;
    std::cout << params_b << std::endl;


    //RowVectorXd params_a{{6.3333}};
    //RowVectorXd params_b{{0.0934}};


    std::shared_ptr<dealii::TableHandler> data_table = std::make_shared<dealii::TableHandler>();

    for(int i=0 ; i < params_a.size() ; i++){
        std::cout << "Index: " << i << std::endl;
        std::cout << "Rewienski a: " << params_a(i) << std::endl;
        std::cout << "Rewienski b: " << params_b(i) << std::endl;

        RowVector2d parameter = {params_a(i), params_b(i)};
        Parameters::AllParameters params = reinitParams(parameter);

        std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_implicit = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params);
        auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::implicit_solver;
        flow_solver_implicit->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_implicit->dg);
        flow_solver_implicit->ode_solver->allocate_ode_system();
        std::shared_ptr<DGBaseState<dim,nstate,double>> dg_state_implicit = std::dynamic_pointer_cast<DGBaseState<dim,nstate,double>>(flow_solver_implicit->dg);
        auto functional_implicit = BurgersRewienskiFunctional<dim,nstate,double>(flow_solver_implicit->dg, dg_state_implicit->pde_physics_fad_fad, true, false);

        std::unique_ptr<FlowSolver<dim,nstate>> flow_solver_standard = FlowSolverFactory<dim,nstate>::create_FlowSolver(&params);
        ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
        std::shared_ptr<ProperOrthogonalDecomposition::CoarseStatePOD<dim>> pod_standard = std::make_shared<ProperOrthogonalDecomposition::CoarseStatePOD<dim>>(flow_solver_standard->dg);
        flow_solver_standard->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_standard->dg, pod_standard);
        flow_solver_standard->ode_solver->allocate_ode_system();
        std::shared_ptr<DGBaseState<dim,nstate,double> > dg_state_standard = std::dynamic_pointer_cast< DGBaseState<dim,nstate,double>>(flow_solver_standard->dg);
        auto functional_standard = BurgersRewienskiFunctional<dim,nstate,double>(flow_solver_standard->dg, dg_state_standard->pde_physics_fad_fad, true, false);

        flow_solver_implicit->ode_solver->steady_state();
        flow_solver_standard->ode_solver->steady_state();

        dealii::LinearAlgebra::distributed::Vector<double> standard_solution(flow_solver_standard->dg->solution);
        dealii::LinearAlgebra::distributed::Vector<double> implicit_solution(flow_solver_implicit->dg->solution);

        double standard_error = ((standard_solution-=implicit_solution).l2_norm()/implicit_solution.l2_norm());
        double standard_func_error = functional_standard.evaluate_functional(false,false) - functional_implicit.evaluate_functional(false,false);

        pcout << "Standard error: " << standard_error << std::endl;
        pcout << "Standard func error: " << std::setprecision(15)  << standard_func_error << std::setprecision(6) << std::endl;

        data_table->add_value("Rewienski a", parameter(0));
        data_table->add_value("Rewienski b", parameter(1));
        data_table->add_value("FOM func", functional_implicit.evaluate_functional(false,false));
        data_table->add_value("ROM func", functional_standard.evaluate_functional(false,false));
        data_table->add_value("Func error", standard_func_error);
        data_table->add_value("State error", standard_error);

        data_table->set_precision("Rewienski a", 16);
        data_table->set_precision("Rewienski b", 16);
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
        parameters.burgers_param.rewienski_a = parameter(0);
        parameters.burgers_param.rewienski_b = parameter(1);
    }
    else{
        std::cout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }

    return parameters;
}


#if PHILIP_DIM==1
template class AdaptiveSamplingTesting<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace

