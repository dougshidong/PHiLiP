#include "hyper_reduction_comparison.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/assemble_ECSW_residual.h"
#include "reduced_order/assemble_ECSW_jacobian.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"
#include "reduced_order/pod_adaptive_sampling.h"
#include "rom_import_helper_functions.h"
#include <eigen/Eigen/Dense>
#include <iostream>
#include <filesystem>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
HyperReductionComparison<dim, nstate>::HyperReductionComparison(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters HyperReductionComparison<dim, nstate>::reinitParams(const int max_iter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    parameters.ode_solver_param.nonlinear_max_iterations = max_iter;
    return parameters;
}

template <int dim, int nstate>
int HyperReductionComparison<dim, nstate>::run_test() const
{
    pcout << "Starting error evaluation for ROM and HROM at one parameter location..." << std::endl;

    // Create implicit solver for comparison
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_implicit = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto functional_implicit = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_implicit->dg);

    // Create POD Petrov-Galerkin ROM without Hyper-reduction
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    
    // Create POD Petrov-Galerkin ROM with Hyper-reduction
    Parameters::AllParameters new_parameters = reinitParams(5000);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_hyper_reduced_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&new_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;

    // Run Adaptive Sampling to choose snapshot locations or load from file
    std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    if (this->all_parameters->hyper_reduction_param.adapt_sampling_bool) {
        parameter_sampling->run_sampling();
    }
    else{
        snapshot_parameters(0,0);
        std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
        bool snap_found = getSnapshotParamsFromFile(snapshot_parameters, path);
        if (snap_found){
            parameter_sampling->snapshot_parameters = snapshot_parameters;
        }
        else{
            std::cout << "File with snapshots not found in folder" << std::endl;
            return -1;
        }
        std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_petrov_galerkin->dg);
        parameter_sampling->current_pod->basis = pod_petrov_galerkin->basis;
        parameter_sampling->current_pod->referenceState = pod_petrov_galerkin->referenceState;
        parameter_sampling->current_pod->snapshotMatrix = pod_petrov_galerkin->snapshotMatrix;
    }
    MatrixXd snapshot_parameters = parameter_sampling->snapshot_parameters;

    // Find C and d for NNLS Problem
    std::cout << "Construct instance of Assembler..."<< std::endl;
    std::shared_ptr<HyperReduction::AssembleECSWBase<dim,nstate>> constructer_NNLS_problem;
    if (this->all_parameters->hyper_reduction_param.training_data == "residual")         
        constructer_NNLS_problem = std::make_shared<HyperReduction::AssembleECSWRes<dim,nstate>>(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod,  parameter_sampling->snapshot_parameters, ode_solver_type);
    else {
        constructer_NNLS_problem = std::make_shared<HyperReduction::AssembleECSWJac<dim,nstate>>(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod,  parameter_sampling->snapshot_parameters, ode_solver_type);
    }
    std::cout << "Build Problem..."<< std::endl;
    constructer_NNLS_problem->build_problem();

    // Transfer b vector (RHS of NNLS problem) to Epetra structure
    Epetra_MpiComm Comm( MPI_COMM_WORLD );
    Epetra_Map bMap = (constructer_NNLS_problem->A->trilinos_matrix()).RowMap();
    Epetra_Vector b_Epetra (bMap);
    auto b = constructer_NNLS_problem->b;
    for(unsigned int i = 0 ; i < b.size() ; i++){
        b_Epetra[i] = b(i);
    }

    // Solve NNLS Problem for ECSW weights
    std::cout << "Create NNLS problem..."<< std::endl;
    NNLS_solver NNLS_prob(all_parameters, parameter_handler, constructer_NNLS_problem->A->trilinos_matrix(), Comm, b_Epetra);
    std::cout << "Solve NNLS problem..."<< std::endl;
    bool exit_con = NNLS_prob.solve();
    std::cout << exit_con << std::endl;

    Epetra_Vector weights = NNLS_prob.getSolution();
    std::cout << "ECSW Weights"<< std::endl;
    std::cout << weights << std::endl;

    // Build ODE for POD Petrov-Galerkin
    ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    flow_solver_petrov_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_petrov_galerkin->dg,  parameter_sampling->current_pod);
    flow_solver_petrov_galerkin->ode_solver->allocate_ode_system();
    auto functional_petrov_galerkin = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_petrov_galerkin->dg);

    // Build ODE for Hyper-Reduced POD Petrov-Galerkin
    ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;
    flow_solver_hyper_reduced_petrov_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_hyper_reduced_petrov_galerkin->dg,  parameter_sampling->current_pod, weights);
    flow_solver_hyper_reduced_petrov_galerkin->ode_solver->allocate_ode_system();
    auto functional_hyper_reduced_petrov_galerkin = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_hyper_reduced_petrov_galerkin->dg);
    
    std::cout << "Implicit Solve Results"<< std::endl;
    flow_solver_implicit->run();
    std::cout << "PG Solve Results"<< std::endl;
    flow_solver_petrov_galerkin->ode_solver->steady_state();
    std::cout << "Hyper Reduced PG Solve Results"<< std::endl;
    flow_solver_hyper_reduced_petrov_galerkin->ode_solver->steady_state();
    
    // Extract Solutions
    dealii::LinearAlgebra::distributed::Vector<double> implicit_solution(flow_solver_implicit->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> petrov_galerkin_solution(flow_solver_petrov_galerkin->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> hyper_reduced_petrov_galerkin_solution(flow_solver_hyper_reduced_petrov_galerkin->dg->solution);

    // Write solution vectors to text files
    dealii::LinearAlgebra::ReadWriteVector<double> write_implicit_solution(flow_solver_implicit->dg->solution.size());
    write_implicit_solution.import(flow_solver_implicit->dg->solution, dealii::VectorOperation::values::insert);
    std::ofstream out_file_imp("implicit_solution.txt");
    for(unsigned int i = 0 ; i < write_implicit_solution.size() ; i++){
        out_file_imp << " " << std::setprecision(17) << write_implicit_solution(i) << " \n";
    }
    out_file_imp.close();

    dealii::LinearAlgebra::ReadWriteVector<double> write_pg_solution(flow_solver_petrov_galerkin->dg->solution.size());
    write_pg_solution.import(flow_solver_petrov_galerkin->dg->solution, dealii::VectorOperation::values::insert);
    std::ofstream out_file_pg("pg_solution.txt");
    for(unsigned int i = 0 ; i < write_pg_solution.size() ; i++){
        out_file_pg << " " << std::setprecision(17) << write_pg_solution(i) << " \n";
    }
    out_file_pg.close();

    dealii::LinearAlgebra::ReadWriteVector<double> write_hyp_solution(flow_solver_hyper_reduced_petrov_galerkin->dg->solution.size());
    write_hyp_solution.import(flow_solver_hyper_reduced_petrov_galerkin->dg->solution, dealii::VectorOperation::values::insert);
    std::ofstream out_file_hyp("hyp_solution.txt");
    for(unsigned int i = 0 ; i < write_hyp_solution.size() ; i++){
        out_file_hyp << " " << std::setprecision(17) << write_hyp_solution(i) << " \n";
    }
    out_file_hyp.close();

    // Check errors in the solution and the functional
    double petrov_galerkin_solution_error = ((petrov_galerkin_solution-=implicit_solution).l2_norm()/implicit_solution.l2_norm());
    double hyper_reduced_solution_error = ((hyper_reduced_petrov_galerkin_solution-=implicit_solution).l2_norm()/implicit_solution.l2_norm());

    double petrov_galerkin_func_error = functional_petrov_galerkin->evaluate_functional(false,false) - functional_implicit->evaluate_functional(false,false);
    double hyper_reduced_func_error = functional_hyper_reduced_petrov_galerkin->evaluate_functional(false,false) - functional_implicit->evaluate_functional(false,false);

    pcout << "Petrov-Galerkin solution error: " << petrov_galerkin_solution_error << std::endl
          << "Petrov-Galerkin functional error: " << petrov_galerkin_func_error << std::endl;

    pcout << "Hyper-Reduced Petrov-Galerkin solution error: " << hyper_reduced_solution_error << std::endl
          << "Hyper-Reduced Petrov-Galerkin functional error: " << hyper_reduced_func_error << std::endl;

    if (std::abs(petrov_galerkin_solution_error) < 1E-6 && std::abs(petrov_galerkin_func_error) < 1E-4 && exit_con){
        pcout << "Passed!";
        return 0;
    }else{
        pcout << "Failed!";
        return -1;
    }
}

#if PHILIP_DIM==1
        template class HyperReductionComparison<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HyperReductionComparison<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
