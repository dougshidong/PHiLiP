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
Parameters::AllParameters HyperReductionComparison<dim, nstate>::reinit_params(const int max_iter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    parameters.ode_solver_param.nonlinear_max_iterations = max_iter;
    return parameters;
}

template <int dim, int nstate>
bool HyperReductionComparison<dim, nstate>::getWeightsFromFile(std::shared_ptr<DGBase<dim,double>> &dg) const{
    bool file_found = false;
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    VectorXd weights_eig;
    int rows = 0;
    std::string path = all_parameters->reduced_order_param.path_to_search; 

    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

    for (const auto & entry : files_in_directory){
        if(std::string(entry.filename()).std::string::find("weights") != std::string::npos){
            pcout << "Processing " << entry << std::endl;
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                pcout << "Error opening file." << std::endl;
                std::abort();
            }
            std::string line;

            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    }
                    else {
                        try{
                            std::stod(field);
                            rows++;
                        } catch (...){
                            continue;
                        }
                    }
                }
            }

            weights_eig.resize(rows);
            int row = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    }
                    else {
                        try{
                            double num_string = std::stod(field);
                            weights_eig(row) = num_string;
                            row++;
                        } catch (...){
                            continue;
                        }
                    }
                }
            }
            myfile.close();
        }
    }

    Epetra_CrsMatrix epetra_system_matrix = dg->system_matrix.trilinos_matrix();
    const int n_quad_pts = dg->volume_quadrature_collection[dg->all_parameters->flow_solver_param.poly_degree].size();
    const int length = epetra_system_matrix.NumMyRows()/(nstate*n_quad_pts);
    int *local_elements = new int[length];
    int ctr = 0;
    for (const auto &cell : dg->dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned()){
            local_elements[ctr] = cell->active_cell_index();
            ctr +=1;
        }
    }

    Epetra_Map ColMap(rows, length, local_elements, 0, epetra_comm);
    ColMap.Print(std::cout);
    Epetra_Vector weights(ColMap);
    for(int i = 0; i < length; i++){
        int global_ind = local_elements[i];
        weights[i] = weights_eig(global_ind);
    }

    ptr_weights = std::make_shared<Epetra_Vector>(weights);
    return file_found;
}

template <int dim, int nstate>
int HyperReductionComparison<dim, nstate>::run_test() const
{
    pcout << "Starting error evaluation for ROM and HROM at one parameter location..." << std::endl;

    Epetra_MpiComm Comm( MPI_COMM_WORLD );

    // Create implicit solver for comparison
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_implicit = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto functional_implicit = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_implicit->dg);

    // Create POD Petrov-Galerkin ROM without Hyper-reduction
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    
    // Create POD Petrov-Galerkin ROM with Hyper-reduction
    Parameters::AllParameters new_parameters = reinit_params(100);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_hyper_reduced_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&new_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;

    // Run Adaptive Sampling to choose snapshot locations or load from file
    std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    bool exit_con;
    if (this->all_parameters->hyper_reduction_param.adapt_sampling_bool) {
        parameter_sampling->run_sampling();
        
        // Find C and d for NNLS Problem
        pcout << "Construct instance of Assembler..."<< std::endl;
        std::shared_ptr<HyperReduction::AssembleECSWBase<dim,nstate>> constructor_NNLS_problem;
        if (this->all_parameters->hyper_reduction_param.training_data == "residual")         
            constructor_NNLS_problem = std::make_shared<HyperReduction::AssembleECSWRes<dim,nstate>>(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod,  parameter_sampling->snapshot_parameters, ode_solver_type, Comm);
        else {
            constructor_NNLS_problem = std::make_shared<HyperReduction::AssembleECSWJac<dim,nstate>>(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod,  parameter_sampling->snapshot_parameters, ode_solver_type, Comm);
        }
        pcout << "Build Problem..."<< std::endl;
        constructor_NNLS_problem->build_problem();

        // Transfer b vector (RHS of NNLS problem) to Epetra structure
        // bMap is the same map of b from the constructer_NNLS_problem, when the vector is allocated onto on core
        const int rank = Comm.MyPID();
        int rows = (constructor_NNLS_problem->A_T->trilinos_matrix()).NumGlobalCols();
        Epetra_Map bMap(rows, (rank == 0) ? rows: 0, 0, Comm);
        Epetra_Vector b_Epetra(bMap);
        auto b = constructor_NNLS_problem->b;
        unsigned int local_length = bMap.NumMyElements();
        for(unsigned int i = 0 ; i < local_length ; i++){
            b_Epetra[i] = b(i);
        }

        // Solve NNLS Problem for ECSW weights
        pcout << "Create NNLS problem..."<< std::endl;
        NNLSSolver NNLS_prob(all_parameters, parameter_handler, constructor_NNLS_problem->A_T->trilinos_matrix(), true, Comm, b_Epetra);
        pcout << "Solve NNLS problem..."<< std::endl;
        exit_con = NNLS_prob.solve();
        pcout << exit_con << std::endl;

        *ptr_weights = NNLS_prob.get_solution();
        pcout << "ECSW Weights"<< std::endl;
        pcout << *ptr_weights << std::endl;
    }
    else{
        exit_con = true;
        snapshot_parameters(0,0);
        std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
        bool snap_found = getSnapshotParamsFromFile(snapshot_parameters, path);
        if (snap_found){
            parameter_sampling->snapshot_parameters = snapshot_parameters;
        }
        else{
            pcout << "File with snapshots not found in folder" << std::endl;
            return -1;
        }
        std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_petrov_galerkin->dg);
        parameter_sampling->current_pod->basis = pod_petrov_galerkin->basis;
        parameter_sampling->current_pod->referenceState = pod_petrov_galerkin->referenceState;
        parameter_sampling->current_pod->snapshotMatrix = pod_petrov_galerkin->snapshotMatrix;

        bool weights_found = getWeightsFromFile(flow_solver_hyper_reduced_petrov_galerkin->dg);
        if (weights_found){
            pcout << "ECSW Weights" << std::endl;
            pcout << *ptr_weights << std::endl;
        }
        else{
            pcout << "File with weights not found in folder" << std::endl;
            return -1;
        }
    }
    MatrixXd snapshot_parameters = parameter_sampling->snapshot_parameters;

    // Build ODE for POD Petrov-Galerkin
    ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    flow_solver_petrov_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_petrov_galerkin->dg,  parameter_sampling->current_pod);
    flow_solver_petrov_galerkin->ode_solver->allocate_ode_system();
    auto functional_petrov_galerkin = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_petrov_galerkin->dg);

    // Build ODE for Hyper-Reduced POD Petrov-Galerkin
    ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;
    flow_solver_hyper_reduced_petrov_galerkin->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_hyper_reduced_petrov_galerkin->dg,  parameter_sampling->current_pod, *ptr_weights);
    flow_solver_hyper_reduced_petrov_galerkin->ode_solver->allocate_ode_system();
    auto functional_hyper_reduced_petrov_galerkin = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_hyper_reduced_petrov_galerkin->dg);
    
    pcout << "Implicit Solve Results"<< std::endl;
    flow_solver_implicit->run();
    pcout << "PG Solve Results"<< std::endl;
    flow_solver_petrov_galerkin->ode_solver->steady_state();
    pcout << "Hyper Reduced PG Solve Results"<< std::endl;
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
