#include "hyper_reduction_post_sampling.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_factory.h"
#include "hyper_reduction/assemble_problem_ECSW.h"
#include "hyper_reduction/assemble_ECSW_jacobian.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"
#include "pod_adaptive_sampling.h"
#include "hyper_reduction/hyper_reduced_adaptive_sampling.h"
#include <eigen/Eigen/Dense>
#include <iostream>
#include <filesystem>

namespace PHiLiP {
namespace Tests {


template <int dim, int nstate>
HyperReductionPostSampling<dim, nstate>::HyperReductionPostSampling(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters HyperReductionPostSampling<dim, nstate>::reinitParams(const int max_iter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    parameters.ode_solver_param.nonlinear_max_iterations = max_iter;
    return parameters;
}

template <int dim, int nstate>
bool HyperReductionPostSampling<dim, nstate>::getSnapshotParamsFromFile() const{
    bool file_found = false;
    snapshot_parameters(0,0);
    std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"

    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

    for (const auto & entry : files_in_directory){
        if(std::string(entry.filename()).std::string::find("snapshot_table") != std::string::npos){
            pcout << "Processing " << entry << std::endl;
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                pcout << "Error opening file." << std::endl;
                std::abort();
            }
            std::string line;
            int rows = 0;
            int cols = 0;
            //First loop set to count rows and columns
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                cols = 0;
                bool any_entry = false;
                while (getline(stream, field,' ')){ //parse data values on each line
                    if (field.empty()){ //due to whitespace
                        continue;
                    } try{
                        std::stod(field);
                        cols++;
                        any_entry = true;
                    } catch (...){
                        continue;
                    } 
                }
                if (any_entry){
                    rows++;
                }
                
            }

            snapshot_parameters.conservativeResize(rows, snapshot_parameters.cols()+cols);

            int row = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                int col = 0;
                bool any_entry = false;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    }
                    try{
                        double num_string = std::stod(field);
                        std::cout << field << std::endl;
                        snapshot_parameters(row, col) = num_string;
                        col++;
                        any_entry = true;
                    } catch (...){
                        continue;
                    }
                }
                if (any_entry){
                    row++;
                }
            }
            myfile.close();
        }
    }
    return file_found;
}

template <int dim, int nstate>
bool HyperReductionPostSampling<dim, nstate>::getROMParamsFromFile() const{
    bool file_found = false;
    rom_points(0,0);
    std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"

    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

    for (const auto & entry : files_in_directory){
        if(std::string(entry.filename()).std::string::find("rom_table") != std::string::npos){
            pcout << "Processing " << entry << std::endl;
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                pcout << "Error opening file." << std::endl;
                std::abort();
            }
            std::string line;
            int rows = 0;
            int cols = 0;
            //First loop set to count rows and columns
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                cols = 0;
                bool any_entry = false;
                bool boundary_tol = false;
                while (getline(stream, field,' ')){ //parse data values on each line
                    if (field.empty()){ //due to whitespace
                        continue;
                    } try{
                        std::stod(field);
                        cols++;
                        if (cols > 2){
                            if(abs(std::stod(field)) > 3e-5){
                                boundary_tol = true;
                                std::cout << field << std::endl;
                            }
                        }
                        any_entry = true;
                    } catch (...){
                        continue;
                    } 
                }
                if (any_entry && boundary_tol){
                    rows++;
                }
                
            }
            cols = cols -1;
            rom_points.conservativeResize(rows, rom_points.cols()+cols);

            int row = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                int col = 0;
                bool any_entry = false;
                bool boundary_tol = false;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    }
                    else if (col > 1){
                        try{
                            if(abs(std::stod(field)) > 3e-5){
                                boundary_tol = true; 
                            }
                        } catch (...){
                        continue;
                        }
                    }
                    else {
                        try{
                            double num_string = std::stod(field);
                            // std::cout << field << std::endl;
                            rom_points(row, col) = num_string;
                            col++;
                            any_entry = true;
                        } catch (...){
                            continue;
                        }
                    }
                }
                if (any_entry && boundary_tol){
                    row++;
                }
                if (row == rows){
                    break;
                }
            }
            myfile.close();
        }
    }
    return file_found;
}

template <int dim, int nstate>
int HyperReductionPostSampling<dim, nstate>::run_test() const
{
    pcout << "Starting hyper-reduction test..." << std::endl;

    // Create implicit solver for comparison
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_implicit = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto functional_implicit = FunctionalFactory<dim,nstate,double>::create_Functional(all_parameters->functional_param, flow_solver_implicit->dg);

    // Create POD Petrov-Galerkin ROM without Hyper-reduction
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    
    // Create POD Petrov-Galerkin ROM with Hyper-reduction
    Parameters::AllParameters new_parameters = reinitParams(50);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_hyper_reduced_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&new_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;

    // Run Adaptive Sampling to choose snapshot locations and create POD basis
    //Parameters::AllParameters new_parameters_2 = reinitParams(1);
    //std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(&new_parameters_2, parameter_handler);
    std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    /*
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_petrov_galerkin->dg);
    parameter_sampling->current_pod->basis = pod_petrov_galerkin->basis;
    parameter_sampling->current_pod->referenceState = pod_petrov_galerkin->referenceState;
    parameter_sampling->current_pod->snapshotMatrix = pod_petrov_galerkin->snapshotMatrix;
    bool snap_found = getSnapshotParamsFromFile();
    if (snap_found){
        parameter_sampling->snapshot_parameters = snapshot_parameters;
        std::cout << "snapshot_parameters" << std::endl;
        std::cout << snapshot_parameters << std::endl;
    }
    else{
        return -1;
    }
    getROMParamsFromFile();
    std::cout << "ROM Locations" << std::endl;
    std::cout << rom_points << std::endl; 
    parameter_sampling->placeROMLocations(rom_points); */

    parameter_sampling->run_test();
    MatrixXd snapshot_parameters = parameter_sampling->snapshot_parameters;

    // Find C and d for NNLS Problem
    std::cout << "Construct instance of Assembler..."<< std::endl;
    HyperReduction::AssembleECSWJac<dim,nstate> constructer_NNLS_problem(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod, parameter_sampling, ode_solver_type);
    std::cout << "Build Problem..."<< std::endl;
    constructer_NNLS_problem.build_problem();

    // Transfer b vector (RHS of NNLS problem) to Epetra structure
    Epetra_MpiComm Comm( MPI_COMM_WORLD );
    Epetra_Map bMap = (constructer_NNLS_problem.A->trilinos_matrix()).RowMap();
    Epetra_Vector b_Epetra (bMap);
    auto b = constructer_NNLS_problem.b;
    for(unsigned int i = 0 ; i < b.size() ; i++){
        b_Epetra[i] = b(i);
    }

    // Solve NNLS Problem for ECSW weights
    std::cout << "Create NNLS problem..."<< std::endl;
    NNLS_solver NNLS_prob(all_parameters, parameter_handler, constructer_NNLS_problem.A->trilinos_matrix(), Comm, b_Epetra);
    std::cout << "Solve NNLS problem..."<< std::endl;
    bool exit_con = NNLS_prob.solve();
    std::cout << exit_con << std::endl;

    Epetra_Vector weights = NNLS_prob.getSolution();
    std::cout << "ECSW Weights"<< std::endl;
    std::cout << weights << std::endl;

    // SOLVE FOR ERROR AT ROM POINTS WITH HYPER-REDUCED WEIGHTS
    std::shared_ptr<HyperReduction::HyperreducedAdaptiveSampling<dim,nstate>> hyper_reduced_ROM_solver = std::make_unique<HyperReduction::HyperreducedAdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    hyper_reduced_ROM_solver->current_pod = parameter_sampling->current_pod;
    MatrixXd rom_points(0,0);
    for(auto it = parameter_sampling->rom_locations.begin(); it != parameter_sampling->rom_locations.end(); ++it){
        rom_points.conservativeResize(rom_points.rows()+1, it->get()->parameter.cols());
        rom_points.row(rom_points.rows()-1) = it->get()->parameter;
    }
    hyper_reduced_ROM_solver->placeROMLocations(rom_points, weights);

    hyper_reduced_ROM_solver->outputIterationData(1000);

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

    if (std::abs(petrov_galerkin_solution_error) < 1E-6 && std::abs(petrov_galerkin_func_error) < 1E-5 && std::abs(hyper_reduced_solution_error) < 1E-6 && std::abs(hyper_reduced_func_error) < 1E-5 && exit_con){
        pcout << "Passed!";
        return 0;
    }else{
        pcout << "Failed!";
        return -1;
    }
}

#if PHILIP_DIM==1
        template class HyperReductionPostSampling<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HyperReductionPostSampling<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
