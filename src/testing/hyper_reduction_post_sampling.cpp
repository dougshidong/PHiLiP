#include "hyper_reduction_post_sampling.h"
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
#include "reduced_order/hyper_reduced_adaptive_sampling.h"
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
            unsigned int rows = 0;
            unsigned int cols = 0;
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
                        if (cols > (all_parameters->reduced_order_param.parameter_names.size())){
                            if(abs(std::stod(field)) > all_parameters->hyper_reduction_param.ROM_error_tol){
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

            unsigned int row = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                unsigned int col = 0;
                bool any_entry = false;
                bool boundary_tol = false;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    }
                    else if (col > (all_parameters->reduced_order_param.parameter_names.size()-1)){
                        try{
                            if(abs(std::stod(field)) > all_parameters->hyper_reduction_param.ROM_error_tol){
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
    pcout << "Starting hyperreduction test..." << std::endl;

    // Create POD Petrov-Galerkin ROM without Hyperreduction
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    
    // Create POD Petrov-Galerkin ROM with Hyperreduction
    Parameters::AllParameters new_parameters = reinitParams(50);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_hyper_reduced_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&new_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;

    // Run Adaptive Sampling to choose snapshot locations and create POD basis
    std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    
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

    // Find C and d for NNLS Problem
    std::cout << "Construct instance of Assembler..."<< std::endl;
    std::shared_ptr<HyperReduction::AssembleECSWBase<dim,nstate>> constructer_NNLS_problem;
    if (this->all_parameters->hyper_reduction_param.training_data == "residual")         
        constructer_NNLS_problem = std::make_shared<HyperReduction::AssembleECSWRes<dim,nstate>>(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod, parameter_sampling->snapshot_parameters, ode_solver_type);
    else {
        constructer_NNLS_problem = std::make_shared<HyperReduction::AssembleECSWJac<dim,nstate>>(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod, parameter_sampling->snapshot_parameters, ode_solver_type);
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

    // SOLVE FOR ERROR AT ROM POINTS WITH HYPER-REDUCED WEIGHTS
    std::shared_ptr<HyperreducedAdaptiveSampling<dim,nstate>> hyper_reduced_ROM_solver = std::make_unique<HyperreducedAdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    hyper_reduced_ROM_solver->current_pod = parameter_sampling->current_pod;
    hyper_reduced_ROM_solver->snapshot_parameters = parameter_sampling->snapshot_parameters;
    hyper_reduced_ROM_solver->placeROMLocations(rom_points, weights);
    hyper_reduced_ROM_solver->outputIterationData(1000);
    
    return 0;
}

#if PHILIP_DIM==1
        template class HyperReductionPostSampling<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HyperReductionPostSampling<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
