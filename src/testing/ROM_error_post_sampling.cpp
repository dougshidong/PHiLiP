#include "ROM_error_post_sampling.h"
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
ROMErrorPostSampling<dim, nstate>::ROMErrorPostSampling(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters ROMErrorPostSampling<dim, nstate>::reinitParams(std::string path) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    parameters.reduced_order_param.path_to_search = path;
    return parameters;
}

template <int dim, int nstate>
bool ROMErrorPostSampling<dim, nstate>::getSnapshotParamsFromFile() const{
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
void ROMErrorPostSampling<dim, nstate>::getROMPoints() const{
    const double pi = atan(1.0) * 4.0;
    rom_points.conservativeResize(400, 2);
    RowVectorXd parameter1_range;
    parameter1_range.resize(2);
    parameter1_range << all_parameters->reduced_order_param.parameter_min_values[0], all_parameters->reduced_order_param.parameter_max_values[0];

    RowVectorXd parameter2_range;
    parameter2_range.resize(2);
    parameter2_range << all_parameters->reduced_order_param.parameter_min_values[1], all_parameters->reduced_order_param.parameter_max_values[1];
    if(all_parameters->reduced_order_param.parameter_names[1] == "alpha"){
        parameter2_range *= pi/180; //convert to radians
    }
    double step_1 = (parameter1_range[1] - parameter1_range[0]) / (20 - 1);
    double step_2 = (parameter2_range[1] - parameter2_range[0]) / (20 - 1);

    std::cout << step_1 << std::endl;
    std::cout << step_2 << std::endl;

    int row = 0;
    for (int i = 0; i < 20; i++){
        for(int j = 0; j < 20; j++){
            rom_points(row, 0) =  parameter1_range[0] + (step_1 * i);
            rom_points(row, 1) =  parameter2_range[0] + (step_2 * j);

            std::cout << rom_points(row, 0)  << std::endl;
            std::cout << rom_points(row, 1)  << std::endl;
            row ++;
        }
    }
}

template <int dim, int nstate>
int ROMErrorPostSampling<dim, nstate>::run_test() const
{
    pcout << "Starting error analysis for ROM..." << std::endl;

    // Create POD Petrov-Galerkin ROM from Offline POD Files
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_petrov_galerkin->dg);
    
    // Create Instance of Adaptive Sampling to calculate the error between the FOM and ROM at the points from getROMPoints
    std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
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
    getROMPoints();
    std::cout << "ROM Locations" << std::endl;
    std::cout << rom_points << std::endl; 
    parameter_sampling->placeROMLocations(rom_points);

    // Output Error Table like an iteration in the adaptive sampling procedure
    parameter_sampling->outputIterationData(1000);
    return 0;
}

#if PHILIP_DIM==1
        template class ROMErrorPostSampling<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class ROMErrorPostSampling<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
