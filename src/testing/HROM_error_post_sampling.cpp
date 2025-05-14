#include "HROM_error_post_sampling.h"
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
#include "rom_import_helper_functions.h"
#include <eigen/Eigen/Dense>
#include <iostream>
#include <filesystem>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {


template <int dim, int nstate>
HROMErrorPostSampling<dim, nstate>::HROMErrorPostSampling(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
Parameters::AllParameters HROMErrorPostSampling<dim, nstate>::reinit_params(std::string path) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    parameters.reduced_order_param.path_to_search = path;
    return parameters;
}

template <int dim, int nstate>
bool HROMErrorPostSampling<dim, nstate>::getWeightsFromFile(std::shared_ptr<DGBase<dim,double>> &dg) const{
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
int HROMErrorPostSampling<dim, nstate>::run_test() const
{
    pcout << "Starting error analysis for HROM..." << std::endl;

    // Create POD Petrov-Galerkin ROM with Hyper-reduction
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_hyper_reduced_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);

    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_hyper_reduced_petrov_galerkin->dg);
    std::shared_ptr<HyperreducedAdaptiveSampling<dim,nstate>> hyper_reduced_ROM_solver = std::make_unique<HyperreducedAdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    hyper_reduced_ROM_solver->current_pod->basis = pod_petrov_galerkin->basis;
    hyper_reduced_ROM_solver->current_pod->referenceState = pod_petrov_galerkin->referenceState;
    hyper_reduced_ROM_solver->current_pod->snapshotMatrix = pod_petrov_galerkin->snapshotMatrix;
    snapshot_parameters(0,0);
    std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
    bool snap_found = getSnapshotParamsFromFile(snapshot_parameters, path);
    if (snap_found){
        hyper_reduced_ROM_solver->snapshot_parameters = snapshot_parameters;
        std::cout << "snapshot_parameters" << std::endl;
        std::cout << snapshot_parameters << std::endl;
    }
    else{
        std::cout << "File with snapshots not found in folder" << std::endl;
        return -1;
    }
    getROMPoints(rom_points, all_parameters);
    std::cout << "ROM Locations" << std::endl;
    std::cout << rom_points << std::endl; 

    bool weights_found = getWeightsFromFile(flow_solver_hyper_reduced_petrov_galerkin->dg);
    if (weights_found){
        std::cout << "ECSW Weights" << std::endl;
        std::cout << *ptr_weights << std::endl;
    }
    else{
        std::cout << "File with weights not found in folder" << std::endl;
        return -1;
    }

    hyper_reduced_ROM_solver->trueErrorROM(rom_points, *ptr_weights);

    return 0;
}

#if PHILIP_DIM==1
        template class HROMErrorPostSampling<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HROMErrorPostSampling<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace
