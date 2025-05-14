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
#include "rom_import_helper_functions.h"
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
Parameters::AllParameters ROMErrorPostSampling<dim, nstate>::reinit_params(std::string path) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    parameters.reduced_order_param.path_to_search = path;
    return parameters;
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
    snapshot_parameters(0,0);
    std::string path = all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
    bool snap_found = getSnapshotParamsFromFile(snapshot_parameters, path);
    if (snap_found){
        parameter_sampling->snapshot_parameters = snapshot_parameters;
        pcout << "snapshot_parameters" << std::endl;
        pcout << snapshot_parameters << std::endl;
    }
    else{
        pcout << "File with snapshots not found in folder" << std::endl;
        return -1;
    }
    getROMPoints(rom_points, all_parameters);
    pcout << "ROM Locations" << std::endl;
    pcout << rom_points << std::endl; 
    parameter_sampling->trueErrorROM(rom_points);

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
