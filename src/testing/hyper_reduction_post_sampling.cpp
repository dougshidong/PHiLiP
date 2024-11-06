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
#include "rom_import_helper_functions.h"
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
Epetra_Vector HyperReductionPostSampling<dim,nstate>::allocateVectorToSingleCore(const Epetra_Vector &b) const{
    // Gather Vector Information
    const Epetra_SerialComm sComm;
    const int b_size = b.GlobalLength();
    // Create new map for one core and gather old map
    Epetra_Map single_core_b (b_size, b_size, 0, sComm);
    Epetra_BlockMap old_map_b = b.Map();
    // Create Epetra_importer object
    Epetra_Import b_importer(single_core_b, old_map_b);
    // Create new b vector
    Epetra_Vector b_temp (single_core_b); 
    // Load the data from vector b (Multi core) into b_temp (Single core)
    b_temp.Import(b, b_importer, Epetra_CombineMode::Insert);
    return b_temp;
}

template <int dim, int nstate>
int HyperReductionPostSampling<dim, nstate>::run_test() const
{
    pcout << "Starting hyperreduction test..." << std::endl;

    Epetra_MpiComm Comm( MPI_COMM_WORLD );
    // Create POD Petrov-Galerkin ROM with Hyperreduction
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_hyper_reduced_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;

    // Run Adaptive Sampling to choose snapshot locations and create POD basis
    std::shared_ptr<AdaptiveSampling<dim,nstate>> parameter_sampling = std::make_unique<AdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    parameter_sampling->run_sampling();

    // Find C and d for NNLS Problem
    std::cout << "Construct instance of Assembler..."<< std::endl;
    std::shared_ptr<HyperReduction::AssembleECSWBase<dim,nstate>> constructer_NNLS_problem;
    if (this->all_parameters->hyper_reduction_param.training_data == "residual")         
        constructer_NNLS_problem = std::make_shared<HyperReduction::AssembleECSWRes<dim,nstate>>(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod, parameter_sampling->snapshot_parameters, ode_solver_type, Comm);
    else {
        constructer_NNLS_problem = std::make_shared<HyperReduction::AssembleECSWJac<dim,nstate>>(all_parameters, parameter_handler, flow_solver_hyper_reduced_petrov_galerkin->dg, parameter_sampling->current_pod, parameter_sampling->snapshot_parameters, ode_solver_type, Comm);
    }
    for (unsigned int j = 0 ; j < parameter_sampling->fom_locations.size() ; j++ ){
        constructer_NNLS_problem->updateSnapshots(parameter_sampling->fom_locations[j]);
    }
    std::cout << "Build Problem..."<< std::endl;
    constructer_NNLS_problem->build_problem();

    // Transfer b vector (RHS of NNLS problem) to Epetra structure
    const int rank = Comm.MyPID();
    int rows = (constructer_NNLS_problem->A_T->trilinos_matrix()).NumGlobalCols();
    Epetra_Map bMap(rows, (rank == 0) ? rows: 0, 0, Comm);
    Epetra_Vector b_Epetra(bMap);
    auto b = constructer_NNLS_problem->b;
    unsigned int local_length = bMap.NumMyElements();
    for(unsigned int i = 0 ; i < local_length ; i++){
        b_Epetra[i] = b(i);
    }

    // Solve NNLS Problem for ECSW weights
    std::cout << "Create NNLS problem..."<< std::endl;
    NNLS_solver NNLS_prob(all_parameters, parameter_handler, constructer_NNLS_problem->A_T->trilinos_matrix(), true, Comm, b_Epetra);
    std::cout << "Solve NNLS problem..."<< std::endl;
    bool exit_con = NNLS_prob.solve();
    std::cout << exit_con << std::endl;

    std::shared_ptr<Epetra_Vector> ptr_weights = std::make_shared<Epetra_Vector>(NNLS_prob.getSolution());
    std::cout << "ECSW Weights"<< std::endl;
    std::cout << *ptr_weights << std::endl;

    Epetra_Vector local_weights = allocateVectorToSingleCore(*ptr_weights);
    std::unique_ptr<dealii::TableHandler> weights_table = std::make_unique<dealii::TableHandler>();
    for(int i = 0 ; i < local_weights.MyLength() ; i++){
        weights_table->add_value("ECSW Weights", local_weights[i]);
        weights_table->set_precision("ECSW Weights", 16);
    }
    std::ofstream weights_table_file("weights_table_iteration_HROM_post_sampling.txt");
    weights_table->write_text(weights_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
    weights_table_file.close();

    // Solve for the DWR Error at the ROM points with the hyperreduced weights
    std::shared_ptr<HyperreducedAdaptiveSampling<dim,nstate>> hyper_reduced_ROM_solver = std::make_unique<HyperreducedAdaptiveSampling<dim,nstate>>(all_parameters, parameter_handler);
    hyper_reduced_ROM_solver->current_pod = parameter_sampling->current_pod;
    hyper_reduced_ROM_solver->snapshot_parameters = parameter_sampling->snapshot_parameters;
    MatrixXd rom_points(0, hyper_reduced_ROM_solver->snapshot_parameters.cols());
    for(auto it = parameter_sampling->rom_locations.begin(); it != parameter_sampling->rom_locations.end(); ++it){
        rom_points.conservativeResize(rom_points.rows()+1, rom_points.cols());
        Eigen::RowVectorXd rom = it->get()->parameter;
        rom_points.row(rom_points.rows()-1) = rom;
    }
    hyper_reduced_ROM_solver->placeROMLocations(rom_points, *ptr_weights);
    hyper_reduced_ROM_solver->outputIterationData("HROM_post_sampling");
    
    // True Error for ROM and HROM at 20 points
    MatrixXd rom_true_error_points(0,0);
    getROMPoints(rom_true_error_points, all_parameters);
    parameter_sampling->trueErrorROM(rom_true_error_points);
    hyper_reduced_ROM_solver->trueErrorROM(rom_true_error_points, *ptr_weights);

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
