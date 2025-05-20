#include "hyper_reduced_adaptive_sampling.h"
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "reduced_order/reduced_order_solution.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <cmath>
#include "reduced_order/rbf_interpolation.h"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"
#include "ROL_Bounds.hpp"
#include "reduced_order/halton.h"
#include "reduced_order/min_max_scaler.h"
#include "pod_adaptive_sampling.h"
#include "assemble_ECSW_residual.h"
#include "assemble_ECSW_jacobian.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"

namespace PHiLiP {

template<int dim, int nstate>
HyperreducedAdaptiveSampling<dim, nstate>::HyperreducedAdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input,
                                                const dealii::ParameterHandler &parameter_handler_input)
        : AdaptiveSamplingBase<dim, nstate>(parameters_input, parameter_handler_input)
{   }

template <int dim, int nstate>
int HyperreducedAdaptiveSampling<dim, nstate>::run_sampling() const
{
    this->pcout << "Starting adaptive sampling process" << std::endl;
    auto stream = this->pcout;
    dealii::TimerOutput timer(stream,dealii::TimerOutput::summary,dealii::TimerOutput::wall_times);
    int iteration = 0;
    timer.enter_subsection ("Iteration " + std::to_string(iteration));
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, this->parameter_handler);

    this->placeInitialSnapshots();
    this->current_pod->computeBasis();

    auto ode_solver_type_HROM = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;
    
    // Find C and d for NNLS Problem
    Epetra_MpiComm Comm( MPI_COMM_WORLD );
    this->pcout << "Construct instance of Assembler..."<< std::endl;  
    std::unique_ptr<HyperReduction::AssembleECSWBase<dim,nstate>> constructor_NNLS_problem;
    if (this->all_parameters->hyper_reduction_param.training_data == "residual")         
        constructor_NNLS_problem = std::make_unique<HyperReduction::AssembleECSWRes<dim,nstate>>(this->all_parameters, this->parameter_handler, flow_solver->dg, this->current_pod, this->snapshot_parameters, ode_solver_type_HROM, Comm);
    else {
        constructor_NNLS_problem = std::make_unique<HyperReduction::AssembleECSWJac<dim,nstate>>(this->all_parameters, this->parameter_handler, flow_solver->dg, this->current_pod, this->snapshot_parameters, ode_solver_type_HROM, Comm);
    }

    for (int k = 0; k < this->snapshot_parameters.rows(); k++){
        constructor_NNLS_problem->update_snapshots(std::move(this->fom_locations[k]));
    }    

    this->pcout << "Build Problem..."<< std::endl;
    constructor_NNLS_problem->build_problem();

    // Transfer b vector (RHS of NNLS problem) to Epetra structure
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
    this->pcout << "Create NNLS problem..."<< std::endl;
    NNLSSolver NNLS_prob(this->all_parameters, this->parameter_handler, constructor_NNLS_problem->A_T->trilinos_matrix(), true,  Comm, b_Epetra);
    this->pcout << "Solve NNLS problem..."<< std::endl;
    bool exit_con = NNLS_prob.solve();
    this->pcout << exit_con << std::endl;

    ptr_weights = std::make_shared<Epetra_Vector>(NNLS_prob.get_solution());

    MatrixXd rom_points = this->nearest_neighbors->kPairwiseNearestNeighborsMidpoint();
    this->pcout << "ROM Points"<< std::endl;
    this->pcout << rom_points << std::endl;

    this->placeROMLocations(rom_points, *ptr_weights);

    RowVectorXd max_error_params = this->getMaxErrorROM();

    RowVectorXd functional_ROM = this->readROMFunctionalPoint();

    this->pcout << "Solving FOM at " << functional_ROM << std::endl;

    Parameters::AllParameters params = this->reinit_params(functional_ROM);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_FOM = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::implicit_solver;
    flow_solver_FOM->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_FOM->dg);
    flow_solver_FOM->ode_solver->allocate_ode_system();
    flow_solver_FOM->run();

    // Create functional
    std::shared_ptr<Functional<dim,nstate,double>> functional_FOM = FunctionalFactory<dim,nstate,double>::create_Functional(params.functional_param, flow_solver_FOM->dg);
    this->pcout << "FUNCTIONAL FROM FOM" << std::endl;
    this->pcout << functional_FOM->evaluate_functional(false, false) << std::endl;

    solveFunctionalHROM(functional_ROM, *ptr_weights);
    
    while(this->max_error > this->all_parameters->reduced_order_param.adaptation_tolerance){
        Epetra_Vector local_weights = allocateVectorToSingleCore(*ptr_weights);
        this->outputIterationData(std::to_string(iteration));
        std::unique_ptr<dealii::TableHandler> weights_table = std::make_unique<dealii::TableHandler>();
        for(int i = 0 ; i < local_weights.MyLength() ; i++){
            weights_table->add_value("ECSW Weights", local_weights[i]);
            weights_table->set_precision("ECSW Weights", 16);
        }

        std::ofstream weights_table_file("weights_table_iteration_" + std::to_string(iteration) + ".txt");
        weights_table->write_text(weights_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
        weights_table_file.close();

        dealii::Vector<double> weights_dealii(local_weights.MyLength());
        for(int j = 0 ; j < local_weights.MyLength() ; j++){
            weights_dealii[j] = local_weights[j];
        } 
        flow_solver->dg->reduced_mesh_weights = weights_dealii;
        flow_solver->dg->output_results_vtk(iteration);

        timer.leave_subsection();
        timer.enter_subsection ("Iteration " + std::to_string(iteration+1));

        this->pcout << "Sampling snapshot at " << max_error_params << std::endl;
        dealii::LinearAlgebra::distributed::Vector<double> fom_solution = this->solveSnapshotFOM(max_error_params);
        this->snapshot_parameters.conservativeResize(this->snapshot_parameters.rows()+1, this->snapshot_parameters.cols());
        this->snapshot_parameters.row(this->snapshot_parameters.rows()-1) = max_error_params;
        this->nearest_neighbors->update_snapshots(this->snapshot_parameters, fom_solution);
        this->current_pod->addSnapshot(fom_solution);
        this->fom_locations.emplace_back(fom_solution);
        this->current_pod->computeBasis();

        // Find C and d for NNLS Problem
        this->pcout << "Update Assembler..."<< std::endl;
        constructor_NNLS_problem->update_POD_snaps(this->current_pod, this->snapshot_parameters);
        constructor_NNLS_problem->update_snapshots(fom_solution);
        this->pcout << "Build Problem..."<< std::endl;
        constructor_NNLS_problem->build_problem();

        // Transfer b vector (RHS of NNLS problem) to Epetra structure
        int rows = (constructor_NNLS_problem->A_T->trilinos_matrix()).NumGlobalCols();
        Epetra_Map bMap(rows, (rank == 0) ? rows: 0, 0, Comm);
        Epetra_Vector b_Epetra(bMap);
        auto b = constructor_NNLS_problem->b;
        unsigned int local_length = bMap.NumMyElements();
        for(unsigned int i = 0 ; i < local_length ; i++){
            b_Epetra[i] = b(i);
        }

        // Solve NNLS Problem for ECSW weights
        this->pcout << "Create NNLS problem..."<< std::endl;
        NNLSSolver NNLS_prob (this->all_parameters, this->parameter_handler, constructor_NNLS_problem->A_T->trilinos_matrix(), true,  Comm, b_Epetra);
        this->pcout << "Solve NNLS problem..."<< std::endl;
        bool exit_con = NNLS_prob.solve();
        this->pcout << exit_con << std::endl;
        
        ptr_weights = std::make_shared<Epetra_Vector>(NNLS_prob.get_solution());

        // Update previous ROM errors with updated current_pod
        for(auto it = this->rom_locations.begin(); it != this->rom_locations.end(); ++it){
            it->get()->compute_initial_rom_to_final_rom_error(this->current_pod);
            it->get()->compute_total_error();
        }

        this->updateNearestExistingROMs(max_error_params, *ptr_weights);

        rom_points = this->nearest_neighbors->kNearestNeighborsMidpoint(max_error_params);
        this->pcout << rom_points << std::endl;

        this->placeROMLocations(rom_points, *ptr_weights);

        // Update max error
        max_error_params = this->getMaxErrorROM();

        this->pcout << "Max error is: " << this->max_error << std::endl;

        solveFunctionalHROM(functional_ROM, *ptr_weights);

        this->pcout << "FUNCTIONAL FROM ROMs" << std::endl;
        std::ofstream output_file("rom_functional" + std::to_string(iteration+1) +".txt");

        std::ostream_iterator<double> output_iterator(output_file, "\n");
        std::copy(std::begin(rom_functional), std::end(rom_functional), output_iterator);

        iteration++;

        // Exit statement for loop if the total number of iterations is greater than the FOM dimension N (i.e. the reduced-order basis dimension n is equal to N)
        if (iteration > local_weights.MyLength()){
            break;
        }
    }

    Epetra_Vector local_weights = allocateVectorToSingleCore(*ptr_weights);
    this->outputIterationData("final");
    std::unique_ptr<dealii::TableHandler> weights_table = std::make_unique<dealii::TableHandler>();
    for(int i = 0 ; i < local_weights.MyLength() ; i++){
        weights_table->add_value("ECSW Weights", local_weights[i]);
        weights_table->set_precision("ECSW Weights", 16);
    }

    dealii::Vector<double> weights_dealii(local_weights.MyLength());
    for(int j = 0 ; j < local_weights.MyLength() ; j++){
        weights_dealii[j] = local_weights[j];
    } 
    std::ofstream weights_table_file("weights_table_iteration_final.txt");
    weights_table->write_text(weights_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
    weights_table_file.close();

    flow_solver->dg->reduced_mesh_weights = weights_dealii;
    flow_solver->dg->output_results_vtk(iteration);

    timer.leave_subsection();

    this->pcout << "FUNCTIONAL FROM ROMs" << std::endl;
    std::ofstream output_file("rom_functional.txt");

    std::ostream_iterator<double> output_iterator(output_file, "\n");
    std::copy(std::begin(rom_functional), std::end(rom_functional), output_iterator);

    return 0;
}

template <int dim, int nstate>
bool HyperreducedAdaptiveSampling<dim, nstate>::placeROMLocations(const MatrixXd& rom_points, Epetra_Vector weights) const{
    bool error_greater_than_tolerance = false;
    for(auto midpoint : rom_points.rowwise()){

        // Check if ROM point already exists as another ROM point
        auto element = std::find_if(this->rom_locations.begin(), this->rom_locations.end(), [&midpoint](std::unique_ptr<ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>>& location){ return location->parameter.isApprox(midpoint);} );

        // Check if ROM point already exists as a snapshot
        bool snapshot_exists = false;
        for(auto snap_param : this->snapshot_parameters.rowwise()){
            if(snap_param.isApprox(midpoint)){
                snapshot_exists = true;
            }
        }

        if(element == this->rom_locations.end() && snapshot_exists == false){
            std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>> rom_solution = solveSnapshotROM(midpoint, weights);
            this->rom_locations.emplace_back(std::make_unique<ProperOrthogonalDecomposition::ROMTestLocation<dim,nstate>>(midpoint, std::move(rom_solution)));
            if(abs(this->rom_locations.back()->total_error) > this->all_parameters->reduced_order_param.adaptation_tolerance){
                error_greater_than_tolerance = true;
            }
        }
        else{
            this->pcout << "ROM already computed." << std::endl;
        }
    }
    return error_greater_than_tolerance;
}

template <int dim, int nstate>
void HyperreducedAdaptiveSampling<dim, nstate>::trueErrorROM(const MatrixXd& rom_points, Epetra_Vector weights) const{

    std::unique_ptr<dealii::TableHandler> rom_table = std::make_unique<dealii::TableHandler>();

    for(auto rom : rom_points.rowwise()){
        for(int i = 0 ; i < rom_points.cols() ; i++){
            rom_table->add_value(this->all_parameters->reduced_order_param.parameter_names[i], rom(i));
            rom_table->set_precision(this->all_parameters->reduced_order_param.parameter_names[i], 16);
        }
        double error = solveSnapshotROMandFOM(rom, weights);
        this->pcout << "Error in the functional: " << error << std::endl;
        rom_table->add_value("ROM_errors", error);
        rom_table->set_precision("ROM_errors", 16);
    }

    std::ofstream rom_table_file("true_error_table_iteration_HROM_post_sampling.txt");
    rom_table->write_text(rom_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
    rom_table_file.close();
}

template <int dim, int nstate>
double HyperreducedAdaptiveSampling<dim, nstate>::solveSnapshotROMandFOM(const RowVectorXd& parameter, Epetra_Vector weights) const{
    this->pcout << "Solving HROM at " << parameter << std::endl;
    Parameters::AllParameters params = this->reinit_params(parameter);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_ROM = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);

    // Solve implicit solution
    auto ode_solver_type_ROM = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;
    flow_solver_ROM->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type_ROM, flow_solver_ROM->dg, this->current_pod, weights);
    flow_solver_ROM->ode_solver->allocate_ode_system();
    flow_solver_ROM->ode_solver->steady_state();

    this->pcout << "Done solving HROM." << std::endl;

    // Create functional
    std::shared_ptr<Functional<dim,nstate,double>> functional_ROM = FunctionalFactory<dim,nstate,double>::create_Functional(params.functional_param, flow_solver_ROM->dg);

    this->pcout << "Solving FOM at " << parameter << std::endl;

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_FOM = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::implicit_solver;
    flow_solver_FOM->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver_FOM->dg);
    flow_solver_FOM->ode_solver->allocate_ode_system();
    flow_solver_FOM->run();

    // Create functional
    std::shared_ptr<Functional<dim,nstate,double>> functional_FOM = FunctionalFactory<dim,nstate,double>::create_Functional(params.functional_param, flow_solver_FOM->dg);

    this->pcout << "Done solving FOM." << std::endl;
    return functional_ROM->evaluate_functional(false, false) - functional_FOM->evaluate_functional(false, false);
}

template <int dim, int nstate>
void HyperreducedAdaptiveSampling<dim, nstate>::solveFunctionalHROM(const RowVectorXd& parameter, Epetra_Vector weights) const{
    this->pcout << "Solving HROM at " << parameter << std::endl;
    Parameters::AllParameters params = this->reinit_params(parameter);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_ROM = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);

    // Solve 
    auto ode_solver_type_ROM = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;
    flow_solver_ROM->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type_ROM, flow_solver_ROM->dg, this->current_pod, weights);
    flow_solver_ROM->ode_solver->allocate_ode_system();
    flow_solver_ROM->ode_solver->steady_state();

    this->pcout << "Done solving HROM." << std::endl;

    // Create functional
    std::shared_ptr<Functional<dim,nstate,double>> functional_ROM = FunctionalFactory<dim,nstate,double>::create_Functional(params.functional_param, flow_solver_ROM->dg);

    rom_functional.emplace_back(functional_ROM->evaluate_functional(false, false));
}

template <int dim, int nstate>
void HyperreducedAdaptiveSampling<dim, nstate>::updateNearestExistingROMs(const RowVectorXd& /*parameter*/, Epetra_Vector weights) const{

    this->pcout << "Verifying ROM points for recomputation." << std::endl;
    // Assemble ROM points in a matrix
    MatrixXd rom_points(0,0);
    for(auto it = this->rom_locations.begin(); it != this->rom_locations.end(); ++it){
        rom_points.conservativeResize(rom_points.rows()+1, it->get()->parameter.cols());
        rom_points.row(rom_points.rows()-1) = it->get()->parameter;
    }

    // Get distances between each ROM point and all other ROM points
    for(auto point : rom_points.rowwise()) {
        ProperOrthogonalDecomposition::MinMaxScaler scaler;
        MatrixXd scaled_rom_points = scaler.fit_transform(rom_points);
        RowVectorXd scaled_point = scaler.transform(point);

        VectorXd distances = (scaled_rom_points.rowwise() - scaled_point).rowwise().squaredNorm();

        std::vector<int> index(distances.size());
        std::iota(index.begin(), index.end(), 0);

        std::sort(index.begin(), index.end(),
                  [&](const int &a, const int &b) {
                      return distances[a] < distances[b];
                  });

        this->pcout << "Searching ROM points near: " << point << std::endl;
        double local_mean_error = 0;
        for (int i = 1; i < rom_points.cols() + 2; i++) {
            local_mean_error = local_mean_error + std::abs(this->rom_locations[index[i]]->total_error);
        }
        local_mean_error = local_mean_error / (rom_points.cols() + 1);
        if ((std::abs(this->rom_locations[index[0]]->total_error) > this->all_parameters->reduced_order_param.recomputation_coefficient * local_mean_error) || (std::abs(this->rom_locations[index[0]]->total_error) < (1/this->all_parameters->reduced_order_param.recomputation_coefficient) * local_mean_error)) {
            this->pcout << "Total error greater than tolerance. Recomputing ROM solution" << std::endl;
            std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>> rom_solution = solveSnapshotROM(this->rom_locations[index[0]]->parameter, weights);
            std::unique_ptr<ProperOrthogonalDecomposition::ROMTestLocation<dim, nstate>> rom_location = std::make_unique<ProperOrthogonalDecomposition::ROMTestLocation<dim, nstate>>(this->rom_locations[index[0]]->parameter, std::move(rom_solution));
            this->rom_locations[index[0]] = std::move(rom_location);
        }
    }
}

template <int dim, int nstate>
std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> HyperreducedAdaptiveSampling<dim, nstate>::solveSnapshotROM(const RowVectorXd& parameter, Epetra_Vector weights) const{
    this->pcout << "Solving ROM at " << parameter << std::endl;
    Parameters::AllParameters params = this->reinit_params(parameter);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);

    // Solve implicit solution
    auto ode_solver_type_HROM = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;
    flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type_HROM, flow_solver->dg, this->current_pod, weights);
    flow_solver->ode_solver->allocate_ode_system();
    flow_solver->ode_solver->steady_state();

    // Create functional
    std::shared_ptr<Functional<dim,nstate,double>> functional = FunctionalFactory<dim,nstate,double>::create_Functional(params.functional_param, flow_solver->dg);
    functional->evaluate_functional( true, false, false);

    dealii::LinearAlgebra::distributed::Vector<double> solution(flow_solver->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> gradient(functional->dIdw);

    std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> rom_solution = std::make_unique<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>>(params, solution, gradient);
    this->pcout << "Done solving ROM." << std::endl;

    return rom_solution;
}

template <int dim, int nstate>
Epetra_Vector HyperreducedAdaptiveSampling<dim,nstate>::allocateVectorToSingleCore(const Epetra_Vector &b) const{
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

#if PHILIP_DIM==1
        template class HyperreducedAdaptiveSampling<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HyperreducedAdaptiveSampling<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}