#include "hyper_reduced_sampling_error_updated.h"
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "reduced_order_solution.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <cmath>
#include "rbf_interpolation.h"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"
#include "ROL_Bounds.hpp"
#include "halton.h"
#include "min_max_scaler.h"
#include "pod_adaptive_sampling.h"
#include "assemble_ECSW_residual.h"
#include "assemble_ECSW_jacobian.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"

namespace PHiLiP {

template<int dim, int nstate>
HyperreducedSamplingErrorUpdated<dim, nstate>::HyperreducedSamplingErrorUpdated(const PHiLiP::Parameters::AllParameters *const parameters_input,
                                                const dealii::ParameterHandler &parameter_handler_input)
        : AdaptiveSamplingBase<dim, nstate>(parameters_input, parameter_handler_input)
{   }

template <int dim, int nstate>
int HyperreducedSamplingErrorUpdated<dim, nstate>::run_sampling() const
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

        dealii::Vector<double> weights_dealii(ptr_weights->MyLength());
        for(int j = 0 ; j < ptr_weights->MyLength() ; j++){
            weights_dealii[j] = (*ptr_weights)[j];
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
        NNLSSolver NNLS_prob(this->all_parameters, this->parameter_handler, constructor_NNLS_problem->A_T->trilinos_matrix(), true,  Comm, b_Epetra);
        this->pcout << "Solve NNLS problem..."<< std::endl;
        bool exit_con = NNLS_prob.solve();
        this->pcout << exit_con << std::endl;
        
        ptr_weights = std::make_shared<Epetra_Vector>(NNLS_prob.get_solution());

        // Update previous ROM errors with updated current_pod
        for(auto it = hrom_locations.begin(); it != hrom_locations.end(); ++it){
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
RowVectorXd HyperreducedSamplingErrorUpdated<dim, nstate>::getMaxErrorROM() const{
    this->pcout << "Updating RBF interpolation..." << std::endl;

    int n_rows = this->snapshot_parameters.rows() + hrom_locations.size();
    MatrixXd parameters(n_rows, this->snapshot_parameters.cols());
    VectorXd errors(n_rows);

    int i;
    // Loop through FOM snapshot locations and add zero error to errors vector
    for(i = 0 ; i < this->snapshot_parameters.rows() ; i++){
        errors(i) = 0;
        parameters.row(i) = this->snapshot_parameters.row(i);
    }
    this->pcout << i << std::endl;
    // Loop through ROM points and add total error to errors vector (both FOM snaps and ROM points are used to build RBF)
    for(auto it = hrom_locations.begin(); it != hrom_locations.end(); ++it){
        parameters.row(i) = it->get()->parameter.array();
        errors(i) = it->get()->total_error;
        i++;
    }

    // Must scale both axes between [0,1] for the 2d rbf interpolation to work optimally
    ProperOrthogonalDecomposition::MinMaxScaler scaler;
    MatrixXd parameters_scaled = scaler.fit_transform(parameters);

    // Construct radial basis function
    std::string kernel = "thin_plate_spline";
    ProperOrthogonalDecomposition::RBFInterpolation rbf = ProperOrthogonalDecomposition::RBFInterpolation(parameters_scaled, errors, kernel);

    // Set parameters.
    ROL::ParameterList parlist;
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Backtracking");
    parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Quasi-Newton Method");
    parlist.sublist("Step").sublist("Line Search").sublist("Secant").set("Type","Limited-Memory BFGS");
    parlist.sublist("Status Test").set("Gradient Tolerance",1.e-10);
    parlist.sublist("Status Test").set("Step Tolerance",1.e-14);
    parlist.sublist("Status Test").set("Iteration Limit",100);

    // Find max error and parameters by minimizing function starting at each ROM location
    RowVectorXd max_error_params(parameters.cols());
    this->max_error = 0;

    for(auto it = hrom_locations.begin(); it != hrom_locations.end(); ++it){

        Eigen::RowVectorXd rom_unscaled = it->get()->parameter;
        Eigen::RowVectorXd rom_scaled = scaler.transform(rom_unscaled);

        //start bounds
        int dimension = parameters.cols();
        ROL::Ptr<std::vector<double>> l_ptr = ROL::makePtr<std::vector<double>>(dimension,0.0);
        ROL::Ptr<std::vector<double>> u_ptr = ROL::makePtr<std::vector<double>>(dimension,1.0);
        ROL::Ptr<ROL::Vector<double>> lo = ROL::makePtr<ROL::StdVector<double>>(l_ptr);
        ROL::Ptr<ROL::Vector<double>> up = ROL::makePtr<ROL::StdVector<double>>(u_ptr);
        ROL::Bounds<double> bcon(lo,up);
        //end bounds

        ROL::Ptr<ROL::Step<double>> step = ROL::makePtr<ROL::LineSearchStep<double>>(parlist);
        ROL::Ptr<ROL::StatusTest<double>> status = ROL::makePtr<ROL::StatusTest<double>>(parlist);
        ROL::Algorithm<double> algo(step,status,false);

        // Iteration Vector
        ROL::Ptr<std::vector<double>> x_ptr = ROL::makePtr<std::vector<double>>(dimension, 0.0);

        // Set Initial Guess
        for(int j = 0 ; j < dimension ; j++){
            (*x_ptr)[j] = rom_scaled(j);
        }

        this->pcout << "Unscaled parameter: " << rom_unscaled << std::endl;
        this->pcout << "Scaled parameter: ";
        for(int j = 0 ; j < dimension ; j++){
            this->pcout << (*x_ptr)[j] << " ";
        }
        this->pcout << std::endl;

        ROL::StdVector<double> x(x_ptr);

        // Run Algorithm
        algo.run(x, rbf, bcon,false);

        ROL::Ptr<std::vector<double>> x_min = x.getVector();

        for(int j = 0 ; j < dimension ; j++){
            rom_scaled(j) = (*x_min)[j];
        }

        RowVectorXd rom_unscaled_optim = scaler.inverse_transform(rom_scaled);

        this->pcout << "Parameters of optimization convergence: " << rom_unscaled_optim << std::endl;

        double error = std::abs(rbf.evaluate(rom_scaled));
        this->pcout << "RBF error at optimization convergence: " << error << std::endl;
        if(error > this->max_error){
            this->pcout << "RBF error is greater than current max error. Updating max error." << std::endl;
            this->max_error = error;
            max_error_params = rom_unscaled_optim;
            this->pcout << "RBF Max error: " << this->max_error << std::endl;
        }
    }

    // Check if max_error_params is a ROM point
    for(auto it = hrom_locations.begin(); it != hrom_locations.end(); ++it){
        if(max_error_params.isApprox(it->get()->parameter)){
            this->pcout << "Max error location is approximately the same as a ROM location. Removing ROM location." << std::endl;
            hrom_locations.erase(it);
            break;
        }
    }

    return max_error_params;
}


template <int dim, int nstate>
bool HyperreducedSamplingErrorUpdated<dim, nstate>::placeROMLocations(const MatrixXd& rom_points, Epetra_Vector weights) const{
    bool error_greater_than_tolerance = false;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, this->parameter_handler);

    for(auto midpoint : rom_points.rowwise()){

        // Check if ROM point already exists as another ROM point
        auto element = std::find_if(hrom_locations.begin(), hrom_locations.end(), [&midpoint](std::unique_ptr<ProperOrthogonalDecomposition::HROMTestLocation<dim,nstate>>& location){ return location->parameter.isApprox(midpoint);} );

        // Check if ROM point already exists as a snapshot
        bool snapshot_exists = false;
        for(auto snap_param : this->snapshot_parameters.rowwise()){
            if(snap_param.isApprox(midpoint)){
                snapshot_exists = true;
            }
        }

        if(element == hrom_locations.end() && snapshot_exists == false){
            std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>> rom_solution = solveSnapshotROM(midpoint, weights);
            hrom_locations.emplace_back(std::make_unique<ProperOrthogonalDecomposition::HROMTestLocation<dim,nstate>>(midpoint, std::move(rom_solution), flow_solver->dg, weights));
            if(abs(hrom_locations.back()->total_error) > this->all_parameters->reduced_order_param.adaptation_tolerance){
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
void HyperreducedSamplingErrorUpdated<dim, nstate>::trueErrorROM(const MatrixXd& rom_points, Epetra_Vector weights) const{

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
double HyperreducedSamplingErrorUpdated<dim, nstate>::solveSnapshotROMandFOM(const RowVectorXd& parameter, Epetra_Vector weights) const{
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
void HyperreducedSamplingErrorUpdated<dim, nstate>::solveFunctionalHROM(const RowVectorXd& parameter, Epetra_Vector weights) const{
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
void HyperreducedSamplingErrorUpdated<dim, nstate>::updateNearestExistingROMs(const RowVectorXd& /*parameter*/, Epetra_Vector weights) const{
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, this->parameter_handler);

    this->pcout << "Verifying ROM points for recomputation." << std::endl;
    // Assemble ROM points in a matrix
    MatrixXd rom_points(0,0);
    for(auto it = hrom_locations.begin(); it != hrom_locations.end(); ++it){
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
            local_mean_error = local_mean_error + std::abs(hrom_locations[index[i]]->total_error);
        }
        local_mean_error = local_mean_error / (rom_points.cols() + 1);
        if ((std::abs(hrom_locations[index[0]]->total_error) > this->all_parameters->reduced_order_param.recomputation_coefficient * local_mean_error) || (std::abs(hrom_locations[index[0]]->total_error) < (1/this->all_parameters->reduced_order_param.recomputation_coefficient) * local_mean_error)) {
            this->pcout << "Total error greater than tolerance. Recomputing ROM solution" << std::endl;
            std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>> rom_solution = solveSnapshotROM(hrom_locations[index[0]]->parameter, weights);
            std::unique_ptr<ProperOrthogonalDecomposition::HROMTestLocation<dim, nstate>> rom_location = std::make_unique<ProperOrthogonalDecomposition::HROMTestLocation<dim, nstate>>(hrom_locations[index[0]]->parameter, std::move(rom_solution), flow_solver->dg, weights);
            hrom_locations[index[0]] = std::move(rom_location);
        }
    }
}

template <int dim, int nstate>
std::unique_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> HyperreducedSamplingErrorUpdated<dim, nstate>::solveSnapshotROM(const RowVectorXd& parameter, Epetra_Vector weights) const{
    this->pcout << "Solving ROM at " << parameter << std::endl;
    Parameters::AllParameters params = this->reinit_params(parameter);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_petrov_galerkin_solver;
    flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver->dg, this->current_pod, weights);
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
Epetra_Vector HyperreducedSamplingErrorUpdated<dim,nstate>::allocateVectorToSingleCore(const Epetra_Vector &b) const{
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
void HyperreducedSamplingErrorUpdated<dim, nstate>::outputIterationData(std::string iteration) const{
    std::unique_ptr<dealii::TableHandler> snapshot_table = std::make_unique<dealii::TableHandler>();

    std::ofstream solution_out_file("solution_snapshots_iteration_" +  iteration + ".txt");
    unsigned int precision = 16;
    this->current_pod->dealiiSnapshotMatrix.print_formatted(solution_out_file, precision);
    solution_out_file.close();

    for(auto parameters : this->snapshot_parameters.rowwise()){
        for(int i = 0 ; i < this->snapshot_parameters.cols() ; i++){
            snapshot_table->add_value(this->all_parameters->reduced_order_param.parameter_names[i], parameters(i));
            snapshot_table->set_precision(this->all_parameters->reduced_order_param.parameter_names[i], 16);
        }
    }

    std::ofstream snapshot_table_file("snapshot_table_iteration_" + iteration + ".txt");
    snapshot_table->write_text(snapshot_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
    snapshot_table_file.close();

    std::unique_ptr<dealii::TableHandler> rom_table = std::make_unique<dealii::TableHandler>();

    for(auto it = hrom_locations.begin(); it != hrom_locations.end(); ++it){
        for(int i = 0 ; i < this->snapshot_parameters.cols() ; i++){
            rom_table->add_value(this->all_parameters->reduced_order_param.parameter_names[i], it->get()->parameter(i));
            rom_table->set_precision(this->all_parameters->reduced_order_param.parameter_names[i], 16);
        }
        rom_table->add_value("ROM_errors", it->get()->total_error);
        rom_table->set_precision("ROM_errors", 16);
    }

    std::ofstream rom_table_file("rom_table_iteration_" + iteration + ".txt");
    rom_table->write_text(rom_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
    rom_table_file.close();
}

#if PHILIP_DIM==1
        template class HyperreducedSamplingErrorUpdated<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HyperreducedSamplingErrorUpdated<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}