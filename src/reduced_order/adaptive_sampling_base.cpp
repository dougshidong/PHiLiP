#include "adaptive_sampling_base.h"
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "reduced_order_solution.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "testing/rom_import_helper_functions.h"
#include <cmath>
#include "rbf_interpolation.h"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"
#include "ROL_Bounds.hpp"
#include "halton.h"
#include "min_max_scaler.h"

namespace PHiLiP {

template<int dim, int nstate>
AdaptiveSamplingBase<dim, nstate>::AdaptiveSamplingBase(const PHiLiP::Parameters::AllParameters *const parameters_input,
                                                const dealii::ParameterHandler &parameter_handler_input)
        : all_parameters(parameters_input)
        , parameter_handler(parameter_handler_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
{
    configureInitialParameterSpace();
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
    system_matrix->copy_from(flow_solver->dg->system_matrix);
    current_pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(system_matrix);
    nearest_neighbors = std::make_shared<ProperOrthogonalDecomposition::NearestNeighbors>();
}

template <int dim, int nstate>
void AdaptiveSamplingBase<dim, nstate>::outputIterationData(std::string iteration) const{
    std::unique_ptr<dealii::TableHandler> snapshot_table = std::make_unique<dealii::TableHandler>();

    std::ofstream solution_out_file("solution_snapshots_iteration_" +  iteration + ".txt");
    unsigned int precision = 16;
    current_pod->dealiiSnapshotMatrix.print_formatted(solution_out_file, precision);
    solution_out_file.close();

    for(auto parameters : snapshot_parameters.rowwise()){
        for(int i = 0 ; i < snapshot_parameters.cols() ; i++){
            snapshot_table->add_value(all_parameters->reduced_order_param.parameter_names[i], parameters(i));
            snapshot_table->set_precision(all_parameters->reduced_order_param.parameter_names[i], 16);
        }
    }

    std::ofstream snapshot_table_file("snapshot_table_iteration_" + iteration + ".txt");
    snapshot_table->write_text(snapshot_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
    snapshot_table_file.close();

    std::unique_ptr<dealii::TableHandler> rom_table = std::make_unique<dealii::TableHandler>();

    for(auto it = rom_locations.begin(); it != rom_locations.end(); ++it){
        for(int i = 0 ; i < snapshot_parameters.cols() ; i++){
            rom_table->add_value(all_parameters->reduced_order_param.parameter_names[i], it->get()->parameter(i));
            rom_table->set_precision(all_parameters->reduced_order_param.parameter_names[i], 16);
        }
        rom_table->add_value("ROM_errors", it->get()->total_error);
        rom_table->set_precision("ROM_errors", 16);
    }

    std::ofstream rom_table_file("rom_table_iteration_" + iteration + ".txt");
    rom_table->write_text(rom_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);
    rom_table_file.close();
}


template <int dim, int nstate>
RowVectorXd AdaptiveSamplingBase<dim, nstate>::readROMFunctionalPoint() const{
    RowVectorXd params(1);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;

    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if(this->all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(this->all_parameters->reduced_order_param.parameter_names[0] == "rewienski_a"){
                params(0) = this->all_parameters->burgers_param.rewienski_a;
            }
            else if(this->all_parameters->reduced_order_param.parameter_names[0] == "rewienski_b"){
                params(0) = this->all_parameters->burgers_param.rewienski_b;
            }
        }
        else{
            params.resize(2);
            params(0) = this->all_parameters->burgers_param.rewienski_a;
            params(1) = this->all_parameters->burgers_param.rewienski_b;
        }
    }
    else if (flow_type == FlowCaseEnum::naca0012){
        if(this->all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(this->all_parameters->reduced_order_param.parameter_names[0] == "mach"){
                params(0) = this->all_parameters->euler_param.mach_inf;
            }
            else if(this->all_parameters->reduced_order_param.parameter_names[0] == "alpha"){
                params(0) = this->all_parameters->euler_param.angle_of_attack;
            }
        }
        else{
            params.resize(2);
            params(0) = this->all_parameters->euler_param.mach_inf;
            params(1) = this->all_parameters->euler_param.angle_of_attack; //radians!
        }
    }
    else if (flow_type == FlowCaseEnum::gaussian_bump){
        if(this->all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(this->all_parameters->reduced_order_param.parameter_names[0] == "mach"){
                params(0) = this->all_parameters->euler_param.mach_inf;
            }
        }
    }
    else{
        this->pcout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return params;
}

template <int dim, int nstate>
RowVectorXd AdaptiveSamplingBase<dim, nstate>::getMaxErrorROM() const{
    this->pcout << "Updating RBF interpolation..." << std::endl;

    int n_rows = snapshot_parameters.rows() + rom_locations.size();
    MatrixXd parameters(n_rows, snapshot_parameters.cols());
    VectorXd errors(n_rows);

    int i;
    // Loop through FOM snapshot locations and add zero error to errors vector
    for(i = 0 ; i < snapshot_parameters.rows() ; i++){
        errors(i) = 0;
        parameters.row(i) = snapshot_parameters.row(i);
    }
    this->pcout << i << std::endl;
    // Loop through ROM points and add total error to errors vector (both FOM snaps and ROM points are used to build RBF)
    for(auto it = rom_locations.begin(); it != rom_locations.end(); ++it){
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
    max_error = 0;

    for(auto it = rom_locations.begin(); it != rom_locations.end(); ++it){

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
        if(error > max_error){
            this->pcout << "RBF error is greater than current max error. Updating max error." << std::endl;
            max_error = error;
            max_error_params = rom_unscaled_optim;
            this->pcout << "RBF Max error: " << max_error << std::endl;
        }
    }

    // Check if max_error_params is a ROM point
    for(auto it = rom_locations.begin(); it != rom_locations.end(); ++it){
        if(max_error_params.isApprox(it->get()->parameter)){
            this->pcout << "Max error location is approximately the same as a ROM location. Removing ROM location." << std::endl;
            rom_locations.erase(it);
            break;
        }
    }

    return max_error_params;
}

template <int dim, int nstate>
void AdaptiveSamplingBase<dim, nstate>::placeInitialSnapshots() const{
    for(auto snap_param : snapshot_parameters.rowwise()){
        this->pcout << "Sampling initial snapshot at " << snap_param << std::endl;
        dealii::LinearAlgebra::distributed::Vector<double> fom_solution = solveSnapshotFOM(snap_param);
        nearest_neighbors->update_snapshots(snapshot_parameters, fom_solution);
        current_pod->addSnapshot(fom_solution);
        this->fom_locations.emplace_back(fom_solution);
    }
}

template <int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> AdaptiveSamplingBase<dim, nstate>::solveSnapshotFOM(const RowVectorXd& parameter) const{
    this->pcout << "Solving FOM at " << parameter << std::endl;
    Parameters::AllParameters params = reinit_params(parameter);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::implicit_solver;
    flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver->dg);
    flow_solver->ode_solver->allocate_ode_system();
    flow_solver->run();

    this->pcout << "Done solving FOM." << std::endl;
    return flow_solver->dg->solution;
}

template <int dim, int nstate>
Parameters::AllParameters AdaptiveSamplingBase<dim, nstate>::reinit_params(const RowVectorXd& parameter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;

    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "rewienski_a"){
                parameters.burgers_param.rewienski_a = parameter(0);
            }
            else if(all_parameters->reduced_order_param.parameter_names[0] == "rewienski_b"){
                parameters.burgers_param.rewienski_b = parameter(0);
            }
        }
        else{
            parameters.burgers_param.rewienski_a = parameter(0);
            parameters.burgers_param.rewienski_b = parameter(1);
        }
    }
    else if (flow_type == FlowCaseEnum::naca0012){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "mach"){
                parameters.euler_param.mach_inf = parameter(0);
            }
            else if(all_parameters->reduced_order_param.parameter_names[0] == "alpha"){
                parameters.euler_param.angle_of_attack = parameter(0); //radians!
            }
        }
        else{
            parameters.euler_param.mach_inf = parameter(0);
            parameters.euler_param.angle_of_attack = parameter(1); //radians!
        }
    }
    else if (flow_type == FlowCaseEnum::gaussian_bump){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "mach"){
                parameters.euler_param.mach_inf = parameter(0);
            }
        }
    }
    else{
        this->pcout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return parameters;
}

template <int dim, int nstate>
void AdaptiveSamplingBase<dim, nstate>::configureInitialParameterSpace() const
{
    const double pi = atan(1.0) * 4.0;

    int n_halton = all_parameters->reduced_order_param.num_halton;

    if(all_parameters->reduced_order_param.parameter_names.size() == 1){
        RowVectorXd parameter1_range;
        parameter1_range.resize(2);
        parameter1_range << all_parameters->reduced_order_param.parameter_min_values[0], all_parameters->reduced_order_param.parameter_max_values[0];
        if(all_parameters->reduced_order_param.parameter_names[0] == "alpha"){
            parameter1_range *= pi/180; // convert to radians
        }

        // Place parameters at 2 ends and center
        snapshot_parameters.resize(3,1);
        snapshot_parameters  << parameter1_range[0],
                                parameter1_range[1],
                                (parameter1_range[0]+parameter1_range[1])/2;

        snapshot_parameters.conservativeResize(snapshot_parameters.rows() + n_halton, snapshot_parameters.cols());

        double *seq = nullptr;
        for (int i = 0; i < n_halton; i++)
        {
            seq = ProperOrthogonalDecomposition::halton(i+2, 1); //ignore the first two Halton point as they are the left end and center
                snapshot_parameters(i+3) = seq[0]*(parameter1_range[1] - parameter1_range[0])+parameter1_range[0];
        }
        delete [] seq;

        this->pcout << snapshot_parameters << std::endl;
    }
    else if(all_parameters->reduced_order_param.parameter_names.size() == 2){
        RowVectorXd parameter1_range;
        parameter1_range.resize(2);
        parameter1_range << all_parameters->reduced_order_param.parameter_min_values[0], all_parameters->reduced_order_param.parameter_max_values[0];

        RowVectorXd parameter2_range;
        parameter2_range.resize(2);
        parameter2_range << all_parameters->reduced_order_param.parameter_min_values[1], all_parameters->reduced_order_param.parameter_max_values[1];
        if(all_parameters->reduced_order_param.parameter_names[1] == "alpha"){
            parameter2_range *= pi/180; // convert to radians
        }

        // Place 9 parameters in a grid
        snapshot_parameters.resize(9,2);
        snapshot_parameters  << parameter1_range[0], parameter2_range[0],
                                parameter1_range[0], parameter2_range[1],
                                parameter1_range[1], parameter2_range[1],
                                parameter1_range[1], parameter2_range[0],
                                0.5*(parameter1_range[1] - parameter1_range[0])+parameter1_range[0], parameter2_range[0],
                                0.5*(parameter1_range[1] - parameter1_range[0])+parameter1_range[0], parameter2_range[1],
                                parameter1_range[0], 0.5*(parameter2_range[1] - parameter2_range[0])+parameter2_range[0],
                                parameter1_range[1], 0.5*(parameter2_range[1] - parameter2_range[0])+parameter2_range[0],
                                0.5*(parameter1_range[1] - parameter1_range[0])+parameter1_range[0], 0.5*(parameter2_range[1] - parameter2_range[0])+parameter2_range[0];

        snapshot_parameters.conservativeResize(snapshot_parameters.rows() + n_halton, snapshot_parameters.cols());

        double *seq = nullptr;
        for (int i = 0; i < n_halton; i++)
        {
            seq = ProperOrthogonalDecomposition::halton(i+1, 2); //ignore the first Halton point as it is one of the corners
            for (int j = 0; j < 2; j++)
            {
                if(j == 0){
                    snapshot_parameters(i+9, j) = seq[j]*(parameter1_range[1] - parameter1_range[0])+parameter1_range[0];
                }
                else if(j == 1){
                    snapshot_parameters(i+9, j) = seq[j]*(parameter2_range[1] - parameter2_range[0])+parameter2_range[0];
                }
            }
        }
        delete [] seq;

        std::string path = all_parameters->reduced_order_param.file_path_for_snapshot_locations;
        this->pcout << path << std::endl;
        if(!path.empty()){
            this->pcout << "LHS Points " << std::endl;
            std::string path = all_parameters->reduced_order_param.file_path_for_snapshot_locations;
            Tests::getSnapshotParamsFromFile(snapshot_parameters, path);
        }

        this->pcout << snapshot_parameters << std::endl;
    }
}

#if PHILIP_DIM==1
    template class AdaptiveSamplingBase<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
    template class AdaptiveSamplingBase<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}