#include "pod_adaptive_sampling.h"

namespace PHiLiP {
namespace Tests {

template<int dim, int nstate>
AdaptiveSampling<dim, nstate>::AdaptiveSampling(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{
    std::shared_ptr<Tests::BurgersRewienskiSnapshot<dim, nstate>> flow_solver_case = std::make_shared<Tests::BurgersRewienskiSnapshot<dim,nstate>>(all_parameters);
    std::unique_ptr<Tests::FlowSolver<dim,nstate>> flow_solver = std::make_unique<Tests::FlowSolver<dim,nstate>>(all_parameters, flow_solver_case);
    current_pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(flow_solver->dg);
}

template <int dim, int nstate>
int AdaptiveSampling<dim, nstate>::run_test() const
{
    std::cout << "Starting adaptive sampling process" << std::endl;

    placeInitialSnapshots();
    current_pod->computeBasis();
    placeROMs();

    RowVector2d max_error_params = getMaxErrorROM();
    double tolerance = 1E-03;

    Parameters::AllParameters params = reinitParams(max_error_params);
    std::shared_ptr<Tests::BurgersRewienskiSnapshot<dim, nstate>> flow_solver_case = std::make_shared<Tests::BurgersRewienskiSnapshot<dim,nstate>>(&params);
    std::unique_ptr<Tests::FlowSolver<dim,nstate>> flow_solver = std::make_unique<Tests::FlowSolver<dim,nstate>>(&params, flow_solver_case);
    const dealii::LinearAlgebra::distributed::Vector<double> initial_conditions = flow_solver->dg->solution;

    while(max_error > tolerance){

        std::cout << "Sampling snapshot at " << max_error_params << std::endl;

        std::shared_ptr<ProperOrthogonalDecomposition::FOMSolution<dim,nstate>> fom_solution = solveSnapshotFOM(max_error_params);
        dealii::LinearAlgebra::distributed::Vector<double> state_tmp = fom_solution->state;
        snapshot_parameters.conservativeResize(snapshot_parameters.rows()+1, 2);
        snapshot_parameters.row(snapshot_parameters.rows()-1) = max_error_params;
        current_pod->addSnapshot(state_tmp -= initial_conditions);
        current_pod->computeBasis();

        //Update previous ROM errors with updated current_pod
        for(auto& [key, value] : rom_locations){
            value.compute_initial_rom_to_final_rom_error(current_pod);
            value.compute_total_error();
        }

        //Find and compute new ROM locations
        placeROMs();

        //Update max error
        max_error_params = getMaxErrorROM();

        std::cout << "Max error is: " << max_error << std::endl;
    }

    std::shared_ptr<dealii::TableHandler> snapshot_table = std::make_shared<dealii::TableHandler>();

    for(auto parameters : snapshot_parameters.rowwise()){
        snapshot_table->add_value("Rewienski a", parameters(0));
        snapshot_table->add_value("Rewienski b", parameters(1));
        snapshot_table->set_precision("Rewienski a", 16);
        snapshot_table->set_precision("Rewienski b", 16);
    }

    std::ofstream snapshot_table_file("adaptive_sampling_snapshot_table.txt");
    snapshot_table->write_text(snapshot_table_file);

    std::shared_ptr<dealii::TableHandler> rom_table = std::make_shared<dealii::TableHandler>();

    for(auto& [key, value] : rom_locations){
        rom_table->add_value("Rewienski a", value.parameter(0));
        rom_table->set_precision("Rewienski a", 16);

        rom_table->add_value("Rewienski b", value.parameter(1));
        rom_table->set_precision("Rewienski b", 16);

        rom_table->add_value("ROM errors", value.total_error);
        rom_table->set_precision("ROM errors", 16);
    }

    std::ofstream rom_table_file("adaptive_sampling_rom_table.txt");
    rom_table->write_text(rom_table_file);

    return 0;
}


template <int dim, int nstate>
RowVector2d AdaptiveSampling<dim, nstate>::getMaxErrorROM() const{
    std::cout << "Updating RBF interpolation..." << std::endl;

    int n_rows = snapshot_parameters.rows() + rom_locations.size();
    MatrixXd parameters(n_rows, 2);
    VectorXd errors(n_rows);

    int i;
    for(i = 0 ; i < snapshot_parameters.rows() ; i++){
        errors(i) = 0;
        parameters.row(i) = snapshot_parameters.row(i);
        std::cout << i << std::endl;
    }
    std::cout << i << std::endl;
    for(auto& rom: rom_locations){
        std::cout << i << std::endl;
        parameters.row(i) = rom.first;
        errors(i) = rom.second.total_error;
        i++;
    }

    //Construct radial basis function
    std::string kernel = "thin_plate_spline";
    ProperOrthogonalDecomposition::RBFInterpolation rbf = ProperOrthogonalDecomposition::RBFInterpolation(parameters, errors, kernel);

    //Find max error and parameters by minimizing function starting at each ROM location
    RowVector2d max_error_params;
    max_error = 0;
    Eigen::NumericalDiff<ProperOrthogonalDecomposition::RBFInterpolation> numericalDiffMyFunctor(rbf);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ProperOrthogonalDecomposition::RBFInterpolation>, double> levenbergMarquardt(numericalDiffMyFunctor);
    levenbergMarquardt.parameters.ftol = 1e-6;
    levenbergMarquardt.parameters.xtol = 1e-6;
    levenbergMarquardt.parameters.maxfev = 100; // Max iterations
    for(auto& rom: rom_locations){
        Eigen::VectorXd x = rom.first.transpose();
        levenbergMarquardt.minimize(x);

        double error = std::abs(rbf.evaluate(x.transpose()).value());
        if(error > max_error){
            max_error = error;
            std::cout << "Max error: " << max_error << std::endl;
            max_error_params = x.transpose();
        }
    }
    std::cout << "Max error: " << max_error << std::endl;

    //Check if max_error_params is a ROM point
    for(auto& rom: rom_locations){
        if(max_error_params.isApprox(rom.first)){
            std::cout << "Max error location is approximately the same as a ROM location. Removing ROM location." << std::endl;
            rom_locations.erase(rom.first);
            break;
        }
    }

    return max_error_params;
}

template <int dim, int nstate>
void AdaptiveSampling<dim, nstate>::placeInitialSnapshots() const{
    std::vector<double> rewienski_a_range = {2, 10};
    std::vector<double> rewienski_b_range = {0.01, 0.1};

    snapshot_parameters.resize(5,2);
    snapshot_parameters << rewienski_a_range[0], rewienski_b_range[0],
                           rewienski_a_range[0], rewienski_b_range[1],
                           rewienski_a_range[1], rewienski_b_range[1],
                           rewienski_a_range[1], rewienski_b_range[0],
                           (rewienski_a_range[0] + rewienski_a_range[1])/2, (rewienski_b_range[0] + rewienski_b_range[1])/2;

    //Get initial conditions
    Parameters::AllParameters params = reinitParams(snapshot_parameters.row(0));
    std::shared_ptr<Tests::BurgersRewienskiSnapshot<dim, nstate>> flow_solver_case = std::make_shared<Tests::BurgersRewienskiSnapshot<dim,nstate>>(&params);
    std::unique_ptr<Tests::FlowSolver<dim,nstate>> flow_solver = std::make_unique<Tests::FlowSolver<dim,nstate>>(&params, flow_solver_case);
    const dealii::LinearAlgebra::distributed::Vector<double> initial_conditions = flow_solver->dg->solution;

    for(auto snap_param : snapshot_parameters.rowwise()){
        std::cout << "Sampling initial snapshot at " << snap_param << std::endl;
        std::shared_ptr<ProperOrthogonalDecomposition::FOMSolution<dim,nstate>> fom_solution = solveSnapshotFOM(snap_param);
        dealii::LinearAlgebra::distributed::Vector<double> state_tmp = fom_solution->state;
        current_pod->addSnapshot(state_tmp -= initial_conditions);
    }
}

template <int dim, int nstate>
void AdaptiveSampling<dim, nstate>::placeROMs() const{
    ProperOrthogonalDecomposition::Delaunay delaunay(snapshot_parameters);

    for(auto centroid : delaunay.centroids.rowwise()){
        auto element = rom_locations.find(centroid);
        if(element == rom_locations.end()){
            std::cout << "Computing ROM at " << centroid << std::endl;
            std::shared_ptr<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>> rom_solution = solveSnapshotROM(centroid);
            ProperOrthogonalDecomposition::ROMTestLocation < dim,nstate > rom_location = ProperOrthogonalDecomposition::ROMTestLocation < dim, nstate>(centroid, rom_solution);
            rom_locations.emplace(centroid, rom_location);
        }
        else{
            std::cout << "ROM already computed." << std::endl;
        }
    }
}

template <int dim, int nstate>
std::shared_ptr<ProperOrthogonalDecomposition::FOMSolution<dim,nstate>> AdaptiveSampling<dim, nstate>::solveSnapshotFOM(RowVector2d parameter) const{
    std::cout << "Solving FOM at " << parameter << std::endl;
    Parameters::AllParameters params = reinitParams(parameter);

    std::shared_ptr<Tests::BurgersRewienskiSnapshot<dim, nstate>> flow_solver_case = std::make_shared<Tests::BurgersRewienskiSnapshot<dim,nstate>>(&params);
    std::unique_ptr<Tests::FlowSolver<dim,nstate>> flow_solver = std::make_unique<Tests::FlowSolver<dim,nstate>>(&params, flow_solver_case);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::implicit_solver;
    flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver->dg);
    flow_solver->ode_solver->allocate_ode_system();
    flow_solver->ode_solver->steady_state();
    flow_solver->flow_solver_case->steady_state_postprocessing(flow_solver->dg);
    // Casting to dg state
    std::shared_ptr< DGBaseState<dim,nstate,double>> dg_state = std::dynamic_pointer_cast< DGBaseState<dim,nstate,double> >(flow_solver->dg);
    // Create functional
    auto functional = BurgersRewienskiFunctional<dim,nstate,double>(flow_solver->dg,dg_state->pde_physics_fad_fad,true,false);
    //Get sensitivity from FlowSolver
    std::shared_ptr<ProperOrthogonalDecomposition::FOMSolution<dim,nstate>> fom_solution = std::make_shared<ProperOrthogonalDecomposition::FOMSolution<dim, nstate>>(flow_solver->dg, functional);

    std::cout << "Done solving FOM." << std::endl;
    return fom_solution;
}

template <int dim, int nstate>
std::shared_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> AdaptiveSampling<dim, nstate>::solveSnapshotROM(RowVector2d parameter) const{
    std::cout << "Solving ROM at " << parameter << std::endl;
    Parameters::AllParameters params = reinitParams(parameter);

    std::shared_ptr<Tests::BurgersRewienskiSnapshot<dim, nstate>> flow_solver_case = std::make_shared<Tests::BurgersRewienskiSnapshot<dim,nstate>>(&params);
    std::unique_ptr<Tests::FlowSolver<dim,nstate>> flow_solver = std::make_unique<Tests::FlowSolver<dim,nstate>>(&params, flow_solver_case);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver->dg, current_pod);
    flow_solver->ode_solver->allocate_ode_system();
    flow_solver->ode_solver->steady_state();

    // Casting to dg state
    std::shared_ptr< DGBaseState<dim,nstate,double>> dg_state = std::dynamic_pointer_cast< DGBaseState<dim,nstate,double>>(flow_solver->dg);

    // Create functional
    auto functional = BurgersRewienskiFunctional<dim,nstate,double>(flow_solver->dg,dg_state->pde_physics_fad_fad,true,false);

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix_transpose = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
    system_matrix_transpose->copy_from(flow_solver->dg->system_matrix_transpose);

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> pod_basis = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
    pod_basis->copy_from(*current_pod->getPODBasis());

    std::shared_ptr<ProperOrthogonalDecomposition::ROMSolution<dim,nstate>> rom_solution = std::make_shared<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>>(flow_solver->dg, system_matrix_transpose,functional, pod_basis);
    std::cout << "Done solving ROM." << std::endl;
    return rom_solution;
}

template <int dim, int nstate>
Parameters::AllParameters AdaptiveSampling<dim, nstate>::reinitParams(RowVector2d parameter) const{
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    PHiLiP::Parameters::AllParameters parameters;
    parameters.parse_parameters(parameter_handler);

    // Copy all parameters
    parameters.manufactured_convergence_study_param = this->all_parameters->manufactured_convergence_study_param;
    parameters.ode_solver_param = this->all_parameters->ode_solver_param;
    parameters.linear_solver_param = this->all_parameters->linear_solver_param;
    parameters.euler_param = this->all_parameters->euler_param;
    parameters.navier_stokes_param = this->all_parameters->navier_stokes_param;
    parameters.reduced_order_param= this->all_parameters->reduced_order_param;
    parameters.burgers_param = this->all_parameters->burgers_param;
    parameters.grid_refinement_study_param = this->all_parameters->grid_refinement_study_param;
    parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
    parameters.flow_solver_param = this->all_parameters->flow_solver_param;
    parameters.mesh_adaptation_param = this->all_parameters->mesh_adaptation_param;
    parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
    parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
    parameters.artificial_dissipation_param = this->all_parameters->artificial_dissipation_param;
    parameters.dimension = this->all_parameters->dimension;
    parameters.pde_type = this->all_parameters->pde_type;
    parameters.use_weak_form = this->all_parameters->use_weak_form;
    parameters.use_collocated_nodes = this->all_parameters->use_collocated_nodes;

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        parameters.burgers_param.rewienski_b = parameter(0);
        parameters.burgers_param.rewienski_b = parameter(1);
    }
    else{
        std::cout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }

    return parameters;
}




#if PHILIP_DIM==1
template class AdaptiveSampling<PHILIP_DIM, PHILIP_DIM>;
#endif


}
}