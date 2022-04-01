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
    placeInitialROMs();
    placeTriangulationROMs();

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
        placeTriangulationROMs();

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
    snapshot_table->write_text(snapshot_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);

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
    rom_table->write_text(rom_table_file, dealii::TableHandler::TextOutputFormat::org_mode_table);

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
    for(auto& [key, value] : rom_locations){
        std::cout << "Index: " << i << std::endl;
        std::cout << "Parameters: " << key << std::endl;
        std::cout << "Parameters array: " << key.array() << std::endl;
        std::cout << "Error: " << value.total_error << std::endl;
        parameters.row(i) = key.array();
        errors(i) = value.total_error;
        i++;
    }

    std::cout << "Parameters: " << parameters << std::endl;

    //Must scale both axes between [0,1] for the 2d rbf interpolation to work optimally
    MatrixXd parameters_scaled(n_rows, 2);
    for(int j = 0 ; j < parameters.cols() ; j++){
        double min = parameters.col(j).minCoeff();
        double max = parameters.col(j).maxCoeff();
        for(int k = 0 ; k < parameters.rows() ; k++){
            parameters_scaled(k, j) = (parameters(k, j) - min) / (max - min);
        }
    }

    std::cout << "Parameters scaled: " <<parameters_scaled << std::endl;

    //Construct radial basis function
    std::string kernel = "thin_plate_spline";
    ProperOrthogonalDecomposition::RBFInterpolation rbf = ProperOrthogonalDecomposition::RBFInterpolation(parameters_scaled, errors, kernel);

    // Set parameters.
    ROL::ParameterList parlist;
    //parlist.sublist("General").set("Recompute Objective Function",false);
    //parlist.sublist("Step").sublist("Line Search").set("Initial Step Size", 1.0);
    //parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Backtracking");
    parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Newton-Krylov");
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Null Curvature Condition");
    parlist.sublist("Status Test").set("Gradient Tolerance",1.e-10);
    parlist.sublist("Status Test").set("Step Tolerance",1.e-14);
    parlist.sublist("Status Test").set("Iteration Limit",100);

    //Find max error and parameters by minimizing function starting at each ROM location
    RowVector2d max_error_params(2);
    max_error = 0;

    for(auto& [key, value] : rom_locations){

        Eigen::RowVector2d rom_unscaled = key;
        Eigen::VectorXd rom_scaled;
        rom_scaled.resize(rom_unscaled.size());

        //Scale ROM location
        for(int k = 0 ; k < parameters.cols() ; k++){
            double min = parameters.col(k).minCoeff();
            double max = parameters.col(k).maxCoeff();
            rom_scaled(k) = (rom_unscaled(k) - min) / (max - min);
        }

        ROL::Ptr<ROL::Step<double>> step = ROL::makePtr<ROL::LineSearchStep<double>>(parlist);
        ROL::Ptr<ROL::StatusTest<double>> status = ROL::makePtr<ROL::StatusTest<double>>(parlist);
        ROL::Algorithm<double> algo(step,status,false);

        // Iteration Vector
        int dimension = 2;
        ROL::Ptr<std::vector<double> > x_ptr = ROL::makePtr<std::vector<double>>(dimension, 0.0);

        // Set Initial Guess
        (*x_ptr)[0] = rom_scaled(0);
        (*x_ptr)[1] = rom_scaled(1);

        std::cout << "Unscaled parameter: " << rom_unscaled << std::endl;
        std::cout << "Scaled parameter: " << (*x_ptr)[0] << " " << (*x_ptr)[1] << std::endl;

        ROL::StdVector<double> x(x_ptr);

        // Run Algorithm
        ROL::Ptr<std::ostream> outStream;
        outStream = ROL::makePtrFromRef(std::cout);
        algo.run(x, rbf, true, *outStream);

        ROL::Ptr<std::vector<double>> x_min = x.getVector();

        rom_scaled(0) = (*x_min)[0];
        rom_scaled(1) = (*x_min)[1];

        //Ensure that optimization did not converge outside of the domain or diverge.
        RowVector2d rom_unscaled_optim(2);
        for(int k = 0 ; k < 2 ; k++){
            if(rom_scaled(k) > 1){
                rom_scaled(k) = 1;
            }
            if(rom_scaled(k) < 0){
                rom_scaled(k) = 0;
            }
            double min = parameters.col(k).minCoeff();
            double max = parameters.col(k).maxCoeff();
            rom_unscaled_optim(k) = (rom_scaled(k)*(max - min)) + min;
        }

        std::cout << "Parameters of optimization convergence: " << rom_unscaled_optim << std::endl;

        double error = std::abs(rbf.evaluate(rom_scaled.transpose()).value());
        std::cout << "RBF error at optimization convergence: " << error << std::endl;
        if(error > max_error){
            std::cout << "RBF error is greater than current max error. Updating max error." << std::endl;
            max_error = error;
            max_error_params = rom_unscaled_optim;
            std::cout << "RBF Max error: " << max_error << std::endl;
        }
    }

    //Check if max_error_params is a ROM point
    for(auto& [key, value] : rom_locations){
        if(max_error_params.isApprox(key)){
            std::cout << "Max error location is approximately the same as a ROM location. Removing ROM location." << std::endl;
            rom_locations.erase(key);
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
                           0.5*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.5*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0];


                           //0.5*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.33*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.25*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.67*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.75*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.11*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.125*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.44*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.625*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.78*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.375*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.22*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0];
                           //0.875*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.56*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.0625*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.89*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.5625*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.03704*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.3125*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.37037*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
                           //0.8125*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], 0.7037*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0];
    std::cout << snapshot_parameters << std::endl;
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
void AdaptiveSampling<dim, nstate>::placeInitialROMs() const{
    std::vector<double> rewienski_a_range = {2, 10};
    std::vector<double> rewienski_b_range = {0.01, 0.1};

    MatrixXd initial_rom_parameters;
    initial_rom_parameters.resize(8,2);
    initial_rom_parameters <<
            rewienski_a_range[0], 0.25*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
            rewienski_a_range[0], 0.75*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
            rewienski_a_range[1], 0.25*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
            rewienski_a_range[1], 0.75*(rewienski_b_range[1] - rewienski_b_range[0])+rewienski_b_range[0],
            0.25*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], rewienski_b_range[0],
            0.75*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], rewienski_b_range[0],
            0.25*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], rewienski_b_range[1],
            0.75*(rewienski_a_range[1] - rewienski_a_range[0])+rewienski_a_range[0], rewienski_b_range[1];

    for(auto rom_param : initial_rom_parameters.rowwise()){
        std::cout << "Sampling initial ROM at " << rom_param << std::endl;
        std::shared_ptr<ProperOrthogonalDecomposition::ROMSolution<dim, nstate>> rom_solution = solveSnapshotROM(rom_param);
        ProperOrthogonalDecomposition::ROMTestLocation < dim,nstate > rom_location = ProperOrthogonalDecomposition::ROMTestLocation < dim, nstate>(rom_param, rom_solution);
        rom_locations.emplace(rom_param, rom_location);
    }
}

template <int dim, int nstate>
void AdaptiveSampling<dim, nstate>::placeTriangulationROMs() const{
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
        parameters.burgers_param.rewienski_a = parameter(0);
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