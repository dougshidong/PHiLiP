#include "mesh_optimizer.hpp"
#include "functional/dual_weighted_residual_obj_func1.h"
#include "functional/dual_weighted_residual_obj_func2.h"
#include "functional/implicit_shocktracking_functional.h"
#include "optimization/design_parameterization/inner_vol_parameterization.hpp"
#include "optimization/design_parameterization/sliding_boundary_parameterization.hpp"
#include "optimization/design_parameterization/box_bounded_parameterization.hpp"
#include "optimization/design_parameterization/specific_nodes_parameterization.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"

#include "optimization/full_space_step.hpp"
#include "global_counter.hpp"

namespace PHiLiP {

template<int dim, int nstate>
MeshOptimizer<dim,nstate>::MeshOptimizer(
    std::shared_ptr<DGBase<dim,double>> dg_input,
    const Parameters::AllParameters *const parameters_input, 
    const bool _use_full_space_method)
    : dg(dg_input)
    , all_parameters(parameters_input)
    , use_full_space_method(_use_full_space_method)
    , std_outstream(nullptr)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    MPI_Comm_rank(mpi_communicator, &mpi_rank);
    MPI_Comm_size(mpi_communicator, &n_mpi);

    initialize_objfunc_and_design_parameterization();
    initialize_state_design_and_dual_variables();
    initialize_output_stream();
    n_preconditioner_calls = 0;
}

template<int dim, int nstate>
void MeshOptimizer<dim,nstate>::initialize_objfunc_and_design_parameterization()
{
//    design_parameterization = std::make_shared<BoxBoundedParameterization<dim>>(dg->high_order_grid); 
    design_parameterization = std::make_shared<SpecificNodesParameterization<dim>>(dg->high_order_grid); 
//    design_parameterization = std::make_shared<InnerVolParameterization<dim>>(dg->high_order_grid); 
//    design_parameterization = std::make_shared<SlidingBoundaryParameterization<dim>>(dg->high_order_grid); 

    bool uses_coarse_residual = false;
    const bool uses_solution_values = true;
    const bool uses_solution_gradient = false;
    if(use_full_space_method) {uses_coarse_residual = false;}

//    objective_function = std::make_shared<ImplicitShockTrackingFunctional<dim, nstate, double>> (dg,uses_solution_values,uses_solution_gradient,uses_coarse_residual);
    objective_function = std::make_shared<DualWeightedResidualObjFunc2<dim, nstate, double>> (dg,uses_solution_values,uses_solution_gradient,uses_coarse_residual);
}

template<int dim, int nstate>
void MeshOptimizer<dim,nstate>::initialize_state_design_and_dual_variables()
{
    dg->solution.update_ghost_values();
    dg->set_dual(dg->solution);

    state_variables = dg->solution;
    design_parameterization->initialize_design_variables(design_variables);
    dual_variables = dg->dual;

    state_variables.update_ghost_values();
    dual_variables.update_ghost_values();
    design_variables.update_ghost_values();
}

template<int dim, int nstate>
void MeshOptimizer<dim,nstate>::initialize_output_stream()
{
    std::string optimization_type;
    if(use_full_space_method)
    {
        optimization_type = "full_space";
    }
    else
    {
        optimization_type = "reduced_space";
    }
    

    if (this->mpi_rank == 0) filebuffer.open(optimization_type+"_optimization_run.log", std::ios::out);
    std_outstream.rdbuf(&filebuffer);
    
    if (mpi_rank == 0) {rcp_outstream = ROL::makePtrFromRef(std_outstream);} // processor #0 outputs in file.
    else if (mpi_rank == 1) {rcp_outstream = ROL::makePtrFromRef(std::cout);} // processor #1 outputs on screen.
    else rcp_outstream = ROL::makePtrFromRef(null_stream);
}
    
template<int dim, int nstate>
Teuchos::ParameterList MeshOptimizer<dim,nstate>::get_parlist()
{
    Teuchos::ParameterList parlist;
    
    // ROL set optimization parameters.
    parlist.sublist("General").set("Print Verbosity", 1);
    parlist.sublist("Status Test").set("Gradient Tolerance", all_parameters->optimization_param.gradient_tolerance);
    parlist.sublist("Status Test").set("Iteration Limit", all_parameters->optimization_param.max_design_cycles);

    parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size", all_parameters->optimization_param.initial_step_size);
    parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit", all_parameters->optimization_param.functional_evaluation_limit); 
    parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type", all_parameters->optimization_param.line_search_method);
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type", all_parameters->optimization_param.line_search_curvature);


    parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
    parlist.sublist("General").sublist("Secant").set("Maximum Storage", all_parameters->optimization_param.max_design_cycles);

    if(use_full_space_method)
    {
        parlist.sublist("Full Space").set("Preconditioner", all_parameters->optimization_param.full_space_preconditioner);
        parlist.sublist("Full Space").set("Linear iteration Limit", all_parameters->optimization_param.linear_iteration_limit);
        parlist.sublist("Full Space").set("regularization_parameter", all_parameters->optimization_param.regularization_parameter);
        parlist.sublist("Full Space").set("regularization_scaling", all_parameters->optimization_param.regularization_scaling);
        parlist.sublist("Full Space").set("regularization_tol_low", all_parameters->optimization_param.regularization_tol_low);
        parlist.sublist("Full Space").set("regularization_tol_high", all_parameters->optimization_param.regularization_tol_high);
    }
    else
    {
        // Parlist for reduced-space.
        parlist.sublist("Step").set("Type","Line Search");
        parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", all_parameters->optimization_param.reduced_space_descent_method);

        if (all_parameters->optimization_param.reduced_space_descent_method == "Newton-Krylov") {
            parlist.sublist("General").sublist("Secant").set("Use as Preconditioner", true);
            //parlist.sublist("General").sublist("Krylov").set("Type","Conjugate Gradients");
            parlist.sublist("General").sublist("Krylov").set("Type","GMRES");
            parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1.0e-8);
            parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", 1.0e-4);
            parlist.sublist("General").sublist("Krylov").set("Iteration Limit", all_parameters->optimization_param.linear_iteration_limit);
            parlist.sublist("General").set("Inexact Hessian-Times-A-Vector",false);
        }
    }

    return parlist;
}

template<int dim, int nstate>
void MeshOptimizer<dim,nstate>::run_full_space_optimizer()
{
    //==============================================================================================================================
    // Setup vector_ptrs
    const bool has_ownership = false;
    VectorAdaptor state_variables_rol(Teuchos::rcp(&state_variables, has_ownership));
    VectorAdaptor design_variables_rol(Teuchos::rcp(&design_variables, has_ownership));
    VectorAdaptor dual_variables_rol(Teuchos::rcp(&dual_variables, has_ownership));

    ROL::Ptr<ROL::Vector<double>> state_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(state_variables_rol);
    ROL::Ptr<ROL::Vector<double>> design_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(design_variables_rol);
    ROL::Ptr<ROL::Vector<double>> dual_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(dual_variables_rol);
    auto state_design_rol_ptr = ROL::makePtr<ROL::Vector_SimOpt<double>>(state_variables_rol_ptr, design_variables_rol_ptr);
    //==============================================================================================================================
    
    Teuchos::ParameterList parlist = get_parlist();
    
    auto objective_function_rol_ptr = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>(*objective_function, design_parameterization);
    auto flow_constraints_rol_ptr  = ROL::makePtr<FlowConstraints<dim>>(dg, design_parameterization); // Constraint of Residual = 0

    const double timing_start = MPI_Wtime();
    
    // Full space Newton
    *rcp_outstream << "Starting Full Space mesh optimization..."<<std::endl;
    auto full_space_step = ROL::makePtr<ROL::FullSpace_BirosGhattas<double>>(parlist);
    auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
    const bool printHeader = true;
    ROL::Algorithm<double> algorithm(full_space_step, status_test, printHeader);
    const bool print  = true;
    algorithm.run(*state_design_rol_ptr, 
                  *dual_variables_rol_ptr, 
                  *objective_function_rol_ptr, 
                  *flow_constraints_rol_ptr, 
                  print, 
                  *rcp_outstream);
    
    ROL::Ptr<const ROL::AlgorithmState <double>> algo_state = algorithm.getState();

    const double timing_end = MPI_Wtime();
    *rcp_outstream << "The process took "<<timing_end - timing_start << " seconds to run."<<std::endl;
    *rcp_outstream << "n_preconditioner_calls = "<<n_preconditioner_calls << std::endl;
    *rcp_outstream << "n_design_variables = "<< design_variables.size() << std::endl;
    filebuffer.close(); 
}


template<int dim, int nstate>
void MeshOptimizer<dim,nstate>::run_reduced_space_optimizer()
{
    //==============================================================================================================================
    // Setup vector_ptrs
    const bool has_ownership = false;
    VectorAdaptor state_variables_rol(Teuchos::rcp(&state_variables, has_ownership));
    VectorAdaptor design_variables_rol(Teuchos::rcp(&design_variables, has_ownership));
    VectorAdaptor dual_variables_rol(Teuchos::rcp(&dual_variables, has_ownership));

    ROL::Ptr<ROL::Vector<double>> state_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(state_variables_rol);
    ROL::Ptr<ROL::Vector<double>> design_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(design_variables_rol);
    ROL::Ptr<ROL::Vector<double>> dual_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(dual_variables_rol);
    //==============================================================================================================================
    
    Teuchos::ParameterList parlist = get_parlist();
    
    auto objective_function_rol_ptr = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>(*objective_function, design_parameterization);
    auto flow_constraints_rol_ptr  = ROL::makePtr<FlowConstraints<dim>>(dg, design_parameterization); // Constraint of Residual = 0

    const double timing_start = MPI_Wtime();

    // Reduced space Newton
    const bool storage = true;
    const bool useFDHessian = false;
    auto reduced_objective_rol_ptr = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>(
                                                                objective_function_rol_ptr,
                                                                flow_constraints_rol_ptr,
                                                                state_variables_rol_ptr,
                                                                design_variables_rol_ptr,
                                                                dual_variables_rol_ptr,
                                                                storage,
                                                                useFDHessian);

    ROL::OptimizationProblem<double> optimization_problem = ROL::OptimizationProblem<double>(reduced_objective_rol_ptr, design_variables_rol_ptr);
    ROL::EProblem problemType = optimization_problem.getProblemType();
    std::cout << ROL::EProblemToString(problemType) << std::endl;

    *rcp_outstream << "Starting Reduced Space mesh optimization..."<<std::endl;
    ROL::OptimizationSolver<double> solver(optimization_problem, parlist);
    solver.solve(*rcp_outstream);
    ROL::Ptr<const ROL::AlgorithmState <double>> algo_state = solver.getAlgorithmState();

    const double timing_end = MPI_Wtime();
    *rcp_outstream << "The process took "<<timing_end - timing_start << " seconds to run."<<std::endl;
    *rcp_outstream << "n_preconditioner_calls = "<<n_preconditioner_calls << std::endl;
    *rcp_outstream << "n_design_variables = "<< design_variables.size() << std::endl;
    filebuffer.close();
}

template class MeshOptimizer <PHILIP_DIM, 1>;
template class MeshOptimizer <PHILIP_DIM, 2>;
template class MeshOptimizer <PHILIP_DIM, 3>;
template class MeshOptimizer <PHILIP_DIM, 4>;
template class MeshOptimizer <PHILIP_DIM, 5>;
} // PHiLiP namespace
