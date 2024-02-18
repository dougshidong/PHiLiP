#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_BoundConstraint_SimOpt.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Reduced_Constraint_SimOpt.hpp"

#include "ROL_Constraint_Partitioned.hpp"

#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"

#include "ROL_SingletonVector.hpp"
#include <ROL_AugmentedLagrangian_SimOpt.hpp>

#include "euler_naca0012_optimization.hpp"

#include "physics/euler.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver.h"

#include "functional/target_boundary_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"
#include "optimization/constraintfromobjective_simopt.hpp"

#include "optimization/primal_dual_active_set.hpp"
#include "optimization/full_space_step.hpp"
#include "optimization/sequential_quadratic_programming.hpp"

#include "mesh/gmsh_reader.hpp"
#include "functional/lift_drag.hpp"
#include "functional/geometric_volume.hpp"
#include "functional/target_wall_pressure.hpp"

#include "global_counter.hpp"

namespace {
enum class OptimizationAlgorithm { full_space_birosghattas, reduced_space_bfgs, reduced_space_newton};
enum class PreconditionerType { P2, P2A, P4, P4A, identity };

const std::vector<PreconditionerType> precond_list { PreconditionerType::P4 };
//const std::vector<PreconditionerType> precond_list { PreconditionerType::P2, PreconditionerType::P2A, PreconditionerType::P4, PreconditionerType::P4A };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::full_space_birosghattas, OptimizationAlgorithm::reduced_space_bfgs, OptimizationAlgorithm::reduced_space_newton };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_bfgs };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_newton };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::full_space_birosghattas };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_bfgs };
const std::vector<OptimizationAlgorithm> opt_list {
    //OptimizationAlgorithm::reduced_space_newton,
    OptimizationAlgorithm::full_space_birosghattas,
    OptimizationAlgorithm::reduced_space_bfgs,
    };

const unsigned int POLY_START = 0;
const unsigned int POLY_END = 1; // Can do until at least P2

const unsigned int n_des_var_start = 10;//20;
const unsigned int n_des_var_end   = 20;//100;
const unsigned int n_des_var_step  = 10;//20;

const int max_design_cycle = 1000;

const double FD_TOL = 1e-6;
const double CONSISTENCY_ABS_TOL = 1e-10;

bool USE_BFGS = false;
const int LINESEARCH_MAX_ITER = 10;
const double BACKTRACKING_RATE = 0.5;
const int PDAS_MAX_ITER = 1;

const std::string line_search_curvature = "Null Curvature Condition";
const std::string line_search_method = "Backtracking";
}

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
OptimizationSetup<dim,nstate>::
OptimizationSetup(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
ROL::Ptr<ROL::Objective<double>> 
OptimizationSetup<dim,nstate>::
getObjective(
    const ROL::Ptr<ROL::Objective_SimOpt<double>> drag_objective,
    const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
    const ROL::Ptr<ROL::Vector<double>> design_simulation,
    const ROL::Ptr<ROL::Vector<double>> design_control,
    const bool is_reduced_space) const
{
    const auto state_constraints = ROL::makePtrFromRef<PHiLiP::FlowConstraints<PHILIP_DIM>>(dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*flow_constraints));
    ROL::Ptr<ROL::Vector<double>> drag_adjoint = design_simulation->clone();
    // int objective_check_error = check_objective<PHILIP_DIM,PHILIP_DIM+2>( drag_objective, state_constraints, design_simulation, design_control, drag_adjoint);
    // (void) objective_check_error;

    if (!is_reduced_space) return drag_objective;

    const bool storage = true; const bool useFDHessian = false;
    return ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( drag_objective, flow_constraints, design_simulation, design_control, drag_adjoint, storage, useFDHessian);
}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::Constraint<double>>>
OptimizationSetup<dim,nstate>::
getInequalityConstraint(
    const ROL::Ptr<ROL::Objective_SimOpt<double>> lift_objective,
    const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
    const ROL::Ptr<ROL::Vector<double>> design_simulation,
    const ROL::Ptr<ROL::Vector<double>> design_control,
    const double lift_target,
    const ROL::Ptr<ROL::Objective_SimOpt<double>> volume_objective,
    const bool is_reduced_space,
    const double volume_target
    ) const
{
    std::vector<ROL::Ptr<ROL::Constraint<double> > > cvec;
    ROL::Ptr<ROL::Constraint<double>> lift_constraint;
    if (is_reduced_space) {
        ROL::Ptr<ROL::Vector<double>> lift_adjoint = design_simulation->clone();
        const bool storage = true;
        const bool useFDHessian = false;
        auto reduced_lift_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( lift_objective, flow_constraints, design_simulation, design_control, lift_adjoint, storage, useFDHessian);

        const auto state_constraints = ROL::makePtrFromRef<PHiLiP::FlowConstraints<PHILIP_DIM>>(dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*flow_constraints));
        //int objective_check_error = check_objective<PHILIP_DIM,PHILIP_DIM+2>( lift_objective, state_constraints, design_simulation, design_control, lift_adjoint);
        //(void) objective_check_error;
        (void) lift_target;
        (void) volume_target;
        ROL::Ptr<ROL::Constraint<double>> reduced_lift_constraint = ROL::makePtr<ROL::ConstraintFromObjective<double>> (reduced_lift_objective, 0.0);
        lift_constraint = reduced_lift_constraint;

        cvec.push_back(lift_constraint);
        //if (volume_target > 0) {
            ROL::Ptr<ROL::Vector<double>> volume_adjoint = design_simulation->clone();
            auto reduced_volume_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( volume_objective, flow_constraints, design_simulation, design_control, volume_adjoint, storage, useFDHessian);
            const ROL::Ptr<ROL::Constraint<double>> volume_constraint = ROL::makePtr<ROL::ConstraintFromObjective<double>> (reduced_volume_objective, 0.0);
            cvec.push_back(volume_constraint);
        //}
    } else {
        ROL::Ptr<ROL::Constraint<double>> lift_constraint_simpopt = ROL::makePtr<PHiLiP::ConstraintFromObjective_SimOpt<double>> (lift_objective, 0.0);
        lift_constraint = lift_constraint_simpopt;

        cvec.push_back(lift_constraint);
        //if (volume_target > 0) {
            const ROL::Ptr<ROL::Constraint<double>> volume_constraint = ROL::makePtr<PHiLiP::ConstraintFromObjective_SimOpt<double>> (volume_objective, 0.0);
            cvec.push_back(volume_constraint);
        //}
    }

    return cvec;

}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::Vector<double>>> 
OptimizationSetup<dim,nstate>::
getInequalityMultiplier(const double volume_target) const
{
    const ROL::Ptr<ROL::SingletonVector<double>> lift_constraint_dual = ROL::makePtr<ROL::SingletonVector<double>> (1.0);
    std::vector<ROL::Ptr<ROL::Vector<double>>> emul;
    emul.push_back(lift_constraint_dual);
    (void) volume_target;
    //if (volume_target > 0) {
        const ROL::Ptr<ROL::SingletonVector<double>> volume_constraint_dual = ROL::makePtr<ROL::SingletonVector<double>> (1.0);
        emul.push_back(volume_constraint_dual);
    //}
    return emul;
}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::BoundConstraint<double>>>
OptimizationSetup<dim,nstate>::
getSlackBoundConstraint(const double lift_target, const double volume_target) const
{

    (void) lift_target;
    //const ROL::Ptr<ROL::SingletonVector<double>> lift_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (lift_target);
    // Constraint is already (lift - target_lift), therefore, the lower bound is 0.0.
    const ROL::Ptr<ROL::SingletonVector<double>> lift_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (lift_target+0.0);
    const ROL::Ptr<ROL::SingletonVector<double>> lift_upper_bound = ROL::makePtr<ROL::SingletonVector<double>> (lift_target+1.0);
    //const bool isLower = true;
    double scale = 1;
    double feasTol = 1e-4;
    auto lift_bounds = ROL::makePtr<ROL::Bounds<double>> (lift_lower_bound, lift_upper_bound, scale, feasTol);


    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> bcon;
    bcon.push_back(lift_bounds);
    //if (volume_target > 0) {
        const ROL::Ptr<ROL::SingletonVector<double>> volume_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (volume_target-1e-4);
        const ROL::Ptr<ROL::SingletonVector<double>> volume_upper_bound = ROL::makePtr<ROL::SingletonVector<double>> (volume_target+1e+4);
        auto volume_bounds = ROL::makePtr<ROL::Bounds<double>> (volume_lower_bound, volume_upper_bound, scale, feasTol);
        bcon.push_back(volume_bounds);
    //}
    return bcon;
}


template<int dim, int nstate>
int OptimizationSetup<dim,nstate>
::run_test () const
{
    int test_error = 0;
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization.log", std::ios::out);
    if (this->mpi_rank == 0) filebuffer.close();

    for (unsigned int poly_degree = POLY_START; poly_degree <= POLY_END; ++poly_degree) {
        for (unsigned int n_des_var = n_des_var_start; n_des_var <= n_des_var_end; n_des_var += n_des_var_step) {
        //for (unsigned int n_des_var = n_des_var_start; n_des_var <= n_des_var_end; n_des_var *= 2) {
            // assert(n_des_var%2 == 0);
            // assert(n_des_var>=2);
            // const unsigned int nx_ffd = n_des_var / 2 + 2;
            const unsigned int nx_ffd = n_des_var + 2;
            test_error += optimize(nx_ffd, poly_degree);
        }
    }
    return test_error;
}

std::string get_preconditioner_type_string (const PreconditionerType preconditioner_type)
{
    std::string preconditioner_string = "";
    switch(precond_type) {
        case PreconditionerType::P2: return "P2";
        case PreconditionerType::P2A: return "P2A";
        case PreconditionerType::P4: return "P4";
        case PreconditionerType::P4A: return "P4A";
        case PreconditionerType::identity: return "identity";
    }
}
std::string get_optimization_type_string(
    const OptimizationAlgorithm optimization_type,
    const unsigned int nx_ffd,
    const unsigned int poly_degree)
{
    switch(opt_type) {
        case OptimizationAlgorithm::full_space_birosghattas: return "full_space";
        case OptimizationAlgorithm::reduced_space_bfgs:      return "reduced_space_bfgs";
        case OptimizationAlgorithm::reduced_space_newton:    return "reduced_space_newton";
    }
}
template<int dim, int nstate>
int OptimizationSetup<dim,nstate>
::optimize (const unsigned int nx_ffd, const unsigned int poly_degree) const
{
    int test_error = 0;

    for (auto const opt_type : opt_list) {
    for (auto const precond_type : precond_list) {

    std::string optimization_type_string = get_optimization_type_string(opt_type);
    std::string preconditioner_type_string = get_preconditioner_type_string(precond_type);

    std::string optimization_output_filename = "optimization_"
    optimization_output_filename += optimization_type_string;
    if (OptimizationAlgorithm::full_space_birosghattas) {
        optimization_output_filename += preconditioner_type_string;
    }
    optimization_output_filename += "_" + "P" + std::to_string(poly_degree);
    optimization_output_filename += "_" + std::to_string(nx_ffd-2) + ".log";

    // Output stream
    Teuchos::RCP<std::ostream> outStream;

    ROL::nullstream bhs; // outputs nothing
    std::filebuf filebuffer;
    std::ostream ostr(&filebuffer);
    if (this->mpi_rank == 0) {
        // Write to specific optimization file.
        filebuffer.open (optimization_output_filename, std::ios::out);
        outStream = ROL::makePtrFromRef(ostr);
    } else if (this->mpi_rank == 1) 
        // Print to console
        outStream = ROL::makePtrFromRef(std::cout);
    } else if (this->mpi_rank == 2) {
        // Write to general optimization.log file.
        filebuffer.open ("optimization.log", std::ios::out|std::ios::app);
        outStream = ROL::makePtrFromRef(ostr);
    } else {
        // Output nothing.
        outStream = ROL::makePtrFromRef(bhs);
    }


    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using RolVectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;

    Parameters::AllParameters param = *(TestsBase::all_parameters);
    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(param.pde_type == param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());

    const dealii::Point<dim> ffd_origin(0.0,-0.061);
    const std::array<double,dim> ffd_rectangle_lengths = {{0.999,0.122}};
    const std::array<unsigned int,dim> ffd_ndim_control_pts = {{nx_ffd,3}};
    FreeFormDeformation<dim> ffd( ffd_origin, ffd_rectangle_lengths, ffd_ndim_control_pts);

    unsigned int n_design_variables = 0;
    // Vector of ijk indices and dimension.
    // Each entry in the vector points to a design variable's ijk ctl point and its acting dimension.
    std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {

        const std::array<unsigned int,dim> ijk = ffd.global_to_grid ( i_ctl );
        for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {

            if (   ijk[0] == 0 // Constrain first column of FFD points.
                || ijk[0] == ffd_ndim_control_pts[0] - 1  // Constrain last column of FFD points.
                || ijk[1] == 1 // Constrain middle row of FFD points.
                || d_ffd == 0 // Constrain x-direction of FFD points.
               ) {
                continue;
            }
            ++n_design_variables;
            ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
        }
    }

    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_design_variables);
    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);
    DealiiVector ffd_design_variables(row_part,ghost_row_part,MPI_COMM_WORLD);

    ffd.get_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    const auto initial_design_variables = ffd_design_variables;

    // Initial optimization point
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation> (
        this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    grid->clear();
    dealii::GridGenerator::hyper_cube(*grid);

    ffd_design_variables = initial_design_variables;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, grid);

    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012.msh");
    dg->set_high_order_grid(naca0012_mesh);
    dg->allocate_system ();

    Physics::Euler<dim,nstate,double> euler_physics_double = Physics::Euler<dim, nstate, double>(
        param.euler_param.ref_length,
        param.euler_param.gamma_gas,
        param.euler_param.mach_inf,
        param.euler_param.angle_of_attack,
        param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->initialize_steady_polynomial_ramping (poly_degree);
    ode_solver->steady_state();

    const bool has_ownership_vector = false;

    DealiiVector design_control_dealii = initial_design_variables;
    RolVectorAdaptor design_control_rol(Teuchos::rcp(&design_control_dealii, has_ownership_vector));
    ROL::Ptr<ROL::Vector<double>> design_control = ROL::makePtr<RolVectorAdaptor>(design_control_rol);

    DealiiVector design_simulation_dealii = dg->solution;
    RolVectorAdaptor design_simulation_rol(Teuchos::rcp(&design_simulation_dealii, has_ownership_vector));
    ROL::Ptr<ROL::Vector<double>> design_simulation = ROL::makePtr<RolVectorAdaptor>(design_simulation_rol);

    DealiiVector dual_equality_state_dealii = dg->dual;
    des_var_adj.add(0.1);
    RolVectorAdaptor dual_equality_state_rol(Teuchos::rcp(&dual_equality_state_dealii, has_ownership_vector));
    ROL::Ptr<ROL::Vector<double>> dual_equality_state  = ROL::makePtr<RolVectorAdaptor>(dual_equality_state_rol);

    LiftDragFunctional<dim,nstate,double> lift_functional( dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::lift );
    LiftDragFunctional<dim,nstate,double> drag_functional( dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::drag );
    GeometricVolume<dim,nstate,double> volume_functional( dg );

    std::cout << " Current lift = " << lift_functional.evaluate_functional()
              << ". Current drag = " << drag_functional.evaluate_functional()
              << std::endl;

    double lift_target = lift_functional.evaluate_functional() * 1.0;
    double volume_target = volume_functional.evaluate_functional() * 1.0;

    auto flow_constraints  = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);
    // int flow_constraints_check_error
    //     = check_flow_constraints<dim,nstate>( nx_ffd,
    //                                            flow_constraints,
    //                                            design_simulation,
    //                                            design_control,
    //                                            dual_equality_state);
    // (void) flow_constraints_check_error;

    auto drag_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( drag_functional, ffd, ffd_design_variables_indices_dim, &(flow_constraints->dXvdXp) );
    auto lift_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( lift_functional, ffd, ffd_design_variables_indices_dim, &(flow_constraints->dXvdXp) );
    auto volume_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( volume_functional, ffd, ffd_design_variables_indices_dim, &(flow_constraints->dXvdXp) );

    const unsigned int n_other_constraints = 1;
    const dealii::IndexSet constraint_row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_other_constraints);
    dealii::IndexSet constraint_ghost_row_part(n_other_constraints);
    constraint_ghost_row_part.add_range(0,n_other_constraints);

    double tol = 0.0;
    std::cout << "Drag value= " << drag_objective->value(*design_simulation, *design_control, tol) << std::endl;

    dg->output_results_vtk(9999);

    double timing_start, timing_end;
    timing_start = MPI_Wtime();
    // Verbosity setting
    Teuchos::ParameterList parlist;
    parlist.sublist("General").set("Print Verbosity", 1);

    //parlist.sublist("Status Test").set("Gradient Tolerance", 1e-9);
    parlist.sublist("Status Test").set("Gradient Tolerance", 1e-7);
    parlist.sublist("Status Test").set("Iteration Limit", max_design_cycle);

    parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
    parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit",LINESEARCH_MAX_ITER); // 0.5^30 ~  1e-10
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").get("Backtracking Rate", BACKTRACKING_RATE);
    parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


    parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
    //parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory SR1");
    //parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
    parlist.sublist("General").sublist("Secant").set("Maximum Storage", 100);
    parlist.sublist("General").sublist("Secant").set("Use as Hessian", USE_BFGS);

    parlist.sublist("Full Space").set("Preconditioner",preconditioner_string);

    ROL::Ptr< const ROL::AlgorithmState <double> > algo_state;
    n_vmult = 0;
    dRdW_form = 0;
    dRdW_mult = 0;
    dRdX_mult = 0;
    d2R_mult = 0;

    *outStream << "Starting optimization with " << n_design_variables << "..." << std::endl;
    switch (opt_type) {
        case OptimizationAlgorithm::reduced_space_bfgs:
            USE_BFGS = true;
            parlist.sublist("General").sublist("Secant").set("Use as Hessian", USE_BFGS);
            [[fallthrough]];
        case OptimizationAlgorithm::reduced_space_newton: {

            const bool is_reduced_space = true;
            ROL::Ptr<ROL::Vector<double>>                       design_variables               = getDesignVariables(design_simulation, design_control, is_reduced_space);
            ROL::Ptr<ROL::BoundConstraint<double>>              design_bounds                  = getDesignBoundConstraint(design_simulation, design_control, is_reduced_space);
            ROL::Ptr<ROL::Objective<double>>                    reduced_drag_objective         = getObjective(drag_objective, flow_constraints, design_simulation, design_control, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Constraint<double>>>      reduced_inequality_constraints = getInequalityConstraint(lift_objective, flow_constraints, design_simulation, design_control, lift_target, volume_objective, is_reduced_space, volume_target);
            std::vector<ROL::Ptr<ROL::Vector<double>>>          dual_inequality                = getInequalityMultiplier(volume_target);
            std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> inequality_bounds              = getSlackBoundConstraint(lift_target, volume_target);

            ROL::OptimizationProblem<double> optimization_problem = ROL::OptimizationProblem<double> (
                reduced_drag_objective, design_variables, design_bounds,
                reduced_inequality_constraints, dual_inequality, inequality_bounds);
            ROL::EProblem problem_type_opt = optimization_problem.getProblemType();
            ROL::EProblem problem_type = ROL::TYPE_EB;
            if (problem_type_opt != problem_type) std::abort();

            parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-10);
            parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-8);
            parlist.sublist("General").sublist("Krylov").set("Iteration Limit", 300);


            // This step transforms the inequality into equality + slack variables with box constraints.
            auto x      = optimization_problem.getSolutionVector();
            auto g      = x->dual().clone();
            auto l      = optimization_problem.getMultiplierVector();
            auto c      = l->dual().clone();
            auto obj    = optimization_problem.getObjective();
            auto con    = optimization_problem.getConstraint();
            auto bnd    = optimization_problem.getBoundConstraint();

            for (auto &constraint_dual : dual_inequality) {
                constraint_dual->zero();
            }

            auto pdas_step = ROL::makePtr<PHiLiP::PrimalDualActiveSetStep<double>>(parlist);
            auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
            const bool printHeader = true;

            const ROL::Ptr<ROL::Algorithm<double>> algorithm = ROL::makePtr<ROL::Algorithm<double>>( pdas_step, status_test, printHeader );
            algorithm->run(*x, *g, *l, *c, *obj, *con, *bnd, true, *outStream);
            algo_state = algorithm->getState();

            break;
        } case OptimizationAlgorithm::full_space_birosghattas: {

            const bool is_reduced_space = false;

            ROL::Ptr<ROL::Vector<double>>                       design_variables               = getDesignVariables(design_simulation, design_control, is_reduced_space);
            ROL::Ptr<ROL::BoundConstraint<double>>              design_bounds                  = getDesignBoundConstraint(design_simulation, design_control, is_reduced_space);
            ROL::Ptr<ROL::Objective<double>>                    drag_objective_simopt          = getObjective(drag_objective, flow_constraints, design_simulation, design_control, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Constraint<double>>>      inequality_constraints         = getInequalityConstraint(lift_objective, flow_constraints, design_simulation, design_control, lift_target, volume_objective, is_reduced_space, volume_target);
            std::vector<ROL::Ptr<ROL::Vector<double>>>          dual_inequality                = getInequalityMultiplier(volume_target);
            std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> inequality_bounds              = getSlackBoundConstraint(lift_target, volume_target);

            ROL::Ptr<ROL::Constraint<double>>                   equality_constraints           = flow_constraints;
            ROL::Ptr<ROL::Vector<double>>                       dual_equality                  = design_simulation->clone();
            dual_equality->zero();

            ROL::OptimizationProblem<double> optimization_problem = ROL::OptimizationProblem<double> (
                drag_objective_simopt, design_variables, design_bounds,
                equality_constraints, dual_equality,
                inequality_constraints, dual_inequality, inequality_bounds);
            ROL::EProblem problem_type_opt = optimization_problem.getProblemType();
            ROL::EProblem problem_type = ROL::TYPE_EB;
            if (problem_type_opt != problem_type) std::abort();

            parlist.sublist("Step").sublist("Primal Dual Active Set").set("Iteration Limit",PDAS_MAX_ITER);
            parlist.sublist("General").sublist("Secant").set("Use as Preconditioner", true);
            parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-12);
            parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-4);
            parlist.sublist("General").sublist("Krylov").set("Iteration Limit", 400);


            // This step transforms the inequality into equality + slack variables with box constraints.
            auto x      = optimization_problem.getSolutionVector();
            auto g      = x->dual().clone();
            auto l      = optimization_problem.getMultiplierVector();
            auto c      = l->dual().clone();
            auto obj    = optimization_problem.getObjective();
            auto con    = optimization_problem.getConstraint();
            auto bnd    = optimization_problem.getBoundConstraint();

            for (auto &constraint_dual : dual_inequality) {
                constraint_dual->zero();
            }

            auto pdas_step = ROL::makePtr<PHiLiP::PrimalDualActiveSetStep<double>>(parlist);
            auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
            const bool printHeader = true;

            const ROL::Ptr<ROL::Algorithm<double>> algorithm = ROL::makePtr<ROL::Algorithm<double>>( pdas_step, status_test, printHeader );
            algorithm->run(*x, *g, *l, *c, *obj, *con, *bnd, true, *outStream);
            algo_state = algorithm->getState();

            break;
        }
    }
    std::cout << " Current lift = " << lift_functional.evaluate_functional()
              << ". Current drag = " << drag_functional.evaluate_functional()
              << ". Drag with quadratic lift penalty = " << drag_objective->value(*design_simulation, *design_control, tol);
    static int resulting_optimization = 5000;
    std::cout << "Outputting final grid resulting_optimization: " << resulting_optimization << std::endl;
    dg->output_results_vtk(resulting_optimization++);


    timing_end = MPI_Wtime();
    *outStream << "The process took " << timing_end - timing_start << " seconds to run." << std::endl;

    *outStream << "Total n_vmult for algorithm " << n_vmult << std::endl;

    test_error += algo_state->statusFlag;

    filebuffer.close();

    if (opt_type != OptimizationAlgorithm::full_space_birosghattas) break;
    }
    }

    return test_error;
}


#if PHILIP_DIM==2
    template class OptimizationSetup <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace



