#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
//#include "ROL_StatusTest.hpp"

#include "euler_bump_optimization.h"

#include "physics/euler.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "functional/target_boundary_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"

namespace PHiLiP {
namespace Tests {


const int POLY_DEGREE = 1;
const double BUMP_HEIGHT = 0.0625;
const double TARGET_BUMP_HEIGHT = 0.5*BUMP_HEIGHT;
const double CHANNEL_LENGTH = 3.0;
const double CHANNEL_HEIGHT = 0.8;
const unsigned int NY_CELL = 5;
const unsigned int NX_CELL = 9*NY_CELL;


template <int dim, int nstate>
EulerBumpOptimization<dim,nstate>::EulerBumpOptimization(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
int EulerBumpOptimization<dim,nstate>
::run_test () const
{
    int test_error = 0;
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization.log", std::ios::out);
    if (this->mpi_rank == 0) filebuffer.close();

    const unsigned int n_des_var_start = 10;
    const unsigned int n_des_var_end   = 50;
    const unsigned int n_des_var_step  = 10;
    for (unsigned int n_des_var = n_des_var_start; n_des_var <= n_des_var_end; n_des_var += n_des_var_step) {
        assert(n_des_var%2 == 0);
        assert(n_des_var>=2);
        const unsigned int nx_ffd = n_des_var / 2 + 2;
        test_error += optimize_target_bump(nx_ffd);
    }
    return test_error;
}

template<int dim, int nstate>
int EulerBumpOptimization<dim,nstate>
::optimize_target_bump (const unsigned int nx_ffd) const
{
    int test_error = 0;

    // Output stream
    ROL::nullstream bhs; // outputs nothing
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization.log", std::ios::out|std::ios::app);
    std::ostream ostr(&filebuffer);

    Teuchos::RCP<std::ostream> outStream;
    if (this->mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (this->mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);


    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(param.pde_type == param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;


    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);

    std::vector<unsigned int> n_subdivisions(dim);

    //const int n_1d_cells = manu_grid_conv_param.initial_grid_size;
    //n_subdivisions[1] = n_1d_cells; // y-direction
    //n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction

    // n_subdivisions[1] = 5; //20;// y-direction
    // n_subdivisions[0] = 9*n_subdivisions[1]; // x-direction
    n_subdivisions[1] = NY_CELL;
    n_subdivisions[0] = NX_CELL;

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation> (
        this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));


    const dealii::Point<dim> ffd_origin(-1.4,-0.1);
    const std::array<double,dim> ffd_rectangle_lengths = {2.8,0.6};
    const std::array<unsigned int,dim> ffd_ndim_control_pts = {nx_ffd,2};
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

    // Create target nonlinear bump solution
    DealiiVector target_bump_solution;
    {
        grid->clear();
        Grids::gaussian_bump(*grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, 0.5*BUMP_HEIGHT);
        // Create DG object
        std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, grid);

        // Initialize coarse grid solution with free-stream
        dg->allocate_system ();
        dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
        // Create ODE solver and ramp up the solution from p0
        std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver->initialize_steady_polynomial_ramping (POLY_DEGREE);
        // Solve the steady state problem
        ode_solver->steady_state();
        // Output target_bump_solution
        dg->output_results_vtk(9998);

        target_bump_solution = dg->solution;
    }

    // Generate target design vector.
    DealiiVector target_ffd_solution;
    {
        // Initial optimization point
        grid->clear();
        Grids::gaussian_bump(*grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, BUMP_HEIGHT);

        ffd_design_variables = initial_design_variables;
        ffd_design_variables.update_ghost_values();
        ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

        // Initialize flow solution with free-stream
        std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, grid);
        dg->allocate_system ();
        dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
        // Create ODE solver and ramp up the solution from p0
        std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver->initialize_steady_polynomial_ramping (POLY_DEGREE);
        // Solve the steady state problem
        ode_solver->steady_state();
        // Output target_ffd_solution
        dg->output_results_vtk(9999);
        dg->set_dual(dg->solution);

        // Copy vector to be used by optimizer.
        DealiiVector des_var_sim = dg->solution;
        DealiiVector des_var_ctl = initial_design_variables;
        DealiiVector des_var_adj = dg->dual;

        const bool has_ownership = false;
        VectorAdaptor des_var_sim_rol(Teuchos::rcp(&des_var_sim, has_ownership));
        VectorAdaptor des_var_ctl_rol(Teuchos::rcp(&des_var_ctl, has_ownership));
        VectorAdaptor des_var_adj_rol(Teuchos::rcp(&des_var_adj, has_ownership));

        ROL::Ptr<ROL::Vector<double>> des_var_sim_rol_p = ROL::makePtr<VectorAdaptor>(des_var_sim_rol);
        ROL::Ptr<ROL::Vector<double>> des_var_ctl_rol_p = ROL::makePtr<VectorAdaptor>(des_var_ctl_rol);
        ROL::Ptr<ROL::Vector<double>> des_var_adj_rol_p = ROL::makePtr<VectorAdaptor>(des_var_adj_rol);

        // Reduced space problem
        const bool functional_uses_solution_values = true, functional_uses_solution_gradient = false;
        TargetBoundaryFunctional<dim,nstate,double> target_bump_functional(dg, target_bump_solution, functional_uses_solution_values, functional_uses_solution_gradient);
        auto obj  = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( target_bump_functional, ffd, ffd_design_variables_indices_dim );
        auto con  = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);
        const bool storage = false;
        const bool useFDHessian = false;
        auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( obj, con, des_var_sim_rol_p, des_var_ctl_rol_p, des_var_adj_rol_p, storage, useFDHessian);

        ROL::OptimizationProblem<double> opt = ROL::OptimizationProblem<double> ( robj, des_var_ctl_rol_p );
        ROL::EProblem problemType = opt.getProblemType();
        std::cout << ROL::EProblemToString(problemType) << std::endl;

        Teuchos::ParameterList parlist;
        parlist.sublist("Status Test").set("Gradient Tolerance", 1e-8);
        parlist.sublist("Status Test").set("Iteration Limit", 5000);

        parlist.sublist("Step").set("Type","Line Search");
        parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Quasi-Newton Method");
        parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
        parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
        parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Cubic Interpolation");
        parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Goldstein Conditions");

        parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
        parlist.sublist("General").sublist("Secant").set("Maximum Storage",5000);

        *outStream << "Starting optimization with " << n_design_variables << "..." << std::endl;
        *outStream << "Optimizing FFD points to obtain target Gaussian bump..." << std::endl;
        ROL::OptimizationSolver<double> solver( opt, parlist );
        solver.solve( *outStream );

        ROL::Ptr< const ROL::AlgorithmState <double> > opt_state = solver.getAlgorithmState();

        test_error += opt_state->statusFlag;

        target_ffd_solution = dg->solution;
    }
    *outStream << " Done target optimization..." << std::endl;
    *outStream << " Resetting problem to initial conditions and target previous FFD points "<< std::endl;
    *outStream << " and aim for machine precision functional value...." << std::endl;

    // Initial optimization point
    grid->clear();
    Grids::gaussian_bump(*grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, BUMP_HEIGHT);

    ffd_design_variables = initial_design_variables;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Initialize flow solution with free-stream
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, grid);
    dg->allocate_system ();
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->initialize_steady_polynomial_ramping (POLY_DEGREE);
    // Solve the steady state problem
    ode_solver->steady_state();

    // Reset to initial_grid
    DealiiVector des_var_sim = dg->solution;
    DealiiVector des_var_ctl = initial_design_variables;
    DealiiVector des_var_adj = dg->dual;

    const bool has_ownership = false;
    VectorAdaptor des_var_sim_rol(Teuchos::rcp(&des_var_sim, has_ownership));
    VectorAdaptor des_var_ctl_rol(Teuchos::rcp(&des_var_ctl, has_ownership));
    VectorAdaptor des_var_adj_rol(Teuchos::rcp(&des_var_adj, has_ownership));

    ROL::Ptr<ROL::Vector<double>> des_var_sim_rol_p = ROL::makePtr<VectorAdaptor>(des_var_sim_rol);
    ROL::Ptr<ROL::Vector<double>> des_var_ctl_rol_p = ROL::makePtr<VectorAdaptor>(des_var_ctl_rol);
    ROL::Ptr<ROL::Vector<double>> des_var_adj_rol_p = ROL::makePtr<VectorAdaptor>(des_var_adj_rol);

    ROL::OptimizationProblem<double> opt;
    Teuchos::ParameterList parlist;

    // Reduced space problem
    const bool functional_uses_solution_values = true, functional_uses_solution_gradient = false;
    TargetBoundaryFunctional<dim,nstate,double> target_ffd_functional(dg, target_ffd_solution, functional_uses_solution_values, functional_uses_solution_gradient);
    auto obj  = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( target_ffd_functional, ffd, ffd_design_variables_indices_dim );
    auto con  = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);

    const bool full_space = false;
    if (full_space) {
        // Full space problem
        auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(des_var_sim_rol_p, des_var_ctl_rol_p);
        auto dual_sim_p = des_var_sim_rol_p->clone();
        //auto dual_sim_p = ROL::makePtrFromRef(dual_sim);
        opt = ROL::OptimizationProblem<double> ( obj, des_var_p, con, dual_sim_p );
        ROL::EProblem problemType = opt.getProblemType();
        std::cout << ROL::EProblemToString(problemType) << std::endl;

        // Set parameters.
        //parlist.sublist("Secant").set("Use as Preconditioner", false);
        parlist.sublist("Status Test").set("Gradient Tolerance", 1e-14);
        parlist.sublist("Status Test").set("Iteration Limit", 5000);

        //parlist.sublist("Step").sublist("Interior Point").set("Initial Step Size",0.1);

        parlist.sublist("Step").set("Type","Composite Step");
        ROL::ParameterList& steplist = parlist.sublist("Step").sublist("Composite Step");
        steplist.sublist("Optimality System Solver").set("Nominal Relative Tolerance", 1e-8);
        steplist.sublist("Optimality System Solver").set("Fix Tolerance", true);
        steplist.sublist("Tangential Subproblem Solver").set("Iteration Limit", 20);
        steplist.sublist("Tangential Subproblem Solver").set("Relative Tolerance", 1e-2);
        steplist.set("Initial Radius", 1e2);
        steplist.set("Use Constraint Hessian", true); // default is true

        steplist.set("Output Level", 1);


        parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
        //parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
        parlist.sublist("General").sublist("Secant").set("Maximum Storage",5000);

    } else { 
        // Reduced space problem
        const bool storage = false;
        const bool useFDHessian = false;
        auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( obj, con, des_var_sim_rol_p, des_var_ctl_rol_p, des_var_adj_rol_p, storage, useFDHessian);
        opt = ROL::OptimizationProblem<double> ( robj, des_var_ctl_rol_p );
        ROL::EProblem problemType = opt.getProblemType();
        std::cout << ROL::EProblemToString(problemType) << std::endl;

        parlist.sublist("Status Test").set("Gradient Tolerance", 1e-10);
        parlist.sublist("Status Test").set("Iteration Limit", 5000);
        parlist.sublist("Step").set("Type","Line Search");
        parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
        parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);

        //parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Iteration Scaling");
        //parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Backtracking");
        parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Cubic Interpolation");

        //parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Null Curvature Condition");
        //parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Strong Wolfe Conditions");
        parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Goldstein Conditions");

        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Newton's Method");

        // {
        //     parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Newton-Krylov");
        //     parlist.sublist("General").sublist("Secant").set("Use as Preconditioner", true);
        //     const double em4 = 1e-4, em2 = 1e-2;
        //     const int cg_iteration_limit = 100;
        //     parlist.sublist("General").sublist("Krylov").set("Type","Conjugate Gradients");
        //     parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", em4);
        //     parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", em2);
        //     parlist.sublist("General").sublist("Krylov").set("Iteration Limit", cg_iteration_limit);
        //     parlist.sublist("General").set("Inexact Hessian-Times-A-Vector",false);
        // }

        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Steepest Descent");

        parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Quasi-Newton Method");
        parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
        parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
        parlist.sublist("General").sublist("Secant").set("Maximum Storage",5000);
    }

    *outStream << "Starting optimization with " << n_design_variables << "..." << std::endl;
    ROL::OptimizationSolver<double> solver( opt, parlist );
    solver.solve( *outStream );

    ROL::Ptr< const ROL::AlgorithmState <double> > opt_state = solver.getAlgorithmState();

    test_error += opt_state->statusFlag;

    filebuffer.close();

    return test_error;
}


#if PHILIP_DIM==2
    template class EulerBumpOptimization <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

