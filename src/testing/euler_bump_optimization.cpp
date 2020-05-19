#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"

#include "euler_bump_optimization.h"

#include "physics/euler.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "functional/target_boundary_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"

#include "parameters/all_parameters.h"


#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"

namespace PHiLiP {
namespace Tests {


const int POLY_DEGREE = 4;
const double BUMP_HEIGHT = 0.0625;
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
    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;


    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }


    std::vector<unsigned int> n_subdivisions(dim);

    //const int n_1d_cells = manu_grid_conv_param.initial_grid_size;
    //n_subdivisions[1] = n_1d_cells; // y-direction
    //n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction

    // n_subdivisions[1] = 5; //20;// y-direction
    // n_subdivisions[0] = 9*n_subdivisions[1]; // x-direction
    n_subdivisions[1] = NY_CELL;
    n_subdivisions[0] = NX_CELL;

    dealii::parallel::distributed::Triangulation<dim> grid(this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));


    const dealii::Point<dim> ffd_origin(-1.4,-0.1);
    const std::array<double,dim> ffd_rectangle_lengths = {2.8,0.6};
    const std::array<unsigned int,dim> ffd_ndim_control_pts = {10,2};
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

    //const std::vector<dealii::IndexSet> row_parts = dealii::Utilities::MPI::create_evenly_distributed_partitioning(this->mpi_communicator, n_design_variables);
    //const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator);
    //const dealii::IndexSet &row_part = row_parts[this_mpi_process];
    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_design_variables);

    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);

    DealiiVector ffd_design_variables(row_part,ghost_row_part,MPI_COMM_WORLD);

    ffd_design_variables.print(std::cout);


    ffd.get_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    const auto initial_design_variables = ffd_design_variables;
    ffd_design_variables[0] = -1.558e-01;
    ffd_design_variables[1] = -2.189e-01;
    ffd_design_variables[2] = -2.338e-01;
    ffd_design_variables[3] = -1.691e-01;
    ffd_design_variables[4] = -1.806e-01;
    ffd_design_variables[5] = -2.294e-01;
    ffd_design_variables[6] = -2.243e-01;
    ffd_design_variables[7] = -1.552e-01;
    ffd_design_variables[8] = 7.737e-01;
    ffd_design_variables[9] = 1.110e+00;
    ffd_design_variables[10] = 1.141e+00;
    ffd_design_variables[11] = 8.757e-01;
    ffd_design_variables[12] = 8.825e-01;
    ffd_design_variables[13] = 1.159e+00;
    ffd_design_variables[14] = 1.119e+00;
    ffd_design_variables[15] = 7.769e-01;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Create Target solution
    DealiiVector target_solution;
    {
        grid.clear();
        Grids::gaussian_bump(grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, BUMP_HEIGHT);
        // Create DG object
        std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, &grid);

        ffd.deform_mesh(dg->high_order_grid);

        // Initialize coarse grid solution with free-stream
        dg->allocate_system ();
        dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
        // Create ODE solver and ramp up the solution from p0
        std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver->initialize_steady_polynomial_ramping (POLY_DEGREE);
        // Solve the steady state problem
        ode_solver->steady_state();
        // Output target solution
        dg->output_results_vtk(9998);

        target_solution = dg->solution;
    }

    // Initial optimization point
    grid.clear();
    Grids::gaussian_bump(grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, BUMP_HEIGHT);

    ffd_design_variables = initial_design_variables;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, &grid);

    // Initialize coarse grid solution with free-stream
    dg->allocate_system ();
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->initialize_steady_polynomial_ramping (POLY_DEGREE);
    // Solve the steady state problem
    ode_solver->steady_state();
    // Output initial solution
    dg->output_results_vtk(9999);

    const bool functional_uses_solution_values = true, functional_uses_solution_gradient = false;
    TargetBoundaryFunctional<dim,nstate,double> functional(dg, target_solution, functional_uses_solution_values, functional_uses_solution_gradient);

    const bool has_ownership = false;
    DealiiVector des_var_sim = dg->solution;
    DealiiVector des_var_ctl = ffd_design_variables;
    DealiiVector des_var_adj = dg->dual;
    Teuchos::RCP<DealiiVector> des_var_sim_rcp = Teuchos::rcp(&des_var_sim, has_ownership);
    Teuchos::RCP<DealiiVector> des_var_ctl_rcp = Teuchos::rcp(&des_var_ctl, has_ownership);
    Teuchos::RCP<DealiiVector> des_var_adj_rcp = Teuchos::rcp(&des_var_adj, has_ownership);
    dg->set_dual(dg->solution);

    using VectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;

    VectorAdaptor des_var_sim_rol(des_var_sim_rcp);
    VectorAdaptor des_var_ctl_rol(des_var_ctl_rcp);
    VectorAdaptor des_var_adj_rol(des_var_adj_rcp);

    ROL::Ptr<ROL::Vector<double>> des_var_sim_rol_p = ROL::makePtr<VectorAdaptor>(des_var_sim_rol);
    ROL::Ptr<ROL::Vector<double>> des_var_ctl_rol_p = ROL::makePtr<VectorAdaptor>(des_var_ctl_rol);
    ROL::Ptr<ROL::Vector<double>> des_var_adj_rol_p = ROL::makePtr<VectorAdaptor>(des_var_adj_rol);


    // Output stream
    ROL::nullstream bhs; // outputs nothing
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization.log",std::ios::out);
    std::ostream ostr(&filebuffer);

    Teuchos::RCP<std::ostream> outStream;
    if (this->mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (this->mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    auto obj  = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( functional, ffd, ffd_design_variables_indices_dim );
    auto con  = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);
    const bool storage = false;
    const bool useFDHessian = false;
    auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( obj, con, des_var_sim_rol_p, des_var_ctl_rol_p, des_var_adj_rol_p, storage, useFDHessian);
    //const bool full_space = true;
    const bool full_space = false;
    ROL::OptimizationProblem<double> opt;
    // Set parameters.
    Teuchos::ParameterList parlist;

    auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(des_var_sim_rol_p, des_var_ctl_rol_p);

    if (full_space) {
        // Full space problem
        auto dual_sim_p = des_var_sim_rol_p->clone();
        //auto dual_sim_p = ROL::makePtrFromRef(dual_sim);
        opt = ROL::OptimizationProblem<double> ( obj, des_var_p, con, dual_sim_p );
        ROL::EProblem problemType = opt.getProblemType();
        std::cout << ROL::EProblemToString(problemType) << std::endl;

        // Set parameters.
        //parlist.sublist("Secant").set("Use as Preconditioner", false);
        parlist.sublist("Status Test").set("Gradient Tolerance", 1e-14);
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

        parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Quasi-Newton Method");
        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Steepest Descent");

        //parlist.sublist("Step").sublist("Interior Point").set("Initial Step Size",0.1);

        parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
        //parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
        parlist.sublist("General").sublist("Secant").set("Maximum Storage",5000);

    } else { 
        // Reduced space problem
        opt = ROL::OptimizationProblem<double> ( robj, des_var_ctl_rol_p );
        ROL::EProblem problemType = opt.getProblemType();
        std::cout << ROL::EProblemToString(problemType) << std::endl;

        {
            const auto u = des_var_sim_rol_p->clone();
            const auto z = des_var_ctl_rol_p->clone();
            const auto v = u->clone();
            const auto jv = v->clone();

            v->zero();
            v->setScalar(1.0);

            std::vector<double> steps;
            for (int i = -2; i > -9; i--) {
                steps.push_back(std::pow(10,i));
            }
            const int order = 2;
            *outStream << "con->checkApplyJacobian_1..." << std::endl;
            con->checkApplyJacobian_1(*u, *z, *v, *jv, steps, true, *outStream, order);

            *outStream << "con->checkInverseJacobian_1..." << std::endl;
            con->checkInverseJacobian_1(*jv, *v, *u, *z, true, *outStream);

            *outStream << "con->checkInverseAdjointJacobian_1..." << std::endl;
            con->checkInverseAdjointJacobian_1(*jv, *v, *u, *z, true, *outStream);

            const auto direction = des_var_p->clone();
            *outStream << "obj->checkGradient..." << std::endl;
            obj->checkGradient( *des_var_p, *direction, steps, true, *outStream, order);

            *outStream << "robj->checkGradient..." << std::endl;
            const auto direction_ctl = des_var_ctl_rol_p->clone();
            robj->checkGradient( *des_var_ctl_rol_p, *direction_ctl, steps, true, *outStream, order);

        }
        {
            const auto u = des_var_sim_rol_p->clone();
            const auto z = des_var_ctl_rol_p->clone();
            const auto v = z->clone();
            const auto jv = u->clone();

            std::vector<double> steps;
            for (int i = -2; i > -9; i--) {
                steps.push_back(std::pow(10,i));
            }
            const int order = 2;
            *outStream << "con->checkApplyJacobian_2..." << std::endl;
            con->checkApplyJacobian_2(*u, *z, *v, *jv, steps, true, *outStream, order);

        }
        {
            auto dual = des_var_sim_rol_p->clone();
            dual->set(*des_var_sim_rol_p);
            const auto v = des_var_p->clone();
            v->set(*des_var_p);
            const auto hv = des_var_p->clone();

            std::vector<double> steps;
            for (int i = -1; i > -13; i--) {
                steps.push_back(std::pow(10,i));
            }
            const int order = 2; (void) order;

            *outStream << "con->checkApplyAdjointHessian..." << std::endl;
            (void) con->checkApplyAdjointHessian(*des_var_p, *dual, *v, *hv, steps, true, *outStream, order);

            const auto direction_1 = des_var_p->clone();
            auto direction_2 = des_var_p->clone();
            direction_2->scale(0.5);
            *outStream << "obj->checkHessVec..." << std::endl;
            obj->checkHessVec( *des_var_p, *direction_1, steps, true, *outStream, order);
            *outStream << "obj->checkHessSym..." << std::endl;
            obj->checkHessSym( *des_var_p, *direction_1, *direction_2, true, *outStream);

            const auto direction_ctl_1 = des_var_ctl_rol_p->clone();
            auto direction_ctl_2 = des_var_ctl_rol_p->clone();
            direction_ctl_2->scale(0.5);
            *outStream << "robj->checkHessVec..." << std::endl;
            robj->checkHessVec( *des_var_ctl_rol_p, *direction_ctl_1, steps, true, *outStream, order);

            //  *outStream << "Outputting Hessian..." << std::endl;
            //  dealii::FullMatrix<double> hessian(n_design_variables, n_design_variables);
            //  for (unsigned int i=0; i<n_design_variables; ++i) {
            //      pcout << "Column " << i << " out of " << n_design_variables << std::endl;
            //      auto direction_unit = des_var_ctl_rol_p->basis(i);
            //      auto hv = des_var_ctl_rol_p->clone();
            //      double tol = 1e-6;
            //      robj->hessVec( *hv, *direction_unit, *des_var_ctl_rol_p, tol );

            //      auto result = ROL_vector_to_dealii_vector_reference(*hv);
            //      result.update_ghost_values();

            //      for (unsigned int j=0; j<result.size(); ++j) {
            //          hessian[j][i] = result[j];
            //      }
            //  }
            //  if (mpi_rank == 0) hessian.print_formatted(*outStream, 3, true, 10, "0", 1., 0.);

            *outStream << "robj->checkHessSym..." << std::endl;
            robj->checkHessSym( *des_var_ctl_rol_p, *direction_ctl_1, *direction_ctl_2, true, *outStream);

        }


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

        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Newton-Krylov");
        //parlist.sublist("General").sublist("Secant").set("Use as Preconditioner", false);


        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Steepest Descent");

        parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Quasi-Newton Method");
        parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
        parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
        parlist.sublist("General").sublist("Secant").set("Maximum Storage",5000);

    }

    *outStream << "Starting optimization..." << std::endl;
    ROL::OptimizationSolver<double> solver( opt, parlist );
    solver.solve( *outStream );

    ROL::Ptr< const ROL::AlgorithmState <double> > opt_state = solver.getAlgorithmState();

    ROL::EExitStatus opt_exit_state = opt_state->statusFlag;

    filebuffer.close();

    return opt_exit_state;
}


#if PHILIP_DIM==2
    template class EulerBumpOptimization <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

