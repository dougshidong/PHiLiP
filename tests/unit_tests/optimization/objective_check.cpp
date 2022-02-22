
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
//#include "ROL_StatusTest.hpp"

#include "physics/initial_conditions/initial_condition.h"
#include "physics/euler.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"

#include "functional/target_boundary_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"


const double TOL = 1e-7;

const int dim = 2;
const int nstate = 4;
const int POLY_DEGREE = 2;
const int MESH_DEGREE = POLY_DEGREE+1;
const double BUMP_HEIGHT = 0.0625;
const double CHANNEL_LENGTH = 3.0;
const double CHANNEL_HEIGHT = 0.8;
const unsigned int NY_CELL = 3;
const unsigned int NX_CELL = 5*NY_CELL;

double check_max_rel_error(std::vector<std::vector<double>> rol_check_results) {
    double max_rel_err = 999999;
    for (unsigned int i = 0; i < rol_check_results.size(); ++i) {
        const double abs_val_ad = std::abs(rol_check_results[i][1]);
        const double abs_val_fd = std::abs(rol_check_results[i][2]);
        const double abs_err    = std::abs(rol_check_results[i][3]);
        const double rel_err    = abs_err / std::max(abs_val_ad,abs_val_fd);
        max_rel_err = std::min(max_rel_err, rel_err);
    }
    return max_rel_err;
}

int test(const unsigned int nx_ffd)
{
    int test_error = 0;
    using namespace PHiLiP;
    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;

    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    parameter_handler.set("pde_type", "euler");
    parameter_handler.set("conv_num_flux", "roe");
    parameter_handler.set("dimension", (long int)dim);

    parameter_handler.enter_subsection("euler");
    parameter_handler.set("mach_infinity", 0.3);
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("ODE solver");

    parameter_handler.set("nonlinear_max_iterations", (long int) 500);
    parameter_handler.set("nonlinear_steady_residual_tolerance", 1e-14);
    //parameter_handler.set("output_solution_every_x_steps", (long int) 1);

    parameter_handler.set("ode_solver_type", "implicit");
    parameter_handler.set("initial_time_step", 10.);
    parameter_handler.set("time_step_factor_residual", 25.0);
    parameter_handler.set("time_step_factor_residual_exp", 4.0);

    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("linear solver");
    parameter_handler.enter_subsection("gmres options");
    parameter_handler.set("linear_residual_tolerance", 1e-8);
    parameter_handler.leave_subsection();
    parameter_handler.leave_subsection();


    Parameters::AllParameters param;
    param.parse_parameters (parameter_handler);

    param.euler_param.parse_parameters (parameter_handler);

    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);

    std::vector<unsigned int> n_subdivisions(dim);

    n_subdivisions[1] = NY_CELL;
    n_subdivisions[0] = NX_CELL;

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));



    // Create Target solution
    DealiiVector target_solution;
    {
        grid->clear();
        Grids::gaussian_bump(*grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, 0.5*BUMP_HEIGHT);
        // Create DG object
        std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, POLY_DEGREE, MESH_DEGREE, grid);

        // Initialize coarse grid solution with free-stream
        dg->allocate_system ();
        dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
        // Create ODE solver and ramp up the solution from p0
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver->initialize_steady_polynomial_ramping (POLY_DEGREE);
        // Solve the steady state problem
        ode_solver->steady_state();
        // Output target solution
        dg->output_results_vtk(9998);

        target_solution = dg->solution;
    }

    // Initial optimization point
    grid->clear();
    Grids::gaussian_bump(*grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, BUMP_HEIGHT);

    const dealii::Point<dim> ffd_origin(-1.4,-0.1);
    const std::array<double,dim> ffd_rectangle_lengths = {{2.8,0.6}};
    const std::array<unsigned int,dim> ffd_ndim_control_pts = {{nx_ffd,2}};
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
                || ijk[1] == 0 // Constrain first row of FFD points.
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
    ffd_design_variables.update_ghost_values();

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, grid);

    // Initialize coarse grid solution with free-stream
    dg->allocate_system ();
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
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
    if (mpi_rank == 0) filebuffer.open ("objective_check_"+std::to_string(nx_ffd)+".log",std::ios::out);
    std::ostream ostr(&filebuffer);

    Teuchos::RCP<std::ostream> outStream;
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    auto obj  = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( functional, ffd, ffd_design_variables_indices_dim );
    auto con  = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);
    const bool storage = false;
    const bool useFDHessian = false;
    auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( obj, con, des_var_sim_rol_p, des_var_ctl_rol_p, des_var_adj_rol_p, storage, useFDHessian);
    //const bool full_space = true;
    ROL::OptimizationProblem<double> opt;
    // Set parameters.
    Teuchos::ParameterList parlist;

    auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(des_var_sim_rol_p, des_var_ctl_rol_p);

    std::vector<double> steps;
    for (int i = -2; i > -9; i--) {
        steps.push_back(std::pow(10,i));
    }
    const int order = 2;
    {
        const auto direction = des_var_p->clone();
        *outStream << "obj->checkGradient..." << std::endl;
        std::vector<std::vector<double>> results
            = obj->checkGradient( *des_var_p, *direction, steps, true, *outStream, order);

        const double max_rel_err = check_max_rel_error(results);
        if (max_rel_err > TOL) test_error++;
    }
    {
        const auto direction_1 = des_var_p->clone();
        auto direction_2 = des_var_p->clone();
        direction_2->scale(0.5);
        *outStream << "obj->checkHessVec..." << std::endl;
        std::vector<std::vector<double>> results
            = obj->checkHessVec( *des_var_p, *direction_1, steps, true, *outStream, order);

        const double max_rel_err = check_max_rel_error(results);
        if (max_rel_err > TOL) test_error++;

        *outStream << "obj->checkHessSym..." << std::endl;
        std::vector<double> results_HessSym = obj->checkHessSym( *des_var_p, *direction_1, *direction_2, true, *outStream);
        double wHv       = std::abs(results_HessSym[0]);
        double vHw       = std::abs(results_HessSym[1]);
        double abs_error = std::abs(wHv - vHw);
        double rel_error = abs_error / std::max(wHv, vHw);
        if (rel_error > TOL) test_error++;
    }

    {
        const auto direction_ctl = des_var_ctl_rol_p->clone();
        *outStream << "robj->checkGradient..." << std::endl;
        dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
        std::vector<std::vector<double>> results
            = robj->checkGradient( *des_var_ctl_rol_p, *direction_ctl, steps, true, *outStream, order);

        const double max_rel_err = check_max_rel_error(results);
        if (max_rel_err > TOL) test_error++;

    }
    // Takes a really long time to evaluate the reduced Hessian.
    // {

    //     const auto direction_ctl_1 = des_var_ctl_rol_p->clone();
    //     auto direction_ctl_2 = des_var_ctl_rol_p->clone();
    //     direction_ctl_2->scale(0.5);
    //     *outStream << "robj->checkHessVec..." << std::endl;
    //     robj->checkHessVec( *des_var_ctl_rol_p, *direction_ctl_1, steps, true, *outStream, order);

    //     *outStream << "robj->checkHessSym..." << std::endl;
    //     robj->checkHessSym( *des_var_ctl_rol_p, *direction_ctl_1, *direction_ctl_2, true, *outStream);
    // }
    {
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
    }


    filebuffer.close();

    return test_error;
}

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int test_error = 0;
    try {
         test_error += test(5);
         test_error += test(10);
    }
    catch (std::exception &exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }
    catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }

    return test_error;
}

