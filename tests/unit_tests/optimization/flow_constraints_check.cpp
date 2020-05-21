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

#include "physics/euler.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "functional/target_boundary_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"

const int dim = 2;
const int nstate = 4;
const int POLY_DEGREE = 1;
const double BUMP_HEIGHT = 0.0625;
const double CHANNEL_LENGTH = 3.0;
const double CHANNEL_HEIGHT = 0.8;
const unsigned int NY_CELL = 5;
const unsigned int NX_CELL = 9*NY_CELL;

int test(const unsigned int nx_ffd)
{
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
    parameter_handler.set("nonlinear_steady_residual_tolerance", 1e-12);
    parameter_handler.set("initial_time_step", 0.05);
    parameter_handler.set("time_step_factor_residual", 25.0);
    parameter_handler.set("time_step_factor_residual_exp", 4.0);
    parameter_handler.leave_subsection();

    Parameters::AllParameters param;
    param.parse_parameters (parameter_handler);

    param.euler_param.parse_parameters (parameter_handler);
    param.euler_param.mach_inf = 0.3;
    Physics::Euler<dim,nstate,double> euler_physics_double = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);

    std::vector<unsigned int> n_subdivisions(dim);
    n_subdivisions[1] = NY_CELL;
    n_subdivisions[0] = NX_CELL;

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

    // Initial optimization point
    dealii::parallel::distributed::Triangulation<dim> grid(MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    Grids::gaussian_bump(grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, BUMP_HEIGHT);

    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, &grid);
    dg->allocate_system ();

    // Initialize coarse grid solution with free-stream
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->initialize_steady_polynomial_ramping (POLY_DEGREE);
    // Solve the steady state problem
    ode_solver->steady_state();
    // Output initial solution
    dg->output_results_vtk(9999);

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

    const ROL::Ptr<ROL::Vector<double>> des_var_sim_rol_p = ROL::makePtr<VectorAdaptor>(des_var_sim_rol);
    const ROL::Ptr<ROL::Vector<double>> des_var_ctl_rol_p = ROL::makePtr<VectorAdaptor>(des_var_ctl_rol);
    const ROL::Ptr<ROL::Vector<double>> des_var_adj_rol_p = ROL::makePtr<VectorAdaptor>(des_var_adj_rol);

    // Output stream
    ROL::nullstream bhs; // outputs nothing
    std::filebuf filebuffer;
    if (mpi_rank == 0) filebuffer.open ("flow_constraints_check.log",std::ios::out);
    std::ostream ostr(&filebuffer);

    Teuchos::RCP<std::ostream> outStream;
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    auto con  = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);

    const ROL::Ptr<ROL::Vector_SimOpt<double>> des_var_rol_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(des_var_sim_rol_p, des_var_ctl_rol_p);

    // Temporary vectors
    const auto temp_sim = des_var_sim_rol_p->clone();
    const auto temp_ctl = des_var_ctl_rol_p->clone();
    const auto v1 = temp_sim->clone();
    const auto v2 = temp_ctl->clone();

    const auto jv1 = temp_sim->clone();
    const auto jv2 = temp_sim->clone();

    v1->zero();
    v1->setScalar(1.0);
    v2->zero();
    v2->setScalar(1.0);

    std::vector<double> steps;
    for (int i = -2; i > -12; i--) {
        steps.push_back(std::pow(10,i));
    }
    const int order = 2;
    *outStream << "con->checkApplyJacobian_1..." << std::endl;
    con->checkApplyJacobian_1(*temp_sim, *temp_ctl, *v1, *jv1, steps, true, *outStream, order);

    *outStream << "con->checkApplyJacobian_2..." << std::endl;
    con->checkApplyJacobian_2(*temp_sim, *temp_ctl, *v2, *jv2, steps, true, *outStream, order);

    *outStream << "con->checkInverseJacobian_1..." << std::endl;
    con->checkInverseJacobian_1(*jv1, *v1, *temp_sim, *temp_ctl, true, *outStream);

    *outStream << "con->checkInverseAdjointJacobian_1..." << std::endl;
    con->checkInverseAdjointJacobian_1(*jv1, *v1, *temp_sim, *temp_ctl, true, *outStream);

    //const auto dual = des_var_sim_rol_p->clone();
    //const auto temp_sim_ctl = des_var_rol_p->clone();
    //const auto v3 = des_var_rol_p->clone();
    //const auto hv3 = des_var_rol_p->clone();

    //*outStream << "con->checkApplyAdjointHessian..." << std::endl;
    //(void) con->checkApplyAdjointHessian(*des_var_rol_p, *dual, *v3, *hv3, steps, true, *outStream, order);

    {
        *outStream << "con->checkAdjointConsistencyJacobian..." << std::endl;
        *outStream << "Checks (w J v) versus (v Jt w)  ..." << std::endl;
        const auto w = des_var_adj_rol_p->clone();
        const auto v = des_var_rol_p->clone();
        const auto x = des_var_rol_p->clone();
        const auto temp_Jv = des_var_adj_rol_p->clone();
        const auto temp_Jtw = des_var_rol_p->clone();
        double tol = 1e-8;
        con->applyJacobian(*temp_Jv,*v,*x,tol);
        con->applyAdjointJacobian(*temp_Jtw,*w,*x,tol);

        const bool printToStream = true;
        con->checkAdjointConsistencyJacobian (*w, *v, *x, *temp_Jv, *temp_Jtw, printToStream, *outStream);
    }

    // const auto direction_1 = des_var_rol_p->clone();
    // auto direction_2 = des_var_rol_p->clone();
    // direction_2->scale(0.5);

    // *outStream << "obj->checkHessVec..." << std::endl;
    // obj->checkHessVec( *des_var_rol_p, *direction_1, steps, true, *outStream, order);

    // *outStream << "obj->checkHessSym..." << std::endl;
    // obj->checkHessSym( *des_var_rol_p, *direction_1, *direction_2, true, *outStream);

    // //  *outStream << "Outputting Hessian..." << std::endl;
    // //  dealii::FullMatrix<double> hessian(n_design_variables, n_design_variables);
    // //  for (unsigned int i=0; i<n_design_variables; ++i) {
    // //      pcout << "Column " << i << " out of " << n_design_variables << std::endl;
    // //      auto direction_unit = des_var_ctl_rol_p->basis(i);
    // //      auto hv3 = des_var_ctl_rol_p->clone();
    // //      double tol = 1e-6;
    // //      robj->hessVec( *hv3, *direction_unit, *des_var_ctl_rol_p, tol );

    // //      auto result = ROL_vector_to_dealii_vector_reference(*hv3);
    // //      result.update_ghost_values();

    // //      for (unsigned int j=0; j<result.size(); ++j) {
    // //          hessian[j][i] = result[j];
    // //      }
    // //  }
    // //  if (mpi_rank == 0) hessian.print_formatted(*outStream, 3, true, 10, "0", 1., 0.);

    // *outStream << "robj->checkHessSym..." << std::endl;
    // robj->checkHessSym( *des_var_ctl_rol_p, *direction_ctl_1, *direction_ctl_2, true, *outStream);

    // *outStream << "Starting optimization..." << std::endl;
    // ROL::OptimizationSolver<double> solver( opt, parlist );
    // solver.solve( *outStream );

    // ROL::Ptr< const ROL::AlgorithmState <double> > opt_state = solver.getAlgorithmState();

    // ROL::EExitStatus opt_exit_state = opt_state->statusFlag;

    filebuffer.close();

    return 0;
}


int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int test_error = false;
    try {
         test_error += test(10);
         test_error += test(20);
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

