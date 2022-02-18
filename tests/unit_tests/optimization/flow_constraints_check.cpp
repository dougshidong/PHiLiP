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

const double FD_TOL = 1e-6;
const double CONSISTENCY_ABS_TOL = 1e-10;

const int dim = 2;
const int nstate = 4;
const int POLY_DEGREE = 2;
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
    parameter_handler.set("nonlinear_steady_residual_tolerance", 1e-12);
    parameter_handler.set("initial_time_step", 10.);
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
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    Grids::gaussian_bump(*grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, BUMP_HEIGHT);

    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, grid);
    dg->allocate_system ();

    // Initialize coarse grid solution with free-stream
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->initialize_steady_polynomial_ramping (POLY_DEGREE);
    // Solve the steady state problem
    ode_solver->steady_state();
    // Output initial solution
    dg->output_results_vtk(9999);
    dg->set_dual(dg->solution);

    const bool has_ownership = false;
    DealiiVector des_var_sim = dg->solution;
    DealiiVector des_var_ctl = ffd_design_variables;
    DealiiVector des_var_adj = dg->dual;
    Teuchos::RCP<DealiiVector> des_var_sim_rcp = Teuchos::rcp(&des_var_sim, has_ownership);
    Teuchos::RCP<DealiiVector> des_var_ctl_rcp = Teuchos::rcp(&des_var_ctl, has_ownership);
    Teuchos::RCP<DealiiVector> des_var_adj_rcp = Teuchos::rcp(&des_var_adj, has_ownership);

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
    if (mpi_rank == 0) filebuffer.open ("flow_constraints_check"+std::to_string(nx_ffd)+".log",std::ios::out);
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
    *outStream << "Checks dRdW * v1 against R(w+h*v1,x)/h  ..." << std::endl;
    {
        std::vector<std::vector<double>> results
            = con->checkApplyJacobian_1(*temp_sim, *temp_ctl, *v1, *jv1, steps, true, *outStream, order);

        const double max_rel_err = check_max_rel_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed con->checkApplyJacobian_1..." << std::endl;
        }
    }

    *outStream << "con->checkApplyJacobian_2..." << std::endl;
    *outStream << "Checks dRdX * v2 against R(w,x+h*v2)/h  ..." << std::endl;
    {
        std::vector<std::vector<double>> results
            = con->checkApplyJacobian_2(*temp_sim, *temp_ctl, *v2, *jv2, steps, true, *outStream, order);

        const double max_rel_err = check_max_rel_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed con->checkApplyJacobian_2..." << std::endl;
        }
    }

    *outStream << "con->checkInverseJacobian_1..." << std::endl;
    *outStream << "Checks || v - Jinv J v || == 0  ..." << std::endl;
    {
        const double v_minus_Jinv_J_v = con->checkInverseJacobian_1(*jv1, *v1, *temp_sim, *temp_ctl, true, *outStream);
        const double normalized_v_minus_Jinv_J_v = v_minus_Jinv_J_v / v1->norm();
        if (normalized_v_minus_Jinv_J_v > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed con->checkInverseJacobian_1..." << std::endl;
        }
    }

    *outStream << "con->checkInverseAdjointJacobian_1..." << std::endl;
    *outStream << "Checks || v - Jtinv Jt v || == 0  ..." << std::endl;
    {
        const double v_minus_Jinv_J_v = con->checkInverseAdjointJacobian_1(*jv1, *v1, *temp_sim, *temp_ctl, true, *outStream);
        const double normalized_v_minus_Jinv_J_v = v_minus_Jinv_J_v / v1->norm();
        if (normalized_v_minus_Jinv_J_v > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed con->checkInverseAdjointJacobian_1..." << std::endl;
        }

    }

    *outStream << "con->checkAdjointConsistencyJacobian..." << std::endl;
    *outStream << "Checks (w J v) versus (v Jt w)  ..." << std::endl;
    {
        const auto w = des_var_adj_rol_p->clone();
        const auto v = des_var_rol_p->clone();
        const auto x = des_var_rol_p->clone();
        const auto temp_Jv = des_var_adj_rol_p->clone();
        const auto temp_Jtw = des_var_rol_p->clone();
        const bool printToStream = true;
        const double wJv_minus_vJw = con->checkAdjointConsistencyJacobian (*w, *v, *x, *temp_Jv, *temp_Jtw, printToStream, *outStream);
        if (wJv_minus_vJw > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed con->checkAdjointConsistencyJacobian..." << std::endl;
        }
    }

    *outStream << "con->checkApplyAdjointHessian..." << std::endl;
    *outStream << "Checks (w H v) versus FD approximation  ..." << std::endl;
    {
        const auto dual = des_var_sim_rol_p->clone();
        const auto temp_sim_ctl = des_var_rol_p->clone();
        const auto v3 = des_var_rol_p->clone();
        const auto hv3 = des_var_rol_p->clone();

        std::vector<std::vector<double>> results
            = con->checkApplyAdjointHessian(*des_var_rol_p, *dual, *v3, *hv3, steps, true, *outStream, order);

        const double max_rel_err = check_max_rel_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed con->checkApplyAdjointHessian..." << std::endl;
        }
    }

    filebuffer.close();

    return test_error;
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

