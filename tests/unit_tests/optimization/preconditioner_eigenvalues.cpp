// The KKT system preconditioned with Biros and Ghattas' preconditioners P2 and P4
// should be perfectly conditioned if the exact Jacobians are used
// except for the eigenvalues from the control variables.
// Therefore, we check the eigenvalues of the system precond(KKT)^{-1} * KKT * Identity.
//
// We use a basic Euler test case and the Identity matrix to form the resulting matrix.

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
//#include "ROL_StatusTest.hpp"

#include "physics/euler.h"
#include "physics/initial_conditions/initial_condition.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_base.h"
#include "ode_solver/ode_solver_factory.h"

#include "functional/target_boundary_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"

#include "optimization/full_space_step.hpp"
#include "optimization/kkt_operator.hpp"
#include "optimization/kkt_birosghattas_preconditioners.hpp"

#include <deal.II/lac/lapack_full_matrix.h>

#include <string>


const double TOL = 1e-7;

const int dim = 2;
const int nstate = 4;
const int POLY_DEGREE = 0;
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
template<typename PrecType>
int check_preconditioned_system_eigenvalues(
    const ROL::Ptr<ROL::Vector_SimOpt<double>> design_variables,
    const ROL::Ptr<ROL::Vector<double>> lagrange_mult,
    const KKT_Operator<double> &kkt_operator,
    const PrecType &kkt_preconditioner,
    const unsigned int n_expected_eig,
    std::string &output)
{
    ROL::Ptr<ROL::Vector<double> > rhs1 = design_variables->clone();
    ROL::Ptr<ROL::Vector<double> > rhs2 = lagrange_mult->clone();
    ROL::Vector_SimOpt rhs_rol(rhs1, rhs2);
    dealiiSolverVectorWrappingROL<double> right_hand_side(makePtrFromRef(rhs_rol));
    dealiiSolverVectorWrappingROL<double> column_of_kkt_operator, column_of_precond_kkt_operator;
    column_of_kkt_operator.reinit(right_hand_side);
    column_of_precond_kkt_operator.reinit(right_hand_side);

    const unsigned int n = right_hand_side.size();
    dealii::FullMatrix<double> fullA(n);

    for (unsigned int i = 0; i < n; ++i) {
        std::cout << "COLUMN NUMBER: " << i+1 << " OUT OF " << n << std::endl;
        auto basis = right_hand_side.basis(i);
        MPI_Barrier(MPI_COMM_WORLD);
        {
            kkt_operator.vmult(column_of_kkt_operator,*basis);
            kkt_preconditioner.vmult(column_of_precond_kkt_operator,column_of_kkt_operator);
        }
        //kkt_preconditioner.vmult(column_of_precond_kkt_operator,*basis);
        for (unsigned int j = 0; j < n; ++j) {
            fullA[j][i] = column_of_precond_kkt_operator[j];
            //fullA[j][i] = column_of_kkt_operator[j];
        }
    }
    dealii::LAPACKFullMatrix< double > lapack_fullmatrix(n,n);
    lapack_fullmatrix = fullA;
    lapack_fullmatrix.compute_eigenvalues();

    unsigned int n_nonunity_eig = 0;
    for(unsigned int i = 0; i < n; ++i) {
        const std::complex<double> eig = lapack_fullmatrix.eigenvalue(i);

        if (std::abs(std::real(eig) - 1.0) > TOL
            || std::abs(std::real(eig) - 1.0) > TOL ) {

            std::string out = " Eigenvalue " + std::to_string(i) + " is not 1.0. It is ("
                                + std::to_string(std::real(eig))
                                + "  ,   "
                                + std::to_string(std::imag(eig))
                                + ")\n";
            std::cout << out;
            output.append(out);
            //std::cout << " Eigenvalue " << i << " is not 1.0. It is " << eig << std::endl;
            n_nonunity_eig++;

        }
        
    }
    //std::cout << "Total of " << n_nonunity_eig << " non 1.0 eigenvalues, where " << n_expected_eig << " are expected." << std::endl;
    std::string out = "Total of " + std::to_string(n_nonunity_eig) 
                      + " non 1.0 eigenvalues, where "
                      + std::to_string(n_expected_eig) 
                      + " are expected. \n";

    std::cout << out;
    output.append(out);

    //std::cout<<"Dense matrix:"<<std::endl;
    //fullA.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
    //std::abort();


    return (n_nonunity_eig - n_expected_eig);
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
    FreeStreamInitialConditions<dim,nstate,double> initial_conditions(euler_physics_double);

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

    auto objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( functional, ffd, ffd_design_variables_indices_dim );
    auto flow_constraints = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);

    auto design_variables = ROL::makePtr<ROL::Vector_SimOpt<double>>(des_var_sim_rol_p, des_var_ctl_rol_p);
    auto lagrange_mult    = des_var_adj_rol_p;
    lagrange_mult->set(*des_var_sim_rol_p);

    KKT_Operator<double> kkt_operator( objective, flow_constraints, design_variables, lagrange_mult);

    Teuchos::ParameterList parlist;
    auto secant = ROL::SecantFactory<double>(parlist);



    std::string output;
    output.append("\n\n Summary *************************************** \n\n");
    output.append("P4 preconditioner eigenvalues output: \n");
    {
        // P4 preconditioner
        const unsigned int n_expected_eig = nx_ffd - 2;
        const bool use_second_order_terms = true;
        const bool use_approximate_preconditioner = false;
        KKT_P24_Preconditioner<double> kkt_preconditioner(
            objective,
            flow_constraints,
            design_variables,
            lagrange_mult,
            secant,
            use_second_order_terms,
            use_approximate_preconditioner);
        test_error += check_preconditioned_system_eigenvalues(design_variables, lagrange_mult, kkt_operator, kkt_preconditioner, n_expected_eig, output);
    }
    output.append("\n\n ");
    output.append("P2 preconditioner eigenvalues output: \n");
    {
        // P2 preconditioner
        const unsigned int n_expected_eig = nx_ffd - 2;
        const bool use_second_order_terms = false;
        const bool use_approximate_preconditioner = false;
        KKT_P24_Preconditioner<double> kkt_preconditioner(
            objective,
            flow_constraints,
            design_variables,
            lagrange_mult,
            secant,
            use_second_order_terms,
            use_approximate_preconditioner);
        test_error += check_preconditioned_system_eigenvalues(design_variables, lagrange_mult, kkt_operator, kkt_preconditioner, n_expected_eig, output);
    }
    std::cout << output;


    filebuffer.close();

    return test_error;
}

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    assert (1 == dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    int test_error = 0;
    try {
         test_error += test(5);
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


