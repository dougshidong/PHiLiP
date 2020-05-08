#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/solver_gmres.h>

#include "linear_solver.h"
namespace PHiLiP {

std::pair<unsigned int, double>
solve_linear (
    dealii::TrilinosWrappers::SparseMatrix &system_matrix,
    dealii::LinearAlgebra::distributed::Vector<double> &right_hand_side,
    dealii::LinearAlgebra::distributed::Vector<double> &solution,
    const Parameters::LinearSolverParam &param)
{

    // if (pcout.is_active()) system_matrix.print(pcout.get_stream(), true);
    // if (pcout.is_active()) solution.print(pcout.get_stream());

    Parameters::LinearSolverParam::LinearSolverEnum direct_type = Parameters::LinearSolverParam::LinearSolverEnum::direct;
    Parameters::LinearSolverParam::LinearSolverEnum gmres_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;

    if (param.linear_solver_output == Parameters::OutputEnum::verbose) {
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        if (pcout.is_active()) right_hand_side.print(pcout.get_stream());
        if (pcout.is_active()) solution.print(pcout.get_stream());
        dealii::FullMatrix<double> fullA(system_matrix.m());
        fullA.copy_from(system_matrix);
        pcout<<"Dense matrix:"<<std::endl;
        if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);
    }
    if (param.linear_solver_type == direct_type) {

        dealii::SolverControl solver_control(1, 0);
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data(false);
        //dealii::TrilinosWrappers::SolverDirect::AdditionalData data(parameters.output == Parameters::Solver::verbose);
        dealii::TrilinosWrappers::SolverDirect direct(solver_control, data);

        direct.solve(system_matrix, solution, right_hand_side);
        return {solver_control.last_step(), solver_control.last_value()};
    } else if (param.linear_solver_type == gmres_type) {
        Epetra_Vector x(View,
                        system_matrix.trilinos_matrix().DomainMap(),
                        solution.begin());
        Epetra_Vector b(View,
                        system_matrix.trilinos_matrix().RangeMap(),
                        right_hand_side.begin());
        AztecOO solver;
        solver.SetAztecOption( AZ_output, (param.linear_solver_output ? AZ_all : AZ_none));
        solver.SetAztecOption(AZ_solver, AZ_gmres);
        solver.SetAztecOption(AZ_kspace, param.restart_number);
        solver.SetRHS(&b);
        solver.SetLHS(&x);
        solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
        solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
        solver.SetAztecOption(AZ_overlap, 0);
        solver.SetAztecOption(AZ_reorder, 1); // RCM re-ordering
  
        const double 
          ilut_drop = param.ilut_drop,
          ilut_rtol = param.ilut_rtol,//0.0,//1.1,
          ilut_atol = param.ilut_atol,//0.0,//1e-9,
          linear_residual = param.linear_residual;//1e-4;
        const int 
          ilut_fill = param.ilut_fill,//1,
          max_iterations = param.max_iterations;//200
  
        solver.SetAztecParam(AZ_drop, ilut_drop);
        solver.SetAztecParam(AZ_ilut_fill, ilut_fill);
        solver.SetAztecParam(AZ_athresh, ilut_atol);
        solver.SetAztecParam(AZ_rthresh, ilut_rtol);
        solver.SetUserMatrix(const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix()));
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        pcout << " Linear solver max its: " << max_iterations 
              << " Linear residual tolerance: " << linear_residual << std::endl;
        solver.Iterate(max_iterations,
                       linear_residual);
  
        pcout << " Linear solver took " << solver.NumIters()
              << " iterations resulting in a linear residual of " << solver.ScaledResidual() << std::endl
              << " Current RHS norm: " << right_hand_side.l2_norm()
              << " Newton update norm: " << solution.l2_norm() << std::endl;

        return {solver.NumIters(), solver.TrueResidual()};
    }
    return {-1.0, -1.0};
}

// std::pair<unsigned int, double>
// solve_linear (
//     const dealii::TrilinosWrappers::SparseMatrix &system_matrix,
//     const dealii::LinearAlgebra::distributed::Vector<double> &right_hand_side,
//     dealii::LinearAlgebra::distributed::Vector<double> &solution,
//     const Parameters::LinearSolverParam &param)
// {
//     const unsigned int ilut_drop = param.ilut_drop;
//     const unsigned int ilut_fill = param.ilut_fill;
//     const double ilut_atol = 1e-4;// param.ilut_atol;
//     const double ilut_rtol = 1.01;//param.ilut_rtol;
//     const unsigned int overlap = 1;
//     dealii::TrilinosWrappers::PreconditionILUT::AdditionalData additional_data(ilut_drop, ilut_fill, ilut_atol, ilut_rtol, overlap);
//     dealii::TrilinosWrappers::PreconditionILUT preconditioner;
// 
//     //dealii::TrilinosWrappers::PreconditionIdentity preconditioner;
//     //dealii::TrilinosWrappers::PreconditionIdentity::AdditionalData additional_data;
// 
//     preconditioner.initialize(system_matrix, additional_data);
// 
//         // dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
//         // if (pcout.is_active()) right_hand_side.print(pcout.get_stream());
//         // if (pcout.is_active()) solution.print(pcout.get_stream());
//         // dealii::FullMatrix<double> fullA(system_matrix.m());
//         // fullA.copy_from(system_matrix);
//         // pcout<<"Dense matrix:"<<std::endl;
//         // if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);
// 
//     dealii::SolverControl solver_control(param.max_iterations, param.linear_residual);
// 
//     const bool output_solver_details=false;
//     const unsigned int restart_parameter=100;
//     dealii::TrilinosWrappers::SolverGMRES::AdditionalData solver_add_data(output_solver_details, restart_parameter);
// 
//     dealii::TrilinosWrappers::SolverGMRES solver(solver_control, solver_add_data);
//     solver.solve(system_matrix, solution, right_hand_side, preconditioner);
//     return {solver_control.last_step(), solver_control.last_value()};
// }

std::pair<unsigned int, double>
solve_linear_2(
    const dealii::TrilinosWrappers::SparseMatrix &system_matrix,
    const dealii::LinearAlgebra::distributed::Vector<double> &right_hand_side,
    dealii::LinearAlgebra::distributed::Vector<double> &solution,
    const Parameters::LinearSolverParam &param)
{
    std::cout << "Not working. Don't use until it is debugged." << std::endl;
    std::abort();
    //const unsigned int ilut_drop = param.ilut_drop;
    //const unsigned int ilut_fill = param.ilut_fill;
    //const double ilut_atol = param.ilut_atol;
    //const double ilut_rtol = param.ilut_rtol;
    //const unsigned int overlap = 1;
    //dealii::TrilinosWrappers::PreconditionILUT::AdditionalData additional_data(ilut_drop, ilut_fill, ilut_atol, ilut_rtol, overlap);
    //dealii::TrilinosWrappers::PreconditionILUT preconditioner;

    //dealii::TrilinosWrappers::PreconditionIdentity preconditioner;
    //dealii::TrilinosWrappers::PreconditionIdentity::AdditionalData additional_data;

    //preconditioner.initialize(system_matrix, additional_data);

    dealii::TrilinosWrappers::PreconditionJacobi preconditioner;
    preconditioner.initialize(system_matrix);

    // dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    // if (pcout.is_active()) right_hand_side.print(pcout.get_stream());
    // if (pcout.is_active()) solution.print(pcout.get_stream());
    // dealii::FullMatrix<double> fullA(system_matrix.m());
    // fullA.copy_from(system_matrix);
    // pcout<<"Dense matrix:"<<std::endl;
    // if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);

    bool log_history = true;
    bool log_result = true;
    dealii::SolverControl solver_control(param.max_iterations, param.linear_residual, log_history, log_result);
    dealii::deallog.depth_console(10);

    //const bool output_solver_details=false;
    //const unsigned int restart_parameter=100;
    //dealii::TrilinosWrappers::SolverGMRES::AdditionalData solver_add_data(output_solver_details, restart_parameter);
    //dealii::TrilinosWrappers::SolverGMRES solver(solver_control, solver_add_data);
    dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver( solver_control, dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>>::AdditionalData(100));

    solver.solve(system_matrix, solution, right_hand_side, preconditioner);
    return {solver_control.last_step(), solver_control.last_value()};
}

} // PHiLiP namespace
