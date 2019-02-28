#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include "linear_solver.h"
namespace PHiLiP
{
    using namespace dealii;


    template std::pair<unsigned int, double>
    //LinearSolver<double>::solve_linear (
    solve_linear (
        TrilinosWrappers::SparseMatrix &system_matrix,
        Vector<double> &right_hand_side, 
        Vector<double> &solution);

    template <typename real>
    std::pair<unsigned int, double>
    //LinearSolver<real>::solve_linear (
    solve_linear (
        TrilinosWrappers::SparseMatrix &system_matrix,
        Vector<real> &right_hand_side, 
        Vector<real> &solution)
    {
        //SolverControl solver_control(1, 0);
        //TrilinosWrappers::SolverDirect::AdditionalData data(true);
        ////TrilinosWrappers::SolverDirect::AdditionalData data(parameters.output == Parameters::Solver::verbose);
        //TrilinosWrappers::SolverDirect direct(solver_control, data);

        ////system_matrix.print(std::cout, true);
        ////solution.print(std::cout);
        ////right_hand_side.print(std::cout);

        //direct.solve(system_matrix, solution, right_hand_side);
        //return {solver_control.last_step(), solver_control.last_value()};

        {
          Epetra_Vector x(View,
                          system_matrix.trilinos_matrix().DomainMap(),
                          solution.begin());
          Epetra_Vector b(View,
                          system_matrix.trilinos_matrix().RangeMap(),
                          right_hand_side.begin());
          AztecOO solver;
          solver.SetAztecOption(
            AZ_output,
            (true ? AZ_none : AZ_all));
          solver.SetAztecOption(AZ_solver, AZ_gmres);
          solver.SetRHS(&b);
          solver.SetLHS(&x);
          solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
          solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
          solver.SetAztecOption(AZ_overlap, 0);
          solver.SetAztecOption(AZ_reorder, 0);

          const double 
            ilut_drop = 1e-10,
            ilut_rtol = 1.1,
            ilut_atol = 1e-9,
            linear_residual = 1e-4;
          const int 
            ilut_fill = 2,
            max_iterations = 1000
            ;

          //solver.SetAztecParam(AZ_drop, parameters.ilut_drop);
          //solver.SetAztecParam(AZ_ilut_fill, parameters.ilut_fill);
          //solver.SetAztecParam(AZ_athresh, parameters.ilut_atol);
          //solver.SetAztecParam(AZ_rthresh, parameters.ilut_rtol);
          //solver.SetUserMatrix(
          //  const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix()));
          //solver.Iterate(parameters.max_iterations,
          //               parameters.linear_residual);
          solver.SetAztecParam(AZ_drop, ilut_drop);
          solver.SetAztecParam(AZ_ilut_fill, ilut_fill);
          solver.SetAztecParam(AZ_athresh, ilut_atol);
          solver.SetAztecParam(AZ_rthresh, ilut_rtol);
          solver.SetUserMatrix(
            const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix()));
          solver.Iterate(max_iterations,
                         linear_residual);

        std::cout << " Linear solber iteration: " << solver.NumIters() 
                  << " Residual: " << solver.TrueResidual()
                  << " RHS norm: " << right_hand_side.l2_norm()
                  << " Newton update norm: " << solution.l2_norm()
                  << std::endl;
          return {solver.NumIters(), solver.TrueResidual()};
        }
    }
}
