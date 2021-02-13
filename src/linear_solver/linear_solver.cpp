#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include "Ifpack.h"
#include <Ifpack_ILU.h>

#include <deal.II/lac/solver_gmres.h>

#include "linear_solver.h"

#include "global_counter.hpp"

namespace PHiLiP {

class PreconditionILU_RAS: public dealii::TrilinosWrappers::PreconditionILU {

    using dealii::TrilinosWrappers::PreconditionILU::AdditionalData;

public:
    void initialize(const dealii::TrilinosWrappers::SparseMatrix &matrix, const AdditionalData &additional_data) 
    {
        preconditioner.reset();

        matrix.trilinos_matrix().Print(std::cout);

        Epetra_CrsMatrix * epetra_matrix = const_cast<Epetra_CrsMatrix *>(&(matrix.trilinos_matrix()));

        Ifpack_Preconditioner *ifpack = dynamic_cast<Ifpack_Preconditioner *>(preconditioner.get());

        preconditioner.reset(Ifpack().Create("ILU", epetra_matrix, additional_data.overlap));

        std::cout << ifpack << std::endl;
        ifpack = dynamic_cast<Ifpack_Preconditioner *>(preconditioner.get());
        std::cout << ifpack << std::endl;

        Assert(ifpack != nullptr, dealii::ExcMessage("Trilinos could not create this " "preconditioner"));

        int ierr;

        Teuchos::ParameterList parameter_list;
        parameter_list.set("fact: level-of-fill", static_cast<int>(additional_data.ilu_fill));
        parameter_list.set("fact: absolute threshold", additional_data.ilu_atol);
        parameter_list.set("fact: relative threshold", additional_data.ilu_rtol);
        //parameter_list.set("schwarz: combine mode", "Add");
        parameter_list.set("schwarz: reordering type", "rcm");

        ierr = ifpack->SetParameters(parameter_list);
        AssertThrow(ierr == 0, dealii::ExcTrilinosError(ierr));

        ierr = ifpack->Initialize();
        AssertThrow(ierr == 0, dealii::ExcTrilinosError(ierr));

        ierr = ifpack->Compute();
        AssertThrow(ierr == 0, dealii::ExcTrilinosError(ierr));

        ifpack->Print(std::cout);

        Ifpack_ILU *ifpack_ilu = dynamic_cast<Ifpack_ILU *>(ifpack);

        {
            const Epetra_CrsMatrix &L = ifpack_ilu->L();
            const Epetra_Vector &D = ifpack_ilu->D();
            const Epetra_CrsMatrix &U = ifpack_ilu->U();

            L.Print(std::cout);
            D.Print(std::cout);
            U.Print(std::cout);
        }


        Ifpack Factory;

        Teuchos::ParameterList List;

        const std::string PrecType = "ILU"; 
        List.set("fact: level-of-fill", 0);

        List.set("schwarz: reordering type", "rcm");
        const int OverlapLevel = 1; // one row of overlap among the processes
        Ifpack_Preconditioner *jacobian_prec = Factory.Create(PrecType, epetra_matrix, OverlapLevel);
        jacobian_prec->Print(std::cout);
        assert (jacobian_prec != 0);


        ierr = jacobian_prec->SetParameters(List);
        AssertThrow(ierr == 0, dealii::ExcTrilinosError(ierr));
        ierr = jacobian_prec->Initialize();
        AssertThrow(ierr == 0, dealii::ExcTrilinosError(ierr));
        ierr = jacobian_prec->Compute();
        AssertThrow(ierr == 0, dealii::ExcTrilinosError(ierr));

        preconditioner.reset(jacobian_prec);


        {
            ifpack = dynamic_cast<Ifpack_Preconditioner *>(preconditioner.get());
            ifpack_ilu = dynamic_cast<Ifpack_ILU *>(ifpack);
            const Epetra_CrsMatrix &L = ifpack_ilu->L();
            const Epetra_Vector &D = ifpack_ilu->D();
            const Epetra_CrsMatrix &U = ifpack_ilu->U();

            L.Print(std::cout);
            D.Print(std::cout);
            U.Print(std::cout);
        }
    };
};

std::pair<unsigned int, double>
solve_linear3 (
    const dealii::TrilinosWrappers::SparseMatrix &system_matrix,
    dealii::LinearAlgebra::distributed::Vector<double> &right_hand_side,
    dealii::LinearAlgebra::distributed::Vector<double> &solution,
    const Parameters::LinearSolverParam &param)
{
    std::shared_ptr<dealii::TrilinosWrappers::PreconditionBase> preconditioner;
    // ILU Preconditioner
    if (param.ilut_fill < 1) {
        typedef dealii::TrilinosWrappers::PreconditionILU::AdditionalData AddiData_ILU;
        const unsigned int overlap = 1;
        AddiData_ILU precond_settings(std::abs(param.ilut_fill), param.ilut_atol, param.ilut_rtol, overlap);

        std::shared_ptr<dealii::TrilinosWrappers::PreconditionILU> ilu_preconditioner = std::make_shared<dealii::TrilinosWrappers::PreconditionILU> ();
        ilu_preconditioner->initialize(system_matrix, precond_settings);

        preconditioner = ilu_preconditioner;
    } else {
        typedef dealii::TrilinosWrappers::PreconditionILUT::AdditionalData AddiData_ILUT;
        const unsigned int overlap = 1;
        AddiData_ILUT precond_settings(param.ilut_fill, param.ilut_atol, param.ilut_rtol, overlap);

        std::shared_ptr<dealii::TrilinosWrappers::PreconditionILUT> ilut_preconditioner = std::make_shared<dealii::TrilinosWrappers::PreconditionILUT> ();
        ilut_preconditioner->initialize(system_matrix, precond_settings);

        preconditioner = ilut_preconditioner;
    }

    // Solver convergence settings
    const double rhs_norm = right_hand_side.l2_norm();
    const double linear_residual_tolerance = param.linear_residual * rhs_norm;
    const int max_iterations = param.max_iterations;

    const bool log_history = (param.linear_solver_output == Parameters::OutputEnum::verbose);
    const bool log_result = true;//(param.linear_solver_output == Parameters::OutputEnum::verbose);
    dealii::SolverControl solver_control(max_iterations, linear_residual_tolerance, log_history, log_result);
    if (log_history) solver_control.enable_history_data();

    const bool     right_preconditioning = false; // default: false
    const bool     use_default_residual = true;//false; // default: true
    const bool     force_re_orthogonalization = false; // default: false
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    typedef typename dealii::SolverGMRES<VectorType>::AdditionalData AddiData_GMRES;
    AddiData_GMRES add_data_gmres( param.restart_number, right_preconditioning, use_default_residual, force_re_orthogonalization);
    dealii::SolverGMRES<VectorType> solver_gmres(solver_control, add_data_gmres);


    solver_gmres.solve(system_matrix, solution, right_hand_side, *preconditioner);

    return {solver_control.last_step(), solver_control.last_value()};

}

std::pair<unsigned int, double>
solve_linear (
    const dealii::TrilinosWrappers::SparseMatrix &system_matrix,
    dealii::LinearAlgebra::distributed::Vector<double> &right_hand_side,
    dealii::LinearAlgebra::distributed::Vector<double> &solution,
    const Parameters::LinearSolverParam &param)
{

    // if (pcout.is_active()) system_matrix.print(pcout.get_stream(), true);
    // if (pcout.is_active()) solution.print(pcout.get_stream());

    Parameters::LinearSolverParam::LinearSolverEnum direct_type = Parameters::LinearSolverParam::LinearSolverEnum::direct;
    Parameters::LinearSolverParam::LinearSolverEnum gmres_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;

    //if (param.linear_solver_output == Parameters::OutputEnum::verbose) {
    //    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    //    if (pcout.is_active()) right_hand_side.print(pcout.get_stream());
    //    if (pcout.is_active()) solution.print(pcout.get_stream());
    //    dealii::FullMatrix<double> fullA(system_matrix.m());
    //    fullA.copy_from(system_matrix);
    //    pcout<<"Dense matrix:"<<std::endl;
    //    if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 12, true, 10, "0", 1., 0.);
    //}
    if (param.linear_solver_type == direct_type) {

        dealii::SolverControl solver_control(1, 0);
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data(false);
        //dealii::TrilinosWrappers::SolverDirect::AdditionalData data(parameters.output == Parameters::Solver::verbose);
        dealii::TrilinosWrappers::SolverDirect direct(solver_control, data);

        direct.solve(system_matrix, solution, right_hand_side);
        return {solver_control.last_step(), solver_control.last_value()};
    } else if (param.linear_solver_type == gmres_type) {
        //solution = right_hand_side;
        //solution *= 1e-3;
        solution *= 0.0;
        Epetra_Vector x(View,
                        system_matrix.trilinos_matrix().DomainMap(),
                        solution.begin());
        Epetra_Vector b(View,
                        system_matrix.trilinos_matrix().RangeMap(),
                        right_hand_side.begin());
        AztecOO solver;
        solver.SetAztecOption( AZ_output, (param.linear_solver_output ? AZ_all : AZ_last));
        solver.SetAztecOption(AZ_solver, AZ_gmres);
        //solver.SetAztecOption(AZ_solver, AZ_bicgstab);
        //solver.SetAztecOption(AZ_solver, AZ_cg);
        solver.SetAztecOption(AZ_kspace, param.restart_number);
        solver.SetRHS(&b);
        solver.SetLHS(&x);


        const double rhs_norm = right_hand_side.l2_norm();
        const double linear_residual = param.linear_residual * rhs_norm;//1e-4;
        const int max_iterations = param.max_iterations;//200
        solver.SetUserMatrix(const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix()));
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        pcout << " Solving linear system with max_iterations = " << max_iterations
              << " and linear residual tolerance: " << linear_residual << std::endl;


        //solver.SetAztecOption(AZ_orthog, AZ_modified);
        solver.SetAztecOption(AZ_orthog, AZ_classic);
        solver.SetAztecOption(AZ_conv, AZ_rhs);

        solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
        solver.SetAztecOption(AZ_overlap, 1);
        solver.SetAztecOption(AZ_reorder, 1); // RCM re-ordering
        const int ilut_fill = param.ilut_fill;
        if (ilut_fill < -99) {
            // // Jacobi preconditioner.
            //solver.SetAztecOption(AZ_precond, AZ_Jacobi);
            //solver.SetAztecOption(AZ_poly_ord, 1);

            // No Preconditioner.
            solver.SetAztecOption(AZ_precond, AZ_none);
        } else if (ilut_fill < 1) {
            solver.SetAztecOption(AZ_subdomain_solve, AZ_ilu);
            solver.SetAztecOption(AZ_graph_fill, std::abs(ilut_fill));

            double ilut_rtol = param.ilut_rtol;//0.0,//1.1,
            double ilut_atol = param.ilut_atol;//0.0,//1e-9,
            solver.SetAztecParam(AZ_athresh, ilut_atol);
            solver.SetAztecParam(AZ_rthresh, ilut_rtol);
        } else {
            const double ilut_drop = param.ilut_drop;
            double ilut_rtol = param.ilut_rtol;//0.0,//1.1,
            double ilut_atol = param.ilut_atol;//0.0,//1e-9,
            solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);

            solver.SetAztecParam(AZ_drop, ilut_drop);
            solver.SetAztecParam(AZ_ilut_fill, ilut_fill);
            solver.SetAztecParam(AZ_athresh, ilut_atol);
            solver.SetAztecParam(AZ_rthresh, ilut_rtol);
        }

        unsigned int n_iterations = 0;
        const int n_solves = 2;
        for (int i_solve = 0; i_solve < n_solves; ++i_solve) {
            solver.Iterate(max_iterations,
                           linear_residual);
            n_iterations += solver.NumIters();
            pcout << " Solve #" << i_solve + 1 << " out of " << n_solves << "."
                  << " Linear solver took " << solver.NumIters()
                  << " iterations resulting in a linear residual of " << solver.ScaledResidual()
                  << std::endl;
        }

        pcout << " Totalling " << n_iterations
              << " iterations resulting in a linear residual of " << solver.ScaledResidual() << std::endl
              << " Current RHS norm: " << right_hand_side.l2_norm()
              << " Linear solution norm: " << solution.l2_norm() << std::endl;

        //n_vmult += 3*solver.NumIters();
        //dRdW_mult += 3*solver.NumIters();
        n_vmult += 7*solver.NumIters();
        dRdW_mult += 7*solver.NumIters();

        //std::abort();
        return {solver.NumIters(), solver.TrueResidual()};
    }
    return {-1.0, -1.0};
}


} // PHiLiP namespace
