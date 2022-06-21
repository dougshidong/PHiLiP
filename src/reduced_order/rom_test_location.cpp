#include "rom_test_location.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim, int nstate>
ROMTestLocation<dim, nstate>::ROMTestLocation(const RowVectorXd& parameter, std::shared_ptr<ROMSolution<dim, nstate>> rom_solution)
        : parameter(parameter)
        , rom_solution(rom_solution)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
{
    pcout << "Creating ROM test location..." << std::endl;
    compute_FOM_to_initial_ROM_error();
    initial_rom_to_final_rom_error = 0;
    compute_total_error();
    pcout << "ROM test location created. Error estimate updated." << std::endl;
}

template <int dim, int nstate>
void ROMTestLocation<dim, nstate>::compute_FOM_to_initial_ROM_error(){
    pcout << "Computing adjoint-based error estimate between ROM and FOM..." << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> gradient(rom_solution->right_hand_side);
    dealii::LinearAlgebra::distributed::Vector<double> adjoint(rom_solution->right_hand_side);
    //adjoint.update_ghost_values();
    //gradient.update_ghost_values();

    gradient = rom_solution->gradient;

    Parameters::LinearSolverParam linear_solver_param;
    linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::direct;
    solve_linear(*rom_solution->system_matrix_transpose, gradient*=-1.0, adjoint, linear_solver_param);

    //Compute dual weighted residual
    fom_to_initial_rom_error = 0;
    fom_to_initial_rom_error = -(adjoint * rom_solution->right_hand_side);
    //std::cout << "fom_to_initial_rom_error: " << fom_to_initial_rom_error << std::endl;
    //fom_to_initial_rom_error  = dealii::Utilities::MPI::sum(fom_to_initial_rom_error, MPI_COMM_WORLD);
    //std::cout << "fom_to_initial_rom_error: " << fom_to_initial_rom_error << std::endl;
    pcout << "Parameter: " << parameter << ". Error estimate between ROM and FOM: " << fom_to_initial_rom_error << std::endl;
}

template <int dim, int nstate>
void ROMTestLocation<dim, nstate>::compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod_updated){

    pcout << "Computing adjoint-based error estimate between initial ROM and updated ROM..." << std::endl;

    Epetra_CrsMatrix *epetra_pod_basis = const_cast<Epetra_CrsMatrix *>(&(pod_updated->getPODBasis()->trilinos_matrix()));
    Epetra_CrsMatrix *epetra_system_matrix_transpose = const_cast<Epetra_CrsMatrix *>(&(rom_solution->system_matrix_transpose->trilinos_matrix()));

    Epetra_CrsMatrix epetra_petrov_galerkin_basis(Epetra_DataAccess::View, epetra_system_matrix_transpose->DomainMap(), pod_updated->getPODBasis()->n());
    EpetraExt::MatrixMatrix::Multiply(*epetra_system_matrix_transpose, true, *epetra_pod_basis, false, epetra_petrov_galerkin_basis, true);

    Epetra_Vector epetra_gradient(Epetra_DataAccess::View, epetra_pod_basis->RowMap(), const_cast<double *>(rom_solution->gradient.begin()));
    Epetra_Vector epetra_reduced_gradient(epetra_pod_basis->DomainMap());

    epetra_pod_basis->Multiply(true, epetra_gradient, epetra_reduced_gradient);

    Epetra_CrsMatrix epetra_reduced_jacobian_transpose(Epetra_DataAccess::View, epetra_petrov_galerkin_basis.DomainMap(), pod_updated->getPODBasis()->n());
    EpetraExt::MatrixMatrix::Multiply(epetra_petrov_galerkin_basis, true, epetra_petrov_galerkin_basis, false, epetra_reduced_jacobian_transpose);

    Epetra_Vector epetra_reduced_adjoint(epetra_reduced_jacobian_transpose.DomainMap());
    epetra_reduced_gradient.Scale(-1);
    Epetra_LinearProblem linearProblem(&epetra_reduced_jacobian_transpose, &epetra_reduced_adjoint, &epetra_reduced_gradient);

    Amesos_BaseSolver* Solver;
    Amesos Factory;
    std::string SolverType = "Klu";
    Solver = Factory.Create(SolverType, linearProblem);
    Teuchos::ParameterList List;
    Solver->SetParameters(List);
    Solver->SymbolicFactorization();
    Solver->NumericFactorization();
    Solver->Solve();

    Epetra_Vector epetra_reduced_residual(epetra_petrov_galerkin_basis.DomainMap());
    Epetra_Vector epetra_residual(Epetra_DataAccess::View, epetra_petrov_galerkin_basis.RangeMap(), const_cast<double *>(rom_solution->right_hand_side.begin()));
    epetra_petrov_galerkin_basis.Multiply(true, epetra_residual, epetra_reduced_residual);

    //pcout << "Reduced adjoint: " << std::endl;
    //epetra_reduced_adjoint.Print(std::cout);

    //pcout << "Reduced gradient: " << std::endl;
    //epetra_reduced_gradient.Print(std::cout);

    //pcout << "Reduced residual: " << std::endl;
    //epetra_reduced_residual.Print(std::cout);


    //Compute dual weighted residual
    initial_rom_to_final_rom_error = 0;
    epetra_reduced_adjoint.Dot(epetra_reduced_residual, &initial_rom_to_final_rom_error);
    initial_rom_to_final_rom_error *= -1;
    //std::cout << "initial_rom_to_final_rom_error: " << initial_rom_to_final_rom_error << std::endl;
    //initial_rom_to_final_rom_error = dealii::Utilities::MPI::sum(initial_rom_to_final_rom_error, MPI_COMM_WORLD);
    //std::cout << "initial_rom_to_final_rom_error: " << initial_rom_to_final_rom_error << std::endl;
    pcout << "Parameter: " << parameter << ". Error estimate between initial ROM and updated ROM: " << initial_rom_to_final_rom_error << std::endl;

    delete Solver;
}

template <int dim, int nstate>
void ROMTestLocation<dim, nstate>::compute_total_error(){
    pcout << "Computing total error estimate between FOM and updated ROM..." << std::endl;
    total_error = fom_to_initial_rom_error - initial_rom_to_final_rom_error;
    pcout << "Parameter: " << parameter <<  ". Total error estimate between FOM and updated ROM: " << total_error << std::endl;
}


template class ROMTestLocation <PHILIP_DIM, 1>;
template class ROMTestLocation <PHILIP_DIM, 2>;
template class ROMTestLocation <PHILIP_DIM, 3>;
template class ROMTestLocation <PHILIP_DIM, 4>;
template class ROMTestLocation <PHILIP_DIM, 5>;

}
}