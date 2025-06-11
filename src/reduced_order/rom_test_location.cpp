#include "rom_test_location.h"
#include <iostream>
#include <filesystem>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "parameters/all_parameters.h"
#include "pod_basis_base.h"
#include "reduced_order_solution.h"
#include "linear_solver/linear_solver.h"
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_LinearProblem.h>
#include "Amesos.h"
#include <Amesos_Lapack.h>
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim, int nstate>
ROMTestLocation<dim, nstate>::ROMTestLocation(const RowVectorXd& parameter, std::unique_ptr<ROMSolution<dim, nstate>> rom_solution)
        :  TestLocationBase<dim, nstate>(parameter, std::move(rom_solution))
{
}

template <int dim, int nstate>
void ROMTestLocation<dim, nstate>::compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod_updated){

    this->pcout << "Computing adjoint-based error estimate between initial ROM and updated ROM..." << std::endl;

    dealii::ParameterHandler dummy_handler;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&this->rom_solution->params, dummy_handler);
    flow_solver->dg->solution = this->rom_solution->solution;
    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);

    const Epetra_CrsMatrix epetra_pod_basis = pod_updated->getPODBasis()->trilinos_matrix();
    const Epetra_CrsMatrix epetra_system_matrix_transpose = flow_solver->dg->system_matrix_transpose.trilinos_matrix();

    Epetra_CrsMatrix epetra_petrov_galerkin_basis(Epetra_DataAccess::Copy, epetra_system_matrix_transpose.DomainMap(), pod_updated->getPODBasis()->n());
    EpetraExt::MatrixMatrix::Multiply(epetra_system_matrix_transpose, true, epetra_pod_basis, false, epetra_petrov_galerkin_basis, true);

    Epetra_Vector epetra_gradient(epetra_pod_basis.RowMap());

    // Iterate over local indices and copy manually
    for (unsigned int i = 0; i < this->rom_solution->gradient.local_size(); ++i){
        epetra_gradient.ReplaceMyValue(static_cast<int>(i), 0, this->rom_solution->gradient.local_element(i));
    }
    
    Epetra_Vector epetra_reduced_gradient(epetra_pod_basis.DomainMap());

    epetra_pod_basis.Multiply(true, epetra_gradient, epetra_reduced_gradient);

    Epetra_CrsMatrix epetra_reduced_jacobian_transpose(Epetra_DataAccess::Copy, epetra_petrov_galerkin_basis.DomainMap(), pod_updated->getPODBasis()->n());
    EpetraExt::MatrixMatrix::Multiply(epetra_petrov_galerkin_basis, true, epetra_petrov_galerkin_basis, false, epetra_reduced_jacobian_transpose);

    Epetra_Vector epetra_reduced_adjoint(epetra_reduced_jacobian_transpose.DomainMap());
    epetra_reduced_gradient.Scale(-1);
    if (this->rom_solution->params.reduced_order_param.residual_error_bool == true){
        epetra_reduced_adjoint.PutScalar(0);
    }
    else{
        Epetra_LinearProblem linearProblem(&epetra_reduced_jacobian_transpose, &epetra_reduced_adjoint, &epetra_reduced_gradient);

        Amesos_Lapack Solver(linearProblem);

        Teuchos::ParameterList List;
        Solver.SetParameters(List);
        Solver.SymbolicFactorization();
        Solver.NumericFactorization();
        Solver.Solve();
    }

    Epetra_Vector epetra_reduced_residual(epetra_petrov_galerkin_basis.DomainMap());
    Epetra_Vector epetra_residual(Epetra_DataAccess::Copy, epetra_petrov_galerkin_basis.RangeMap(), const_cast<double *>(flow_solver->dg->right_hand_side.begin()));
    epetra_petrov_galerkin_basis.Multiply(true, epetra_residual, epetra_reduced_residual);

    //Compute dual weighted residual
    this->initial_rom_to_final_rom_error = 0;
    epetra_reduced_adjoint.Dot(epetra_reduced_residual, &this->initial_rom_to_final_rom_error);
    this->initial_rom_to_final_rom_error *= -1;

    this->pcout << "Parameter: " << this->parameter << ". Error estimate between initial ROM and updated ROM: " << this->initial_rom_to_final_rom_error << std::endl;
}

#if PHILIP_DIM==1
        template class ROMTestLocation<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class ROMTestLocation<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}
}