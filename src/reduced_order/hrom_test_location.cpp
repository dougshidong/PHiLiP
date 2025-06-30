#include "hrom_test_location.h"
#include <iostream>
#include <filesystem>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "parameters/all_parameters.h"
#include "pod_basis_base.h"
#include "multi_core_helper_functions.h"
#include "reduced_order_solution.h"
#include "linear_solver/linear_solver.h"
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_SerialComm.h>
#include "Amesos.h"
#include <Amesos_Lapack.h>
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim, int nstate>
HROMTestLocation<dim, nstate>::HROMTestLocation(const RowVectorXd& parameter, std::unique_ptr<ROMSolution<dim, nstate>> rom_solution, std::shared_ptr< DGBase<dim, double> > dg_input, Epetra_Vector weights)
        : TestLocationBase<dim, nstate>(parameter, std::move(rom_solution))
        , dg(dg_input)
        , ECSW_weights(weights)
{
}

template <int dim, int nstate>
void HROMTestLocation<dim, nstate>::compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod_updated){

    this->pcout << "Computing adjoint-based error estimate between initial ROM and updated ROM..." << std::endl;

    dealii::ParameterHandler dummy_handler;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&this->rom_solution->params, dummy_handler);
    flow_solver->dg->solution = this->rom_solution->solution;
    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);

    // Build hyperreduced Jacobian
    Epetra_CrsMatrix epetra_system_matrix = flow_solver->dg->system_matrix.trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> reduced_system_matrix = generate_hyper_reduced_jacobian(epetra_system_matrix);

    // Find test basis W with hyperreduced Jacobian
    const Epetra_CrsMatrix epetra_pod_basis = pod_updated->getPODBasis()->trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> epetra_petrov_galerkin_basis_ptr = generate_test_basis(*reduced_system_matrix, epetra_pod_basis);
    Epetra_CrsMatrix epetra_petrov_galerkin_basis = *epetra_petrov_galerkin_basis_ptr;

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

    // Compute dual weighted residual
    this->initial_rom_to_final_rom_error = 0;
    epetra_reduced_adjoint.Dot(epetra_reduced_residual, &this->initial_rom_to_final_rom_error);
    this->initial_rom_to_final_rom_error *= -1;

    this->pcout << "Parameter: " << this->parameter << ". Error estimate between initial ROM and updated ROM: " << this->initial_rom_to_final_rom_error << std::endl;
}

template <int dim, int nstate>
std::shared_ptr<Epetra_CrsMatrix> HROMTestLocation<dim, nstate>::generate_hyper_reduced_jacobian(const Epetra_CrsMatrix &system_matrix)
{
    /* Refer to Equation (12) in:
    https://onlinelibrary.wiley.com/doi/10.1002/nme.6603 (includes definitions of matrices used below such as L_e and L_e_PLUS)
    Create empty Hyper-reduced Jacobian Epetra structure */
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
    Epetra_CrsMatrix local_system_matrix = copy_matrix_to_all_cores(system_matrix);
    Epetra_CrsMatrix reduced_jacobian(Epetra_DataAccess::Copy, system_matrix_rowmap, system_matrix.NumGlobalCols());
    int N = system_matrix.NumGlobalRows();
    Epetra_BlockMap element_map = ECSW_weights.Map();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbour_dofs_indices(max_dofs_per_cell);

    // Loop through elements 
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        // Add the contributilons of an element if the weight from the NNLS is non-zero
        if (cell->is_locally_owned()){
            int global = cell->active_cell_index();
            const int local_element = element_map.LID(global);
            if (ECSW_weights[local_element] != 0){
                const int fe_index_curr_cell = cell->active_fe_index();
                const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
                const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

                current_dofs_indices.resize(n_dofs_curr_cell);
                cell->get_dof_indices(current_dofs_indices);

                int numE;
                int row_i = current_dofs_indices[0];
                double *row = new double[local_system_matrix.NumGlobalCols()];
                int *global_indices = new int[local_system_matrix.NumGlobalCols()];
                // Use the Jacobian to determine the stencil around the current element
                local_system_matrix.ExtractGlobalRowCopy(row_i, local_system_matrix.NumGlobalCols(), numE, row, global_indices);
                int neighbour_dofs_curr_cell = 0;
                for (int i = 0; i < numE; i++){
                    neighbour_dofs_curr_cell +=1;
                    neighbour_dofs_indices.resize(neighbour_dofs_curr_cell);
                    neighbour_dofs_indices[neighbour_dofs_curr_cell-1] = global_indices[i];
                }

                delete[] row;
                delete[] global_indices;

                // Create L_e matrix and transposed L_e matrix for current cell
                const Epetra_SerialComm sComm;
                Epetra_Map LeRowMap(n_dofs_curr_cell, 0, sComm);
                Epetra_Map LeTRowMap(N, 0, sComm);
                Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, LeTRowMap, 1);
                Epetra_CrsMatrix L_e_T(Epetra_DataAccess::Copy, LeTRowMap, n_dofs_curr_cell);
                Epetra_Map LePLUSRowMap(neighbour_dofs_curr_cell, 0, sComm);
                Epetra_CrsMatrix L_e_PLUS(Epetra_DataAccess::Copy, LePLUSRowMap, LeTRowMap, 1);
                double posOne = 1.0;

                for(int i = 0; i < n_dofs_curr_cell; i++){
                    const int col = current_dofs_indices[i];
                    L_e.InsertGlobalValues(i, 1, &posOne , &col);
                    L_e_T.InsertGlobalValues(col, 1, &posOne , &i);
                }
                L_e.FillComplete(LeTRowMap, LeRowMap);
                L_e_T.FillComplete(LeRowMap, LeTRowMap);

                for(int i = 0; i < neighbour_dofs_curr_cell; i++){
                    const int col = neighbour_dofs_indices[i];
                    L_e_PLUS.InsertGlobalValues(i, 1, &posOne , &col);
                }
                L_e_PLUS.FillComplete(LeTRowMap, LePLUSRowMap);

                // Find contribution of element to the Jacobian
                Epetra_CrsMatrix J_L_e_T(Epetra_DataAccess::Copy, local_system_matrix.RowMap(), neighbour_dofs_curr_cell);
                Epetra_CrsMatrix J_e_m(Epetra_DataAccess::Copy, LeRowMap, neighbour_dofs_curr_cell);
                EpetraExt::MatrixMatrix::Multiply(local_system_matrix, false, L_e_PLUS, true, J_L_e_T, true);
                EpetraExt::MatrixMatrix::Multiply(L_e, false, J_L_e_T, false, J_e_m, true);

                // Jacobian for this element in the global dimensions
                Epetra_CrsMatrix J_temp(Epetra_DataAccess::Copy, LeRowMap, N);
                Epetra_CrsMatrix J_global_e(Epetra_DataAccess::Copy, LeTRowMap, N);
                EpetraExt::MatrixMatrix::Multiply(J_e_m, false, L_e_PLUS, false, J_temp, true);
                EpetraExt::MatrixMatrix::Multiply(L_e_T, false, J_temp, false, J_global_e, true);

                // Add the contribution of the element to the hyper-reduced Jacobian with scaling from the weights
                double scaling = ECSW_weights[local_element];
                EpetraExt::MatrixMatrix::Add(J_global_e, false, scaling, reduced_jacobian, 1.0);
            }
        }
    }
    reduced_jacobian.FillComplete();
    return std::make_shared<Epetra_CrsMatrix>(reduced_jacobian);
}

template <int dim, int nstate>
std::shared_ptr<Epetra_CrsMatrix> HROMTestLocation<dim, nstate>::generate_test_basis(Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis)
{
    Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
    Epetra_CrsMatrix petrov_galerkin_basis(Epetra_DataAccess::Copy, system_matrix_rowmap, pod_basis.NumGlobalCols());
    EpetraExt::MatrixMatrix::Multiply(system_matrix, false, pod_basis, false, petrov_galerkin_basis, true);

    return std::make_shared<Epetra_CrsMatrix>(petrov_galerkin_basis);
}

#if PHILIP_DIM==1
        template class HROMTestLocation<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HROMTestLocation<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}
}