#include "hrom_test_location.h"
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
HROMTestLocation<dim, nstate>::HROMTestLocation(const RowVectorXd& parameter, std::unique_ptr<ROMSolution<dim, nstate>> rom_solution, std::shared_ptr< DGBase<dim, double> > dg_input, Epetra_Vector weights)
        : parameter(parameter)
        , rom_solution(std::move(rom_solution))
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
        , dg(dg_input)
        , xi(weights)
{
    pcout << "Creating ROM test location..." << std::endl;
    compute_FOM_to_initial_ROM_error();
    initial_rom_to_final_rom_error = 0;
    total_error = fom_to_initial_rom_error;

    pcout << "ROM test location created. Error estimate updated." << std::endl;
}

template <int dim, int nstate>
void HROMTestLocation<dim, nstate>::compute_FOM_to_initial_ROM_error(){
    pcout << "Computing adjoint-based error estimate between ROM and FOM..." << std::endl;

    dealii::ParameterHandler dummy_handler;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&rom_solution->params, dummy_handler);
    flow_solver->dg->solution = rom_solution->solution;
    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);
    dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose = dealii::TrilinosWrappers::SparseMatrix();
    system_matrix_transpose.copy_from(flow_solver->dg->system_matrix_transpose);

    // Initialize with same parallel layout as dg->right_hand_side
    dealii::LinearAlgebra::distributed::Vector<double> adjoint(flow_solver->dg->right_hand_side);

    dealii::LinearAlgebra::distributed::Vector<double> gradient(rom_solution->gradient);

    Parameters::LinearSolverParam linear_solver_param;

    linear_solver_param.max_iterations = 1000;
    linear_solver_param.restart_number = 200;
    linear_solver_param.linear_residual = 1e-17;
    linear_solver_param.ilut_fill = 50;
    linear_solver_param.ilut_drop = 1e-8;
    linear_solver_param.ilut_atol = 1e-5;
    linear_solver_param.ilut_rtol = 1.0+1e-2;
    //linear_solver_param.linear_solver_output = Parameters::OutputEnum::verbose;
    linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;

    //linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::direct;
    solve_linear(system_matrix_transpose, gradient*=-1.0, adjoint, linear_solver_param);

    //Compute dual weighted residual
    fom_to_initial_rom_error = 0;
    fom_to_initial_rom_error = -(adjoint * flow_solver->dg->right_hand_side);

    pcout << "Parameter: " << parameter << ". Error estimate between ROM and FOM: " << fom_to_initial_rom_error << std::endl;
}

template <int dim, int nstate>
void HROMTestLocation<dim, nstate>::compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod_updated){

    pcout << "Computing adjoint-based error estimate between initial ROM and updated ROM..." << std::endl;

    dealii::ParameterHandler dummy_handler;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&rom_solution->params, dummy_handler);
    flow_solver->dg->solution = rom_solution->solution;
    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);

    // Build hyperreduced Jacobian
    Epetra_CrsMatrix epetra_system_matrix = flow_solver->dg->system_matrix.trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> reduced_system_matrix = generate_hyper_reduced_jacobian(epetra_system_matrix);

    // Find test basis W with hyperreduced Jacobian
    const Epetra_CrsMatrix epetra_pod_basis = pod_updated->getPODBasis()->trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> epetra_petrov_galerkin_basis_ptr = generate_test_basis(*reduced_system_matrix, epetra_pod_basis);
    Epetra_CrsMatrix epetra_petrov_galerkin_basis = *epetra_petrov_galerkin_basis_ptr;

    Epetra_Vector epetra_gradient(Epetra_DataAccess::Copy, epetra_pod_basis.RowMap(), const_cast<double *>(rom_solution->gradient.begin()));
    Epetra_Vector epetra_reduced_gradient(epetra_pod_basis.DomainMap());

    epetra_pod_basis.Multiply(true, epetra_gradient, epetra_reduced_gradient);

    Epetra_CrsMatrix epetra_reduced_jacobian_transpose(Epetra_DataAccess::Copy, epetra_petrov_galerkin_basis.DomainMap(), pod_updated->getPODBasis()->n());
    EpetraExt::MatrixMatrix::Multiply(epetra_petrov_galerkin_basis, true, epetra_petrov_galerkin_basis, false, epetra_reduced_jacobian_transpose);

    Epetra_Vector epetra_reduced_adjoint(epetra_reduced_jacobian_transpose.DomainMap());
    epetra_reduced_gradient.Scale(-1);
    Epetra_LinearProblem linearProblem(&epetra_reduced_jacobian_transpose, &epetra_reduced_adjoint, &epetra_reduced_gradient);

    Amesos_Lapack Solver(linearProblem);

    Teuchos::ParameterList List;
    Solver.SetParameters(List);
    Solver.SymbolicFactorization();
    Solver.NumericFactorization();
    Solver.Solve();

    Epetra_Vector epetra_reduced_residual(epetra_petrov_galerkin_basis.DomainMap());
    Epetra_Vector epetra_residual(Epetra_DataAccess::Copy, epetra_petrov_galerkin_basis.RangeMap(), const_cast<double *>(flow_solver->dg->right_hand_side.begin()));
    epetra_petrov_galerkin_basis.Multiply(true, epetra_residual, epetra_reduced_residual);

    //Compute dual weighted residual
    initial_rom_to_final_rom_error = 0;
    epetra_reduced_adjoint.Dot(epetra_reduced_residual, &initial_rom_to_final_rom_error);
    initial_rom_to_final_rom_error *= -1;

    pcout << "Parameter: " << parameter << ". Error estimate between initial ROM and updated ROM: " << initial_rom_to_final_rom_error << std::endl;
}

template <int dim, int nstate>
void HROMTestLocation<dim, nstate>::compute_total_error(){
    pcout << "Computing total error estimate between FOM and updated ROM..." << std::endl;
    total_error = fom_to_initial_rom_error - initial_rom_to_final_rom_error;
    pcout << "Parameter: " << parameter <<  ". Total error estimate between FOM and updated ROM: " << total_error << std::endl;
}

template <int dim, int nstate>
std::shared_ptr<Epetra_CrsMatrix> HROMTestLocation<dim, nstate>::generate_hyper_reduced_jacobian(const Epetra_CrsMatrix &system_matrix)
{
    // NOTE: ASSUMING L_e matrix is equal to the L_e_plus matrix (i.e. no neighbouring elements considered in stencil)
    // Create empty Hyper-reduced Jacobian Epetra structure
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
    Epetra_CrsMatrix reduced_jacobian(Epetra_DataAccess::Copy, system_matrix_rowmap, system_matrix.NumGlobalCols());
    int N = system_matrix.NumGlobalRows();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbour_dofs_indices(max_dofs_per_cell);

    // Loop through elements 
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        // Add the contributilons of an element if the weight from the NNLS is non-zero
        if (xi[cell->active_cell_index()] != 0){
            const int fe_index_curr_cell = cell->active_fe_index();
            const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
            const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

            current_dofs_indices.resize(n_dofs_curr_cell);
            cell->get_dof_indices(current_dofs_indices);

            double *row = new double[system_matrix.NumGlobalCols()];
            int *global_indices = new int[system_matrix.NumGlobalCols()];
            int numE;
            int row_num = current_dofs_indices[0];
            system_matrix.ExtractGlobalRowCopy(row_num, system_matrix.NumGlobalCols(), numE, row, global_indices);
            int neighbour_dofs_curr_cell = 0;
            for (int i = 0; i < numE; i++){
                neighbour_dofs_curr_cell +=1;
                neighbour_dofs_indices.resize(neighbour_dofs_curr_cell);
                neighbour_dofs_indices[neighbour_dofs_curr_cell-1] = global_indices[i];
            }

            // Create L_e matrix and transposed L_e matrixfor current cell
            Epetra_Map LeRowMap(n_dofs_curr_cell, 0, epetra_comm);
            Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, N);
            Epetra_CrsMatrix L_e_T(Epetra_DataAccess::Copy, system_matrix_rowmap, n_dofs_curr_cell);
            Epetra_Map LePLUSRowMap(neighbour_dofs_curr_cell, 0, epetra_comm);
            Epetra_CrsMatrix L_e_PLUS(Epetra_DataAccess::Copy, LePLUSRowMap, N);
            double posOne = 1.0;

            for(int i = 0; i < n_dofs_curr_cell; i++){
                const int col = current_dofs_indices[i];
                L_e.InsertGlobalValues(i, 1, &posOne , &col);
                L_e_T.InsertGlobalValues(col, 1, &posOne , &i);
            }
            L_e.FillComplete(system_matrix_rowmap, LeRowMap);
            L_e_T.FillComplete(LeRowMap, system_matrix_rowmap);

            for(int i = 0; i < neighbour_dofs_curr_cell; i++){
                const int col = neighbour_dofs_indices[i];
                L_e_PLUS.InsertGlobalValues(i, 1, &posOne , &col);
            }
            L_e_PLUS.FillComplete(system_matrix_rowmap, LePLUSRowMap);

            // Find contribution of element to the Jacobian
            Epetra_CrsMatrix J_L_e_T(Epetra_DataAccess::Copy, system_matrix_rowmap, neighbour_dofs_curr_cell);
            Epetra_CrsMatrix J_e_m(Epetra_DataAccess::Copy, LeRowMap, neighbour_dofs_curr_cell);
            EpetraExt::MatrixMatrix::Multiply(system_matrix, false, L_e_PLUS, true, J_L_e_T, true);
            EpetraExt::MatrixMatrix::Multiply(L_e, false, J_L_e_T, false, J_e_m, true);

            // Jacobian for this element in the global dimensions
            Epetra_CrsMatrix J_temp(Epetra_DataAccess::Copy, LeRowMap, N);
            Epetra_CrsMatrix J_global_e(Epetra_DataAccess::Copy, system_matrix_rowmap, N);
            EpetraExt::MatrixMatrix::Multiply(J_e_m, false, L_e_PLUS, false, J_temp, true);
            EpetraExt::MatrixMatrix::Multiply(L_e_T, false, J_temp, false, J_global_e, true);

            // Add the contribution of the element to the hyper-reduced Jacobian with scaling from the weights
            double scaling = xi[cell->active_cell_index()];
            EpetraExt::MatrixMatrix::Add(J_global_e, false, scaling, reduced_jacobian, 1.0);
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