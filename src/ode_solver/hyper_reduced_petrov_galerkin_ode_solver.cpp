#include "hyper_reduced_petrov_galerkin_ode_solver.h"
#include "linear_solver/linear_solver.h"
#include "reduced_order/multi_core_helper_functions.h"
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <Epetra_Vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <Epetra_LinearProblem.h>
#include "Amesos_BaseSolver.h"
#include <Amesos_Lapack.h>
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_SerialComm.h>
#include <Epetra_Comm.h>
#include <Epetra_Export.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
HyperReducedODESolver<dim,real,MeshType>::HyperReducedODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, Epetra_Vector weights)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , pod(pod)
        , ECSW_weights(weights)
{}

template <int dim, typename real, typename MeshType>
int HyperReducedODESolver<dim,real,MeshType>::steady_state ()
{
    this->pcout << " Performing steady state analysis... " << std::endl;

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose
        && (this->current_iteration%this->ode_param.print_iteration_modulo) == 0)
    {
        this->pcout.set_condition(true);
    } else {
        this->pcout.set_condition(false);
    }

    this->current_iteration = 0;

    this->pcout << " Evaluating right-hand side and setting system_matrix to Jacobian before starting iterations... " << std::endl;
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);

    // Build hyper-reduced Jacobian
    const Epetra_CrsMatrix epetra_system_matrix = this->dg->system_matrix.trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> reduced_system_matrix = generate_hyper_reduced_jacobian(epetra_system_matrix);

    // Find test basis W with hyper-reduced Jacobian
    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = generate_test_basis(*reduced_system_matrix, epetra_pod_basis);

    // Build hyper-reduced residual
    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_system_matrix.RowMap(), this->dg->right_hand_side.begin());
    std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);

    hyper_reduced_rhs->Norm2(&this->initial_residual_norm);
    this->initial_residual_norm /= this->dg->right_hand_side.size();

    this->pcout << " ********************************************************** "
                << std::endl
                << " Initial absolute residual norm: " << this->initial_residual_norm
                << std::endl;

    this->residual_norm = 1;

    while (this->residual_norm > this->all_parameters->reduced_order_param.reduced_residual_tolerance && this->current_iteration < this->ode_param.nonlinear_max_iterations)
    {
        if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose
            && (this->current_iteration%this->ode_param.print_iteration_modulo) == 0
            && dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0 )
        {
            this->pcout.set_condition(true);
        } else {
            this->pcout.set_condition(false);
        }
        this->pcout << " ********************************************************** "
                    << std::endl
                    << " Nonlinear iteration: " << this->current_iteration
                    << " Residual norm (normalized) : " << this->residual_norm
                    << std::endl;

        if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
            (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
            this->pcout << " Evaluating right-hand side and setting system_matrix to Jacobian... " << std::endl;
        }

        //Dummy CFL and pseudotime
        double ramped_CFL = 0;
        const bool pseudotime = true;
        step_in_time(ramped_CFL, pseudotime);

        if (this->ode_param.output_solution_every_x_steps > 0) {
            const bool is_output_iteration = (this->current_iteration % this->ode_param.output_solution_every_x_steps == 0);
            if (is_output_iteration) {
                const int file_number = this->current_iteration / this->ode_param.output_solution_every_x_steps;
                this->dg->output_results_vtk(file_number);
            }
        }
    }

    this->pcout << " ********************************************************** "
                << std::endl
                << " ODESolver steady_state stopped at"
                << std::endl
                << " Nonlinear iteration: " << this->current_iteration
                << " residual norm: " << this->residual_norm
                << std::endl
                << " ********************************************************** "
                << std::endl;

    return 0;
}

template <int dim, typename real, typename MeshType>
void HyperReducedODESolver<dim,real,MeshType>::step_in_time (real /*dt*/, const bool /*pseudotime*/)
{
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Evaluating system update... " << std::endl;
    }
    // Build hyperreduced Jacobian
    Epetra_CrsMatrix epetra_system_matrix = this->dg->system_matrix.trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> reduced_system_matrix = generate_hyper_reduced_jacobian(epetra_system_matrix);

    // Find test basis W with hyperreduced Jacobian
    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = generate_test_basis(*reduced_system_matrix, epetra_pod_basis);

    // Build hyperreduced residual
    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_system_matrix.RowMap(), this->dg->right_hand_side.begin());
    std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
    hyper_reduced_rhs->Scale(-1.0);

    // Form (A p^k = b) where A = W^T * W and b = - W^T * R
    std::shared_ptr<Epetra_CrsMatrix> epetra_reduced_lhs = generate_reduced_lhs(*epetra_test_basis);
    Epetra_Vector epetra_reduced_solution_update(epetra_reduced_lhs->DomainMap());
    Epetra_LinearProblem linearProblem(epetra_reduced_lhs.get(), &epetra_reduced_solution_update, hyper_reduced_rhs.get());

    Amesos_Lapack Solver(linearProblem);
    Teuchos::ParameterList List;
    Solver.SetParameters(List);
    Solver.SymbolicFactorization();
    Solver.NumericFactorization();
    Solver.Solve();

    const dealii::LinearAlgebra::distributed::Vector<double> old_solution(this->dg->solution);
    
    // Line search parameters (currently identical to reduced-order ode solver values)
    double step_length = 1.0;
    const double step_reduction = 0.5;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;
    const double reduction_tolerance_2 = 2.0;

    double initial_residual;
    hyper_reduced_rhs->Norm2(&initial_residual);
    initial_residual /= this->dg->right_hand_side.size();

    Epetra_Vector epetra_solution(Epetra_DataAccess::View, epetra_pod_basis.RangeMap(), this->dg->solution.begin());
    Epetra_Vector epetra_solution_update(epetra_pod_basis.RangeMap());

    epetra_pod_basis.Multiply(false, epetra_reduced_solution_update, epetra_solution_update);
    epetra_solution.Update(1, epetra_solution_update, 1);
    this->dg->assemble_residual();
    hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
    double new_residual;
    hyper_reduced_rhs->Norm2(&new_residual);
    new_residual /= this->dg->right_hand_side.size();

    // Note that line search is in the same function to avoid having to recompute test basis in a new function or store it as a member variable

    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        this->dg->solution = old_solution;
        Epetra_Vector epetra_linesearch_reduced_solution_update(epetra_reduced_solution_update);
        epetra_linesearch_reduced_solution_update.Scale(step_length);
        epetra_pod_basis.Multiply(false, epetra_linesearch_reduced_solution_update, epetra_solution_update);
        epetra_solution.Update(1, epetra_solution_update, 1);
        this->dg->assemble_residual();
        std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
        hyper_reduced_rhs->Norm2(&new_residual);
        new_residual /= this->dg->right_hand_side.size();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }


    if (iline == maxline) {
        step_length = 1.0;
        this->pcout << " Line Search (Case 2): Increase nonlinear residual tolerance by a factor " << std::endl;
        this->pcout << " Line search failed. Will accept any valid residual less than " << reduction_tolerance_2 << " times the current " << initial_residual << "residual. " << std::endl;
        epetra_solution.Update(1, epetra_solution_update, 1);
        this->dg->assemble_residual();
        std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
        hyper_reduced_rhs->Norm2(&new_residual);
        new_residual /= this->dg->right_hand_side.size();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            Epetra_Vector epetra_linesearch_reduced_solution_update(epetra_reduced_solution_update);
            epetra_linesearch_reduced_solution_update.Scale(step_length);
            epetra_pod_basis.Multiply(false, epetra_linesearch_reduced_solution_update, epetra_solution_update);
            epetra_solution.Update(1, epetra_solution_update, 1);
            this->dg->assemble_residual();
            std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
            hyper_reduced_rhs->Norm2(&new_residual);
            new_residual /= this->dg->right_hand_side.size();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }

    if (iline == maxline) {
        this->pcout << " Line Search (Case 3): Reverse Search Direction " << std::endl;
        step_length = -1.0;
        epetra_solution.Update(-1, epetra_solution_update, 1);
        this->dg->assemble_residual();
        std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
        hyper_reduced_rhs->Norm2(&new_residual);
        new_residual /= this->dg->right_hand_side.size();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            Epetra_Vector epetra_linesearch_reduced_solution_update(epetra_reduced_solution_update);
            epetra_linesearch_reduced_solution_update.Scale(step_length);
            epetra_pod_basis.Multiply(false, epetra_linesearch_reduced_solution_update, epetra_solution_update);
            epetra_solution.Update(1, epetra_solution_update, 1);
            this->dg->assemble_residual();
            std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
            hyper_reduced_rhs->Norm2(&new_residual);
            new_residual /= this->dg->right_hand_side.size();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }

    if (iline == maxline) {
        this->pcout << " Line Search (Case 4): Reverse Search Direction AND Increase nonlinear residual tolerance by a factor " << std::endl;
        step_length = -1.0;
        epetra_solution.Update(-1, epetra_solution_update, 1);
        this->dg->assemble_residual();
        std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
        hyper_reduced_rhs->Norm2(&new_residual);
        new_residual /= this->dg->right_hand_side.size();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            Epetra_Vector epetra_linesearch_reduced_solution_update(epetra_reduced_solution_update);
            epetra_linesearch_reduced_solution_update.Scale(step_length);
            epetra_pod_basis.Multiply(false, epetra_linesearch_reduced_solution_update, epetra_solution_update);
            epetra_solution.Update(1, epetra_solution_update, 1);
            this->dg->assemble_residual();
            std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
            hyper_reduced_rhs->Norm2(&new_residual);
            new_residual /= this->dg->right_hand_side.size();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }

    if(iline == maxline){
        this->pcout << "Line search failed. Returning to old solution." << std::endl;
        this->dg->solution = old_solution;
    }

    this->pcout << "Full-order residual norm: " << this->dg->get_residual_l2norm() << std::endl;

    this->residual_norm = new_residual;

    ++(this->current_iteration);
}

template <int dim, typename real, typename MeshType>
void HyperReducedODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    /*Projection of initial conditions on reduced-order subspace, refer to Equation 19 in:
    Washabaugh, K. M., Zahr, M. J., & Farhat, C. (2016).
    On the use of discrete nonlinear reduced-order models for the prediction of steady-state flows past parametrically deformed complex geometries.
    In 54th AIAA Aerospace Sciences Meeting (p. 1814).
    */
    this->pcout << "Allocating ODE system..." << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> reference_solution(this->dg->solution);
    reference_solution.import(pod->getReferenceState(), dealii::VectorOperation::values::insert);

    dealii::LinearAlgebra::distributed::Vector<double> initial_condition(this->dg->solution);
    initial_condition -= reference_solution;

    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    Epetra_Vector epetra_reduced_solution(epetra_pod_basis.DomainMap());
    Epetra_Vector epetra_initial_condition(Epetra_DataAccess::Copy, epetra_pod_basis.RangeMap(), initial_condition.begin());

    epetra_pod_basis.Multiply(true, epetra_initial_condition, epetra_reduced_solution);

    Epetra_Vector epetra_projection_tmp(epetra_pod_basis.RangeMap());
    epetra_pod_basis.Multiply(false, epetra_reduced_solution, epetra_projection_tmp);

    Epetra_Vector epetra_solution(Epetra_DataAccess::View, epetra_pod_basis.RangeMap(), this->dg->solution.begin());

    epetra_solution = epetra_projection_tmp;
    this->dg->solution += reference_solution;
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedODESolver<dim,real,MeshType>::generate_test_basis(Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis)
{
    Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
    Epetra_CrsMatrix petrov_galerkin_basis(Epetra_DataAccess::Copy, system_matrix_rowmap, pod_basis.NumGlobalCols());
    EpetraExt::MatrixMatrix::Multiply(system_matrix, false, pod_basis, false, petrov_galerkin_basis, true);

    return std::make_shared<Epetra_CrsMatrix>(petrov_galerkin_basis);
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedODESolver<dim,real,MeshType>::generate_hyper_reduced_jacobian(const Epetra_CrsMatrix &system_matrix)
{
    /* Refer to Equation (12) in:
    https://onlinelibrary.wiley.com/doi/10.1002/nme.6603 (includes definitions of matrices used below such as L_e and L_e_PLUS)
    Create empty Hyper-reduced Jacobian Epetra structure */
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
    Epetra_CrsMatrix local_system_matrix = copy_matrix_to_all_cores(system_matrix);
    Epetra_CrsMatrix reduced_jacobian(Epetra_DataAccess::Copy, system_matrix_rowmap, system_matrix.NumGlobalCols());
    const int N = system_matrix.NumGlobalRows();
    Epetra_BlockMap element_map = ECSW_weights.Map();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbour_dofs_indices(max_dofs_per_cell);

    // Loop through elements 
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        // Add the contributions of an element if the weight from the NNLS is non-zero
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

template <int dim, typename real, typename MeshType>
std::shared_ptr<Epetra_Vector> HyperReducedODESolver<dim,real,MeshType>::generate_hyper_reduced_residual(Epetra_Vector epetra_right_hand_side, Epetra_CrsMatrix &test_basis)
{
    /* Refer to Equation (10) in:
    https://onlinelibrary.wiley.com/doi/10.1002/nme.6603 (includes definitions of matrices used below such as L_e and L_e_PLUS)   
    Create empty Hyper-reduced residual Epetra structure */
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map test_basis_colmap = test_basis.DomainMap();
    int *global_ind = new int[test_basis.NumGlobalCols()];
    for (int i = 0; i < test_basis.NumGlobalCols(); i++){
        global_ind[i] = i;
    }
    Epetra_Map POD_dim (-1,test_basis.NumGlobalCols(), global_ind, 0, epetra_comm);
    delete[] global_ind;
    Epetra_Vector hyper_reduced_residual(POD_dim);
    const int N = test_basis.NumGlobalRows();
    Epetra_BlockMap element_map = ECSW_weights.Map();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    Epetra_Vector local_rhs = copy_vector_to_all_cores(epetra_right_hand_side);
    Epetra_CrsMatrix local_test_basis = copy_matrix_to_all_cores(test_basis);

    // Loop through elements
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned()){
            // Add the contributions of an element if the weight from the NNLS is non-zero
            const int global = cell->active_cell_index();
            const int local_element = element_map.LID(global);
            if (ECSW_weights[local_element] != 0){
                const int fe_index_curr_cell = cell->active_fe_index();
                const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
                const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

                current_dofs_indices.resize(n_dofs_curr_cell);
                cell->get_dof_indices(current_dofs_indices);

                // Create L_e matrix for current cell
                const Epetra_SerialComm sComm;
                Epetra_Map LeRowMap(n_dofs_curr_cell,n_dofs_curr_cell, 0, sComm);
                Epetra_Map LeTRowMap(N, 0, sComm);
                Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, LeTRowMap, 1);
                double posOne = 1.0;

                for(int i = 0; i < n_dofs_curr_cell; i++){
                    const int col = current_dofs_indices[i];
                    L_e.InsertGlobalValues(i, 1, &posOne , &col);
                }
                L_e.FillComplete(LeTRowMap, LeRowMap);

                // Find contribution of the current element in the global dimensions
                Epetra_Vector local_r(LeRowMap);
                Epetra_Vector global_r_e(LeTRowMap);
                L_e.Multiply(false, local_rhs, local_r);
                L_e.Multiply(true, local_r, global_r_e);

                // Find reduced representation of residual and scale by weight
                Epetra_Map POD_local (test_basis.NumGlobalCols(), test_basis.NumGlobalCols(), 0, sComm);
                Epetra_Vector reduced_rhs_e(POD_local);
                local_test_basis.Multiply(true, global_r_e, reduced_rhs_e);
                reduced_rhs_e.Scale(ECSW_weights[local_element]);
                double *reduced_rhs_array = new double[reduced_rhs_e.MyLength()];

                // Add to hyper-reduced representation of the residual
                reduced_rhs_e.ExtractCopy(reduced_rhs_array);
                for (int k = 0; k < reduced_rhs_e.MyLength(); ++k){
                    hyper_reduced_residual.SumIntoMyValues(1, &reduced_rhs_array[k], &k);
                }
                delete[] reduced_rhs_array;
            }
        }
    }
    Epetra_BlockMap old_map_b = hyper_reduced_residual.Map();
    Epetra_Export b_importer(old_map_b, test_basis_colmap);
    Epetra_Vector dist_hyper_res (test_basis_colmap); 
    dist_hyper_res.Export(hyper_reduced_residual, b_importer, Add);
    return std::make_shared<Epetra_Vector>(dist_hyper_res);
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedODESolver<dim,real,MeshType>::generate_reduced_lhs(Epetra_CrsMatrix &test_basis)
{
    Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
    EpetraExt::MatrixMatrix::Multiply(test_basis, true, test_basis, false, epetra_reduced_lhs);

    return std::make_shared<Epetra_CrsMatrix>(epetra_reduced_lhs);
}

template class HyperReducedODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class HyperReducedODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
    template class HyperReducedODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace//