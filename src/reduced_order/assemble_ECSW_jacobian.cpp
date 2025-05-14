#include "assemble_ECSW_jacobian.h"
#include <iostream>

#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;

template <int dim, int nstate>
AssembleECSWJac<dim,nstate>::AssembleECSWJac(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input,
    std::shared_ptr<DGBase<dim,double>> &dg_input, 
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, 
    MatrixXd snapshot_parameters_input,
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type,
    Epetra_MpiComm &Comm)
        : AssembleECSWBase<dim, nstate>(parameters_input, parameter_handler_input, dg_input, pod, snapshot_parameters_input, ode_solver_type, Comm)
{
}

template <int dim, int nstate>
void AssembleECSWJac<dim,nstate>::build_problem(){
    this->pcout << "Solve for A and b for the NNLS Problem from POD Snapshots"<< std::endl;
    MatrixXd snapshotMatrix = this->pod->getSnapshotMatrix();
    const Epetra_CrsMatrix epetra_pod_basis = this->pod->getPODBasis()->trilinos_matrix();
    Epetra_CrsMatrix epetra_system_matrix = this->dg->system_matrix.trilinos_matrix();
    Epetra_Map system_matrix_rowmap = epetra_system_matrix.RowMap();
    Epetra_CrsMatrix local_system_matrix = copy_matrix_to_all_cores(epetra_system_matrix);

    // Get dimensions of the problem
    int num_snaps_POD = snapshotMatrix.cols(); // Number of snapshots used to build the POD basis
    int n_reduced_dim_POD = epetra_pod_basis.NumGlobalCols(); // Reduced subspace dimension
    int N_FOM_dim = epetra_pod_basis.NumGlobalRows(); // Length of solution vector
    int num_elements_N_e = this->dg->triangulation->n_active_cells(); // Number of elements (equal to N if there is one DOF per cell)

    // Create empty and temporary C and d structs
    int training_snaps;
    // Check if all or a subset of the snapshots will be used for training
    if (this->all_parameters->hyper_reduction_param.num_training_snaps != 0) {
        this->pcout << "LIMITED NUMBER OF TRAINING SNAPSHOTS" << std::endl;
        training_snaps = this->all_parameters->hyper_reduction_param.num_training_snaps;
    }
    else{
        training_snaps = num_snaps_POD;
    }
    const int rank = this->Comm_.MyPID();
    const int n_quad_pts = this->dg->volume_quadrature_collection[this->all_parameters->flow_solver_param.poly_degree].size();
    const int length = epetra_system_matrix.NumMyRows()/(nstate*n_quad_pts);
    int *local_elements = new int[length];
    int ctr = 0;
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned()){
            local_elements[ctr] = cell->active_cell_index();
            ctr +=1;
        }
    }

    Epetra_Map RowMap((n_reduced_dim_POD*n_reduced_dim_POD*training_snaps),(n_reduced_dim_POD*n_reduced_dim_POD*training_snaps), 0, this->Comm_); // Number of rows in Jacobian based training matrix = n^2 * (number of training snapshots)
    Epetra_Map ColMap(num_elements_N_e, length, local_elements, 0, this->Comm_);
    Epetra_Map dMap((n_reduced_dim_POD*n_reduced_dim_POD*training_snaps), (rank == 0) ?  (n_reduced_dim_POD*n_reduced_dim_POD*training_snaps) : 0,  0, this->Comm_);

    delete[] local_elements;

    Epetra_CrsMatrix C_T(Epetra_DataAccess::Copy, ColMap, RowMap, num_elements_N_e);
    Epetra_Vector d(dMap);

    // Loop through the given number of training snapshots to find Jacobian values
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbour_dofs_indices(max_dofs_per_cell); 
    int row_num = 0;
    int snap_num = 0;
    for(auto snap_param : this->snapshot_parameters.rowwise()){
        this->pcout << "Snapshot Parameter Values" << std::endl;
        this->pcout << snap_param << std::endl;

        // Modifiy parameters for snapshot location and create new flow solver
        Parameters::AllParameters params = this->reinit_params(snap_param);
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, this->parameter_handler);
        this->dg = flow_solver->dg;

        // Set solution to snapshot and re-compute the residual/Jacobian
        this->dg->solution = this->fom_locations[snap_num];
        const bool compute_dRdW = true;
        this->dg->assemble_residual(compute_dRdW);

        // Compute test basis
        epetra_system_matrix = this->dg->system_matrix.trilinos_matrix(); // Jacobian at snapshot location
        system_matrix_rowmap = epetra_system_matrix.RowMap();
        std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = this->local_generate_test_basis(epetra_system_matrix, epetra_pod_basis);
        Epetra_CrsMatrix local_system_matrix = copy_matrix_to_all_cores(epetra_system_matrix);
        Epetra_CrsMatrix local_pod_basis = copy_matrix_to_all_cores(epetra_pod_basis);
        Epetra_CrsMatrix local_test_basis = copy_matrix_to_all_cores(*epetra_test_basis);

        // Loop through elements
        for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned()){
                int cell_num = cell->active_cell_index();
                double *row = new double[local_system_matrix.NumGlobalCols()];
                int *global_indices = new int[local_system_matrix.NumGlobalCols()];

                // Create L_e matrix and transposed L_e matrix for current cell
                const int fe_index_curr_cell = cell->active_fe_index();
                const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
                const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

                current_dofs_indices.resize(n_dofs_curr_cell);
                cell->get_dof_indices(current_dofs_indices);

                int numE;
                int row_i = current_dofs_indices[0];
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

                const Epetra_SerialComm sComm;
                Epetra_Map LeRowMap(n_dofs_curr_cell, 0, sComm);
                Epetra_Map LeTRowMap(N_FOM_dim, 0, sComm);
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
                Epetra_CrsMatrix J_temp(Epetra_DataAccess::Copy, LeRowMap, N_FOM_dim);
                Epetra_CrsMatrix J_global_e(Epetra_DataAccess::Copy, LeTRowMap, N_FOM_dim);
                EpetraExt::MatrixMatrix::Multiply(J_e_m, false, L_e_PLUS, false, J_temp, true);
                EpetraExt::MatrixMatrix::Multiply(L_e_T, false, J_temp, false, J_global_e, true);

                // Post-multiply by the ROB V
                Epetra_CrsMatrix J_e_V(Epetra_DataAccess::Copy, LeTRowMap, n_reduced_dim_POD);
                EpetraExt::MatrixMatrix::Multiply(J_global_e, false, local_pod_basis, false, J_e_V, true);

                // Assemble the transpose of the test basis
                Epetra_CrsMatrix W_T(Epetra_DataAccess::Copy, local_test_basis.ColMap(), N_FOM_dim);
                for(int i =0; i < local_test_basis.NumGlobalRows(); i++){
                    double *row = new double[local_test_basis.NumGlobalCols()];
                    int *global_cols = new int[local_test_basis.NumGlobalCols()];
                    int numE;
                    const int globalRow = local_test_basis.GRID(i);
                    local_test_basis.ExtractGlobalRowCopy(globalRow, local_test_basis.NumGlobalCols(), numE , row, global_cols);
                    for(int j = 0; j < numE; j++){
                        int col = global_cols[j];
                        W_T.InsertGlobalValues(col, 1, &row[j], &i);
                    }
                    delete[] row;
                    delete[] global_cols;
                }
                W_T.FillComplete(local_test_basis.RowMap(), local_test_basis.ColMap());

                // Pre-multiply by the tranpose of the test basis
                Epetra_CrsMatrix W_T_J_e_V(Epetra_DataAccess::Copy, W_T.RowMap(), n_reduced_dim_POD);
                EpetraExt::MatrixMatrix::Multiply(W_T, false, J_e_V, false, W_T_J_e_V , true);

                // Stack into n^2 vector
                Epetra_Map cseRowMap(n_reduced_dim_POD*n_reduced_dim_POD, 0, sComm);
                Epetra_Vector c_se(cseRowMap);
                for(int i =0; i < W_T_J_e_V.NumGlobalRows(); i++){
                    double *row = new double[W_T_J_e_V.NumGlobalCols()];
                    int *global_cols = new int[epetra_system_matrix.NumGlobalCols()];
                    int numE;
                    const int globalRow = W_T_J_e_V.GRID(i);
                    W_T_J_e_V.ExtractGlobalRowCopy(globalRow, W_T_J_e_V.NumGlobalCols(), numE , row, global_cols);
                    for(int j = 0; j < numE; j++){
                        int col = global_cols[j];
                        int idx = col*n_reduced_dim_POD + i; 
                        c_se[idx] = row[j];
                    }
                    delete[] row;
                    delete[] global_cols;
                }

                double *c_se_array = new double[n_reduced_dim_POD*n_reduced_dim_POD];

                c_se.ExtractCopy(c_se_array);
            
                // Sub into entries of C and d
                for (int k = 0; k < (n_reduced_dim_POD*n_reduced_dim_POD); ++k){
                    int place = row_num+k;
                    C_T.InsertGlobalValues(cell_num, 1, &c_se_array[k], &place);
                }
                delete[] c_se_array; 
            }     
        }
        row_num+=(n_reduced_dim_POD*n_reduced_dim_POD);
        snap_num+=1;
        
        // Check if number of training snapshots has been reached
        if (this->all_parameters->hyper_reduction_param.num_training_snaps != 0) {
            this->pcout << "LIMITED NUMBER OF SNAPSHOTS"<< std::endl;
            if (snap_num > (this->all_parameters->hyper_reduction_param.num_training_snaps-1)){
                break;
            }
        }
    }

    C_T.FillComplete(RowMap, ColMap);

    Epetra_CrsMatrix C_single = copy_matrix_to_all_cores(C_T);
    for (int p = 0; p < num_elements_N_e; p++){
        double *row = new double[C_single.NumGlobalCols()];
        int *global_cols = new int[C_single.NumGlobalCols()];
        int numE;
        C_single.ExtractGlobalRowCopy(p, C_single.NumGlobalCols(), numE , row, global_cols);
        for (int o = 0; o < dMap.NumMyElements(); o++){
            int col = dMap.GID(o);
            d.SumIntoMyValues(1, &row[col], &o);
        }
        delete[] row;
        delete[] global_cols;
    }

    // Sub temp C and d into class A and b
    this->A_T->reinit(C_T);
    this->b.reinit(dMap.NumMyElements());
    for(int z = 0 ; z <  dMap.NumMyElements() ; z++){
        this->b(z) = d[z];
    }
}

#if PHILIP_DIM==1
    template class AssembleECSWJac<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
    template class AssembleECSWJac<PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // HyperReduction namespace
} // PHiLiP namespace