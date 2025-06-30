#include "assemble_ECSW_residual.h"
#include <iostream>

#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;

template <int dim, int nstate>
AssembleECSWRes<dim,nstate>::AssembleECSWRes(
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
void AssembleECSWRes<dim,nstate>::build_problem(){
    this->pcout << "Solve for A and b for the NNLS Problem from POD Snapshots"<< std::endl;
    MatrixXd snapshotMatrix = this->pod->getSnapshotMatrix();
    const Epetra_CrsMatrix epetra_pod_basis = this->pod->getPODBasis()->trilinos_matrix();
    Epetra_CrsMatrix epetra_system_matrix = this->dg->system_matrix.trilinos_matrix();

    // Get dimensions of the problem
    int num_snaps_POD = snapshotMatrix.cols(); // Number of snapshots used to build the POD basis
    int n_reduced_dim_POD = epetra_pod_basis.NumGlobalCols(); // Reduced subspace dimension
    int N_FOM_dim = epetra_pod_basis.NumGlobalRows(); // Length of solution vector
    int num_elements_N_e = this->dg->triangulation->n_active_cells(); // Number of elements (equal to N if there is one DOF per cell)

    // Create empty and temporary C and d structs
    int training_snaps;
    // Check if all or a subset of the snapshots will be used for training
    if (this->all_parameters->hyper_reduction_param.num_training_snaps != 0) {
        this->pcout << "LIMITED NUMBER OF SNAPSHOTS"<< std::endl;
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
    Epetra_Map RowMap((n_reduced_dim_POD*training_snaps), (n_reduced_dim_POD*training_snaps), 0, this->Comm_); // Number of rows in residual based training matrix = n * (number of training snapshots)
    Epetra_Map ColMap(num_elements_N_e, length, local_elements, 0, this->Comm_);
    Epetra_Map dMap((n_reduced_dim_POD*training_snaps), (rank == 0) ?  (n_reduced_dim_POD*training_snaps) : 0,  0, this->Comm_);

    delete[] local_elements;

    Epetra_CrsMatrix C_T(Epetra_DataAccess::Copy, ColMap, RowMap, num_elements_N_e);
    Epetra_Vector d(dMap);

    // Loop through the given number of training snapshots to find residuals
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell); 
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
        Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::Copy, epetra_system_matrix.RowMap(), this->dg->right_hand_side.begin());
        Epetra_Vector local_rhs = copy_vector_to_all_cores(epetra_right_hand_side);

        // Compute test basis
        epetra_system_matrix = this->dg->system_matrix.trilinos_matrix();
        std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = this->local_generate_test_basis(epetra_system_matrix, epetra_pod_basis);
        Epetra_CrsMatrix local_test_basis = copy_matrix_to_all_cores(*epetra_test_basis);

        // Loop through the elements
        for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned()){
                int cell_num = cell->active_cell_index();
                const int fe_index_curr_cell = cell->active_fe_index();
                const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
                const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

                current_dofs_indices.resize(n_dofs_curr_cell);
                cell->get_dof_indices(current_dofs_indices);

                // Create L_e matrix for current cell
                const Epetra_SerialComm sComm;
                Epetra_Map LeRowMap(n_dofs_curr_cell, 0, sComm);
                Epetra_Map LeColMap(N_FOM_dim, 0, sComm);
                Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, LeColMap, 1);
                double posOne = 1.0;

                for(int i = 0; i < n_dofs_curr_cell; i++){
                    const int col = current_dofs_indices[i];
                    L_e.InsertGlobalValues(i, 1, &posOne , &col);
                }
                L_e.FillComplete(LeColMap, LeRowMap);

                // Extract residual contributions of the current cell into global dimension
                Epetra_Vector local_r(LeRowMap);
                Epetra_Vector global_r_e(LeColMap);
                L_e.Multiply(false, local_rhs, local_r);
                L_e.Multiply(true, local_r, global_r_e);

                // Find reduced-order representation of contribution
                Epetra_Map cseRowMap(n_reduced_dim_POD, 0, sComm);
                Epetra_Vector c_se(cseRowMap);

                local_test_basis.Multiply(true, global_r_e, c_se);
                double *c_se_array = new double[n_reduced_dim_POD];

                c_se.ExtractCopy(c_se_array);
                
                // Sub into entries of C and d
                for (int k = 0; k < n_reduced_dim_POD; ++k){
                    int place = row_num+k;
                    C_T.InsertGlobalValues(cell_num, 1, &c_se_array[k], &place);
                }
                delete[] c_se_array;
            }
        }
        row_num+=n_reduced_dim_POD;
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
    template class AssembleECSWRes<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
    template class AssembleECSWRes<PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // HyperReduction namespace
} // PHiLiP namespace