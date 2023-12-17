#include "assemble_ECSW_jacobian.h"
#include <eigen/Eigen/Dense>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
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
    std::shared_ptr<Tests::AdaptiveSampling<dim,nstate>> parameter_sampling_input,
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type)
        : all_parameters(parameters_input)
        , parameter_handler(parameter_handler_input)
        , dg(dg_input)
        , pod(pod)
        , parameter_sampling(parameter_sampling_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , ode_solver_type(ode_solver_type)
        , A(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
}

template <int dim, int nstate>
std::shared_ptr<Epetra_CrsMatrix> AssembleECSWJac<dim,nstate>::local_generate_test_basis(Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis){
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver){ 
        return std::make_shared<Epetra_CrsMatrix>(pod_basis);
    }
    else if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver || ode_solver_type == ODEEnum::hyper_reduced_petrov_galerkin_solver){ 
        Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
        Epetra_CrsMatrix petrov_galerkin_basis(Epetra_DataAccess::Copy, system_matrix_rowmap, pod_basis.NumGlobalCols());
        EpetraExt::MatrixMatrix::Multiply(system_matrix, false, pod_basis, false, petrov_galerkin_basis, true);

        return std::make_shared<Epetra_CrsMatrix>(petrov_galerkin_basis);
    }
    else {
        return nullptr;
    }
}

template <int dim, int nstate>
void AssembleECSWJac<dim,nstate>::build_problem(){
    std::cout << "Solve for A and b for NNLS Problem from POD Snapshots"<< std::endl;
    MatrixXd snapshotMatrix = pod->getSnapshotMatrix();
    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    Epetra_CrsMatrix epetra_system_matrix = dg->system_matrix.trilinos_matrix();
    Epetra_Map system_matrix_rowmap = epetra_system_matrix.RowMap();

    // Get dimensions of the problem
    int num_snaps = snapshotMatrix.cols(); // Number of snapshots used to build the POD basis
    int n = epetra_pod_basis.NumGlobalCols(); // Reduced subspace dimension
    int N = epetra_pod_basis.NumGlobalRows(); // Length of solution vector
    int N_e = dg->triangulation->n_active_cells(); // Number of elements (equal to N if there is one DOF per cell)

    // Create empty and temporary C and d structs
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    int training_snaps;
    if (all_parameters->hyper_reduction_param.num_training_snaps != 0) {
        std::cout << "LIMITED NUMBER OF SNAPSHOTS"<< std::endl;
        training_snaps = all_parameters->hyper_reduction_param.num_training_snaps-1;
    }
    else{
        training_snaps = num_snaps;
    }
    Epetra_Map RowMap((n*n*training_snaps), 0, epetra_comm);
    Epetra_Map ColMap(N_e, 0, epetra_comm);

    Epetra_CrsMatrix C(Epetra_DataAccess::Copy, RowMap, N_e);
    Epetra_Vector d(RowMap);

    // Loop through the snapshots used to build the POD to find residuals
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbour_dofs_indices(max_dofs_per_cell); 
    int row_num = 0;
    int snap_num = 0;
    for(auto snap_param : parameter_sampling->snapshot_parameters.rowwise()){
        std::cout << "snap_param" << std::endl;
        std::cout << snap_param << std::endl;
        dealii::LinearAlgebra::ReadWriteVector<double> snapshot_s;
        snapshot_s.reinit(N_e);
        // Extract snapshot from the snapshotMatrix
        for (int snap_row = 0; snap_row < N_e; snap_row++){
            snapshot_s(snap_row) = snapshotMatrix(snap_row, snap_num);
        }
        dealii::LinearAlgebra::distributed::Vector<double> reference_solution(dg->solution);
        reference_solution.import(snapshot_s, dealii::VectorOperation::values::insert);
        
        // Modifiy parameters for snapshot and create new flow solver
        Parameters::AllParameters params = parameter_sampling->reinitParams(snap_param);
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);
        dg = flow_solver->dg;

        // Set solution to snapshot and re-compute the residual
        dg->solution = reference_solution;
        const bool compute_dRdW = true;
        dg->assemble_residual(compute_dRdW);
        Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::Copy, epetra_system_matrix.RowMap(), dg->right_hand_side.begin());

        // Compute test basis
        epetra_system_matrix = dg->system_matrix.trilinos_matrix();
        std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = local_generate_test_basis(epetra_system_matrix, epetra_pod_basis);
        
        // Loop through elements 
        for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
        {
            int cell_num = cell->active_cell_index();
            double *row = new double[epetra_system_matrix.NumGlobalCols()];
            int *global_indices = new int[epetra_system_matrix.NumGlobalCols()];
            /*
            
            int numE;
            int row_i = cell->active_cell_index();
            epetra_system_matrix.ExtractGlobalRowCopy(row_i, epetra_system_matrix.NumGlobalCols(), numE, row, global_indices);
            int neighbour_dofs_curr_cell = 0;
            for (int i = 0; i < numE; i++){
                neighbour_dofs_curr_cell +=1;
                neighbour_dofs_indices.resize(neighbour_dofs_curr_cell);
                // this -> pcout << "col" << global_indices[i]<< std::endl;
                neighbour_dofs_indices[neighbour_dofs_curr_cell-1] = N - global_indices[i] - 1;
                // this -> pcout << "ind " << neighbour_dofs_indices[neighbour_dofs_curr_cell-1] << std::endl;
            }
            const int fe_index_curr_cell = cell->active_fe_index();
            const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
            const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

            current_dofs_indices.resize(n_dofs_curr_cell);
            cell->get_dof_indices(current_dofs_indices);
            */

            // Create L_e matrix and transposed L_e matrixfor current cell

            const int fe_index_curr_cell = cell->active_fe_index();
            const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
            const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

            current_dofs_indices.resize(n_dofs_curr_cell);
            cell->get_dof_indices(current_dofs_indices);


            int numE;
            int row_i = current_dofs_indices[0];
            // this -> pcout << row_num << std::endl;
            epetra_system_matrix.ExtractGlobalRowCopy(row_i, epetra_system_matrix.NumGlobalCols(), numE, row, global_indices);
            int neighbour_dofs_curr_cell = 0;
            for (int i = 0; i < numE; i++){
                neighbour_dofs_curr_cell +=1;
                neighbour_dofs_indices.resize(neighbour_dofs_curr_cell);
                // this -> pcout << "col" << global_indices[i]<< std::endl;
                neighbour_dofs_indices[neighbour_dofs_curr_cell-1] = global_indices[i];
                // this -> pcout << "ind " << neighbour_dofs_indices[neighbour_dofs_curr_cell-1] << std::endl;
            }

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
            // this->pcout << L_e << std::endl;
            L_e_T.FillComplete(LeRowMap, system_matrix_rowmap);

            for(int i = 0; i < neighbour_dofs_curr_cell; i++){
                const int col = neighbour_dofs_indices[i];
                L_e_PLUS.InsertGlobalValues(i, 1, &posOne , &col);
            }
            L_e_PLUS.FillComplete(system_matrix_rowmap, LePLUSRowMap);
            // this->pcout << L_e_PLUS << std::endl;

            // Find contribution of element to the Jacobian
            Epetra_CrsMatrix J_L_e_T(Epetra_DataAccess::Copy, system_matrix_rowmap, neighbour_dofs_curr_cell);
            Epetra_CrsMatrix J_e_m(Epetra_DataAccess::Copy, LeRowMap, neighbour_dofs_curr_cell);
            EpetraExt::MatrixMatrix::Multiply(epetra_system_matrix, false, L_e_PLUS, true, J_L_e_T, true);
            // this->pcout << J_L_e_T << std::endl;
            EpetraExt::MatrixMatrix::Multiply(L_e, false, J_L_e_T, false, J_e_m, true);
            // this->pcout << J_e_m << std::endl;

            // Jacobian for this element in the global dimensions
            Epetra_CrsMatrix J_temp(Epetra_DataAccess::Copy, LeRowMap, N);
            Epetra_CrsMatrix J_global_e(Epetra_DataAccess::Copy, system_matrix_rowmap, N);
            EpetraExt::MatrixMatrix::Multiply(J_e_m, false, L_e_PLUS, false, J_temp, true);
            // this->pcout << J_temp << std::endl;
            EpetraExt::MatrixMatrix::Multiply(L_e_T, false, J_temp, false, J_global_e, true);
            // std::cout << "J_e"<< std::endl;
            // std::cout << J_global_e << std::endl;

            Epetra_CrsMatrix J_e_V(Epetra_DataAccess::Copy, system_matrix_rowmap, n);
            EpetraExt::MatrixMatrix::Multiply(J_global_e, false, epetra_pod_basis, false, J_e_V, true);
            // std::cout << "J_e_V"<< std::endl;
            // std::cout << J_e_V << std::endl;
            
            Epetra_CrsMatrix W_T(Epetra_DataAccess::Copy, epetra_test_basis->ColMap(), N);
            for(int i =0; i < epetra_test_basis->NumGlobalRows(); i++){
                double *row = new double[epetra_test_basis->NumGlobalCols()];
                int *global_cols = new int[epetra_test_basis->NumGlobalCols()];
                int numE;
                const int globalRow = epetra_test_basis->GRID(i);
                epetra_test_basis->ExtractGlobalRowCopy(globalRow, epetra_test_basis->NumGlobalCols(), numE , row, global_cols);
                for(int j = 0; j < numE; j++){
                    int col = global_cols[j];
                    W_T.InsertGlobalValues(col, 1, &row[j], &i);
                }
            }
            W_T.FillComplete(epetra_test_basis->RowMap(), epetra_test_basis->ColMap());
            Epetra_CrsMatrix W_T_J_e_V(Epetra_DataAccess::Copy, W_T.RowMap(), n);
            EpetraExt::MatrixMatrix::Multiply(W_T, false, J_e_V, false, W_T_J_e_V , true);
            // std::cout << "W_T_J_e_V"<< std::endl;
            // std::cout << W_T_J_e_V << std::endl;

            // Stack into n^2 vector
            Epetra_Map cseRowMap(n*n, 0, epetra_comm);
            Epetra_Vector c_se(cseRowMap);
            for(int i =0; i < W_T_J_e_V.NumGlobalRows(); i++){
                double *row = new double[W_T_J_e_V.NumGlobalCols()];
                int *global_cols = new int[epetra_system_matrix.NumGlobalCols()];
                int numE;
                const int globalRow = W_T_J_e_V.GRID(i);
                W_T_J_e_V.ExtractGlobalRowCopy(globalRow, W_T_J_e_V.NumGlobalCols(), numE , row, global_cols);
                for(int j = 0; j < numE; j++){
                    int col = global_cols[j];
                    int idx = col*n + i; 
                    c_se[idx] = row[j];
                }
            }
            // std::cout << "c_se"<< std::endl;
            // std::cout << c_se << std::endl;

            double *c_se_array = new double[n*n];

            c_se.ExtractCopy(c_se_array);
            
            // Sub into entries of C and d
            for (int k = 0; k < (n*n); ++k){
                int place = row_num+k;
                C.InsertGlobalValues(place, 1, &c_se_array[k], &cell_num);
                d.SumIntoGlobalValues(1, &c_se_array[k], &place);
            }       
        }
        row_num+=(n*n);
        snap_num+=1;

        if (all_parameters->hyper_reduction_param.num_training_snaps != 0) {
            std::cout << "LIMITED NUMBER OF SNAPSHOTS"<< std::endl;
            if (snap_num > (all_parameters->hyper_reduction_param.num_training_snaps-1)){
                break;
            }
        }
    }

    C.FillComplete(ColMap, RowMap);

    // std::cout << " Matrix C "<< std::endl;
    // std::cout << C << std::endl;

    // std::cout << " Vector d "<< std::endl;
    // std::cout << d << std::endl;

    // Sub temp C and d into class A and b
    A->reinit(C);
    b.reinit(d.GlobalLength());
    for(int z = 0 ; z < d.GlobalLength() ; z++){
        b(z) = d[z];
    }
}


#if PHILIP_DIM==1
        template class AssembleECSWJac<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class AssembleECSWJac<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}
}