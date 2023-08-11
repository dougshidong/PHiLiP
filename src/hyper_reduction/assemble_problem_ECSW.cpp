#include "assemble_problem_ECSW.h"
#include <eigen/Eigen/Dense>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include <iostream>

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;

template <int dim, int nstate>
AssembleECSW<dim,nstate>::AssembleECSW(
    std::shared_ptr<DGBase<dim,double>> &dg_input, 
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, 
    double *params,
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type)
        : dg(dg_input)
        , pod(pod)
        , params(params)
        , mpi_communicator(MPI_COMM_WORLD)
        , ode_solver_type(ode_solver_type)
        , A(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{
}

template <int dim, int nstate>
std::shared_ptr<Epetra_CrsMatrix> AssembleECSW<dim,nstate>::local_generate_test_basis(Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis){
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver){ 
        return std::make_shared<Epetra_CrsMatrix>(pod_basis);
    }
    else if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver){ 
        Epetra_Map system_matrix_rowmap = system_matrix.RowMap();
        Epetra_CrsMatrix petrov_galerkin_basis(Epetra_DataAccess::Copy, system_matrix_rowmap, pod_basis.NumGlobalCols());
        EpetraExt::MatrixMatrix::Multiply(system_matrix, false, pod_basis, false, petrov_galerkin_basis, true);

        return std::make_shared<Epetra_CrsMatrix>(petrov_galerkin_basis);
    }
    else {
        // display_error_ode_solver_factory(ode_solver_type, true);
        return nullptr;
    }
}

template <int dim, int nstate>
void AssembleECSW<dim,nstate>::build_problem(){
    std::cout << "Get snapshot matrix and pod basis"<< std::endl;
    MatrixXd snapshotMatrix = pod->getSnapshotMatrix();
    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    Epetra_CrsMatrix epetra_system_matrix = dg->system_matrix.trilinos_matrix();

    std::cout << "Get dimensions"<< std::endl;
    int num_snaps = snapshotMatrix.cols(); // Number of snapshots used to build the POD basis
    int n = epetra_pod_basis.NumGlobalCols(); // Reduced subspace dimension
    int N = epetra_pod_basis.NumGlobalRows(); // Length of solution vector
    int N_e = dg->triangulation->n_active_cells(); // Number of elements (? should be the same as N ?)

    std::cout << "CONFIGURE PARAMETER SPACE"<< std::endl;
    //parameter_sampling->configureInitialParameterSpace();

    std::cout << "Build C and d empty structs"<< std::endl;
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map RowMap((n*num_snaps), 0, epetra_comm);
    Epetra_Map ColMap(N_e, 0, epetra_comm);

    Epetra_CrsMatrix C(Epetra_DataAccess::Copy, RowMap, N_e);
    Epetra_Vector d(RowMap);

    std::cout << "Vector for DOFs for local element"<< std::endl;
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell); 
    int row_num = 0;
    for (int j = 0; j < num_snaps; ++j){
        std::cout << "Extract Snapshot from matrix"<< std::endl;
        dealii::LinearAlgebra::ReadWriteVector<double> snapshot_s;
        snapshot_s.reinit(N_e);
        for (int snap_row = 0; snap_row < N_e; snap_row++){
            snapshot_s(snap_row) = snapshotMatrix(snap_row, j);
        }
        dealii::LinearAlgebra::distributed::Vector<double> reference_solution(dg->solution);
        reference_solution.import(snapshot_s, dealii::VectorOperation::values::insert);
        std::cout << "Set dg solution to snapshot"<< std::endl;
        dg->solution = reference_solution;
        reference_solution.print(std::cout, 7);
        const bool compute_dRdW = true;
        std::cout << "Re-compute the residual"<< std::endl;
        dg->assemble_residual(compute_dRdW);

        std::cout << "Place residual in Epetra vector"<< std::endl;
        std::cout << "Residual"<< std::endl;
        dg->right_hand_side.print(std::cout);
        Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::Copy, epetra_system_matrix.RowMap(), dg->right_hand_side.begin());

        std::cout << "Compute test basis with system matrix and pod basis"<< std::endl;
        epetra_system_matrix = dg->system_matrix.trilinos_matrix();
        std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = local_generate_test_basis(epetra_system_matrix, epetra_pod_basis);
        int cell_num = 0;
        for (const auto &cell : dg->dof_handler.active_cell_iterators())
        {
            // std::cout << "Get current cell info"<< std::endl;
            const int fe_index_curr_cell = cell->active_fe_index();
            const dealii::FESystem<dim,dim> &current_fe_ref = dg->fe_collection[fe_index_curr_cell];
            const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
            // std::cout << "DOF Cell  " << n_dofs_curr_cell << std::endl;

            current_dofs_indices.resize(n_dofs_curr_cell);
            cell->get_dof_indices(current_dofs_indices);

            // std::cout << "Create L_e matrix for current cell"<< std::endl;
            Epetra_Map LeRowMap(n_dofs_curr_cell, 0, epetra_comm);
            Epetra_Map LeColMap(N, 0, epetra_comm);
            Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, N);
            double posOne = 1.0;

            for(int i = 0; i < n_dofs_curr_cell; i++){
                const int col = current_dofs_indices[i];
                L_e.InsertGlobalValues(i, 1, &posOne , &col);
            }
            L_e.FillComplete(LeColMap, LeRowMap);
            // std::cout << L_e << std::endl;

            Epetra_Vector local_r(LeRowMap);
            Epetra_Vector global_r_e(LeColMap);
            L_e.Multiply(false, epetra_right_hand_side, local_r);
            // std::cout << local_r << std::endl;
            L_e.Multiply(true, local_r, global_r_e);
            // std::cout << global_r_e << std::endl;

            Epetra_Map cseRowMap(n, 0, epetra_comm);
            Epetra_Vector c_se(cseRowMap);

            //std::cout << *epetra_test_basis << std::endl;
            epetra_test_basis->Multiply(true, global_r_e, c_se);
            // std::cout << c_se << std::endl;
            double *c_se_array = new double[n];

            c_se.ExtractCopy(c_se_array);
            for (int k = 0; k < n; ++k){
                // std::cout << "col: " << cell_num << std::endl;
                int place = row_num+k;
                C.InsertGlobalValues(place, 1, &c_se_array[k], &cell_num);
                d.SumIntoGlobalValues(1, &c_se_array[k], &place);
            }
            cell_num+=1;

            
        }
        row_num+=n;
    }

    C.FillComplete(ColMap, RowMap);

    std::cout << " Matrix C "<< std::endl;
    std::cout << C << std::endl;

    std::cout << " Vector d "<< std::endl;
    std::cout << d << std::endl;

    A->reinit(C);
    b.reinit(d.GlobalLength());

    for(int z = 0 ; z < d.GlobalLength() ; z++){
        b(z) = d[z];
    }
}


#if PHILIP_DIM==1
        template class AssembleECSW<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class AssembleECSW<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}
}