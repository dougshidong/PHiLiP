#include "assemble_ECSW_training_data_base.h"
#include <iostream>

#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace HyperReduction {
using Eigen::MatrixXd;

template <int dim, int nstate>
AssembleECSWBase<dim,nstate>::AssembleECSWBase(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input,
    std::shared_ptr<DGBase<dim,double>> &dg_input, 
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, 
    MatrixXd snapshot_parameters_input,
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type,
    Epetra_MpiComm &Comm)
        : all_parameters(parameters_input)
        , parameter_handler(parameter_handler_input)
        , dg(dg_input)
        , pod(pod)
        , snapshot_parameters(snapshot_parameters_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
        , ode_solver_type(ode_solver_type)
        , Comm_(Comm)
        , A_T(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , fom_locations()
{
}

template <int dim, int nstate>
std::shared_ptr<Epetra_CrsMatrix> AssembleECSWBase<dim,nstate>::local_generate_test_basis(Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &pod_basis){
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
Parameters::AllParameters AssembleECSWBase<dim, nstate>::reinitParams(const RowVectorXd& parameter) const{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;

    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "rewienski_a"){
                parameters.burgers_param.rewienski_a = parameter(0);
            }
            else if(all_parameters->reduced_order_param.parameter_names[0] == "rewienski_b"){
                parameters.burgers_param.rewienski_b = parameter(0);
            }
        }
        else{
            parameters.burgers_param.rewienski_a = parameter(0);
            parameters.burgers_param.rewienski_b = parameter(1);
        }
    }
    else if (flow_type == FlowCaseEnum::naca0012){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "mach"){
                parameters.euler_param.mach_inf = parameter(0);
            }
            else if(all_parameters->reduced_order_param.parameter_names[0] == "alpha"){
                parameters.euler_param.angle_of_attack = parameter(0); //radians!
            }
        }
        else{
            parameters.euler_param.mach_inf = parameter(0);
            parameters.euler_param.angle_of_attack = parameter(1); //radians!
        }
    }
    else if (flow_type == FlowCaseEnum::gaussian_bump){
        if(all_parameters->reduced_order_param.parameter_names.size() == 1){
            if(all_parameters->reduced_order_param.parameter_names[0] == "mach"){
                parameters.euler_param.mach_inf = parameter(0);
            }
        }
    }
    else{
        std::cout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return parameters;
}

template <int dim, int nstate>
Epetra_CrsMatrix AssembleECSWBase<dim,nstate>::copyMatrixToAllCores(const Epetra_CrsMatrix &A){
    // Gather Matrix Information
    const int A_rows = A.NumGlobalRows();
    const int A_cols = A.NumGlobalCols();

    // Create new maps for one core and gather old maps
    const Epetra_SerialComm sComm;
    Epetra_Map single_core_row_A (A_rows, A_rows, 0 , sComm);
    Epetra_Map single_core_col_A (A_cols, A_cols, 0 , sComm);
    Epetra_Map old_row_map_A = A.RowMap();
    Epetra_Map old_col_map_A = A.DomainMap();

    // Create Epetra_importer object
    Epetra_Import A_importer(single_core_row_A,old_row_map_A);

    // Create new A matrix
    Epetra_CrsMatrix A_temp (Epetra_DataAccess::Copy, single_core_row_A, A_cols);
    // Load the data from matrix A (Multi core) into A_temp (Single core)
    A_temp.Import(A, A_importer, Epetra_CombineMode::Insert);
    A_temp.FillComplete(single_core_col_A,single_core_row_A);
    return A_temp;
}

template <int dim, int nstate>
Epetra_Vector AssembleECSWBase<dim,nstate>::copyVectorToAllCores(const Epetra_Vector &b){
    // Gather Vector Information
    const Epetra_SerialComm sComm;
    const int b_size = b.GlobalLength();
    // Create new map for one core and gather old map
    Epetra_Map single_core_b (b_size, b_size, 0, sComm);
    Epetra_BlockMap old_map_b = b.Map();
    // Create Epetra_importer object
    Epetra_Import b_importer(single_core_b, old_map_b);
    // Create new b vector
    Epetra_Vector b_temp (single_core_b); 
    // Load the data from vector b (Multi core) into b_temp (Single core)
    b_temp.Import(b, b_importer, Epetra_CombineMode::Insert);
    return b_temp;
}

template <int dim, int nstate>
void AssembleECSWBase<dim, nstate>::updateSnapshots(dealii::LinearAlgebra::distributed::Vector<double> fom_solution){
    fom_locations.emplace_back(fom_solution);
}

template <int dim, int nstate>
void AssembleECSWBase<dim, nstate>::updatePODSnaps(std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod_update,    
                                                    MatrixXd snapshot_parameters_update) {
    this->pod = pod_update;
    this->snapshot_parameters = snapshot_parameters_update;
}

#if PHILIP_DIM==1
        template class AssembleECSWBase<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class AssembleECSWBase<PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // HyperReduction namespace
} // PHiLiP namespace