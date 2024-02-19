#include "assemble_ECSW_training_data_base.h"
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
AssembleECSWBase<dim,nstate>::AssembleECSWBase(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input,
    std::shared_ptr<DGBase<dim,double>> &dg_input, 
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, 
    MatrixXd snapshot_parameters_input,
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type)
        : all_parameters(parameters_input)
        , parameter_handler(parameter_handler_input)
        , dg(dg_input)
        , pod(pod)
        , snapshot_parameters(snapshot_parameters_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , ode_solver_type(ode_solver_type)
        , A(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
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