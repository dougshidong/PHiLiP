#include "test_location_base.h"
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
TestLocationBase<dim, nstate>::TestLocationBase(const RowVectorXd& parameter, std::unique_ptr<ROMSolution<dim, nstate>> rom_solution)
        : parameter(parameter)
        , rom_solution(std::move(rom_solution))
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
{
    pcout << "Creating ROM test location..." << std::endl;
    compute_FOM_to_initial_ROM_error();
    initial_rom_to_final_rom_error = 0;
    total_error = fom_to_initial_rom_error;

    pcout << "ROM test location created. Error estimate updated." << std::endl;
}

template <int dim, int nstate>
void TestLocationBase<dim, nstate>::compute_FOM_to_initial_ROM_error(){
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

    if (rom_solution->params.reduced_order_param.FOM_error_linear_solver_type == Parameters::ReducedOrderModelParam::LinearSolverEnum::gmres){
        linear_solver_param.max_iterations = 1000;
        linear_solver_param.restart_number = 200;
        linear_solver_param.linear_residual = 1e-17;
        linear_solver_param.ilut_fill = 50;
        linear_solver_param.ilut_drop = 1e-8;
        linear_solver_param.ilut_atol = 1e-5;
        linear_solver_param.ilut_rtol = 1.0+1e-2;
        linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;
    }
    else{
        linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::direct;
    }
    if (rom_solution->params.reduced_order_param.residual_error_bool == true){
        adjoint*=0;
        adjoint.add(1);
    }
    else{
        solve_linear(system_matrix_transpose, gradient*=-1.0, adjoint, linear_solver_param);
    }
    
    //Compute dual weighted residual
    fom_to_initial_rom_error = 0;
    fom_to_initial_rom_error = -(adjoint * flow_solver->dg->right_hand_side);

    pcout << "Parameter: " << parameter << ". Error estimate between ROM and FOM: " << fom_to_initial_rom_error << std::endl;
}

template <int dim, int nstate>
void TestLocationBase<dim, nstate>::compute_total_error(){
    pcout << "Computing total error estimate between FOM and updated ROM..." << std::endl;
    total_error = fom_to_initial_rom_error - initial_rom_to_final_rom_error;
    pcout << "Parameter: " << parameter <<  ". Total error estimate between FOM and updated ROM: " << total_error << std::endl;
}

#if PHILIP_DIM==1
        template class TestLocationBase<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class TestLocationBase<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}
}