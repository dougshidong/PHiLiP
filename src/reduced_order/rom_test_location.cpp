#include "rom_test_location.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim, int nstate>
ROMTestLocation<dim, nstate>::ROMTestLocation(double parameter, std::shared_ptr<ROMSolution<dim, nstate>> rom_solution)
        : parameter(parameter)
        , rom_solution(rom_solution)
{
    std::cout << "Creating ROM test location..." << std::endl;
    compute_FOM_to_initial_ROM_error();
    initial_rom_to_final_rom_error = 0;
    initial_rom_to_final_rom_sensitivity = 0;
    compute_total_error();
    std::cout << "ROM test location created. Error estimate updated." << std::endl;
}

template <int dim, int nstate>
void ROMTestLocation<dim, nstate>::compute_FOM_to_initial_ROM_error(){
    std::cout << "Computing adjoint-based error estimate between ROM and FOM..." << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> gradient(rom_solution->right_hand_side.size());
    dealii::LinearAlgebra::distributed::Vector<double> adjoint(rom_solution->right_hand_side.size());
    dealii::LinearAlgebra::distributed::Vector<double> dualWeightedResidual(rom_solution->right_hand_side.size());

    gradient = rom_solution->gradient;

    //dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;
    //system_matrix_transpose.reinit(rom_solution->system_matrix_transpose);
    //system_matrix_transpose.copy_from(rom_solution->system_matrix_transpose);

    Parameters::LinearSolverParam linear_solver_param;
    linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::direct;
    solve_linear(*rom_solution->system_matrix_transpose, gradient*=-1.0, adjoint, linear_solver_param);

    //Compute dual weighted residual
    fom_to_initial_rom_error = 0;
    //std::cout << std::setw(10) << std::left << "Index" << std::setw(20) << std::left << "Adjoint" << std::setw(20) << std::left << "Residual" << std::setw(20) << std::left << "Dual Weighted Residual" << std::endl;
    for(unsigned int i = 0; i < adjoint.size(); i++){
        dualWeightedResidual[i] = -(adjoint[i] * rom_solution->right_hand_side[i]);
        fom_to_initial_rom_error = fom_to_initial_rom_error + dualWeightedResidual[i];
        //std::cout << std::setw(10) << std::left << i << std::setw(20) << std::left << adjoint[i] << std::setw(20) << std::left << rom_solution->right_hand_side[i] << std::setw(20) << std::left << dualWeightedResidual[i] << std::endl;
    }
    std::cout << "Parameter: " << parameter << ". Error estimate between ROM and FOM: " << fom_to_initial_rom_error << std::endl;

    //Compute estimated sensitivity
    fom_to_initial_rom_sensitivity = 0;
    //std::cout << std::setw(10) << std::left << "Index" << std::setw(20) << std::left << "Adjoint" << std::setw(20) << std::left << "Residual" << std::setw(20) << std::left << "Dual Weighted Residual" << std::endl;
    for(unsigned int i = 0; i < adjoint.size(); i++){
        fom_to_initial_rom_sensitivity = fom_to_initial_rom_sensitivity - (adjoint[i] * rom_solution->sensitivity[i]);
    }
    std::cout << "Parameter: " << parameter << ". Sensitivity estimate between ROM and FOM: " << fom_to_initial_rom_sensitivity << std::endl;

}

template <int dim, int nstate>
void ROMTestLocation<dim, nstate>::compute_initial_rom_to_final_rom_error(std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod_updated){
    std::cout << "Computing adjoint-based error estimate between initial ROM and updated ROM..." << std::endl;

    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    //Initialize
    DealiiVector fineGradient(pod_updated->getPODBasis()->n());
    DealiiVector fineAdjoint(pod_updated->getPODBasis()->n());
    DealiiVector fineResidual(pod_updated->getPODBasis()->n());
    DealiiVector dualWeightedResidual(pod_updated->getPODBasis()->n());

    pod_updated->getPODBasis()->Tvmult(fineGradient, rom_solution->gradient);

    //*******************Compute fine jacobian transpose (Petrov-Galerkin)********************
    dealii::TrilinosWrappers::SparseMatrix petrov_galerkin_basis;
    dealii::TrilinosWrappers::SparseMatrix fineJacobianTranspose;
    rom_solution->system_matrix_transpose->Tmmult(petrov_galerkin_basis, *pod_updated->getPODBasis()); // petrov_galerkin_basis = system_matrix * pod_basis. Note, use transpose in subsequent multiplications
    petrov_galerkin_basis.Tmmult(fineJacobianTranspose, petrov_galerkin_basis); //reduced_lhs = petrov_galerkin_basis^T * petrov_galerkin_basis , equivalent to V^T*J^T*J*V
    //*****************************************************************************************

    //************************Compute fine jacobian transpose (Galerkin)**********************
    //dealii::TrilinosWrappers::SparseMatrix tmp;
    //dealii::TrilinosWrappers::SparseMatrix fineJacobianTranspose;
    //pod_updated->getPODBasis()->Tmmult(tmp, *rom_solution->system_matrix_transpose); //tmp = pod_basis^T * dg->system_matrix_transpose
    //tmp.mmult(fineJacobianTranspose, *pod_updated->getPODBasis()); // reducedJacobianTranspose= tmp*pod_basis
    //****************************************************************************************

    dealii::ParameterHandler parameter_handler;
    Parameters::LinearSolverParam linear_solver_param;
    Parameters::LinearSolverParam::declare_parameters (parameter_handler);
    linear_solver_param.parse_parameters (parameter_handler);
    linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::direct;
    solve_linear(fineJacobianTranspose, fineGradient*=-1.0, fineAdjoint, linear_solver_param);

    //******************Compute fine residual (Galerkin)********************************
    //pod_updated->getPODBasis()->Tvmult(fineResidual, rom_solution->right_hand_side);
    //**********************************************************************************

    //*****************Compute fine residual (Petrov_Galerkin)*************************
    petrov_galerkin_basis.Tvmult(fineResidual, rom_solution->right_hand_side);
    //*********************************************************************************

    //Compute dual weighted residual
    initial_rom_to_final_rom_error = 0;
    //std::cout << std::setw(10) << std::left << "Index" << std::setw(20) << std::left << "Reduced Adjoint" << std::setw(20) << std::left << "Reduced Residual" << std::setw(20) << std::left << "Dual Weighted Residual" << std::endl;
    for(unsigned int i = 0; i < fineAdjoint.size(); i++){
        dualWeightedResidual[i] = -(fineAdjoint[i] * fineResidual[i]);
        initial_rom_to_final_rom_error = initial_rom_to_final_rom_error + dualWeightedResidual[i];
        //std::cout << std::setw(10) << std::left << i << std::setw(20) << std::left << fineAdjoint[i] << std::setw(20) << std::left << fineResidual[i] << std::setw(20) << std::left << dualWeightedResidual[i] << std::endl;
    }
    std::cout << "Parameter: " << parameter << ". Error estimate between initial ROM and updated ROM: " << initial_rom_to_final_rom_error << std::endl;

    //Compute estimated sensitivity
    DealiiVector fineSensitivity(pod_updated->getPODBasis()->n());
    //*****************Compute fine sensitivity (Petrov_Galerkin)*************************
    petrov_galerkin_basis.Tvmult(fineSensitivity, rom_solution->sensitivity);
    //*********************************************************************************
    initial_rom_to_final_rom_sensitivity = 0;
    for(unsigned int i = 0; i < fineAdjoint.size(); i++){
        initial_rom_to_final_rom_sensitivity = initial_rom_to_final_rom_sensitivity - (fineAdjoint[i] * fineSensitivity[i]);
    }
    std::cout << "Parameter: " << parameter << ". Sensitivity estimate between initial ROM and updated ROM: " << initial_rom_to_final_rom_sensitivity << std::endl;

}

template <int dim, int nstate>
void ROMTestLocation<dim, nstate>::compute_total_error(){
    std::cout << "Computing total error estimate between FOM and updated ROM..." << std::endl;
    total_error = fom_to_initial_rom_error - initial_rom_to_final_rom_error;
    std::cout << "Parameter: " << parameter <<  ". Total error estimate between FOM and updated ROM: " << total_error << std::endl;

    total_sensitivity = fom_to_initial_rom_sensitivity - initial_rom_to_final_rom_sensitivity;
    std::cout << "Parameter: " << parameter <<  ". Total sensitivity estimate between FOM and updated ROM: " << total_sensitivity << std::endl;

}


template class ROMTestLocation <PHILIP_DIM, 1>;
template class ROMTestLocation <PHILIP_DIM, 2>;
template class ROMTestLocation <PHILIP_DIM, 3>;
template class ROMTestLocation <PHILIP_DIM, 4>;
template class ROMTestLocation <PHILIP_DIM, 5>;

}
}