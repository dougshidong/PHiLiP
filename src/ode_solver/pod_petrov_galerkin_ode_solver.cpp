#include "pod_petrov_galerkin_ode_solver.h"
#include <deal.II/lac/la_parallel_vector.h>
#include <Epetra_Vector.h>
#include <Epetra_LinearProblem.h>
#include "Amesos.h"
#include "Amesos_BaseSolver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
PODPetrovGalerkinODESolver<dim,real,MeshType>::PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod)
        : ImplicitODESolver<dim,real,MeshType>(dg_input)
        , pod(pod)
{}

template <int dim, typename real, typename MeshType>
int PODPetrovGalerkinODESolver<dim,real,MeshType>::steady_state ()
{
    this->pcout << " Performing steady state analysis... " << std::endl;
    allocate_ode_system ();

    this->current_iteration = 0;

    this->pcout << " Evaluating right-hand side and setting system_matrix to Jacobian before starting iterations... " << std::endl;
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);

    Epetra_CrsMatrix *epetra_system_matrix = const_cast<Epetra_CrsMatrix *>(&(this->dg->system_matrix.trilinos_matrix()));
    Epetra_Map system_matrix_rowmap = epetra_system_matrix->RowMap();
    Epetra_CrsMatrix *epetra_pod_basis = const_cast<Epetra_CrsMatrix *>(&(pod->getPODBasis()->trilinos_matrix()));
    Epetra_CrsMatrix epetra_petrov_galerkin_basis(Epetra_DataAccess::View, system_matrix_rowmap, pod->getPODBasis()->n());
    EpetraExt::MatrixMatrix::Multiply(*epetra_system_matrix, false, *epetra_pod_basis, false, epetra_petrov_galerkin_basis, true);
    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_system_matrix->RowMap(), this->dg->right_hand_side.begin());
    Epetra_Vector epetra_reduced_rhs(epetra_petrov_galerkin_basis.DomainMap());
    epetra_petrov_galerkin_basis.Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);;
    epetra_reduced_rhs.Norm2(&this->initial_residual_norm);

    this->pcout << " ********************************************************** "
          << std::endl
          << " Initial absolute residual norm: " << this->initial_residual_norm
          << std::endl;

    this->residual_norm = 1;
    this->residual_norm_decrease = 1;

    // Initial Courant-Friedrichs-Lax number
    const double initial_CFL = this->all_parameters->ode_solver_param.initial_time_step;
    this->CFL_factor = 1.0;

    while (this->residual_norm > this->ode_param.nonlinear_steady_residual_tolerance)
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

        double ramped_CFL = initial_CFL * this->CFL_factor;
        if (this->residual_norm_decrease < 1.0) {
            ramped_CFL *= pow((1.0-std::log10(this->residual_norm_decrease)*this->ode_param.time_step_factor_residual), this->ode_param.time_step_factor_residual_exp);
            this->pcout << "ramped cfl " << ramped_CFL << std::endl;
        }
        ramped_CFL = std::max(ramped_CFL,initial_CFL*this->CFL_factor);
        this->pcout << "Initial CFL = " << initial_CFL << ". Current CFL = " << ramped_CFL << std::endl;

        /*
        if (this->residual_norm < 1e-12) {
            this->dg->freeze_artificial_dissipation = true;
        } else {
            this->dg->freeze_artificial_dissipation = false;
        }
        */

        const bool pseudotime = true;
        step_in_time(ramped_CFL, pseudotime);

        if (this->ode_param.output_solution_every_x_steps > 0) {
            const bool is_output_iteration = (this->current_iteration % this->ode_param.output_solution_every_x_steps == 0);
            if (is_output_iteration) {
                const int file_number = this->current_iteration / this->ode_param.output_solution_every_x_steps;
                this->dg->output_results_vtk(file_number);
            }
        }

        this->residual_norm_decrease = this->residual_norm / this->initial_residual_norm;
        this->pcout << "this->residual_norm_decrease = " << this->residual_norm_decrease << std::endl;

        //this->residual_norm = this->dg->get_residual_l2norm();
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
void PODPetrovGalerkinODESolver<dim,real,MeshType>::step_in_time (real /*dt*/, const bool /*pseudotime*/)
{
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    //this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw

    this->dg->system_matrix *= -1.0;

    //const double CFL = dt;
    //this->dg->time_scaled_mass_matrices(CFL);
    //this->dg->add_time_scaled_mass_matrices();

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Evaluating system update... " << std::endl;
    }

    /* Reference for Petrov-Galerkin projection: Refer to Equation (23) in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
    */
    //Petrov-Galerkin projection, petrov_galerkin_basis = V^T*J^T, pod basis V, system matrix J
    //V^T*J*V*p = -V^T*R

    Epetra_CrsMatrix *epetra_system_matrix = const_cast<Epetra_CrsMatrix *>(&(this->dg->system_matrix.trilinos_matrix()));
    Epetra_Map system_matrix_rowmap = epetra_system_matrix->RowMap();

    Epetra_CrsMatrix *epetra_pod_basis = const_cast<Epetra_CrsMatrix *>(&(pod->getPODBasis()->trilinos_matrix()));

    Epetra_CrsMatrix epetra_petrov_galerkin_basis(Epetra_DataAccess::View, system_matrix_rowmap, pod->getPODBasis()->n());

    EpetraExt::MatrixMatrix::Multiply(*epetra_system_matrix, false, *epetra_pod_basis, false, epetra_petrov_galerkin_basis, true);

    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_system_matrix->RowMap(), this->dg->right_hand_side.begin());

    Epetra_Vector epetra_reduced_rhs(epetra_petrov_galerkin_basis.DomainMap());

    epetra_petrov_galerkin_basis.Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);

    Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::View, epetra_petrov_galerkin_basis.DomainMap(), pod->getPODBasis()->n());

    EpetraExt::MatrixMatrix::Multiply(epetra_petrov_galerkin_basis, true, epetra_petrov_galerkin_basis, false, epetra_reduced_lhs);

    //Epetra_Vector epetra_reduced_solution_update(epetra_reduced_lhs.DomainMap());
    Epetra_Vector epetra_reduced_solution_update(Epetra_DataAccess::View, epetra_reduced_lhs.DomainMap(), reduced_solution_update.begin());

    Epetra_LinearProblem linearProblem(&epetra_reduced_lhs, &epetra_reduced_solution_update, &epetra_reduced_rhs);

    Amesos_BaseSolver* Solver;
    Amesos Factory;
    std::string SolverType = "Klu";
    Solver = Factory.Create(SolverType, linearProblem);

    Teuchos::ParameterList List;
    Solver->SetParameters(List);

    Solver->SymbolicFactorization();
    Solver->NumericFactorization();
    Solver->Solve();

    this->pcout << "Reduced solution update norm: " << reduced_solution_update.l2_norm() << std::endl;
    //double l2norm;
    //epetra_reduced_rhs.Norm2(&l2norm);
    //this->pcout << "l2 norm of reduced order residual: " << l2norm << std::endl;

    const dealii::LinearAlgebra::distributed::Vector<double> old_reduced_solution(reduced_solution);
    double step_length = 1.0;
    const double step_reduction = 0.5;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;

    double initial_residual;
    epetra_reduced_rhs.Norm2(&initial_residual);

    reduced_solution.add(step_length, this->reduced_solution_update);
    Epetra_Vector epetra_reduced_solution(Epetra_DataAccess::View, epetra_pod_basis->DomainMap(), reduced_solution.begin());
    Epetra_Vector solution(Epetra_DataAccess::View, epetra_pod_basis->RangeMap(), this->dg->solution.begin());
    epetra_pod_basis->Multiply(false, epetra_reduced_solution, solution);
    this->dg->solution += reference_solution;
    this->dg->assemble_residual();
    epetra_petrov_galerkin_basis.Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
    double new_residual;
    epetra_reduced_rhs.Norm2(&new_residual);

    double reduced_l2norm = this->dg->get_reduced_residual_l2norm(epetra_petrov_galerkin_basis);
    this->pcout << "Reduced l2norm: " << reduced_l2norm << std::endl;
    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        reduced_solution = old_reduced_solution;
        reduced_solution.add(step_length, this->reduced_solution_update);
        epetra_pod_basis->Multiply(false, epetra_reduced_solution, solution);
        this->dg->solution += reference_solution;
        this->dg->assemble_residual();
        epetra_petrov_galerkin_basis.Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
        epetra_reduced_rhs.Norm2(&new_residual);
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        reduced_l2norm = this->dg->get_reduced_residual_l2norm(epetra_petrov_galerkin_basis);
        this->pcout << "Reduced l2norm: " << reduced_l2norm << std::endl;
        this->pcout << "Residual l2norm: " << this->dg->get_residual_l2norm() << std::endl;
    }
    if (iline == 0) this->CFL_factor *= 2.0;
    if (iline == maxline) this->CFL_factor *= 0.5;

    this->residual_norm = new_residual;
    ++(this->current_iteration);

    delete Solver;
}

template <int dim, typename real, typename MeshType>
double PODPetrovGalerkinODESolver<dim,real,MeshType>::linesearch()
{
    const dealii::LinearAlgebra::distributed::Vector<double> old_reduced_solution(reduced_solution);
    double step_length = 1.0;

    const double step_reduction = 0.5;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;

    const double initial_residual = this->dg->get_residual_l2norm();
    reduced_solution.add(step_length, this->reduced_solution_update);

    Epetra_CrsMatrix *epetra_pod_basis = const_cast<Epetra_CrsMatrix *>(&(pod->getPODBasis()->trilinos_matrix()));
    Epetra_Vector epetra_reduced_solution(Epetra_DataAccess::View, epetra_pod_basis->DomainMap(), reduced_solution.begin());
    Epetra_Vector solution(Epetra_DataAccess::View, epetra_pod_basis->RangeMap(), this->dg->solution.begin());
    //Epetra_Vector solution(epetra_pod_basis->RangeMap());
    epetra_pod_basis->Multiply(false, epetra_reduced_solution, solution);
    this->dg->solution += reference_solution;

    this->dg->assemble_residual();
    double new_residual = this->dg->get_residual_l2norm();
    this->pcout << "l2 norm of the rhs: " << this->dg->right_hand_side.l2_norm() << std::endl;
    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        reduced_solution = old_reduced_solution;
        reduced_solution.add(step_length, this->reduced_solution_update);

        epetra_pod_basis->Multiply(false, epetra_reduced_solution, solution);
        this->dg->solution += reference_solution;

        this->dg->assemble_residual();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << "l2 norm of the rhs: " << this->dg->right_hand_side.l2_norm() << std::endl;
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }


    return step_length;
}

template <int dim, typename real, typename MeshType>
void PODPetrovGalerkinODESolver<dim,real,MeshType>::allocate_ode_system ()
{

    this->pcout << "Allocating ODE system and evaluating mass matrix..." << std::endl;
    reference_solution = this->dg->solution;
    reference_solution.import(pod->getReferenceState(), dealii::VectorOperation::values::insert);

    dealii::LinearAlgebra::distributed::Vector<double> initial_condition(this->dg->solution);
    initial_condition -= reference_solution;

    Epetra_CrsMatrix *epetra_pod_basis = const_cast<Epetra_CrsMatrix *>(&(pod->getPODBasis()->trilinos_matrix()));
    reduced_solution.reinit(pod->getPODBasis()->n());
    reduced_solution *= 0;
    Epetra_Vector epetra_reduced_solution(Epetra_DataAccess::View, epetra_pod_basis->DomainMap(), reduced_solution.begin());
    Epetra_Vector epetra_initial_condition(Epetra_DataAccess::View, epetra_pod_basis->RangeMap(), initial_condition.begin());

    epetra_pod_basis->Multiply(true, epetra_initial_condition, epetra_reduced_solution);

    dealii::LinearAlgebra::distributed::Vector<double> initial_condition_projected(this->dg->solution);
    initial_condition_projected *= 0;
    Epetra_Vector epetra_projection_tmp(Epetra_DataAccess::View, epetra_pod_basis->RangeMap(), initial_condition_projected.begin());
    epetra_pod_basis->Multiply(false, epetra_reduced_solution, epetra_projection_tmp);
    //reference_solution += projection_tmp;
    initial_condition_projected += reference_solution;
    this->dg->solution = initial_condition_projected;

    reduced_solution_update.reinit(pod->getPODBasis()->n());
    reduced_solution_update *= 0;

    const bool do_inverse_mass_matrix = false;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

}

template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace//