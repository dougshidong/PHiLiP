#include "reduced_order_ode_solver.h"
#include "dg/dg.h"
#include "ode_solver_base.h"
#include "linear_solver/linear_solver.h"
#include "reduced_order/pod_basis_base.h"
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <Epetra_Vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <Epetra_LinearProblem.h>
#include "Amesos_BaseSolver.h"
#include <Amesos_Lapack.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
ReducedOrderODESolver<dim,real,MeshType>::ReducedOrderODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , pod(pod)
{}

template <int dim, typename real, typename MeshType>
int ReducedOrderODESolver<dim,real,MeshType>::steady_state ()
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

    const Epetra_CrsMatrix epetra_system_matrix = this->dg->system_matrix.trilinos_matrix();
    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = generate_test_basis(epetra_system_matrix, epetra_pod_basis);
    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::Copy, epetra_system_matrix.RowMap(), this->dg->right_hand_side.begin());
    Epetra_Vector epetra_reduced_rhs(epetra_test_basis->DomainMap());
    epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
    epetra_reduced_rhs.Norm2(&this->initial_residual_norm);
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
void ReducedOrderODESolver<dim,real,MeshType>::step_in_time (real /*dt*/, const bool /*pseudotime*/)
{
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);

    this->dg->system_matrix *= -1.0;

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Evaluating system update... " << std::endl;
    }

    const Epetra_CrsMatrix epetra_system_matrix = this->dg->system_matrix.trilinos_matrix();
    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = generate_test_basis(epetra_system_matrix, epetra_pod_basis);

    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_system_matrix.RowMap(), this->dg->right_hand_side.begin());
    Epetra_Vector epetra_reduced_rhs(epetra_test_basis->DomainMap());
    epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
    std::shared_ptr<Epetra_CrsMatrix> epetra_reduced_lhs = generate_reduced_lhs(epetra_system_matrix, *epetra_test_basis);
    Epetra_Vector epetra_reduced_solution_update(epetra_reduced_lhs->DomainMap());
    Epetra_LinearProblem linearProblem(epetra_reduced_lhs.get(), &epetra_reduced_solution_update, &epetra_reduced_rhs);

    Amesos_Lapack Solver(linearProblem);
    Teuchos::ParameterList List;
    Solver.SetParameters(List);
    Solver.SymbolicFactorization();
    Solver.NumericFactorization();
    Solver.Solve();

    const dealii::LinearAlgebra::distributed::Vector<double> old_solution(this->dg->solution);
    double step_length = 1.0;
    const double step_reduction = 0.5;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;
    const double reduction_tolerance_2 = 2.0;

    double initial_residual;
    epetra_reduced_rhs.Norm2(&initial_residual);
    initial_residual /= this->dg->right_hand_side.size();
    Epetra_Vector epetra_solution(Epetra_DataAccess::View, epetra_pod_basis.RangeMap(), this->dg->solution.begin());
    Epetra_Vector epetra_solution_update(epetra_pod_basis.RangeMap());
    epetra_pod_basis.Multiply(false, epetra_reduced_solution_update, epetra_solution_update);
    epetra_solution.Update(1, epetra_solution_update, 1);
    this->dg->assemble_residual();
    epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
    double new_residual;
    epetra_reduced_rhs.Norm2(&new_residual);
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
        epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
        epetra_reduced_rhs.Norm2(&new_residual);
        new_residual /= this->dg->right_hand_side.size();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }


    if (iline == maxline) {
        step_length = 1.0;
        this->pcout << " Line Search (Case 2): Increase nonlinear residual tolerance by a factor " << std::endl;
        this->pcout << " Line search failed. Will accept any valid residual less than " << reduction_tolerance_2 << " times the current " << initial_residual << "residual. " << std::endl;
        epetra_solution.Update(1, epetra_solution_update, 1);
        this->dg->assemble_residual();
        epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
        epetra_reduced_rhs.Norm2(&new_residual);
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
            epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
            epetra_reduced_rhs.Norm2(&new_residual);
            new_residual /= this->dg->right_hand_side.size();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }

    if (iline == maxline) {
        this->pcout << " Line Search (Case 3): Reverse Search Direction " << std::endl;
        step_length = -1.0;
        epetra_solution.Update(-1, epetra_solution_update, 1);
        this->dg->assemble_residual();
        epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
        epetra_reduced_rhs.Norm2(&new_residual);
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
            epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
            epetra_reduced_rhs.Norm2(&new_residual);
            new_residual /= this->dg->right_hand_side.size();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }

    if (iline == maxline) {
        this->pcout << " Line Search (Case 4): Reverse Search Direction AND Increase nonlinear residual tolerance by a factor " << std::endl;
        step_length = -1.0;
        epetra_solution.Update(-1, epetra_solution_update, 1);
        this->dg->assemble_residual();
        epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
        epetra_reduced_rhs.Norm2(&new_residual);
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
            epetra_test_basis->Multiply(true, epetra_right_hand_side, epetra_reduced_rhs);
            epetra_reduced_rhs.Norm2(&new_residual);
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
void ReducedOrderODESolver<dim,real,MeshType>::allocate_ode_system ()
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

template class ReducedOrderODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedOrderODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ReducedOrderODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace//