#include "pod_petrov_galerkin_ode_solver.h"
#include <deal.II/lac/householder.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
PODPetrovGalerkinODESolver<dim,real,MeshType>::PODPetrovGalerkinODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod)
        : ImplicitODESolver<dim,real,MeshType>(dg_input)
        , pod(pod)
{}

template <int dim, typename real, typename MeshType>
void PODPetrovGalerkinODESolver<dim,real,MeshType>::step_in_time (real dt, const bool /*pseudotime*/)
{
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw

    this->dg->system_matrix *= -1.0;

    this->dg->add_mass_matrices(1.0/dt);

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
    this->pcout << "here0" << std::endl;
    std::ofstream out_file("system_matrix.txt");
    unsigned int precision1 = 16;
    this->dg->system_matrix.print(out_file, precision1);

    std::ofstream out_file2("pod_basis_petrovgalerkin.txt");
    unsigned int precision2 = 16;
    pod->getPODBasis()->print(out_file2, precision2);
    /*
    Epetra_CrsMatrix *epetra_system_matrix  = const_cast<Epetra_CrsMatrix *>(&(this->dg->system_matrix.trilinos_matrix()));
    Epetra_Map system_matrix_map = epetra_system_matrix->DomainMap();

    Epetra_CrsMatrix *epetra_pod  = const_cast<Epetra_CrsMatrix *>(&(pod->getPODBasis()->trilinos_matrix()));
    //Epetra_Map pod_domainmap = epetra_pod->DomainMap();

    //epetra_pod->FillComplete(pod_domainmap, system_matrix_map);
    std::cout << epetra_pod->ReplaceRowMap(system_matrix_map) << std::endl;
    epetra_pod->FillComplete();

    dealii::TrilinosWrappers::SparseMatrix pod_basis;
    pod_basis.reinit(*epetra_pod, true);

    Epetra_CrsMatrix *epetra_petrovgalerkin  = const_cast<Epetra_CrsMatrix *>(&(this->petrov_galerkin_basis->trilinos_matrix()));
    //epetra_pod->FillComplete(pod_domainmap, system_matrix_map);
    epetra_petrovgalerkin->ReplaceRowMap(system_matrix_map);
    epetra_petrovgalerkin->FillComplete();

    dealii::TrilinosWrappers::SparseMatrix petrovgalerkin;
    petrovgalerkin.reinit(*epetra_petrovgalerkin, true);

    this->dg->system_matrix.mmult(petrovgalerkin, pod_basis); // petrov_galerkin_basis = system_matrix * pod_basis. Note, use transpose in subsequent multiplications
    */
    this->pcout << "here1" << std::endl;
    /*
    dealii::FullMatrix<double> system_matrix_full(this->dg->system_matrix.m(), this->dg->system_matrix.n());
    system_matrix_full.copy_from(this->dg->system_matrix);
    this->pcout << "here2" << std::endl;

    dealii::FullMatrix<double> pod_full(pod->getPODBasis()->m(), pod->getPODBasis()->n());
    pod_full.copy_from(*pod->getPODBasis());
    this->pcout << "here3" << std::endl;

    dealii::FullMatrix<double> result;
    system_matrix_full.mmult(result, pod_full); // petrov_galerkin_basis = system_matrix * pod_basis. Note, use transpose in subsequent multiplications
    this->pcout << "here4" << std::endl;

    dealii::Householder<double> householder (result);
    dealii::Vector<double> leastSquaresSolution(pod->getPODBasis()->n());
    dealii::Vector<double> rhs(pod->getPODBasis()->m());
    this->pcout << "here5" << std::endl;

    for(unsigned int i = 0 ; i < pod->getPODBasis()->m() ; i++){
        rhs(i) = this->dg->right_hand_side(i);
    }
    this->pcout << "here6" << std::endl;

    householder.least_squares(leastSquaresSolution, rhs);
    this->pcout << "here7" << std::endl;

    dealii::LinearAlgebra::distributed::Vector<double> soln(leastSquaresSolution.size());
    for(unsigned int i = 0 ; i < pod->getPODBasis()->m() ; i++){
        soln[i] = leastSquaresSolution[i];
    }
    this->pcout << "here8" << std::endl;

    *reduced_solution_update = soln;
    this->pcout << "here9" << std::endl;
    */
    /*
    this->pcout << "here0.5" << std::endl;


    this->dg->system_matrix.mmult(*this->petrov_galerkin_basis, *pod->getPODBasis()); // petrov_galerkin_basis = system_matrix * pod_basis. Note, use transpose in subsequent multiplications
    this->pcout << "here1" << std::endl;

    this->petrov_galerkin_basis->Tvmult(*this->reduced_rhs, this->dg->right_hand_side); // reduced_rhs = (petrov_galerkin_basis)^T * right_hand_side
    this->pcout << "here2" << std::endl;

    this->petrov_galerkin_basis->Tmmult(*this->reduced_lhs, *this->petrov_galerkin_basis); //reduced_lhs = petrov_galerkin_basis^T * petrov_galerkin_basis , equivalent to V^T*J^T*J*V
    this->pcout << "here3" << std::endl;

    std::ofstream out_file3("reduced_lhs.txt");
    unsigned int precision3 = 16;
    this->reduced_lhs->print(out_file3, precision3);

    solve_linear(
            *this->reduced_lhs,
            *this->reduced_rhs,
            *this->reduced_solution_update,
            this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);
    this->pcout << "here4" << std::endl;
    */

    dealii::LinearAlgebra::distributed::Vector<double> reduced_rhs(pod->getPODBasis()->n());
    dealii::FullMatrix<double> system_matrix_full(this->dg->system_matrix.m(), this->dg->system_matrix.n());
    system_matrix_full.copy_from(this->dg->system_matrix);
    this->pcout << "here2" << std::endl;

    dealii::FullMatrix<double> pod_full(pod->getPODBasis()->m(), pod->getPODBasis()->n());
    pod_full.copy_from(*pod->getPODBasis());
    this->pcout << "here3" << std::endl;

    dealii::FullMatrix<double> reduced_lhs_full(pod->getPODBasis()->m(), pod->getPODBasis()->n());
    system_matrix_full.mmult(reduced_lhs_full, pod_full); // petrov_galerkin_basis = system_matrix * pod_basis. Note, use transpose in subsequent multiplications
    this->pcout << "here4" << std::endl;

    std::ofstream out_file4("reduced_rhs.txt");
    unsigned int precision4 = 16;
    reduced_rhs.print(out_file4, precision4);

    std::ofstream out_file5("reduced_lhs.txt");
    unsigned int precision5 = 16;
    reduced_lhs_full.print(out_file5, precision5);

    this->pcout << "here4" << std::endl;
    dealii::Householder<double> householder (reduced_lhs_full);
    dealii::Vector<double> leastSquaresSolution(pod->getPODBasis()->n());
    dealii::Vector<double> rhs(pod->getPODBasis()->n());
    this->pcout << "here5" << std::endl;

    for(unsigned int i = 0 ; i < pod->getPODBasis()->n() ; i++){
        rhs[i] = reduced_rhs[i];
        std::cout << reduced_rhs[i] << std::endl;
    }
    this->pcout << "here6" << std::endl;

    householder.least_squares(leastSquaresSolution, rhs);
    this->pcout << "here7" << std::endl;

    dealii::LinearAlgebra::distributed::Vector<double> soln(leastSquaresSolution.size());
    for(unsigned int i = 0 ; i < pod->getPODBasis()->n() ; i++){
        soln[i] = leastSquaresSolution[i];
    }
    this->pcout << "here8" << std::endl;

    *reduced_solution_update = soln;
    this->pcout << "here9" << std::endl;

    linesearch();

    this->update_norm = this->solution_update.l2_norm();
    ++(this->current_iteration);
}

template <int dim, typename real, typename MeshType>
double PODPetrovGalerkinODESolver<dim,real,MeshType>::linesearch()
{
    const auto old_reduced_solution = reduced_solution;
    double step_length = 1.0;

    const double step_reduction = 0.5;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;

    const double initial_residual = this->dg->get_residual_l2norm();

    reduced_solution = old_reduced_solution;
    reduced_solution.add(step_length, *this->reduced_solution_update);
    pod->getPODBasis()->vmult(this->dg->solution, reduced_solution);
    this->dg->solution.add(1, reference_solution);

    this->dg->assemble_residual ();
    double new_residual = this->dg->get_residual_l2norm();
    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        reduced_solution = old_reduced_solution;
        reduced_solution.add(step_length, *this->reduced_solution_update);
        pod->getPODBasis()->vmult(this->dg->solution, reduced_solution);
        this->dg->solution.add(1, reference_solution);
        this->dg->assemble_residual();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }
    if (iline == 0) this->CFL_factor *= 2.0;

    return step_length;
}

template <int dim, typename real, typename MeshType>
void PODPetrovGalerkinODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = false;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->solution_update.reinit(this->dg->right_hand_side);

    reduced_solution_update = std::make_unique<dealii::LinearAlgebra::distributed::Vector<double>>(pod->getPODBasis()->n());
    //reduced_rhs = std::make_unique<dealii::LinearAlgebra::distributed::Vector<double>>(pod->getPODBasis()->n());
    petrov_galerkin_basis = std::make_unique<dealii::TrilinosWrappers::SparseMatrix>(pod->getPODBasis()->m(), pod->getPODBasis()->n(), pod->getPODBasis()->n());
    reduced_lhs = std::make_unique<dealii::TrilinosWrappers::SparseMatrix>();
    reference_solution = this->dg->solution; //Set reference solution to initial conditions
    reduced_solution = dealii::LinearAlgebra::distributed::Vector<double>(pod->getPODBasis()->n()); //Zero if reference solution is the initial conditions
}

template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class PODPetrovGalerkinODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace//