#include "JFNK_solver.h"
#include <deal.II/lac/precondition.h>

namespace PHiLiP{
namespace ODE{

template <int dim, typename real, typename MeshType>
JFNKSolver<dim,real,MeshType>::JFNKSolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    , all_parameters(dg_input->all_parameters)
    , linear_param(all_parameters->linear_solver_param) 
    , perturbation_magnitude(linear_param.perturbation_magnitude) 
    , epsilon_Newton(linear_param.newton_residual)
    , epsilon_GMRES(linear_param.linear_residual)
    , max_num_temp_vectors(linear_param.restart_number)
    , max_GMRES_iter(linear_param.max_iterations)
    , max_Newton_iter(linear_param.newton_max_iterations)
    , do_output(linear_param.linear_solver_output == Parameters::OutputEnum::verbose)
    , jacobian_vector_product(dg_input)
    , solver_control(max_GMRES_iter, 
                     epsilon_GMRES,
                     false,         //log_history 
                     do_output)     //log_result 
    , solver_GMRES(solver_control,
            dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>>::AdditionalData(max_num_temp_vectors))
{}

template <int dim, typename real, typename MeshType>
void JFNKSolver<dim,real,MeshType>::solve (real dt,
        dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution)
{ 
    double update_norm = 1.0;
    int Newton_iter_counter = 0;
    
    jacobian_vector_product.reinit_for_next_timestep(dt, perturbation_magnitude, previous_step_solution);
    current_solution_estimate = previous_step_solution;
    solution_update_newton.reinit(previous_step_solution);

    while ((update_norm > epsilon_Newton) && (Newton_iter_counter < max_Newton_iter)){
        jacobian_vector_product.reinit_for_next_Newton_iter(current_solution_estimate);

        solver_GMRES.solve(jacobian_vector_product,
                     solution_update_newton, 
                     jacobian_vector_product.compute_unsteady_residual(current_solution_estimate, true), //do_negate = true
                     dealii::PreconditionIdentity());

        update_norm = solution_update_newton.l2_norm();
        current_solution_estimate += solution_update_newton;
        Newton_iter_counter++;
        if (do_output)      pcout << "Newton residual : " << update_norm << " step " << Newton_iter_counter << std::endl;
    }
    if (Newton_iter_counter == max_Newton_iter){
        pcout << "Maximum number of Newton iterations reached. Aborting..." << std::endl;
        std::abort();
    }


}

template class JFNKSolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class JFNKSolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class JFNKSolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


}
}
