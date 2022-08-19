#include "JFNK_solver.h"
#include <deal.II/lac/precondition.h>

namespace PHiLiP{
namespace ODE{

template <int dim, typename real, typename MeshType>
JFNKSolver<dim,real,MeshType>::JFNKSolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
// Initializing in list for now; should use parameters or other
    , epsilon_jacobian(1.490116119384765625E-8) //sqrt(machine epsilon)
    , epsilon_Newton(1E-7)
    , epsilon_GMRES(1E-6)
    , max_num_temp_vectors(1000) //set high to prevent restart
    , max_GMRES_iter(1000)
    , max_Newton_iter(500)
    , jacobian_vector_product(dg)
    , solver_control(max_GMRES_iter, 
                     epsilon_GMRES,
                     false,     //log_history 
                     true)      //log_result 
    , solver_GMRES(solver_control,
            dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>>::AdditionalData(max_num_temp_vectors))
{}

template <int dim, typename real, typename MeshType>
void JFNKSolver<dim,real,MeshType>::solve (real dt,
        dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution)
{ 
    double update_norm = 1.0;
    int Newton_iter_counter = 0;
    
    jacobian_vector_product.reinit_for_next_timestep(dt, epsilon_jacobian, previous_step_solution);
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
    }
    if (Newton_iter_counter == max_Newton_iter){
        std::cout << "Maximum number of Newton iterations reached. Aborting..." << std::endl; //later: change to pcout
        std::abort();
    }

    //Temp - output results
    std::cout << "Newton residual : " << update_norm << " step " << Newton_iter_counter << std::endl;

}

template class JFNKSolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class JFNKSolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class JFNKSolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


}
}
