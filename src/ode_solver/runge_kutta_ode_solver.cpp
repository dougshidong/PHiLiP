#include "runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType> 
RungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
        std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
        std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
        : RungeKuttaBase<dim,real,n_rk_stages,MeshType>(dg_input, RRK_object_input)
        , butcher_tableau(rk_tableau_input)
{}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::calculate_stage_solution (int istage, real dt, const bool pseudotime)
{
    this->rk_stage[istage]=0.0; //resets all entries to zero
    
    for (int j = 0; j < istage; ++j){
        if (this->butcher_tableau->get_a(istage,j) != 0){
            this->rk_stage[istage].add(this->butcher_tableau->get_a(istage,j), this->rk_stage[j]);
        }
    } //sum(a_ij *k_j), explicit part

    
    if(pseudotime) {
        const double CFL = dt;
        this->dg->time_scale_solution_update(this->rk_stage[istage], CFL);
    }else {
        this->rk_stage[istage]*=dt;
    }//dt * sum(a_ij * k_j)
    
    this->rk_stage[istage].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)
    
    //implicit solve if there is a nonzero diagonal element
    if (!this->butcher_tableau_aii_is_zero[istage]){
        /* // AD version - keeping in comments as it may be useful for future testing
        // Solve (M/dt - dRdW) / a_ii * dw = R
        // w = w + dw
        // Note - need to have assembled residual using this->dg->assemble_residual(true);
        //        and have mass matrix assembled, and include linear_solver
        dealii::LinearAlgebra::distributed::Vector<double> temp_u(this->dg->solution.size());

        this->dg->system_matrix *= -1.0/butcher_tableau_a[istage][istage]; //system_matrix = -1/a_ii*dRdW
        this->dg->add_mass_matrices(1.0/butcher_tableau_a[istage][istage]/dt); //system_matrix = -1/a_ii*dRdW + M/dt/a_ii = A

        solve_linear ( //Solve Ax=b using Aztec00 gmres
                    this->dg->system_matrix, //A = -1/a_ii*dRdW + M/dt/a_ii
                    this->dg->right_hand_side, //b = R
                    temp_u, // result,  x = dw
                    this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

        this->rk_stage[istage].add(1.0, temp_u);
        */

        //JFNK version
        this->solver.solve(dt*this->butcher_tableau->get_a(istage,istage), this->rk_stage[istage]);
        this->rk_stage[istage] = this->solver.current_solution_estimate;

    } // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(istage) <implicit>
    
    // If using the entropy formulation of RRK, solutions must be stored.
    // Call store_stage_solutions before overwriting rk_stage with the derivative.
    this->relaxation_runge_kutta->store_stage_solutions(istage, this->rk_stage[istage]);

    this->dg->solution = this->rk_stage[istage];

}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::calculate_stage_derivative (int istage, real dt)
{
     //set the DG current time for unsteady source terms
    this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(istage)*dt);
    
    //solve the system's right hand side
    this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(istage)))

    if(this->all_parameters->use_inverse_mass_on_the_fly){
        this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, this->rk_stage[istage]); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    } else{
        this->dg->global_inverse_mass_matrix.vmult(this->rk_stage[istage], this->dg->right_hand_side); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::sum_stages (real dt, const bool pseudotime)
{
    //assemble solution from stages
    for (int istage = 0; istage < n_rk_stages; ++istage){
        if (pseudotime){
            const double CFL = this->butcher_tableau->get_b(istage) * dt;
            this->dg->time_scale_solution_update(this->rk_stage[istage], CFL);
            this->solution_update.add(1.0, this->rk_stage[istage]);
        } else {
            this->solution_update.add(dt* this->butcher_tableau->get_b(istage),this->rk_stage[istage]);
        }
    }
}


template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::apply_limiter ()
{
    // Apply limiter at every RK stage
    if (this->limiter) {
        this->limiter->limit(this->dg->solution,
            this->dg->dof_handler,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->dg->high_order_grid->fe_system.tensor_degree(),
            this->dg->max_degree,
            this->dg->oneD_fe_collection_1state,
            this->dg->oneD_quadrature_collection);
    }
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
real RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::adjust_time_step (real dt)
{
    // Calculates relaxation parameter and modify the time step size as dt*=relaxation_parameter.
    // if not using RRK, the relaxation parameter will be set to 1, such that dt is not modified.
    this->relaxation_parameter_RRK_solver = this->relaxation_runge_kutta->update_relaxation_parameter(dt, this->dg, this->rk_stage, this->solution_update);
    dt *= this->relaxation_parameter_RRK_solver;
    this->modified_time_step = dt;
    return dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::allocate_runge_kutta_system ()
{

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int istage=0; istage<n_rk_stages; ++istage) {
        if (this->butcher_tableau->get_a(istage,istage)==0.0)     this->butcher_tableau_aii_is_zero[istage] = true;
    
    }
    if(this->all_parameters->use_inverse_mass_on_the_fly == false) {
        this->pcout << " evaluating inverse mass matrix..." << std::flush;
        this->dg->evaluate_mass_matrices(true); // creates and stores global inverse mass matrix
        //RRK needs both mass matrix and inverse mass matrix
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        ODEEnum ode_type = this->ode_param.ode_solver_type;
        if (ode_type == ODEEnum::rrk_explicit_solver){
            this->dg->evaluate_mass_matrices(false); // creates and stores global mass matrix
        }
    }
}

template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
