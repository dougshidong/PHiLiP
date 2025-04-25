#include "runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType> 
RungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
        std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
        std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , butcher_tableau(rk_tableau_input)
        , relaxation_runge_kutta(RRK_object_input)
        , solver(dg_input)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::step_in_time (real dt, const bool pseudotime)
{
    this->original_time_step = dt;
    this->solution_update = this->dg->solution; //storing u_n

    //calculating stages **Note that rk_stage[i] stores the RHS at a partial time-step (not solution u)
    for (int i = 0; i < n_rk_stages; ++i){

        this->rk_stage[i]=0.0; //resets all entries to zero
        
        for (int j = 0; j < i; ++j){
            if (this->butcher_tableau->get_a(i,j) != 0){
                this->rk_stage[i].add(this->butcher_tableau->get_a(i,j), this->rk_stage[j]);
            }
        } //sum(a_ij *k_j), explicit part

        
        if(pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update(rk_stage[i], CFL);
        }else {
            this->rk_stage[i]*=dt; 
        }//dt * sum(a_ij * k_j)
        
        this->rk_stage[i].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)
       
        //implicit solve if there is a nonzero diagonal element
        if (!this->butcher_tableau_aii_is_zero[i]){
            /* // AD version - keeping in comments as it may be useful for future testing
            // Solve (M/dt - dRdW) / a_ii * dw = R
            // w = w + dw
            // Note - need to have assembled residual using this->dg->assemble_residual(true);
            //        and have mass matrix assembled, and include linear_solver
            dealii::LinearAlgebra::distributed::Vector<double> temp_u(this->dg->solution.size());

            this->dg->system_matrix *= -1.0/butcher_tableau_a[i][i]; //system_matrix = -1/a_ii*dRdW
            this->dg->add_mass_matrices(1.0/butcher_tableau_a[i][i]/dt); //system_matrix = -1/a_ii*dRdW + M/dt/a_ii = A

            solve_linear ( //Solve Ax=b using Aztec00 gmres
                        this->dg->system_matrix, //A = -1/a_ii*dRdW + M/dt/a_ii
                        this->dg->right_hand_side, //b = R
                        temp_u, // result,  x = dw
                        this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

            this->rk_stage[i].add(1.0, temp_u);
            */

            //JFNK version
            solver.solve(dt*this->butcher_tableau->get_a(i,i), this->rk_stage[i]);
            this->rk_stage[i] = solver.current_solution_estimate;

        } // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(i) <implicit>
        
        // If using the entropy formulation of RRK, solutions must be stored.
        // Call store_stage_solutions before overwriting rk_stage with the derivative.
        relaxation_runge_kutta->store_stage_solutions(i, rk_stage[i]);

        this->dg->solution = this->rk_stage[i];

        // Apply limiter at every RK stage
        if (this->limiter) {
            this->limiter->limit(this->dg->solution,
                this->dg->dof_handler,
                this->dg->fe_collection,
                this->dg->volume_quadrature_collection,
                this->dg->high_order_grid->fe_system.tensor_degree(),
                this->dg->max_degree,
                this->dg->oneD_fe_collection_1state,
                this->dg->oneD_quadrature_collection,
                dt);
        }

        //set the DG current time for unsteady source terms
        this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(i)*dt);
        
        //solve the system's right hande side
        this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(i)))

        if(this->all_parameters->use_inverse_mass_on_the_fly){
            this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, this->rk_stage[i]); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
        } else{
            this->dg->global_inverse_mass_matrix.vmult(this->rk_stage[i], this->dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
        }
    }

    // Calculates relaxation parameter and modify the time step size as dt*=relaxation_parameter.
    // if not using RRK, the relaxation parameter will be set to 1, such that dt is not modified.
    this->relaxation_parameter_RRK_solver = relaxation_runge_kutta->update_relaxation_parameter(dt, this->dg, this->rk_stage, this->solution_update);
    dt *= this->relaxation_parameter_RRK_solver;
    this->modified_time_step = dt;

    //assemble solution from stages
    for (int i = 0; i < n_rk_stages; ++i){
        if (pseudotime){
            const double CFL = this->butcher_tableau->get_b(i) * dt;
            this->dg->time_scale_solution_update(this->rk_stage[i], CFL);
            this->solution_update.add(1.0, this->rk_stage[i]);
        } else {
            this->solution_update.add(dt* this->butcher_tableau->get_b(i),this->rk_stage[i]); 
        }
    }
    this->dg->solution = this->solution_update; // u_np1 = u_n + dt* sum(k_i * b_i)

    // Calculate numerical entropy with FR correction. Does nothing if use has not selected param.
    this->FR_entropy_contribution_RRK_solver = relaxation_runge_kutta->compute_FR_entropy_contribution(dt, this->dg, this->rk_stage, true);

    // Apply limiter at every RK stage
    if (this->limiter) {
        this->limiter->limit(this->dg->solution,
            this->dg->dof_handler,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->dg->high_order_grid->fe_system.tensor_degree(),
            this->dg->max_degree,
            this->dg->oneD_fe_collection_1state,
            this->dg->oneD_quadrature_collection,
            dt);
    }
    
    ++(this->current_iteration);
    this->current_time += dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system..." << std::flush;
    this->solution_update.reinit(this->dg->right_hand_side);
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
    this->pcout << std::endl;
    
    this->rk_stage.resize(n_rk_stages);
    for (int i=0; i<n_rk_stages; ++i) {
        this->rk_stage[i].reinit(this->dg->solution);
    }

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int i=0; i<n_rk_stages; ++i) {
        if (this->butcher_tableau->get_a(i,i)==0.0)     this->butcher_tableau_aii_is_zero[i] = true;
    
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
