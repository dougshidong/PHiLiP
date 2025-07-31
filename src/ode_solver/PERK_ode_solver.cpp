#include "PERK_ode_solver.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType> 
PERKODESolver<dim,real,n_rk_stages, MeshType>::PERKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
        std::shared_ptr<PERKTableauBase<dim,real,MeshType>> rk_tableau_input,
        std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
        : RungeKuttaBase<dim,real,n_rk_stages,MeshType>(dg_input, RRK_object_input)
        , butcher_tableau(rk_tableau_input)
{}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim,real,n_rk_stages, MeshType>::calculate_stage_solution (int istage, real dt, const bool pseudotime)
{
    stage_solution *= 0;
    for (size_t k = 0; k < this->group_ID.size(); ++k){ // calculate stage solutions corresponding to tableaus
        if (this->calc_stage[k][istage]==true){
            
            //this->rk_stage_k[k][istage]=0.0; //resets all entries to zero
            for (int j = 0; j < istage; ++j){
                if (this->butcher_tableau->get_a(istage,j, k+1) != 0){
                    //this->rk_stage_k[k][istage].add(this->butcher_tableau->get_a(istage,j, k+1), this->rk_stage_k[k][j]);
                    stage_solution.add(this->butcher_tableau->get_a(istage,j, k+1), this->rk_stage_k[k][j]);
                }
            } //sum(a_ij *k_j), explicit part
            
            if(pseudotime) {
                const double CFL = dt;
                this->dg->time_scale_solution_update(this->rk_stage_k[k][istage], CFL);
            }else {
                //this->rk_stage_k[k][istage]*=dt;
                stage_solution*=dt;
            }//dt * sum(a_ij * k_j)
            
            //this->rk_stage_k[k][istage].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)
            stage_solution.add(1.0, this->solution_update);


            //this->rk_stage_k[0][istage].print(std::cout);

        }
    } 
    this->dg->solution = stage_solution; 

        //std::abort();
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim,real,n_rk_stages,MeshType>::calculate_stage_derivative (int istage, real dt)
{
     //set the DG current time for unsteady source terms
     this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(istage)*dt);

    for (size_t k = 0; k < this->group_ID.size(); ++k){
        if (this->calc_stage[k][istage]==true){
            this->rk_stage_k[k][istage] = this->dg->solution;
            this->dg->assemble_residual(false, false, false, 0.0, this->group_ID[k]); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(istage)))
            if(this->all_parameters->use_inverse_mass_on_the_fly){
                this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, this->rk_stage_k[k][istage]); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            } else{
                this->dg->global_inverse_mass_matrix.vmult(this->rk_stage_k[k][istage], this->dg->right_hand_side); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            }

        // this->pcout << this->dg->right_hand_side.size() << std::endl;
        //     for (unsigned int i = 0 ; i < this->dg->right_hand_side.size(); ++i){
        //         this->pcout << this->dg->right_hand_side(i) << " " ;
        //     }
        // this->pcout << std::endl;

         }

     }
}


template<int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim,real,n_rk_stages,MeshType>::sum_stages (real dt, const bool pseudotime)
{
    //assemble solution from stages
    for (size_t k = 0; k < this->group_ID.size(); ++k){
            for (int istage = 0; istage < n_rk_stages; ++istage){
                if (pseudotime){
                    /*
                    const double CFL = this->butcher_tableau->get_b(istage) * dt;
                    this->dg->time_scale_solution_update(this->rk_stage[istage], CFL);
                    this->solution_update.add(1.0, this->rk_stage[istage]);
                    */
                    std::cout << "not implemented for pseudotime" << std::endl;
                    std::abort();
                } else {
                    if (this->calc_stage[k][istage]==true){
                        this->solution_update.add(dt* this->butcher_tableau->get_b(istage),this->rk_stage_k[k][istage]);
                    }
                }
            }
        }
                        // this->pcout << this->solution_update.size() << std::endl;
                        // for (unsigned int i = 0 ; i < this->solution_update.size(); ++i){
                        //     this->pcout << this->solution_update(i) << " " ;
                        // }
                        // this->pcout << std::endl;
       // std::abort();
    
}        


template<int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim,real,n_rk_stages,MeshType>::apply_limiter ()
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
real PERKODESolver<dim,real,n_rk_stages,MeshType>::adjust_time_step (real dt)
{
    // Calculates relaxation parameter and modify the time step size as dt*=relaxation_parameter.
    // if not using RRK, the relaxation parameter will be set to 1, such that dt is not modified.
    this->relaxation_parameter_RRK_solver = this->relaxation_runge_kutta->update_relaxation_parameter(dt, this->dg, this->rk_stage, this->solution_update);
    dt *= this->relaxation_parameter_RRK_solver;
    this->modified_time_step = dt;
    return dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void PERKODESolver<dim,real,n_rk_stages,MeshType>::allocate_runge_kutta_system ()
{

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int istage=0; istage<n_rk_stages; ++istage) {
        if (this->butcher_tableau->get_a(istage,istage, 1)==0.0)     this->butcher_tableau_aii_is_zero[istage] = true;
    
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

    // store whether or not to calculate stage
    this->calc_stage.resize(this->group_ID.size());
    for (size_t k = 0; k < this->group_ID.size(); ++k) {
        this->calc_stage[k].resize(n_rk_stages);
        for (int j = 0; j < n_rk_stages; ++j) {
            bool calcStage = false;
            for (int i = 0; i < n_rk_stages; ++i) {
                if (this->butcher_tableau->get_a(i, j, k+1) != 0 || this->butcher_tableau->get_b(j) != 0) {
                    calcStage = true;
                    break;
                }
            }
            this->calc_stage[k][j] = calcStage;
        }
   }

   stage_solution.reinit(this->dg->solution);
}

template class PERKODESolver<PHILIP_DIM, double,10, dealii::Triangulation<PHILIP_DIM> >;
template class PERKODESolver<PHILIP_DIM, double,10, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class PERKODESolver<PHILIP_DIM, double,10, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace