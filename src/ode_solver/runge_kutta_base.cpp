#include "runge_kutta_base.h"

namespace PHiLiP {
namespace ODE {

template<int dim, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, real, n_rk_stages, MeshType>::RungeKuttaBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input,
            std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
            : ODESolverBase<dim,real,MeshType>(dg_input, pod)
            , relaxation_runge_kutta(RRK_object_input)
            , solver(dg_input)
{}            

template<int dim, typename real, int n_rk_stages, typename MeshType>
RungeKuttaBase<dim, real, n_rk_stages, MeshType>::RungeKuttaBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
            : RungeKuttaBase(dg_input, RRK_object_input, nullptr)
{}
template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaBase<dim, real, n_rk_stages, MeshType>::step_in_time(real dt, const bool pseudotime)
{
    this->original_time_step = dt;
    this->solution_update = this->dg->solution; //storing u_n
    for (int istage = 0; istage < n_rk_stages; ++istage){
        this->calculate_stage_solution(istage, dt, pseudotime); // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(istage) <implicit>
        this->apply_limiter();
        this->calculate_stage_derivative(istage, dt); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
        // this->pcout<<"RK Stage: "<<istage;
        // this->pcout<<"RK Stage: "<<istage;
        // const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
        // std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
        // auto metric_cell_1 = this->dg->high_order_grid->dof_handler_grid.begin_active();
        // for (auto current_cell = this->dg->dof_handler.begin_active(); current_cell!=this->dg->dof_handler.end(); ++current_cell, ++metric_cell_1) {
        //     if (!current_cell->is_locally_owned()) continue;
        //     const dealii::types::global_dof_index current_cell_index = current_cell->active_cell_index();
        //     if(current_cell_index==1){
        //         const unsigned int n_dofs_cell = this->dg->fe_collection[3].dofs_per_cell;                  
        //         current_dofs_indices.resize(n_dofs_cell);
        //         current_cell->get_dof_indices (current_dofs_indices);             
        //         for(unsigned int idof=0; idof<2; idof++){
        //             this->pcout<<"\nCell 1 solution (idof = "<<idof<<"): "<<this->dg->solution(current_dofs_indices[idof]);
        //             this->pcout<<"\nCell 1 solution update (idof = "<<idof<<"): "<<this->solution_update(current_dofs_indices[idof]);
        //         }
        //     }
        // }
    }
    dt = this->adjust_time_step(dt);
    //this->pcout<<"\nadjusted time step: "<<dt<<"\n";
    this->sum_stages(dt, pseudotime); // u_np1 = u_n + dt* sum(k_i * b_i)
    this->dg->solution = this->solution_update;
    // this->pcout<<"SOLUTION UPDATED\n";
    // const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    // std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    // auto metric_cell_1 = this->dg->high_order_grid->dof_handler_grid.begin_active();
    // for (auto current_cell = this->dg->dof_handler.begin_active(); current_cell!=this->dg->dof_handler.end(); ++current_cell, ++metric_cell_1) {
    //     if (!current_cell->is_locally_owned()) continue;
    //     const dealii::types::global_dof_index current_cell_index = current_cell->active_cell_index();
    //     if(current_cell_index==1){
    //         const unsigned int n_dofs_cell = this->dg->fe_collection[3].dofs_per_cell;                  
    //         current_dofs_indices.resize(n_dofs_cell);
    //         current_cell->get_dof_indices (current_dofs_indices);             
    //         for(unsigned int idof=0; idof<2; idof++){
    //             this->pcout<<"\nCell 1 solution (idof = "<<idof<<"): "<<this->dg->solution(current_dofs_indices[idof]);
    //             this->pcout<<"\nCell 1 solution update (idof = "<<idof<<"): "<<this->solution_update(current_dofs_indices[idof]);
    //         }
    //     }
    // }
     // Calculate numerical entropy with FR correction. Does nothing if use has not selected param.
    this->FR_entropy_contribution_RRK_solver = relaxation_runge_kutta->compute_FR_entropy_contribution(dt, this->dg, this->rk_stage, true);
    this->apply_limiter();
    ++(this->current_iteration);
    this->current_time += dt;
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaBase<dim, real, n_rk_stages, MeshType>::allocate_ode_system()
{
    this->pcout << "Allocating ODE system..." << std::flush;
    this->solution_update.reinit(this->dg->right_hand_side);

    this->pcout << std::endl;

    PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
    using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
    
    if (parameters.ode_solver_param.ode_solver_type != ODESolverEnum::PERK_solver){
        this->rk_stage.resize(n_rk_stages);
        for (int istage=0; istage<n_rk_stages; ++istage) {
            this->rk_stage[istage].reinit(this->dg->solution);
        }
    }

    // allocation for PERK schemes
    if (parameters.ode_solver_param.ode_solver_type == ODESolverEnum::PERK_solver){
        this->rk_stage_k.resize(this->group_ID.size());
        for (size_t i = 0; i < this->group_ID.size(); ++i){
            this->rk_stage_k[i].resize(n_rk_stages);
            for (int j = 0; j < n_rk_stages; ++j){
                this->rk_stage_k[i][j].reinit(this->dg->solution);
            }
        }
    }

    
    this->allocate_runge_kutta_system();
}

/*
Templates with n_rk_stages > 4 are for the LSRK method
*/
template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,5, dealii::Triangulation<PHILIP_DIM> >; 
template class RungeKuttaBase<PHILIP_DIM, double,9, dealii::Triangulation<PHILIP_DIM> >; 
template class RungeKuttaBase<PHILIP_DIM, double,10, dealii::Triangulation<PHILIP_DIM> >; 
template class RungeKuttaBase<PHILIP_DIM, double,16, dealii::Triangulation<PHILIP_DIM> >; 

template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,5, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,9, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,10, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaBase<PHILIP_DIM, double,16, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;

#if PHILIP_DIM != 1
    template class RungeKuttaBase<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,5, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,9, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,10, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaBase<PHILIP_DIM, double,16, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif
} // ODE namespace
} // PHiLiP namespace
