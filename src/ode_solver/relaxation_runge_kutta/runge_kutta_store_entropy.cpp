#include "rrk_ode_solver_base.h"
#include "physics/euler.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType>
RKNumEntropy<dim,real,n_rk_stages,MeshType>::RKNumEntropy(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>(dg_input,rk_tableau_input)
{
    this->rk_stage_solution.resize(n_rk_stages);
   
    // TEMP this should select the actual physics of the problem
    PHiLiP::Parameters::AllParameters parameters_euler = *(this->dg->all_parameters);
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::euler;
    this->euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void RKNumEntropy<dim,real,n_rk_stages,MeshType>::store_stage_solutions(const int istage)
{
    //Store the solution value
    //This function is called before rk_stage is modified to hold the time-derivative
    this->rk_stage_solution[istage]=this->rk_stage[istage]; 
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> RKNumEntropy<dim,real,n_rk_stages,MeshType>::compute_entropy_vars(const dealii::LinearAlgebra::distributed::Vector<double> &u) const
{
    // hard-code nstate for Euler/NS - ODESolverFactory has already ensured that we use Euler/NS
    const unsigned int nstate = dim + 2;
    // Currently only implemented for constant p
    const unsigned int poly_degree = this->dg->get_max_fe_degree();
    if (poly_degree != this->dg->get_min_fe_degree()){
        this->pcout << "Error: Entropy RRK is only implemented for uniform p. Aborting..." << std::endl;
        std::abort();
    }

    const unsigned int n_dofs_cell = this->dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = this->dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, this->dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(this->dg->oneD_fe_collection_1state[poly_degree], this->dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, this->dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(this->dg->oneD_fe_collection_1state[poly_degree], this->dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(this->dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr< Physics::Euler<dim,dim+2,double> > euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(this->dg->all_parameters));

    for (auto cell = this->dg->dof_handler.begin_active(); cell!=this->dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = this->dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = this->dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = u(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(unsigned int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(unsigned int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }

            std::array<double,nstate> entropy_var = euler_physics->compute_entropy_variables(soln_state);

            for(unsigned int istate=0; istate<nstate; istate++){
                if(iquad==0)
                    entropy_var_at_q[istate].resize(n_quad_pts);
                entropy_var_at_q[istate][iquad] = entropy_var[istate];
            }
        }
        for(unsigned int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
            }
        }
    }
    return entropy_var_hat_global;
}


template <int dim, typename real, int n_rk_stages, typename MeshType>
double RKNumEntropy<dim,real,n_rk_stages,MeshType>::compute_FR_entropy_contribution() const
{
    double entropy_contribution = 0;

    for (int istage = 0; istage<n_rk_stages; ++istage){

        // Recall rk_stage is IMM * RHS
        // therefore, RHS = M * rk_stage = M * du/dt
        dealii::LinearAlgebra::distributed::Vector<double> M_matrix_times_rk_stage(this->dg->solution);
        dealii::LinearAlgebra::distributed::Vector<double> MpK_matrix_times_rk_stage(this->dg->solution);
        if(this->dg->all_parameters->use_inverse_mass_on_the_fly)
        {
            this->dg->apply_global_mass_matrix(this->rk_stage[istage],M_matrix_times_rk_stage,
                    this->dg->use_auxiliary_eq, // use_auxiliary_eq,
                    true // use M norm
                    );

            this->dg->apply_global_mass_matrix(this->rk_stage[istage],MpK_matrix_times_rk_stage,
                    this->dg->use_auxiliary_eq, // use_auxiliary_eq,
                    false // use M+K norm
                    );
        } else {
            this->pcout << "ERROR: FR Numerical entropy estimate currently not compatible with use_inverse_mass_matrix = false. Please modify params." << std::endl;
            std::abort();
        }
        
        //transform solution into entropy variables
        dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global = this->compute_entropy_vars(this->rk_stage_solution[istage]);
        
        double entropy_contribution_stage = entropy_var_hat_global * MpK_matrix_times_rk_stage - entropy_var_hat_global * M_matrix_times_rk_stage;
        
        entropy_contribution += this->butcher_tableau->get_b(istage) * entropy_contribution_stage;
    }

    entropy_contribution *= this->modified_time_step;

    return entropy_contribution;
}

template class RKNumEntropy<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RKNumEntropy<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RKNumEntropy<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RKNumEntropy<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RKNumEntropy<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RKNumEntropy<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RKNumEntropy<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RKNumEntropy<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RKNumEntropy<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RKNumEntropy<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RKNumEntropy<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RKNumEntropy<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
