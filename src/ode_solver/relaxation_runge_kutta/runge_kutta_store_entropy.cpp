#include "rrk_ode_solver_base.h"
#include "physics/euler.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
RKNumEntropy<dim,real,MeshType>::RKNumEntropy(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : EmptyRRKBase<dim,real,MeshType>(rk_tableau_input)
        , butcher_tableau(rk_tableau_input)
        , n_rk_stages(butcher_tableau->n_rk_stages)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
{
    this->rk_stage_solution.resize(n_rk_stages);
   
}

template <int dim, typename real, typename MeshType>
void RKNumEntropy<dim,real,MeshType>::store_stage_solutions(const int istage, const dealii::LinearAlgebra::distributed::Vector<double> rk_stage_i)
{
    //Store the solution value
    //This function is called before rk_stage is modified to hold the time-derivative
    this->rk_stage_solution[istage] = rk_stage_i; 
}

template <int dim, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> RKNumEntropy<dim,real,MeshType>::compute_entropy_vars(const dealii::LinearAlgebra::distributed::Vector<double> &u,
        std::shared_ptr<DGBase<dim,real,MeshType>> dg) const
{
    // hard-code nstate for Euler/NS - ODESolverFactory has already ensured that we use Euler/NS
    const unsigned int nstate = dim + 2;
    // Currently only implemented for constant p
    const unsigned int poly_degree = dg->get_max_fe_degree();
    if (poly_degree != dg->get_min_fe_degree()){
        this->pcout << "Error: Entropy RRK is only implemented for uniform p. Aborting..." << std::endl;
        std::abort();
    }
    
    // Select Euler physics. ODESolverFactory has already ensured that we are using Euler orSelect Euler physics. ODESolverFactory has already ensured that we are using Euler or NS, which both use the same entropy variable computation.
    PHiLiP::Parameters::AllParameters parameters_euler = *(dg->all_parameters);
    parameters_euler.pde_type = Parameters::AllParameters::PartialDifferentialEquation::euler;
    std::shared_ptr < Physics::Euler<dim, dim+2, double > > euler_physics = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_euler));

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
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


template <int dim, typename real, typename MeshType>
double RKNumEntropy<dim,real,MeshType>::compute_FR_entropy_contribution(const real dt, 
        std::shared_ptr<DGBase<dim,real,MeshType>> dg,
        const std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
        const bool compute_K_norm) const
{
    double entropy_contribution = 0;

    for (int istage = 0; istage<n_rk_stages; ++istage){

        // Recall rk_stage is IMM * RHS
        // therefore, RHS = M * rk_stage = M * du/dt
        dealii::LinearAlgebra::distributed::Vector<double> M_matrix_times_rk_stage(dg->solution);
        dealii::LinearAlgebra::distributed::Vector<double> MpK_matrix_times_rk_stage(dg->solution);
        if(dg->all_parameters->use_inverse_mass_on_the_fly)
        {
            dg->apply_global_mass_matrix(rk_stage[istage],M_matrix_times_rk_stage,
                    false, // use_auxiliary_eq,
                    true // use M norm
                    );

            dg->apply_global_mass_matrix(rk_stage[istage],MpK_matrix_times_rk_stage,
                    false, // use_auxiliary_eq,
                    false // use M+K norm
                    );
        } else {
            this->pcout << "ERROR: FR Numerical entropy estimate currently not compatible with use_inverse_mass_matrix = false. Please modify params." << std::endl;
            std::abort();
        }
        
        // transform solution into entropy variables based on PDE type
        using PDEEnum = Parameters::AllParameters::PartialDifferentialEquation;
        dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global=0;
        if (dg->all_parameters->pde_type == PDEEnum::burgers_inviscid){
            entropy_var_hat_global = this->rk_stage_solution[istage];
        } else if (dg->all_parameters->pde_type == PDEEnum::euler || dg->all_parameters->pde_type == PDEEnum::navier_stokes) {
            entropy_var_hat_global = this->compute_entropy_vars(this->rk_stage_solution[istage], dg);
        } else {
            this->pcout << "ERROR: Cannot store FR-corrected numerical entropy for this PDE type. Aborting..." << std::endl;
            std::abort();
        }
        
        double entropy_contribution_stage = 0;
        if (compute_K_norm) {
            entropy_contribution_stage = entropy_var_hat_global * MpK_matrix_times_rk_stage - entropy_var_hat_global * M_matrix_times_rk_stage;
        }else {
            entropy_contribution_stage = entropy_var_hat_global * M_matrix_times_rk_stage;
        }
        
        entropy_contribution += this->butcher_tableau->get_b(istage) * entropy_contribution_stage;
    }

    entropy_contribution *= dt;

    return entropy_contribution;
}

template class RKNumEntropy<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RKNumEntropy<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RKNumEntropy<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
