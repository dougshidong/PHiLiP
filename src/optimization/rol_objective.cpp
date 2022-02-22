#include "rol_to_dealii_vector.hpp"
#include "rol_objective.hpp"

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "mesh/meshmover_linear_elasticity.hpp"

#include "global_counter.hpp"

namespace PHiLiP {

template <int dim, int nstate>
ROLObjectiveSimOpt<dim,nstate>::ROLObjectiveSimOpt(
    Functional<dim,nstate,double> &_functional, 
    const FreeFormDeformation<dim> &_ffd,
    std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim,
    dealii::TrilinosWrappers::SparseMatrix *precomputed_dXvdXp)
    : functional(_functional)
    , ffd(_ffd)
    , ffd_design_variables_indices_dim(_ffd_design_variables_indices_dim)
{
    const unsigned int n_design_variables = ffd_design_variables_indices_dim.size();
    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_design_variables);
    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);
    ffd_des_var.reinit(row_part, ghost_row_part, MPI_COMM_WORLD);

    ffd.get_design_variables(ffd_design_variables_indices_dim, ffd_des_var);

    initial_ffd_des_var = ffd_des_var;
    initial_ffd_des_var.update_ghost_values();

    if (precomputed_dXvdXp) {
        if (precomputed_dXvdXp->m() == functional.dg->high_order_grid->volume_nodes.size() && precomputed_dXvdXp->n() == n_design_variables) {
            dXvdXp.copy_from(*precomputed_dXvdXp);
        }
    } else {
        ffd.get_dXvdXp ( *(functional.dg->high_order_grid), ffd_design_variables_indices_dim, dXvdXp);
    }
}


template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::update(
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    bool /*flag*/, int /*iter*/)
{
    functional.set_state(ROL_vector_to_dealii_vector_reference(des_var_sim));

    ffd_des_var =  ROL_vector_to_dealii_vector_reference(des_var_ctl);
    auto current_ffd_des_var = ffd_des_var;
    ffd.get_design_variables( ffd_design_variables_indices_dim, current_ffd_des_var);

    auto diff = ffd_des_var;
    diff -= current_ffd_des_var;
    const double l2_norm = diff.l2_norm();
    if (l2_norm != 0.0) {
        ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);

        //ffd.deform_mesh(*(functional.dg->high_order_grid));
        auto dXp = ffd_des_var;
        dXp -= initial_ffd_des_var;
        dXp.update_ghost_values();
        auto dXv = functional.dg->high_order_grid->volume_nodes;
        dXvdXp.vmult(dXv, dXp);
        dXv.update_ghost_values();
        functional.dg->high_order_grid->volume_nodes = functional.dg->high_order_grid->initial_volume_nodes;
        functional.dg->high_order_grid->volume_nodes += dXv;
        functional.dg->high_order_grid->volume_nodes.update_ghost_values();
    }
}


template <int dim, int nstate>
double ROLObjectiveSimOpt<dim,nstate>::value(
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &tol )
{
    // Tolerance tends to not be used except in the case of a reduced objective function.
    // In that scenario, tol is the constraint norm.
    // If the flow has not converged (>1e-5 or is nan), simply return a high functional.
    // This is likely happening in the linesearch while optimizing in the reduced-space.
    if (tol > 1e-5 || std::isnan(tol)) return 1e200;
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = false;
    return functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
}

template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::gradient_1(
    ROL::Vector<double> &gradient_sim,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = true;
    const bool compute_dIdX = false;
    const bool compute_d2I = false;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
    auto &dIdW = ROL_vector_to_dealii_vector_reference(gradient_sim);
    dIdW = functional.dIdw;
}

template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::gradient_2(
    ROL::Vector<double> &gradient_ctl,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false, compute_dIdX = true, compute_d2I = false;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dIdXv = functional.dIdX;

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(gradient_ctl);
    dXvdXp.Tvmult(dealii_output, dIdXv);

    //n_vmult += 1;

    // auto dIdXvs = dIdXv;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(functional.dg->high_order_grid->triangulation),
    //           functional.dg->high_order_grid->initial_mapping_fe_field,
    //           functional.dg->high_order_grid->dof_handler_grid,
    //           functional.dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs_transpose(dIdXv, dIdXvs);
    // }

    // auto &dIdXp = ROL_vector_to_dealii_vector_reference(gradient_ctl);
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (*(functional.dg->high_order_grid), ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.Tvmult(dIdXp, dIdXvs);
    // }

}

template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::hessVec_11(
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = true;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &hv = ROL_vector_to_dealii_vector_reference(output_vector);

    functional.d2IdWdW.vmult(hv, dealii_input);

    //n_vmult += 1;
}

template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::hessVec_12(
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    // auto dXvsdXp_input = functional.dg->high_order_grid->volume_nodes;
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.vmult(dXvsdXp_input, dealii_input);
    // }

    // auto dXvdXp_input = dXvsdXp_input;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(functional.dg->high_order_grid->triangulation),
    //           functional.dg->high_order_grid->initial_mapping_fe_field,
    //           functional.dg->high_order_grid->dof_handler_grid,
    //           functional.dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    // }

    // auto &d2IdWdXp_input = ROL_vector_to_dealii_vector_reference(output_vector);
    // {
    //     const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
    //     functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
    //     functional.d2IdWdX.vmult(d2IdWdXp_input, dXvdXp_input);
    // }

    auto dXvdXp_input = functional.dg->high_order_grid->volume_nodes;
    dXvdXp.vmult(dXvdXp_input, dealii_input);

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
        functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        functional.d2IdWdX.vmult(dealii_output, dXvdXp_input);
    }

    //n_vmult += 2;

}

template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::hessVec_21(
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = true;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    auto d2IdXdW_input = functional.dg->high_order_grid->volume_nodes;
    functional.d2IdWdX.Tvmult(d2IdXdW_input, dealii_input);

    // auto d2IdXvsdW_input = functional.dg->high_order_grid->volume_nodes;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(functional.dg->high_order_grid->triangulation),
    //           functional.dg->high_order_grid->initial_mapping_fe_field,
    //           functional.dg->high_order_grid->dof_handler_grid,
    //           functional.dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs_transpose(d2IdXdW_input, d2IdXvsdW_input);
    // }

    // auto &d2IdXpdW_input = ROL_vector_to_dealii_vector_reference(output_vector);
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (*(functional.dg->high_order_grid), ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.Tvmult(d2IdXpdW_input, d2IdXvsdW_input);
    // }

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    dXvdXp.Tvmult(dealii_output, d2IdXdW_input);

    //n_vmult += 2;
}

template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::hessVec_22(
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);


    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    // dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    // ffd.get_dXvsdXp (*(functional.dg->high_order_grid), ffd_design_variables_indices_dim, dXvsdXp);

    // auto dXvsdXp_input = functional.dg->high_order_grid->volume_nodes;
    // {
    //     dXvsdXp.vmult(dXvsdXp_input, dealii_input);
    // }

    // auto dXvdXp_input = dXvsdXp_input;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(functional.dg->high_order_grid->triangulation),
    //           functional.dg->high_order_grid->initial_mapping_fe_field,
    //           functional.dg->high_order_grid->dof_handler_grid,
    //           functional.dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    // }

    auto dXvdXp_input = functional.dg->high_order_grid->volume_nodes;
    dXvdXp.vmult(dXvdXp_input, dealii_input);

    auto d2IdXdXp_input = functional.dg->high_order_grid->volume_nodes;
    {
        const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
        functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        functional.d2IdXdX.vmult(d2IdXdXp_input, dXvdXp_input);
    }

    //auto d2IdXvsdXp_input = functional.dg->high_order_grid->volume_nodes;
    //{
    //    dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid->surface_nodes);
    //    MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //        meshmover(*(functional.dg->high_order_grid->triangulation),
    //          functional.dg->high_order_grid->initial_mapping_fe_field,
    //          functional.dg->high_order_grid->dof_handler_grid,
    //          functional.dg->high_order_grid->surface_to_volume_indices,
    //          dummy_vector);
    //    meshmover.apply_dXvdXvs_transpose(d2IdXdXp_input, d2IdXvsdXp_input);
    //}

    //auto &d2IdXpdXp_input = ROL_vector_to_dealii_vector_reference(output_vector);
    //{
    //    dXvsdXp.Tvmult(d2IdXpdXp_input, d2IdXvsdXp_input);
    //}

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    dXvdXp.Tvmult(dealii_output, d2IdXdXp_input);

    //n_vmult += 3;
}

template class ROLObjectiveSimOpt <PHILIP_DIM,1>;
template class ROLObjectiveSimOpt <PHILIP_DIM,2>;
template class ROLObjectiveSimOpt <PHILIP_DIM,3>;
template class ROLObjectiveSimOpt <PHILIP_DIM,4>;
template class ROLObjectiveSimOpt <PHILIP_DIM,5>;

} // PHiLiP namespace
