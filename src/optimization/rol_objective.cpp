#include "rol_to_dealii_vector.hpp"
#include "rol_objective.hpp"

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "mesh/meshmover_linear_elasticity.hpp"

namespace PHiLiP {

template <int dim, int nstate>
ROLObjectiveSimOpt<dim,nstate>::ROLObjectiveSimOpt(
    Functional<dim,nstate,double> &_functional, 
    const FreeFormDeformation<dim> &_ffd,
    std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim)
    : functional(_functional)
    , ffd(_ffd)
    , ffd_design_variables_indices_dim(_ffd_design_variables_indices_dim)
{
    ffd_des_var.reinit(ffd_design_variables_indices_dim.size());
    ffd.get_design_variables(ffd_design_variables_indices_dim, ffd_des_var);
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
        ffd.deform_mesh(functional.dg->high_order_grid);
    }
}


template <int dim, int nstate>
double ROLObjectiveSimOpt<dim,nstate>::value(
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = false;
    return functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
}

template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::gradient_1(
    ROL::Vector<double> &g,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = true;
    const bool compute_dIdX = false;
    const bool compute_d2I = false;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
    auto &dIdW = ROL_vector_to_dealii_vector_reference(g);
    dIdW = functional.dIdw;
}

template <int dim, int nstate>
void ROLObjectiveSimOpt<dim,nstate>::gradient_2(
    ROL::Vector<double> &g,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false, compute_dIdX = true, compute_d2I = false;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dIdXv = functional.dIdX;

    auto dIdXvs = dIdXv;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(functional.dg->high_order_grid.triangulation),
              functional.dg->high_order_grid.initial_mapping_fe_field,
              functional.dg->high_order_grid.dof_handler_grid,
              functional.dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);
        meshmover.apply_dXvdXvs_transpose(dIdXv, dIdXvs);
    }

    auto &dIdXp = ROL_vector_to_dealii_vector_reference(g);
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.Tvmult(dIdXp, dIdXvs);
    }

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

    const auto &v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &hv = ROL_vector_to_dealii_vector_reference(output_vector);

    functional.d2IdWdW.vmult(hv, v);
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

    const auto &v = ROL_vector_to_dealii_vector_reference(input_vector);

    auto dXvsdXp_v = functional.dg->high_order_grid.volume_nodes;
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.vmult(dXvsdXp_v, v);
    }

    auto dXvdXp_v = dXvsdXp_v;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(functional.dg->high_order_grid.triangulation),
              functional.dg->high_order_grid.initial_mapping_fe_field,
              functional.dg->high_order_grid.dof_handler_grid,
              functional.dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);
        meshmover.apply_dXvdXvs(dXvsdXp_v, dXvdXp_v);
    }

    auto &d2IdWdXp_v = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
        functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        functional.d2IdWdX.vmult(d2IdWdXp_v, dXvdXp_v);
    }

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

    const auto &v = ROL_vector_to_dealii_vector_reference(input_vector);

    auto d2IdXdW_v = functional.dg->high_order_grid.volume_nodes;
    functional.d2IdWdX.Tvmult(d2IdXdW_v, v);

    auto d2IdXvsdW_v = functional.dg->high_order_grid.volume_nodes;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(functional.dg->high_order_grid.triangulation),
              functional.dg->high_order_grid.initial_mapping_fe_field,
              functional.dg->high_order_grid.dof_handler_grid,
              functional.dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);
        meshmover.apply_dXvdXvs_transpose(d2IdXdW_v, d2IdXvsdW_v);
    }

    auto &d2IdXpdW_v = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.Tvmult(d2IdXpdW_v, d2IdXvsdW_v);
    }
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


    const auto &v = ROL_vector_to_dealii_vector_reference(input_vector);

    dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    ffd.get_dXvsdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);

    auto dXvsdXp_v = functional.dg->high_order_grid.volume_nodes;
    {
        dXvsdXp.vmult(dXvsdXp_v, v);
    }

    auto dXvdXp_v = dXvsdXp_v;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(functional.dg->high_order_grid.triangulation),
              functional.dg->high_order_grid.initial_mapping_fe_field,
              functional.dg->high_order_grid.dof_handler_grid,
              functional.dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);
        meshmover.apply_dXvdXvs(dXvsdXp_v, dXvdXp_v);
    }


    auto d2IdXdXp_v = functional.dg->high_order_grid.volume_nodes;
    {
        const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
        functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        functional.d2IdXdX.vmult(d2IdXdXp_v, dXvdXp_v);
    }

    auto d2IdXvsdXp_v = functional.dg->high_order_grid.volume_nodes;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(functional.dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(functional.dg->high_order_grid.triangulation),
              functional.dg->high_order_grid.initial_mapping_fe_field,
              functional.dg->high_order_grid.dof_handler_grid,
              functional.dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);
        meshmover.apply_dXvdXvs_transpose(d2IdXdXp_v, d2IdXvsdXp_v);
    }

    auto &d2IdXpdXp_v = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        dXvsdXp.Tvmult(d2IdXpdXp_v, d2IdXvsdXp_v);
    }

}

template class ROLObjectiveSimOpt <PHILIP_DIM,1>;
template class ROLObjectiveSimOpt <PHILIP_DIM,2>;
template class ROLObjectiveSimOpt <PHILIP_DIM,3>;
template class ROLObjectiveSimOpt <PHILIP_DIM,4>;
template class ROLObjectiveSimOpt <PHILIP_DIM,5>;

} // PHiLiP namespace
