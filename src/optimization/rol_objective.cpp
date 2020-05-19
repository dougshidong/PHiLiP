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
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);
    ffd.deform_mesh(functional.dg->high_order_grid);
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

    const bool compute_dIdW = false;
    const bool compute_dIdX = true;
    const bool compute_d2I = false;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    auto &dIdXv = functional.dIdX;

    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

    auto &dIdXp = ROL_vector_to_dealii_vector_reference(g);
    dXvdXp.Tvmult(dIdXp, dIdXv);

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

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = true;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &hv = ROL_vector_to_dealii_vector_reference(output_vector);

    auto dXvdXp_input = functional.dg->high_order_grid.volume_nodes;

    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);
    dXvdXp.vmult(dXvdXp_input, v);
    functional.d2IdWdX.vmult(hv, dXvdXp_input);
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
    auto &hv = ROL_vector_to_dealii_vector_reference(output_vector);

    auto d2IdXdW_input = functional.dg->high_order_grid.volume_nodes;
    functional.d2IdWdX.Tvmult(d2IdXdW_input, v);

    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);
    dXvdXp.Tvmult(hv, d2IdXdW_input);
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

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = true;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &hv = ROL_vector_to_dealii_vector_reference(output_vector);

    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

    auto dXvdXp_input = functional.dg->high_order_grid.volume_nodes;
    dXvdXp.vmult(dXvdXp_input, v);

    auto d2IdXdX_dXvdXp_input = functional.dg->high_order_grid.volume_nodes;
    functional.d2IdXdX.vmult(d2IdXdX_dXvdXp_input, dXvdXp_input);

    dXvdXp.Tvmult(hv, d2IdXdX_dXvdXp_input);
}

template class ROLObjectiveSimOpt <PHILIP_DIM,1>;
template class ROLObjectiveSimOpt <PHILIP_DIM,2>;
template class ROLObjectiveSimOpt <PHILIP_DIM,3>;
template class ROLObjectiveSimOpt <PHILIP_DIM,4>;
template class ROLObjectiveSimOpt <PHILIP_DIM,5>;

} // PHiLiP namespace
