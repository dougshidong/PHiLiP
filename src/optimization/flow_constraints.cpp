#include "optimization/flow_constraints.hpp"

#include "ode_solver/ode_solver.h"

#include <Epetra_RowMatrixTransposer.h>

namespace PHiLiP {

Teuchos::RCP<const dealii_Vector> get_rcp_to_VectorType(const ROL::Vector<double> &x)
{
    return (Teuchos::dyn_cast<const AdaptVector>(x)).getVector();
}

Teuchos::RCP<dealii_Vector> get_rcp_to_VectorType(ROL::Vector<double> &x)
{
    return (Teuchos::dyn_cast<AdaptVector>(x)).getVector();
}

const dealii_Vector & get_ROLvec_to_VectorType(const ROL::Vector<double> &x)
{
    return *(Teuchos::dyn_cast<const AdaptVector>(x)).getVector();
}

dealii_Vector &get_ROLvec_to_VectorType(ROL::Vector<double> &x)
{
    return *(Teuchos::dyn_cast<AdaptVector>(x)).getVector();
}

template<int dim>
FlowConstraints<dim>
::FlowConstraints(std::shared_ptr<DGBase<dim,double>> &_dg, 
                 const FreeFormDeformation<dim> &_ffd,
                 std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim)
    : dg(_dg)
    , ffd(_ffd)
    , ffd_design_variables_indices_dim(_ffd_design_variables_indices_dim)
{
    ffd_des_var.reinit(ffd_design_variables_indices_dim.size());
    ffd.get_design_variables(ffd_design_variables_indices_dim, ffd_des_var);

    dealii::ParameterHandler parameter_handler;
    Parameters::LinearSolverParam::declare_parameters (parameter_handler);
    linear_solver_param.parse_parameters (parameter_handler);
    linear_solver_param.max_iterations = 1000;
    linear_solver_param.restart_number = 200;
    linear_solver_param.linear_residual = 1e-13;
    linear_solver_param.ilut_fill = 0;
    linear_solver_param.ilut_drop = 0.0;
    linear_solver_param.ilut_rtol = 1.0;
    linear_solver_param.ilut_atol = 0.0;
    linear_solver_param.linear_solver_output = Parameters::OutputEnum::quiet;
    linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;
}

template<int dim>
void FlowConstraints<dim>
::update_1( const ROL::Vector<double>& des_var_sim, bool flag, int iter )
{
        (void) flag; (void) iter;
        dg->solution = get_ROLvec_to_VectorType(des_var_sim);
        dg->solution.update_ghost_values();
}

template<int dim>
void FlowConstraints<dim>
::update_2( const ROL::Vector<double>& des_var_ctl, bool flag, int iter )
{
    (void) flag; (void) iter;
    ffd_des_var =  get_ROLvec_to_VectorType(des_var_ctl);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);

    ffd.deform_mesh(dg->high_order_grid);
}

template<int dim>
void FlowConstraints<dim>
::solve(
    ROL::Vector<double>& constraint_values,
    ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl,
    double& /*tol*/
    )
{

    update_2(des_var_ctl);

    dg->output_results_vtk(i_out++);
    ffd.output_ffd_vtu(i_out);
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver_1 = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver_1->steady_state();

    dg->assemble_residual();
    auto &constraint = get_ROLvec_to_VectorType(constraint_values);
    constraint = dg->right_hand_side;
    auto &des_var_sim_v = get_ROLvec_to_VectorType(des_var_sim);
    des_var_sim_v = dg->solution;
}

template<int dim>
void FlowConstraints<dim>
::value(
    ROL::Vector<double>& constraint_values,
    const ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl,
    double &/*tol*/
    )
{

    update_1(des_var_sim);
    update_2(des_var_ctl);

    dg->assemble_residual();
    auto &constraint = get_ROLvec_to_VectorType(constraint_values);
    constraint = dg->right_hand_side;
}
    
template<int dim>
void FlowConstraints<dim>
::applyJacobian_1(
    ROL::Vector<double>& output_vector,
    const ROL::Vector<double>& input_vector,
    const ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl,
    double& /*tol*/
    )
{

    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);
    this->dg->system_matrix.vmult(output_vector_v, input_vector_v);
}

template<int dim>
void FlowConstraints<dim>
::applyInverseJacobian_1(
    ROL::Vector<double>& output_vector,
    const ROL::Vector<double>& input_vector,
    const ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl,
    double& /*tol*/ )
{

    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    std::cout << __PRETTY_FUNCTION__ << std::endl;
    //solve_linear_2 (
    //    this->dg->system_matrix,
    //    input_vector_v,
    //    output_vector_v,
    //    this->linear_solver_param);
    // try {
    //     solve_linear_2 (
    //         this->dg->system_matrix,
    //         input_vector_v,
    //         output_vector_v,
    //         this->linear_solver_param);
    // } catch (...) {
    //     std::cout << "Failed to solve linear system in " << __PRETTY_FUNCTION__ << std::endl;
    //     output_vector.setScalar(1.0);
    // }

    // Input vector is copied into temporary non-const vector.
    auto input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

    solve_linear (dg->system_matrix, input_vector_v, output_vector_v, this->linear_solver_param);
    //solve_linear_2 ( this->dg->system_matrix, input_vector_v, output_vector_v, this->linear_solver_param);
    //try {
    //  solve_linear (dg->system_matrix, input_vector_v, output_vector_v, this->linear_solver_param);
    //} catch (...) {
    //    std::cout << "Failed to solve linear system in " << __PRETTY_FUNCTION__ << std::endl;
    //    output_vector.setScalar(1.0);
    //}
}

template<int dim>
void FlowConstraints<dim>
::applyInverseAdjointJacobian_1( ROL::Vector<double>& output_vector,
const ROL::Vector<double>& input_vector,
const ROL::Vector<double>& des_var_sim,
const ROL::Vector<double>& des_var_ctl,
double& /*tol*/ )
{

    std::cout << __PRETTY_FUNCTION__ << std::endl;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;
    Epetra_CrsMatrix *system_matrix_transpose_tril;
    Epetra_RowMatrixTransposer epmt( const_cast<Epetra_CrsMatrix *>( &( dg->system_matrix.trilinos_matrix() ) ) );
    epmt.CreateTranspose(false, system_matrix_transpose_tril);
    system_matrix_transpose.reinit(*system_matrix_transpose_tril);

    // Input vector is copied into temporary non-const vector.
    auto input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

    std::cout << __PRETTY_FUNCTION__ << std::endl;
    //try {
        solve_linear (system_matrix_transpose, input_vector_v, output_vector_v, this->linear_solver_param);
    //} catch (...) {
    //    std::cout << "Failed to solve linear system in " << __PRETTY_FUNCTION__ << std::endl;
    //    output_vector.setScalar(1.0);
    //}

    // dg->system_matrix.transpose();
    // solve_linear (dg->system_matrix, input_vector_v, output_vector_v, this->linear_solver_param);
    // dg->system_matrix.transpose();
}

template<int dim>
void FlowConstraints<dim>
::applyJacobian_2( ROL::Vector<double>& output_vector,
const ROL::Vector<double>& input_vector,
const ROL::Vector<double>& des_var_sim,
const ROL::Vector<double>& des_var_ctl,
double& /*tol*/ )
{

    std::cout << __PRETTY_FUNCTION__ << std::endl;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

    auto dXvdXp_input = dg->high_order_grid.volume_nodes;

    dXvdXp.vmult(dXvdXp_input, input_vector_v);
    dg->dRdXv.vmult(output_vector_v, dXvdXp_input);

}

template<int dim>
void FlowConstraints<dim>
::applyAdjointJacobian_1( ROL::Vector<double>& output_vector,
const ROL::Vector<double>& input_vector,
const ROL::Vector<double>& des_var_sim,
const ROL::Vector<double>& des_var_ctl,
double& /*tol*/ )
{

    std::cout << __PRETTY_FUNCTION__ << std::endl;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);
    this->dg->system_matrix.Tvmult(output_vector_v, input_vector_v);
}

template<int dim>
void FlowConstraints<dim>
::applyAdjointJacobian_2( ROL::Vector<double>& output_vector,
const ROL::Vector<double>& input_vector,
const ROL::Vector<double>& des_var_sim,
const ROL::Vector<double>& des_var_ctl,
double& /*tol*/ )
{

    std::cout << __PRETTY_FUNCTION__ << std::endl;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

    auto input_dRdXv = dg->high_order_grid.volume_nodes;

    dg->dRdXv.Tvmult(input_dRdXv, input_vector_v);

    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

    dXvdXp.Tvmult(output_vector_v, input_dRdXv);

}

template<int dim>
void FlowConstraints<dim>
::applyAdjointHessian_11 ( ROL::Vector<double> &output_vector,
                                  const ROL::Vector<double> &dual,
                                  const ROL::Vector<double> &input_vector,
                                  const ROL::Vector<double> &des_var_sim,
                                  const ROL::Vector<double> &des_var_ctl,
                                  double &tol)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    (void) tol;
    dg->set_dual(get_ROLvec_to_VectorType(dual));
    dg->dual.update_ghost_values();
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dg->d2RdWdW.vmult(get_ROLvec_to_VectorType(output_vector), get_ROLvec_to_VectorType(input_vector));
}

template<int dim>
void FlowConstraints<dim>
::applyAdjointHessian_12 ( ROL::Vector<double> &output_vector,
                                  const ROL::Vector<double> &dual,
                                  const ROL::Vector<double> &input_vector,
                                  const ROL::Vector<double> &des_var_sim,
                                  const ROL::Vector<double> &des_var_ctl,
                                  double &tol)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    (void) tol;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
    dg->set_dual(get_ROLvec_to_VectorType(dual));
    dg->dual.update_ghost_values();
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

    auto d2RdXdW_input = dg->high_order_grid.volume_nodes;
    dg->d2RdWdX.Tvmult(d2RdXdW_input, input_vector_v);

    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);
    dXvdXp.Tvmult(output_vector_v, d2RdXdW_input);
}

template<int dim>
void FlowConstraints<dim>
::applyAdjointHessian_21 (
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &dual,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &tol
    )
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    (void) tol;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
    dg->set_dual(get_ROLvec_to_VectorType(dual));
    dg->dual.update_ghost_values();
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

    auto dXvdXp_input = dg->high_order_grid.volume_nodes;
    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);
    assert(input_vector_v.size() == dXvdXp.n());
    assert(dXvdXp_input.size() == dXvdXp.m());
    dXvdXp.vmult(dXvdXp_input, input_vector_v);

    dg->d2RdWdX.vmult(output_vector_v, dXvdXp_input);
}


template<int dim>
void FlowConstraints<dim>
::applyAdjointHessian_22 (
    ROL::Vector<double> &output_vector,
    const ROL::Vector<double> &dual,
    const ROL::Vector<double> &input_vector,
    const ROL::Vector<double> &des_var_sim,
    const ROL::Vector<double> &des_var_ctl,
    double &tol
    )
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    (void) tol;

    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
    dg->set_dual(get_ROLvec_to_VectorType(dual));
    dg->dual.update_ghost_values();
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
    auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

    dealii::TrilinosWrappers::SparseMatrix dXvdXp;
    ffd.get_dXvdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

    auto dXvdXp_input = dg->high_order_grid.volume_nodes;
    dXvdXp.vmult(dXvdXp_input, input_vector_v);

    auto d2RdXdX_dXvdXp_input = dg->high_order_grid.volume_nodes;
    dg->d2RdXdX.vmult(d2RdXdX_dXvdXp_input, dXvdXp_input);

    ffd.get_dXvdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);
    dXvdXp.Tvmult(output_vector_v, d2RdXdX_dXvdXp_input);

}

template class FlowConstraints<PHILIP_DIM>;

} // PHiLiP namespace
