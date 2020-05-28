#include "optimization/flow_constraints.hpp"
#include "mesh/meshmover_linear_elasticity.hpp"

#include "rol_to_dealii_vector.hpp"

#include "ode_solver/ode_solver.h"

#include <Epetra_RowMatrixTransposer.h>

namespace PHiLiP {

template<int dim>
FlowConstraints<dim>
::FlowConstraints(std::shared_ptr<DGBase<dim,double>> &_dg, 
                 const FreeFormDeformation<dim> &_ffd,
                 std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim)
    : mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , i_print(mpi_rank==0)
    , dg(_dg)
    , ffd(_ffd)
    , ffd_design_variables_indices_dim(_ffd_design_variables_indices_dim)
{
    ffd_des_var.reinit(ffd_design_variables_indices_dim.size());
    ffd.get_design_variables(ffd_design_variables_indices_dim, ffd_des_var);

    dealii::ParameterHandler parameter_handler;
    Parameters::LinearSolverParam::declare_parameters (parameter_handler);
    this->linear_solver_param.parse_parameters (parameter_handler);
    this->linear_solver_param.max_iterations = 1000;
    this->linear_solver_param.restart_number = 200;
    this->linear_solver_param.linear_residual = 1e-13;
    this->linear_solver_param.ilut_fill = 0;
    this->linear_solver_param.ilut_drop = 0.0;
    this->linear_solver_param.ilut_rtol = 1.0;
    this->linear_solver_param.ilut_atol = 0.0;
    this->linear_solver_param.linear_solver_output = Parameters::OutputEnum::quiet;
    this->linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;
}

template<int dim>
void FlowConstraints<dim>
::update_1( const ROL::Vector<double>& des_var_sim, bool flag, int iter )
{
    (void) flag; (void) iter;
    dg->solution = ROL_vector_to_dealii_vector_reference(des_var_sim);
    dg->solution.update_ghost_values();
}

template<int dim>
void FlowConstraints<dim>
::update_2( const ROL::Vector<double>& des_var_ctl, bool flag, int iter )
{
    (void) flag; (void) iter;
    ffd_des_var =  ROL_vector_to_dealii_vector_reference(des_var_ctl);
    auto current_ffd_des_var = ffd_des_var;
    ffd.get_design_variables( ffd_design_variables_indices_dim, current_ffd_des_var);

    auto diff = ffd_des_var;
    diff -= current_ffd_des_var;
    const double l2_norm = diff.l2_norm();
    if (l2_norm != 0.0) {
        ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);
        ffd.deform_mesh(dg->high_order_grid);
    }
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
    auto &constraint = ROL_vector_to_dealii_vector_reference(constraint_values);
    constraint = dg->right_hand_side;
    auto &des_var_sim_v = ROL_vector_to_dealii_vector_reference(des_var_sim);
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
    auto &constraint = ROL_vector_to_dealii_vector_reference(constraint_values);
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

    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
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

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
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
    //     if(i_print) std::cout << "Failed to solve linear system in " << __PRETTY_FUNCTION__ << std::endl;
    //     output_vector.setScalar(1.0);
    // }

    // Input vector is copied into temporary non-const vector.
    auto input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);

    solve_linear (dg->system_matrix, input_vector_v, output_vector_v, this->linear_solver_param);
    //solve_linear_2 ( this->dg->system_matrix, input_vector_v, output_vector_v, this->linear_solver_param);
    //try {
    //  solve_linear (dg->system_matrix, input_vector_v, output_vector_v, this->linear_solver_param);
    //} catch (...) {
    //    if(i_print) std::cout << "Failed to solve linear system in " << __PRETTY_FUNCTION__ << std::endl;
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

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    // Input vector is copied into temporary non-const vector.
    auto input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);

    dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;
    Epetra_CrsMatrix *system_matrix_transpose_tril;
    Epetra_RowMatrixTransposer epmt( const_cast<Epetra_CrsMatrix *>( &( dg->system_matrix.trilinos_matrix() ) ) );
    epmt.CreateTranspose(false, system_matrix_transpose_tril);
    system_matrix_transpose.reinit(*system_matrix_transpose_tril);
    solve_linear (system_matrix_transpose, input_vector_v, output_vector_v, this->linear_solver_param);

    //try {
    //    solve_linear (system_matrix_transpose, input_vector_v, output_vector_v, this->linear_solver_param, true);
    //} catch (...) {
    //    if(i_print) std::cout << "Failed to solve linear system in " << __PRETTY_FUNCTION__ << std::endl;
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

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);

    auto dXvsdXp_input = dg->high_order_grid.volume_nodes;
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.vmult(dXvsdXp_input,input_vector_v);
    }

    auto dXvdXp_input = dg->high_order_grid.volume_nodes;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(dg->high_order_grid.triangulation),
              dg->high_order_grid.initial_mapping_fe_field,
              dg->high_order_grid.dof_handler_grid,
              dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);

        meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    }

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);

    {
        const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
        dg->dRdXv.vmult(output_vector_v, dXvdXp_input);
    }

}

template<int dim>
void FlowConstraints<dim>
::applyAdjointJacobian_1( ROL::Vector<double>& output_vector,
const ROL::Vector<double>& input_vector,
const ROL::Vector<double>& des_var_sim,
const ROL::Vector<double>& des_var_ctl,
double& /*tol*/ )
{

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
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

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
    update_1(des_var_sim);
    update_2(des_var_ctl);


    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);

    auto input_dRdXv = dg->high_order_grid.volume_nodes;
    {
        const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
        dg->dRdXv.Tvmult(input_dRdXv, input_vector_v);
    }

    auto input_dRdXv_dXvdXvs = dg->high_order_grid.volume_nodes;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(dg->high_order_grid.triangulation),
              dg->high_order_grid.initial_mapping_fe_field,
              dg->high_order_grid.dof_handler_grid,
              dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);
        meshmover.apply_dXvdXvs_transpose(input_dRdXv, input_dRdXv_dXvdXvs);
    }

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.Tvmult(output_vector_v, input_dRdXv_dXvdXvs);
    }

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
    // ROL_vector_to_dealii_vector_reference(output_vector) *= 0.0;
    // return;

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
    (void) tol;
    dg->set_dual(ROL_vector_to_dealii_vector_reference(dual));
    dg->dual.update_ghost_values();
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dg->d2RdWdW.vmult(ROL_vector_to_dealii_vector_reference(output_vector), ROL_vector_to_dealii_vector_reference(input_vector));
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
    // ROL_vector_to_dealii_vector_reference(output_vector) *= 0.0;
    // return;

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
    (void) tol;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    dg->set_dual(ROL_vector_to_dealii_vector_reference(dual));
    dg->dual.update_ghost_values();

    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);

    auto input_d2RdWdX = dg->high_order_grid.volume_nodes;
    {
        const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
        dg->d2RdWdX.Tvmult(input_d2RdWdX, input_vector_v);
    }

    auto input_d2RdWdX_dXvdXvs = dg->high_order_grid.volume_nodes;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(dg->high_order_grid.triangulation),
              dg->high_order_grid.initial_mapping_fe_field,
              dg->high_order_grid.dof_handler_grid,
              dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);
        meshmover.apply_dXvdXvs_transpose(input_d2RdWdX, input_d2RdWdX_dXvdXvs);
    }

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.Tvmult(output_vector_v, input_d2RdWdX_dXvdXvs);
    }
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
    // ROL_vector_to_dealii_vector_reference(output_vector) *= 0.0;
    // return;

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
    (void) tol;
    update_1(des_var_sim);
    update_2(des_var_ctl);

    dg->set_dual(ROL_vector_to_dealii_vector_reference(dual));
    dg->dual.update_ghost_values();

    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);

    auto dXvsdXp_input = dg->high_order_grid.volume_nodes;
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.vmult(dXvsdXp_input,input_vector_v);
    }

    auto dXvdXp_input = dg->high_order_grid.volume_nodes;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(dg->high_order_grid.triangulation),
              dg->high_order_grid.initial_mapping_fe_field,
              dg->high_order_grid.dof_handler_grid,
              dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);

        meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    }

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
        dg->d2RdWdX.vmult(output_vector_v, dXvdXp_input);
    }
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
    // ROL_vector_to_dealii_vector_reference(output_vector) *= 0.0;
    // return;

    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;
    (void) tol;

    update_1(des_var_sim);
    update_2(des_var_ctl);

    dg->set_dual(ROL_vector_to_dealii_vector_reference(dual));
    dg->dual.update_ghost_values();

    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);

    auto dXvsdXp_input = dg->high_order_grid.volume_nodes;
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.vmult(dXvsdXp_input,input_vector_v);
    }

    auto dXvdXp_input = dg->high_order_grid.volume_nodes;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(dg->high_order_grid.triangulation),
              dg->high_order_grid.initial_mapping_fe_field,
              dg->high_order_grid.dof_handler_grid,
              dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);

        meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    }

    auto d2RdXdX_dXvdXp_input = dg->high_order_grid.volume_nodes;
    {
        const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
        dg->d2RdXdX.vmult(d2RdXdX_dXvdXp_input, dXvdXp_input);
    }

    auto dXvdXvsT_d2RdXdX_dXvdXp_input = dg->high_order_grid.volume_nodes;
    {
        dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid.surface_nodes);
        MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
            meshmover(*(dg->high_order_grid.triangulation),
              dg->high_order_grid.initial_mapping_fe_field,
              dg->high_order_grid.dof_handler_grid,
              dg->high_order_grid.surface_to_volume_indices,
              dummy_vector);
        meshmover.apply_dXvdXvs_transpose(d2RdXdX_dXvdXp_input, dXvdXvsT_d2RdXdX_dXvdXp_input);
    }

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
        ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
        dXvsdXp.Tvmult(output_vector_v, dXvdXvsT_d2RdXdX_dXvdXp_input);
    }
}

template class FlowConstraints<PHILIP_DIM>;

} // PHiLiP namespace
