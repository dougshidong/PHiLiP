#include "optimization/flow_constraints.hpp"
#include "mesh/meshmover_linear_elasticity.hpp"

#include "rol_to_dealii_vector.hpp"

#include "ode_solver/ode_solver_factory.h"

#include <Epetra_RowMatrixTransposer.h>

#include "Ifpack.h"

#include "global_counter.hpp"

namespace PHiLiP {

template<int dim>
FlowConstraints<dim>
::FlowConstraints(std::shared_ptr<DGBase<dim,double>> &_dg, 
                 const FreeFormDeformation<dim> &_ffd,
                 std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim,
                 dealii::TrilinosWrappers::SparseMatrix *precomputed_dXvdXp)
    : mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , i_print(mpi_rank==0)
    , dg(_dg)
    , ffd(_ffd)
    , ffd_design_variables_indices_dim(_ffd_design_variables_indices_dim)
    , jacobian_prec(nullptr)
    , adjoint_jacobian_prec(nullptr)
{
    flow_CFL_ = 0.0;

    const unsigned int n_design_variables = ffd_design_variables_indices_dim.size();
    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_design_variables);
    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);
    ffd_des_var.reinit(row_part, ghost_row_part, MPI_COMM_WORLD);

    ffd.get_design_variables(ffd_design_variables_indices_dim, ffd_des_var);

    initial_ffd_des_var = ffd_des_var;
    initial_ffd_des_var.update_ghost_values();

    //if(dXvdXp.m() == 0) ffd.get_dXvdXp ( *(dg->high_order_grid), ffd_design_variables_indices_dim, dXvdXp);
    if (precomputed_dXvdXp) {
        if (precomputed_dXvdXp->m() == dg->high_order_grid->volume_nodes.size() && precomputed_dXvdXp->n() == n_design_variables) {
            dXvdXp.copy_from(*precomputed_dXvdXp);
        }
    } else {
        ffd.get_dXvdXp ( *(dg->high_order_grid), ffd_design_variables_indices_dim, dXvdXp);
    }
    //ffd.get_dXvdXp_FD ( *(dg->high_order_grid), ffd_design_variables_indices_dim, dXvdXp, 1e-6);

    dealii::ParameterHandler parameter_handler;
    Parameters::LinearSolverParam::declare_parameters (parameter_handler);
    this->linear_solver_param.parse_parameters (parameter_handler);
    this->linear_solver_param.max_iterations = 1000;
    this->linear_solver_param.restart_number = 200;
    this->linear_solver_param.linear_residual = 1e-17;
    //this->linear_solver_param.ilut_fill = 1.0;//2; 50
    this->linear_solver_param.ilut_fill = 50;
    this->linear_solver_param.ilut_drop = 1e-8;
    //this->linear_solver_param.ilut_atol = 1e-3;
    //this->linear_solver_param.ilut_rtol = 1.0+1e-2;
    this->linear_solver_param.ilut_atol = 1e-5;
    this->linear_solver_param.ilut_rtol = 1.0+1e-2;
    this->linear_solver_param.linear_solver_output = Parameters::OutputEnum::verbose;
    this->linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;
    //this->linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::direct;
}

// template<int dim>
// FlowConstraints<dim>
// ::FlowConstraints(std::shared_ptr<DGBase<dim,double>> &_dg, 
//                  const FreeFormDeformation<dim> &_ffd,
//                  std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim,
//                  const dealii::TrilinosWrappers::SparseMatrix &_dXvdXp)
//     : FlowConstraints(_dg, _ffd, _ffd_design_variables_indices_dim)
// {
//     dXvdXp = _dXvdXp;
// }

template<int dim>
FlowConstraints<dim>::~FlowConstraints()
{
    destroy_JacobianPreconditioner_1();
    destroy_AdjointJacobianPreconditioner_1();
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
    //if (iter!=-1) {
    //    dg->output_results_vtk(1000000+iter);
    //    ffd.output_ffd_vtu(    1000000+iter);
    //}
    if (l2_norm != 0.0) {

        ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);
        //ffd.deform_mesh(*(dg->high_order_grid));
 
        auto dXp = ffd_des_var;
        dXp -= initial_ffd_des_var;
        dXp.update_ghost_values();
        auto dXv = dg->high_order_grid->volume_nodes;
        dXvdXp.vmult(dXv, dXp);
        dg->high_order_grid->volume_nodes = dg->high_order_grid->initial_volume_nodes;
        dg->high_order_grid->volume_nodes += dXv;
        dg->high_order_grid->volume_nodes.update_ghost_values();

        dg->output_results_vtk(iupdate);
        ffd.output_ffd_vtu(iupdate);
        iupdate++;
    }
}

template<int dim>
void FlowConstraints<dim>
::solve(
    ROL::Vector<double>& constraint_values,
    ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl,
    double& tol
    )
{

    update_2(des_var_ctl);

    dg->output_results_vtk(i_out++);
    ffd.output_ffd_vtu(i_out);
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver_1 = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver_1->steady_state();

    dg->assemble_residual();
    tol = dg->get_residual_l2norm();
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
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);

    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    this->dg->system_matrix.vmult(output_vector_v, input_vector_v);

    n_vmult += 1;
    dRdW_mult += 1;

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
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);

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

    //std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    //std::cout << "System matrix flow_CFL_" << flow_CFL_ << std::endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    //dg->system_matrix.print(std::cout);

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
::destroy_JacobianPreconditioner_1()
{
    delete jacobian_prec;
    jacobian_prec = nullptr;
}
template<int dim>
void FlowConstraints<dim>
::destroy_AdjointJacobianPreconditioner_1()
{
    delete adjoint_jacobian_prec;
    adjoint_jacobian_prec = nullptr;
}

template<int dim>
int FlowConstraints<dim>
::construct_JacobianPreconditioner_1(
    const ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl)
{
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);

    Epetra_CrsMatrix * jacobian = const_cast<Epetra_CrsMatrix *>(&(dg->system_matrix.trilinos_matrix()));

    destroy_JacobianPreconditioner_1();
    Ifpack Factory;

    Teuchos::ParameterList List;

    const std::string PrecType = "ILUT"; 
    List.set("fact: ilut level-of-fill", 50.0);
    List.set("fact: absolute threshold", 1e-3);
    List.set("fact: relative threshold", 1.01);//1.0+1e-2);
    List.set("fact: drop tolerance", 0.0);//1e-12);

    //const std::string PrecType = "ILU"; 
    //List.set("fact: level-of-fill", 0);

    List.set("schwarz: reordering type", "rcm");
    const int OverlapLevel = 1; // one row of overlap among the processes
    jacobian_prec = Factory.Create(PrecType, jacobian, OverlapLevel);
    assert (jacobian_prec != 0);




    IFPACK_CHK_ERR(jacobian_prec->SetParameters(List));
    IFPACK_CHK_ERR(jacobian_prec->Initialize());
    IFPACK_CHK_ERR(jacobian_prec->Compute());

    return 0;

}

template<int dim>
int FlowConstraints<dim>
::construct_AdjointJacobianPreconditioner_1(
    const ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl)
{
    update_1(des_var_sim);
    update_2(des_var_ctl);

    const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);

    Epetra_CrsMatrix * adjoint_jacobian = const_cast<Epetra_CrsMatrix *>(&(dg->system_matrix_transpose.trilinos_matrix()));

    destroy_AdjointJacobianPreconditioner_1();
    Ifpack Factory;

    Teuchos::ParameterList List;

    const std::string PrecType = "ILUT"; 
    List.set("fact: ilut level-of-fill", 50.0);
    List.set("fact: absolute threshold", 1e-3);
    List.set("fact: relative threshold", 1.01);//1.0+1e-2);
    List.set("fact: drop tolerance", 0.0);//1e-12);

    //const std::string PrecType = "ILU"; 
    //List.set("fact: level-of-fill", 0);

    List.set("schwarz: reordering type", "rcm");
    const int OverlapLevel = 1; // one row of overlap among the processes
    adjoint_jacobian_prec = Factory.Create(PrecType, adjoint_jacobian, OverlapLevel);
    assert (adjoint_jacobian_prec != 0);

    IFPACK_CHK_ERR(adjoint_jacobian_prec->SetParameters(List));
    IFPACK_CHK_ERR(adjoint_jacobian_prec->Initialize());
    IFPACK_CHK_ERR(adjoint_jacobian_prec->Compute());

    return 0;

}

template<int dim>
void FlowConstraints<dim>
::applyInverseJacobianPreconditioner_1(
    ROL::Vector<double>& output_vector,
    const ROL::Vector<double>& input_vector,
    const ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl,
    double& /*tol*/ )
{
    (void) des_var_sim; // Preconditioner should be built.
    (void) des_var_ctl; // Preconditioner should be built.
    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;

    // Input vector is copied into temporary non-const vector.
    auto input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);

    Epetra_Vector input_trilinos(View,
                    dg->system_matrix.trilinos_matrix().DomainMap(),
                    input_vector_v.begin());
    Epetra_Vector output_trilinos(View,
                    dg->system_matrix.trilinos_matrix().RangeMap(),
                    output_vector_v.begin());
    jacobian_prec->ApplyInverse (input_trilinos, output_trilinos);

    //n_vmult += 2;
    //dRdW_mult += 2;
    n_vmult += 6;
    dRdW_mult += 6;

}

template<int dim>
void FlowConstraints<dim>
::applyInverseAdjointJacobianPreconditioner_1(
    ROL::Vector<double>& output_vector,
    const ROL::Vector<double>& input_vector,
    const ROL::Vector<double>& des_var_sim,
    const ROL::Vector<double>& des_var_ctl,
    double& /*tol*/ )
{
    (void) des_var_sim; // Preconditioner should be built.
    (void) des_var_ctl; // Preconditioner should be built.
    if(i_print) std::cout << __PRETTY_FUNCTION__ << std::endl;

    // Input vector is copied into temporary non-const vector.
    auto input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);

    Epetra_Vector input_trilinos(View,
                    dg->system_matrix_transpose.trilinos_matrix().DomainMap(),
                    input_vector_v.begin());
    Epetra_Vector output_trilinos(View,
                    dg->system_matrix_transpose.trilinos_matrix().RangeMap(),
                    output_vector_v.begin());
    adjoint_jacobian_prec->ApplyInverse (input_trilinos, output_trilinos);

    //n_vmult += 2;
    //dRdW_mult += 2;
    n_vmult += 6;
    dRdW_mult += 6;
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
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);

    // Input vector is copied into temporary non-const vector.
    auto input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);

    solve_linear (dg->system_matrix_transpose, input_vector_v, output_vector_v, this->linear_solver_param);

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

    //auto dXvsdXp_input = dg->high_order_grid->volume_nodes;
    //{
    //    dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //    ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
    //    dXvsdXp.vmult(dXvsdXp_input,input_vector_v);
    //}

    //auto dXvdXp_input = dg->high_order_grid->volume_nodes;
    //{
    //    dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid->surface_nodes);
    //    MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //        meshmover(*(dg->high_order_grid->triangulation),
    //          dg->high_order_grid->initial_mapping_fe_field,
    //          dg->high_order_grid->dof_handler_grid,
    //          dg->high_order_grid->surface_to_volume_indices,
    //          dummy_vector);

    //    meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    //}

    auto dXvdXp_input = dg->high_order_grid->volume_nodes;
    dXvdXp.vmult(dXvdXp_input, input_vector_v);

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);

    {
        const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);
        dg->dRdXv.vmult(output_vector_v, dXvdXp_input);
    }

    n_vmult += 7;
    dRdX_mult += 1;
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
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);

    const auto &input_vector_v = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    this->dg->system_matrix.Tvmult(output_vector_v, input_vector_v);

    n_vmult += 1;
    dRdW_mult += 1;
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

    auto input_dRdXv = dg->high_order_grid->volume_nodes;
    {
        const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);
        dg->dRdXv.Tvmult(input_dRdXv, input_vector_v);
    }

    // auto input_dRdXv_dXvdXvs = dg->high_order_grid->volume_nodes;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(dg->high_order_grid->triangulation),
    //           dg->high_order_grid->initial_mapping_fe_field,
    //           dg->high_order_grid->dof_handler_grid,
    //           dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs_transpose(input_dRdXv, input_dRdXv_dXvdXvs);
    // }

    // auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.Tvmult(output_vector_v, input_dRdXv_dXvdXvs);
    // }

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    dXvdXp.Tvmult(output_vector_v, input_dRdXv);

    n_vmult += 7;
    dRdX_mult += 1;
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
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);
    dg->d2RdWdW.vmult(ROL_vector_to_dealii_vector_reference(output_vector), ROL_vector_to_dealii_vector_reference(input_vector));

    n_vmult += 6;
    d2R_mult += 1;
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

    auto input_d2RdWdX = dg->high_order_grid->volume_nodes;
    {
        const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);
        dg->d2RdWdX.Tvmult(input_d2RdWdX, input_vector_v);
    }

    // auto input_d2RdWdX_dXvdXvs = dg->high_order_grid->volume_nodes;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(dg->high_order_grid->triangulation),
    //           dg->high_order_grid->initial_mapping_fe_field,
    //           dg->high_order_grid->dof_handler_grid,
    //           dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);
    //     meshmover.apply_dXvdXvs_transpose(input_d2RdWdX, input_d2RdWdX_dXvdXvs);
    // }

    // auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.Tvmult(output_vector_v, input_d2RdWdX_dXvdXvs);
    // }

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    dXvdXp.Tvmult(output_vector_v, input_d2RdWdX);

    n_vmult += 7;
    d2R_mult += 1;
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

    // auto dXvsdXp_input = dg->high_order_grid->volume_nodes;
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.vmult(dXvsdXp_input,input_vector_v);
    // }

    // auto dXvdXp_input = dg->high_order_grid->volume_nodes;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(dg->high_order_grid->triangulation),
    //           dg->high_order_grid->initial_mapping_fe_field,
    //           dg->high_order_grid->dof_handler_grid,
    //           dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);

    //     meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    // }

    auto dXvdXp_input = dg->high_order_grid->volume_nodes;
    dXvdXp.vmult(dXvdXp_input, input_vector_v);

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);
        dg->d2RdWdX.vmult(output_vector_v, dXvdXp_input);
    }

    n_vmult += 7;
    d2R_mult += 1;
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

    // auto dXvsdXp_input = dg->high_order_grid->volume_nodes;
    // {
    //     dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //     ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
    //     dXvsdXp.vmult(dXvsdXp_input,input_vector_v);
    // }

    // auto dXvdXp_input = dg->high_order_grid->volume_nodes;
    // {
    //     dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid->surface_nodes);
    //     MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //         meshmover(*(dg->high_order_grid->triangulation),
    //           dg->high_order_grid->initial_mapping_fe_field,
    //           dg->high_order_grid->dof_handler_grid,
    //           dg->high_order_grid->surface_to_volume_indices,
    //           dummy_vector);

    //     meshmover.apply_dXvdXvs(dXvsdXp_input, dXvdXp_input);
    // }

    auto dXvdXp_input = dg->high_order_grid->volume_nodes;
    dXvdXp.vmult(dXvdXp_input, input_vector_v);

    auto d2RdXdX_dXvdXp_input = dg->high_order_grid->volume_nodes;
    {
        const bool compute_dRdW=false; const bool compute_dRdX=false; const bool compute_d2R=true;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R, flow_CFL_);
        dg->d2RdXdX.vmult(d2RdXdX_dXvdXp_input, dXvdXp_input);
    }

    //auto dXvdXvsT_d2RdXdX_dXvdXp_input = dg->high_order_grid->volume_nodes;
    //{
    //    dealii::LinearAlgebra::distributed::Vector<double> dummy_vector(dg->high_order_grid->surface_nodes);
    //    MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
    //        meshmover(*(dg->high_order_grid->triangulation),
    //          dg->high_order_grid->initial_mapping_fe_field,
    //          dg->high_order_grid->dof_handler_grid,
    //          dg->high_order_grid->surface_to_volume_indices,
    //          dummy_vector);
    //    meshmover.apply_dXvdXvs_transpose(d2RdXdX_dXvdXp_input, dXvdXvsT_d2RdXdX_dXvdXp_input);
    //}

    //auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    //{
    //    dealii::TrilinosWrappers::SparseMatrix dXvsdXp;
    //    ffd.get_dXvsdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvsdXp);
    //    dXvsdXp.Tvmult(output_vector_v, dXvdXvsT_d2RdXdX_dXvdXp_input);
    //}

    auto &output_vector_v = ROL_vector_to_dealii_vector_reference(output_vector);
    dXvdXp.Tvmult(output_vector_v, d2RdXdX_dXvdXp_input);

    n_vmult += 8;
    d2R_mult += 1;
}

// template<int dim>
// void FlowConstraints<dim>
// ::applyPreconditioner(ROL::Vector<double> &pv,
//                          const ROL::Vector<double> &v,
//                          const ROL::Vector<double> &x,
//                          const ROL::Vector<double> &g,
//                          double &tol)
// {
//     Constraint<double>::applyPreconditioner(pv, v, x, g, tol);
//     // try {
//     //     const Vector_SimOpt<double> &xs = dynamic_cast<const Vector_SimOpt<double>&>(x);
//     //     Ptr<Vector<double>> ijv = (xs.get_1())->clone();
//   
//     //     applyInverseJacobian_1_preconditioner(*ijv, v, *(xs.get_1()), *(xs.get_2()), tol);
//     //     const Vector_SimOpt<double> &gs = dynamic_cast<const Vector_SimOpt<double>&>(g);
//     //     Ptr<Vector<double>> ijv_dual = (gs.get_1())->clone();
//     //     ijv_dual->set(ijv->dual());
//     //     applyInverseAdjointJacobian_1_preconditioner(pv, *ijv_dual, *(xs.get_1()), *(xs.get_2()), tol);
//     // }
//     // catch (const std::logic_error &e) {
//     //     Constraint<double>::applyPreconditioner(pv, v, x, g, tol);
//     //     return;
//     // }
// }

// virtual void applyPreconditioner(Vector<Real> &pv,
//                                const Vector<Real> &v,
//                                const Vector<Real> &x,
//                                const Vector<Real> &g,
//                                Real &tol)
// {
//     update(x);
// 
//     const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
//     dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
// 
//     AztecOO solver;
//     solver.SetAztecOption(AZ_output, (param.linear_solver_output ? AZ_all : AZ_none));
//     solver.SetAztecOption(AZ_solver, AZ_gmres);
//     solver.SetAztecOption(AZ_kspace, param.restart_number);
// 
//     solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
//     solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
//     solver.SetAztecOption(AZ_overlap, 0);
//     solver.SetAztecOption(AZ_reorder, 1); // RCM re-ordering
// 
//     const double 
//       ilut_drop = param.ilut_drop,
//       ilut_rtol = param.ilut_rtol,//0.0,//1.1,
//       ilut_atol = param.ilut_atol,//0.0,//1e-9,
//       linear_residual = param.linear_residual;//1e-4;
//     const int ilut_fill = param.ilut_fill,//1,
// 
//     solver.SetAztecParam(AZ_drop, ilut_drop);
//     solver.SetAztecParam(AZ_ilut_fill, ilut_fill);
//     solver.SetAztecParam(AZ_athresh, ilut_atol);
//     solver.SetAztecParam(AZ_rthresh, ilut_rtol);
//     solver.SetUserMatrix(const_cast<Epetra_CrsMatrix *>(&(dg->system_matrix.trilinos_matrix())));
// 
//     double condition_number_estimate;
//     const int precond_error = solver.ConstructPreconditioner (condition_number_estimate);
//     const Epetra_Operator* preconditionner = solver.GetPrecOperator();
// 
// 
//     Epetra_Vector pv_epetra(View,
//                     dg->system_matrix.trilinos_matrix().DomainMap(),
//                     ROL_vector_to_dealii_vector_reference(pv).begin());
//     Epetra_Vector v_epetra(View,
//                     dg->system_matrix.trilinos_matrix().RangeMap(),
//                     ROL_vector_to_dealii_vector_reference(v).begin());
// 
//     preconditionner.applyInverse(
// 
// 
//     pv.set(v.dual());
// }

// std::vector<double> solveAugmentedSystem(
//     ROL::Vector<double> &v1,
//     ROL::Vector<double> &v2,
//     const ROL::Vector<double> &b1,
//     const ROL::Vector<double> &b2,
//     const ROL::Vector<double> &x,
//     double & tol) override
// {
//     ROL::Vector_SimOpt<double> &v1_simctl
//         = dynamic_cast<Vector_SimOpt<double>&>(
//           dynamic_cast<Vector<double>&> (v1));
//     const ROL::Vector_SimOpt<double> &b1_simctl
//         = dynamic_cast<const Vector_SimOpt<double>&>(
//           dynamic_cast<const Vector<double>&>(b));
//     const ROL::Vector<double> &v1_sim = *(v1_simctl.get_1());
//     const ROL::Vector<double> &v1_ctl = *(v1_simctl.get_2());
//     const ROL::Vector<double> &b1_sim = *(b1_simctl.get_1());
//     const ROL::Vector<double> &b1_ctl = *(b1_simctl.get_2());
// }

template class FlowConstraints<PHILIP_DIM>;

} // PHiLiP namespace
