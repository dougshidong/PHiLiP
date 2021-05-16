#ifndef __FULL_SPACE_OPTIMIZATION_H__
#define __FULL_SPACE_OPTIMIZATION_H__

#include "mesh/meshmover_linear_elasticity.hpp"

#define HESSIAN_DIAG 1e0
namespace PHiLiP {
namespace Tests {

template<typename VectorType>
VectorType LBFGS(
    const VectorType &grad_lagrangian,
    const std::list<VectorType> &search_s, // Previous x_kp1 - x_k
    const std::list<VectorType> &dgrad_y, // dgrad
    const std::list<double> &denom_rho) // 1/y^t . s
{
 std::cout << "Applying LBFGS to obtain search direction" << std::endl;
 // Front of the list is most recent iteration
 // back of the list are the oldest iterations (and first ones to be erased).
    VectorType q = grad_lagrangian;

 assert(denom_rho.size() == search_s.size());
 assert(search_s.size() == dgrad_y.size());
 assert(search_s.size() > 0);

 std::list<double> alpha;

 // First recursion loop
 {
  auto rho = denom_rho.begin();
  auto s = search_s.begin();
  auto y = dgrad_y.begin();
  for (; rho != denom_rho.end(); ++rho, ++s, ++y) {
   double a = (*s) * q;
   a *= (*rho);
   alpha.push_front(a);
   q -= a * (*y);
  }
 }
 
 // Search direction
 VectorType r = q;

 // Scale with initial Hessian
 //const double diagonal_scaling = 1.0/HESSIAN_DIAG;
 double diagonal_scaling = denom_rho.front()/dgrad_y.front().norm_sqr(); // 7.20 on Nocedal
 r *= diagonal_scaling;

 // Second recursion loop
 {
  auto rho = denom_rho.rbegin();
  auto s = search_s.rbegin();
  auto y = dgrad_y.rbegin();
  auto a = alpha.rbegin();
  for (; rho != denom_rho.rend(); ++rho, ++s, ++y, ++a) {
   double b = (*y) * r;
   b *= (*rho);
   r += (*s) * (*a - b);
  }
 }
 return r;
}
template<typename VectorType, typename MatrixType>
MatrixType BFGS(
    const MatrixType &oldH,
    const VectorType &oldg,
    const VectorType &currentg,
    const VectorType &searchD)
{
 int n_rows = oldH.rows();
    MatrixType newH(n_rows, n_rows);
    VectorType dg(n_rows), dx(n_rows);
    MatrixType dH(n_rows, n_rows), a(n_rows, n_rows), b(n_rows, n_rows);

    dg = currentg - oldg;
    dx = searchD;
 const double dgdx = dx.dot(dg);

 if(dgdx < 0) {
  printf("Negative curvature. Not updating BFGS \n");
  return oldH;
 }

    a = ((dgdx + dg.transpose() * oldH * dg) * (dx * dx.transpose())) / (dgdx*dgdx);
    b = (oldH * dg * dx.transpose() + dx * dg.transpose() * oldH) / (dgdx);

    dH = a - b;

    newH = oldH + dH;

    return newH;
}


dealii::TrilinosWrappers::SparseMatrix transpose_trilinos_matrix(dealii::TrilinosWrappers::SparseMatrix &input_matrix)
{
 Epetra_CrsMatrix *transpose_CrsMatrix;
 Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&input_matrix.trilinos_matrix()));
 epmt.CreateTranspose(false, transpose_CrsMatrix);
 dealii::TrilinosWrappers::SparseMatrix output_matrix;
 output_matrix.reinit(*transpose_CrsMatrix);
 return output_matrix;
}

template<int dim, int nstate>
dealii::LinearAlgebra::distributed::BlockVector<double> evaluate_kkt_rhs(DGBase<dim,double> &dg,
                   TargetFunctional<dim, nstate, double> &functional,
                   PHiLiP::MeshMover::LinearElasticity<dim, double> &meshmover
                   )
{
    if ( dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0 )
  std::cout << "Evaluating KKT right-hand side: dIdW, dIdX, d2I, Residual..." << std::endl;
 dealii::LinearAlgebra::distributed::BlockVector<double> kkt_rhs(3);

 bool compute_dIdW = true, compute_dIdX = true, compute_d2I = true;
 (void) functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);

 bool compute_dRdW = false, compute_dRdX = false, compute_d2R = false;
 dg.assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

 dealii::LinearAlgebra::distributed::Vector<double> dIdXs;
 dIdXs.reinit(dg.high_order_grid->surface_nodes);
 assert(meshmover.dXvdXs.size() > 0);
 for (unsigned int isurf = 0; isurf < dg.high_order_grid->surface_nodes.size(); ++isurf) {
  const auto scalar_product = meshmover.dXvdXs[isurf] * functional.dIdX;
  if (dIdXs.locally_owned_elements().is_element(isurf)) {
   dIdXs[isurf] = scalar_product;
  }
 }
 dIdXs.update_ghost_values();


 kkt_rhs.block(0) = functional.dIdw;
 kkt_rhs.block(1) = dIdXs;
 kkt_rhs.block(2) = dg.right_hand_side;
 kkt_rhs *= -1.0;

 return kkt_rhs;
}

std::pair<unsigned int, double>
apply_P4 (
    dealii::TrilinosWrappers::BlockSparseMatrix &matrix_A,
    dealii::LinearAlgebra::distributed::BlockVector<double> &right_hand_side,
    dealii::LinearAlgebra::distributed::BlockVector<double> &solution,
    const PHiLiP::Parameters::LinearSolverParam &param)
{
    dealii::deallog.depth_console(3);
 using trilinos_vector_type = dealii::TrilinosWrappers::MPI::Vector;
 using vector_type = dealii::LinearAlgebra::distributed::Vector<double>;
 using matrix_type = dealii::TrilinosWrappers::SparseMatrix;
 using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;

 vector_type &_rhs1 = right_hand_side.block(0);
 vector_type &_rhs2 = right_hand_side.block(1);
 vector_type &_rhs3 = right_hand_side.block(2);

 dealii::IndexSet rhs1_locally_owned = _rhs1.locally_owned_elements();
 dealii::IndexSet rhs1_ghost = _rhs1.get_partitioner()->ghost_indices();
 dealii::IndexSet rhs2_locally_owned = _rhs2.locally_owned_elements();
 dealii::IndexSet rhs2_ghost = _rhs2.get_partitioner()->ghost_indices();
 dealii::IndexSet rhs3_locally_owned = _rhs3.locally_owned_elements();
 dealii::IndexSet rhs3_ghost = _rhs3.get_partitioner()->ghost_indices();

 trilinos_vector_type rhs1(rhs1_locally_owned);
 trilinos_vector_type rhs2(rhs2_locally_owned);
 trilinos_vector_type rhs3(rhs3_locally_owned);
 trilinos_vector_type soln1(_rhs1.locally_owned_elements());
 trilinos_vector_type soln2(_rhs2.locally_owned_elements());
 trilinos_vector_type soln3(_rhs3.locally_owned_elements());

 {
  dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(rhs1_locally_owned);
  rw_vector.import(_rhs1, dealii::VectorOperation::insert);
  rhs1.import(rw_vector, dealii::VectorOperation::insert);
 }
 {
  dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(rhs2_locally_owned);
  rw_vector.import(_rhs2, dealii::VectorOperation::insert);
  rhs2.import(rw_vector, dealii::VectorOperation::insert);
 }
 {
  dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(rhs3_locally_owned);
  rw_vector.import(_rhs3, dealii::VectorOperation::insert);
  rhs3.import(rw_vector, dealii::VectorOperation::insert);
 }
 auto &Wss = matrix_A.block(0,0);
 auto &Wsd = matrix_A.block(0,1);
 auto &AsT = matrix_A.block(0,2);
 auto &Wds = matrix_A.block(1,0);
 auto &Wdd = matrix_A.block(1,1);
 auto &AdT = matrix_A.block(1,2);
 auto &As  = matrix_A.block(2,0);
 auto &Ad  = matrix_A.block(2,1);
 auto &Zer = matrix_A.block(2,2);

 (void) Wss;
 (void) Wsd;
 (void) AsT;
 (void) Wds;
 (void) Wdd;
 (void) AdT;
 (void) As ;
 (void) Ad ;
 (void) Zer;
 (void) param;
 (void) solution;

 // // Block vector y

    dealii::LinearAlgebra::distributed::BlockVector<double> Y;
 Y.reinit(right_hand_side);
 // y1
 Y.block(0) = right_hand_side.block(2);

 // As_inv y1
 vector_type As_inv_y1;
 As_inv_y1.reinit(solution.block(0));
 solve_linear (As, Y.block(0), As_inv_y1, param);

 // y3
 Wss.vmult(Y.block(2), As_inv_y1);
 Y.block(2) *= -1.0;
 Y.block(2) += right_hand_side.block(0);

 // AsT_inv y3
 vector_type AsT_inv_y3;
 AsT_inv_y3.reinit(solution.block(2));
 solve_linear (AsT, Y.block(2), AsT_inv_y3, param);

 // y2
 AdT.vmult(Y.block(1), AsT_inv_y3);
 Wds.vmult_add(Y.block(1), As_inv_y1);
 Y.block(1) *= -1.0;
 Y.block(1) += right_hand_side.block(1);

 // Block vector p
    dealii::LinearAlgebra::distributed::BlockVector<double> P;
 P.reinit(right_hand_side);

 // p2
 const double reduced_hessian_diagonal = HESSIAN_DIAG;
 P.block(1) = Y.block(1);
 P.block(1) /= reduced_hessian_diagonal;

 // As_inv_Ad_p2
 vector_type Ad_p2, As_inv_Ad_p2;
 Ad_p2.reinit(solution.block(0));
 Ad.vmult(Ad_p2, P.block(1));

 As_inv_Ad_p2.reinit(solution.block(0));
 solve_linear(As, Ad_p2, As_inv_Ad_p2, param);

 // p1
 P.block(0) = As_inv_y1;
 P.block(0) -= As_inv_Ad_p2;

 // p3
 vector_type p3_rhs;
 p3_rhs.reinit(solution.block(2));

 Wsd.vmult(p3_rhs, P.block(1));
 p3_rhs *= -1.0;
 Wss.vmult_add(p3_rhs, As_inv_Ad_p2);
 p3_rhs += Y.block(2);

 solve_linear(AsT, p3_rhs, P.block(2), param);

 solution = P;

    return {-1.0, -1.0};
}

} // Tests namespace
} // PHiLiP namespace

#endif
