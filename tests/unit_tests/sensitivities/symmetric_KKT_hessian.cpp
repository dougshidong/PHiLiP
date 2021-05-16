#include <Epetra_RowMatrixTransposer.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "physics/physics_factory.h"
#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "functional/functional.h"

#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>

#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>


#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/trilinos_linear_operator.h>

#include <deal.II/lac/read_write_vector.h>

const double STEPSIZE = 1e-7;
const double TOLERANCE = 1e-6;

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

std::pair<unsigned int, double>
solve_linear (
    dealii::TrilinosWrappers::BlockSparseMatrix &matrix_A,
    dealii::LinearAlgebra::distributed::BlockVector<double> &right_hand_side,
    dealii::LinearAlgebra::distributed::BlockVector<double> &solution,
    const PHiLiP::Parameters::LinearSolverParam &)//param)
{

    {
  using trilinos_vector_type = dealii::TrilinosWrappers::MPI::Vector;
  using vector_type = dealii::LinearAlgebra::distributed::Vector<double>;


  vector_type &_rhs1 = right_hand_side.block(0);
  vector_type &_rhs2 = right_hand_side.block(1);

  dealii::IndexSet rhs1_locally_owned = _rhs1.locally_owned_elements();
  dealii::IndexSet rhs1_ghost = _rhs1.get_partitioner()->ghost_indices();
  //rhs1_locally_relevant.add_indices(_rhs1.get_partitioner()->ghost_indices());
  dealii::IndexSet rhs2_locally_owned = _rhs2.locally_owned_elements();
  dealii::IndexSet rhs2_ghost = _rhs2.get_partitioner()->ghost_indices();
  //rhs2_locally_relevant.add_indices(_rhs2.get_partitioner()->ghost_indices());

  trilinos_vector_type rhs1(rhs1_locally_owned);
  trilinos_vector_type rhs2(rhs2_locally_owned);
  rhs1_locally_owned.print(std::cout);
  std::cout << rhs1.size() << std::endl;
  // trilinos_vector_type rhs1(rhs1_locally_owned, rhs1_ghost);
  // trilinos_vector_type rhs2(rhs2_locally_owned, rhs2_ghost);
  trilinos_vector_type soln1(_rhs1.locally_owned_elements());
  trilinos_vector_type soln2(_rhs2.locally_owned_elements());

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

  //vector_type &soln1 = solution.block(0);
  //vector_type &soln2 = solution.block(1);

  using matrix_type = dealii::TrilinosWrappers::SparseMatrix;
  using payload_type = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;


  const auto &L11 = matrix_A.block(0,0);
  const auto &L12 = matrix_A.block(0,1);
  const auto &L21 = matrix_A.block(1,0);
  const auto &L22 = matrix_A.block(1,1);

  const auto op_L11 = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(L11);
  const auto op_L12 = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(L12);
  const auto op_L21 = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(L21);
  const auto op_L22 = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(L22);

  dealii::ReductionControl reduction_control_L11(2000, 1.0e-15, 1.0e-12);
  dealii::SolverGMRES<trilinos_vector_type> solver_L11(reduction_control_L11);

  dealii::TrilinosWrappers::PreconditionILU preconditioner_L11;
  //dealii::TrilinosWrappers::PreconditionIdentity preconditioner_L11;
  preconditioner_L11.initialize(L11);
  //const dealii::TrilinosWrappers::PreconditionBase &preconditioner_L11_base = preconditioner_L11;

  const auto op_L11_inv = dealii::inverse_operator(op_L11, solver_L11, preconditioner_L11);
  const auto op_Schur = op_L22 - op_L21 * op_L11_inv * op_L12;

  const auto op_preconditioner_L11 = dealii::linear_operator<trilinos_vector_type,trilinos_vector_type,payload_type>(L11,preconditioner_L11);
  const auto op_approxSchur = op_L22 - op_L21 * op_preconditioner_L11 * op_L12;

  const trilinos_vector_type schur_rhs = rhs2 - op_L21 * op_L11_inv * rhs1;
  std::cout << reduction_control_L11.last_step() << " GMRES iterations to solve L11inv*rhs1." << std::endl;

  // Schur inverse preconditioner
  // dealii::IterationNumberControl iteration_number_control_aS(300, 1.e-12);
  // dealii::SolverGMRES<trilinos_vector_type> solver_approxSchur(iteration_number_control_aS);
  // const auto preconditioner_Schur = dealii::inverse_operator(op_approxSchur, solver_approxSchur, dealii::PreconditionIdentity());
  // //const auto preconditioner_Schur = dealii::PreconditionIdentity();

  dealii::TrilinosWrappers::SparseMatrix approxSchur;
  const auto L11_rows = L11.locally_owned_range_indices();
  trilinos_vector_type L11_diag_inv(L11_rows);
  for (auto row = L11_rows.begin(); row != L11_rows.end(); ++row) {
   L11_diag_inv[*row] = 1.0/L11.diag_element(*row);
  }
  L21.mmult(approxSchur, L12, L11_diag_inv);
  approxSchur *= -1.0;
  approxSchur.add(1.0,L22);
  dealii::TrilinosWrappers::PreconditionILU preconditioner_Schur;
  const unsigned int ilu_fill = 20, overlap = 1;
  const double ilu_atol = 1e-5, ilu_rtol = 1e-2;
  preconditioner_Schur.initialize(approxSchur, dealii::TrilinosWrappers::PreconditionILU::AdditionalData(ilu_fill,ilu_atol,ilu_rtol,overlap));

  // Schur inverse operator
  dealii::SolverControl solver_control_Schur(2000, 1.e-15,true);
  //dealii::ReductionControl solver_control_Schur(20000, 1.0e-15, 1.0e-12);
  dealii::SolverGMRES<trilinos_vector_type> solver_Schur(solver_control_Schur,
   dealii::SolverGMRES<trilinos_vector_type>::AdditionalData (2000, false, true, false) );
  const auto op_Schur_inv = dealii::inverse_operator(op_Schur, solver_Schur, preconditioner_Schur);

  soln2 = op_Schur_inv * schur_rhs;
  std::cout << solver_control_Schur.last_step() << " GMRES iterations to obtain convergence." << std::endl;
  soln1 = op_L11_inv * (rhs1 - op_L12 * soln2);

  {
   dealii::LinearAlgebra::ReadWriteVector<double> rw_vector;
   rw_vector.reinit(soln1);
   solution.block(0).import(rw_vector, dealii::VectorOperation::insert);
  }
  {
   dealii::LinearAlgebra::ReadWriteVector<double> rw_vector;
   rw_vector.reinit(soln2);
   solution.block(1).import(rw_vector, dealii::VectorOperation::insert);
  }

  
  // int n_digits = 12;
  // {
  //  dealii::FullMatrix<double> fullA(matrix_A.block(0,0).m(),matrix_A.block(0,0).n());
  //  fullA.copy_from(matrix_A.block(0,0));
  //  std::cout<<"Block 0,0 matrix:"<<std::endl;
  //  fullA.print_formatted(std::cout, n_digits, true, n_digits+7, "0", 1., 0.);
  // }
  // {
  //  dealii::FullMatrix<double> fullA(matrix_A.block(0,1).m(),matrix_A.block(0,1).n());
  //  fullA.copy_from(matrix_A.block(0,1));
  //  std::cout<<"Block 0,1 matrix:"<<std::endl;
  //  fullA.print_formatted(std::cout, n_digits, true, n_digits+7, "0", 1., 0.);
  // }
  // {
  //  dealii::FullMatrix<double> fullA(matrix_A.block(1,0).m(),matrix_A.block(1,0).n());
  //  fullA.copy_from(matrix_A.block(1,0));
  //  std::cout<<"Block 1,0 matrix:"<<std::endl;
  //  fullA.print_formatted(std::cout, n_digits, true, n_digits+7, "0", 1., 0.);
  // }
  // {
  //  dealii::FullMatrix<double> fullA(matrix_A.block(1,1).m(),matrix_A.block(1,1).n());
  //  fullA.copy_from(matrix_A.block(1,1));
  //  std::cout<<"Block 1,1 matrix:"<<std::endl;
  //  fullA.print_formatted(std::cout, n_digits, true, n_digits+7, "0", 1., 0.);
  // }
  // std::cout<<"rhs1:"<<std::endl;
  // rhs1.print(std::cout, n_digits);
  // std::cout<<"rhs2:"<<std::endl;
  // rhs2.print(std::cout, n_digits);
  // std::cout<<"soln1:"<<std::endl;
  // soln1.print(std::cout, n_digits);
  // std::cout<<"soln2:"<<std::endl;
  // soln2.print(std::cout, n_digits);
  
  const trilinos_vector_type r1 = op_L11 * soln1 + op_L12 * soln2 - rhs1;
  const trilinos_vector_type r2 = op_L21 * soln1 + op_L22 * soln2 - rhs2;

  std::cout<<"r1 norm: "<<r1.l2_norm()<<std::endl;
  //r1.print(std::cout, n_digits);
  std::cout<<"r2 norm: "<<r2.l2_norm()<<std::endl;
  //r2.print(std::cout, n_digits);

    }
    //{
 //  dealii::SolverControl solver_control(std::max<std::size_t>(100000, right_hand_side.size() / 10), 1e-10 * right_hand_side.l2_norm(), true, true);
 //  dealii::SolverGMRES<dealii::LinearAlgebra::distributed::BlockVector<double>> solver(solver_control);
 //  //dealii::PreconditionJacobi<dealii::TrilinosWrappers::BlockSparseMatrix<double>> preconditioner;
 //  //dealii::TrilinosWrappers::PreconditionJacobi<dealii::TrilinosWrappers::BlockSparseMatrix> preconditioner;
 //  //preconditioner.initialize(matrix_A, 1.0);
 //  solver.solve(matrix_A, solution, right_hand_side, dealii::PreconditionIdentity());
 //  dealii::LinearAlgebra::distributed::BlockVector<double> residual(right_hand_side);
 //  matrix_A.vmult(residual, solution);
 //  residual -= right_hand_side;
 //  std::cout << "   Iterations required for convergence: "
 //   << solver_control.last_step() << '\n'
 //   << "   Max norm of residual:                "
 //   << residual.linfty_norm() << '\n';

    //}
    return {-1.0, -1.0};
}

/// L2 solution error.
template <int dim, int nstate, typename real>
class L2_Norm_Functional : public PHiLiP::Functional<dim, nstate, real>
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.
public:
    /// Constructor
    L2_Norm_Functional(
        std::shared_ptr<PHiLiP::DGBase<dim,real>> dg_input,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = false)
    : PHiLiP::Functional<dim,nstate,real>(dg_input,uses_solution_values,uses_solution_gradient)
    {}

    /// Templated volume integrand.
    template <typename real2>
    real2 evaluate_volume_integrand(
              const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
              const dealii::Point<dim,real2> &phys_coord,
              const std::array<real2,nstate> &soln_at_q,
              const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/) const
    {
       real2 l2error = 0;
       
       for (int istate=0; istate<nstate; ++istate) {
           const real2 uexact = physics.manufactured_solution_function->value(phys_coord, istate);
           l2error += std::pow(soln_at_q[istate] - uexact, 2);
       }

       return l2error;
    }

    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    template<typename real2>
    real2 evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int /*boundary_id*/,
        const dealii::Point<dim,real2> &phys_coord,
        const dealii::Tensor<1,dim,real2> &/*normal*/,
        const std::array<real2,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/) const
    {
        real2 l1error = 0;
        for (int istate=0; istate<nstate; ++istate) {
            const real2 uexact = physics.manufactured_solution_function->value(phys_coord, istate);
            l1error += std::abs(soln_at_q[istate] - uexact);
        }
        return l1error;
    }

    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,real> &phys_coord,
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const override
    {
        return evaluate_boundary_integrand<real>(
            physics,
            boundary_id,
            phys_coord,
            normal,
            soln_at_q,
            soln_grad_at_q);
    }
    /// Virtual function for Sacado computation of cell boundary functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual FadFadType evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,FadFadType> &phys_coord,
        const dealii::Tensor<1,dim,FadFadType> &normal,
        const std::array<FadFadType,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &soln_grad_at_q) const override
    {
        return evaluate_boundary_integrand<FadFadType>(
            physics,
            boundary_id,
            phys_coord,
            normal,
            soln_at_q,
            soln_grad_at_q);
    }


    /// non-template functions to override the template classes
    real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const dealii::Point<dim,real> &phys_coord,
        const std::array<real,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
    }
       /// non-template functions to override the template classes
    FadFadType evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &physics,
        const dealii::Point<dim,FadFadType> &phys_coord,
        const std::array<FadFadType,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &soln_grad_at_q) const override
    {
        return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
    }

};


template <int dim, int nstate>
void initialize_perturbed_solution(PHiLiP::DGBase<dim,double> &dg, const PHiLiP::Physics::PhysicsBase<dim,nstate,double> &physics)
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg.locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg.dof_handler, *physics.manufactured_solution_function, solution_no_ghost);
    dg.solution = solution_no_ghost;
}

int main(int argc, char *argv[])
{

    const int dim = PHILIP_DIM;
    const int nstate = 1;
    int fail_bool = false;

    // Initializing MPI
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, this_mpi_process==0);

    // Initializing parameter handling
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters(parameter_handler);
    PHiLiP::Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters(parameter_handler);

    // polynomial order and mesh size
    const unsigned poly_degree = 1;

    // creating the grid
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
#if PHILIP_DIM!=1
        MPI_COMM_WORLD,
#endif
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    const unsigned int n_refinements = 2;
    double left = 0.0;
    double right = 2.0;
    const bool colorize = true;

    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
    grid->refine_global(n_refinements);
    const double random_factor = 0.2;
    const bool keep_boundary = false;
    if (random_factor > 0.0) dealii::GridTools::distort_random (random_factor, *grid, keep_boundary);

    pcout << "Grid generated and refined" << std::endl;

    // creating the dg
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, grid);
    pcout << "dg created" << std::endl;

    dg->allocate_system();
    pcout << "dg allocated" << std::endl;

    const int n_refine = 2;
    for (int i=0; i<n_refine;i++) {
        dg->high_order_grid->prepare_for_coarsening_and_refinement();
        grid->prepare_coarsening_and_refinement();
        unsigned int icell = 0;
        for (auto cell = grid->begin_active(); cell!=grid->end(); ++cell) {
            icell++;
            if (!cell->is_locally_owned()) continue;
            if (icell < grid->n_global_active_cells()/2) {
                cell->set_refine_flag();
            }
        }
        grid->execute_coarsening_and_refinement();
        bool mesh_out = (i==n_refine-1);
        dg->high_order_grid->execute_coarsening_and_refinement(mesh_out);
    }
    dg->allocate_system ();

    // manufactured solution function
    std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,double>> physics_double = PHiLiP::Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    pcout << "Physics created" << std::endl;
 
    // performing the interpolation for the intial conditions
    initialize_perturbed_solution(*dg, *physics_double);
    pcout << "solution initialized" << std::endl;

    // evaluating the derivative (using SACADO)
    pcout << std::endl << "Starting Hessian AD... " << std::endl;
    L2_Norm_Functional<dim,nstate,double> functional(dg,true,false);
    const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
    double functional_value = functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
    (void) functional_value;

    // Evaluate residual Hessians
    bool compute_dRdW, compute_dRdX, compute_d2R;

    pcout << "Evaluating RHS only to use as dual variables..." << std::endl;
    compute_dRdW = false; compute_dRdX = false, compute_d2R = false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dealii::LinearAlgebra::distributed::Vector<double> dummy_dual(dg->right_hand_side);
    dg->set_dual(dummy_dual);

    pcout << "Evaluating RHS with d2R..." << std::endl;
    compute_dRdW = true; compute_dRdX = false, compute_d2R = false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    compute_dRdW = false; compute_dRdX = true, compute_d2R = false;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    compute_dRdW = false; compute_dRdX = false, compute_d2R = true;
    dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dealii::LinearAlgebra::distributed::Vector<double> rhs_d2R(dg->right_hand_side);
    // pcout << "*******************************************************************************" << std::endl;

    dealii::TrilinosWrappers::SparseMatrix dRdW_transpose;
    {
        Epetra_CrsMatrix *transpose_CrsMatrix;
        Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&dg->system_matrix.trilinos_matrix()));
        epmt.CreateTranspose(false, transpose_CrsMatrix);
        dRdW_transpose.reinit(*transpose_CrsMatrix);
    }

    dealii::TrilinosWrappers::SparseMatrix dRdX_transpose;
    {
        Epetra_CrsMatrix *transpose_CrsMatrix;
        Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&dg->dRdXv.trilinos_matrix()));
        epmt.CreateTranspose(false, transpose_CrsMatrix);
        dRdX_transpose.reinit(*transpose_CrsMatrix);
    }

    dealii::TrilinosWrappers::SparseMatrix d2RdXdW;
    {
        Epetra_CrsMatrix *transpose_CrsMatrix;
        Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&dg->d2RdWdX.trilinos_matrix()));
        epmt.CreateTranspose(false, transpose_CrsMatrix);
        d2RdXdW.reinit(*transpose_CrsMatrix);
    }
    dealii::TrilinosWrappers::SparseMatrix d2IdXdW;
    {
        Epetra_CrsMatrix *transpose_CrsMatrix;
        Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&functional.d2IdWdX.trilinos_matrix()));
        epmt.CreateTranspose(false, transpose_CrsMatrix);
        d2IdXdW.reinit(*transpose_CrsMatrix);
    }

    // Form Lagrangian Hessian
    functional.d2IdWdW.add(1.0,dg->d2RdWdW);
    functional.d2IdWdX.add(1.0,dg->d2RdWdX);
    d2IdXdW.add(1.0,d2RdXdW);
    functional.d2IdXdX.add(1.0,dg->d2RdXdX);


    // Block Sparsity pattern
    // L_ww  L_wx  R_w^T
    // L_xw  L_xx  R_x^T
    // R_w   R_x   0
    // const unsigned int n_constraints = rhs_only.size();
    // const unsigned int n_flow_var    = dg->dof_handler.n_dofs();
    // const unsigned int n_geom_var    = dg->high_order_grid->dof_handler_grid.n_dofs();
    //BlockDynamicSparsityPattern dsp(3, 3);
    //dsp.block(0, 0).reinit(n_flow_var, n_flow_var);
    //dsp.block(0, 1).reinit(n_flow_var, n_geom_var);
    //dsp.block(0, 2).reinit(n_flow_var, n_constraints);

    //dsp.block(1, 0).reinit(n_geom_var, n_flow_var);
    //dsp.block(1, 1).reinit(n_geom_var, n_geom_var);
    //dsp.block(1, 2).reinit(n_geom_var, n_constraints);

    //dsp.block(2, 0).reinit(n_constraints, n_flow_var);
    //dsp.block(2, 1).reinit(n_constraints, n_geom_var);
    //dsp.block(2, 2).reinit(n_constraints, n_constraints);

    dealii::TrilinosWrappers::BlockSparseMatrix kkt_hessian;
    kkt_hessian.reinit(3,3);
    kkt_hessian.block(0, 0).copy_from( functional.d2IdWdW);
    kkt_hessian.block(0, 1).copy_from( functional.d2IdWdX);
    kkt_hessian.block(0, 2).copy_from( dRdW_transpose);

    kkt_hessian.block(1, 0).copy_from( d2IdXdW);
    kkt_hessian.block(1, 1).copy_from( functional.d2IdXdX);
    kkt_hessian.block(1, 2).copy_from( dRdX_transpose);

    kkt_hessian.block(2, 0).copy_from( dg->system_matrix);
    kkt_hessian.block(2, 1).copy_from( dg->dRdXv);
    dealii::TrilinosWrappers::SparsityPattern zero_sparsity_pattern(dg->locally_owned_dofs, MPI_COMM_WORLD, 0);
    //dealii::TrilinosWrappers::SparseMatrix zero_block;//(n_constraints, n_constraints, 0);
    //zero_block.reinit(dg->locally_owned_dofs, zero_sparsity_pattern, MPI_COMM_WORLD);
    // zero_block.reinit(dg->locally_owned_dofs);
    // dealii::TrilinosWrappers::SparseMatrix zero_block(dg->locally_owned_dofs);
    // kkt_hessian.block(2, 2).copy_from( zero_block );
    zero_sparsity_pattern.compress();
    kkt_hessian.block(2, 2).reinit(zero_sparsity_pattern);

    kkt_hessian.collect_sizes();

    pcout << "kkt_hessian.frobenius_norm()  " << kkt_hessian.frobenius_norm() << std::endl;

    // const int n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    // if (n_mpi_processes == 1) {
    //     dealii::FullMatrix<double> fullA(kkt_hessian.m());
    //     fullA.copy_from(kkt_hessian);
    //     pcout<<"d2IdWdW:"<<std::endl;
    //     if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);
    // }

    dealii::LinearAlgebra::distributed::BlockVector<double> block_vector(3);
    block_vector.block(0) = dg->solution;
    block_vector.block(1) = dg->high_order_grid->volume_nodes;
    block_vector.block(2) = dummy_dual;
    dealii::LinearAlgebra::distributed::BlockVector<double> Hv(3);
    dealii::LinearAlgebra::distributed::BlockVector<double> Htv(3);
    Hv.reinit(block_vector);
    Htv.reinit(block_vector);

    kkt_hessian.vmult(Hv,block_vector);
    kkt_hessian.Tvmult(Htv,block_vector);

    Htv.sadd(-1.0, Hv);

    const double vector_norm = Hv.l2_norm();
    const double vector_abs_diff = Htv.l2_norm();
    const double vector_rel_diff = vector_abs_diff / vector_norm;

    const double tol = 1e-11;
    pcout << "Error: "
                    << " vector_abs_diff: " << vector_abs_diff
                    << " vector_rel_diff: " << vector_rel_diff
                    << std::endl
                    << " vector_abs_diff: " << vector_abs_diff
                    << " vector_rel_diff: " << vector_rel_diff
                    << std::endl;
    if (vector_abs_diff > tol && vector_rel_diff > tol) fail_bool = true;

    const int n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    if (n_mpi_processes == 1) {
        dealii::FullMatrix<double> fullA(kkt_hessian.m());
        fullA.copy_from(kkt_hessian);
  const int n_digits = 8;
        if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), n_digits, true, n_digits+7, "0", 1., 0.);
    }

    dealii::deallog.depth_console(3);
    solve_linear (
        kkt_hessian,
        Hv, // b
        Htv, // x
        all_parameters.linear_solver_param);

    return fail_bool;
}

