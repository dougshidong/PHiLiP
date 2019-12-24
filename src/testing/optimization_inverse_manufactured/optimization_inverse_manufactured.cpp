#include <Epetra_RowMatrixTransposer.h>

#include <stdlib.h>
#include <iostream>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/trilinos_linear_operator.h>

#include "optimization_inverse_manufactured.h"

#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "dg/high_order_grid.h"
#include "ode_solver/ode_solver.h"

#include "functional/target_functional.h"
#include "functional/adjoint.h"

#include "linear_solver/linear_solver.h"

namespace PHiLiP {
namespace Tests {

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
																		 TargetFunctional<dim, nstate, double> &functional)
{
    if ( dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0 )
		std::cout << "Evaluating KKT right-hand side: dIdW, dIdX, d2I, Residual..." << std::endl;
	dealii::LinearAlgebra::distributed::BlockVector<double> kkt_rhs(3);

	bool compute_dIdW = true, compute_dIdX = true, compute_d2I = true;
	(void) functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);

	bool compute_dRdW = false, compute_dRdX = false, compute_d2R = false;
	dg.assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

	kkt_rhs.block(0) = functional.dIdw;
	kkt_rhs.block(1) = functional.dIdX;
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
	const double reduced_hessian_diagonal = 1000000;
	P.block(1) = Y.block(1);
	P.block(1) /= reduced_hessian_diagonal;

	// As_inv_Ad_p2
	vector_type Ad_p2, As_inv_Ad_p2;
	Ad_p2.reinit(solution.block(0));
	Ad.vmult(Ad_p2, P.block(1));

	As_inv_Ad_p2.reinit(solution.block(0));
	solve_linear(As, As_inv_Ad_p2, Ad_p2, param);

	// p1
	P.block(0) = As_inv_y1;
	P.block(0) -= As_inv_Ad_p2;

	// p3
	vector_type p3_rhs;
	p3_rhs.reinit(solution.block(2));

	Wss.vmult(p3_rhs, As_inv_Ad_p2);
	Wsd.vmult_add(p3_rhs, P.block(1));
	p3_rhs *= -1.0;
	p3_rhs += Y.block(2);

	solve_linear(AsT, P.block(2), p3_rhs, param);

	solution = P;

    return {-1.0, -1.0};
}

template <int dim, int nstate, typename real>
class InverseTarget : public TargetFunctional<dim, nstate, real>
{
public:
    InverseTarget(
        std::shared_ptr<DGBase<dim,real>> dg_input,
		const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true)
	: TargetFunctional<dim,nstate,real>(dg_input, target_solution, uses_solution_values, uses_solution_gradient)
	{}

	template <typename real2>
	real2 evaluate_volume_integrand(
		const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/*physics*/,
		const dealii::Point<dim,real2> &/*phys_coord*/,
		const std::array<real2,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
		const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/,
		const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*target_soln_grad_at_q*/)
	{
		real2 l2error = 0;
		
		for (int istate=0; istate<nstate; ++istate) {
			l2error += std::pow(soln_at_q[istate] - target_soln_at_q[istate], 2);
		}
		//std::cout << "l2error: " << l2error << std::endl;

		return l2error;
	}

	// non-template functions to override the template classes
	real evaluate_volume_integrand(
		const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
		const dealii::Point<dim,real> &phys_coord,
		const std::array<real,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
		const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q,
		const std::array<dealii::Tensor<1,dim,real>,nstate> &target_soln_grad_at_q) override
	{
		return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, target_soln_at_q, soln_grad_at_q, target_soln_grad_at_q);
	}
	using ADtype = Sacado::Fad::DFad<real>;
	using ADADtype = Sacado::Fad::DFad<ADtype>;
	ADADtype evaluate_volume_integrand(
		const PHiLiP::Physics::PhysicsBase<dim,nstate,ADADtype> &physics,
		const dealii::Point<dim,ADADtype> &phys_coord,
		const std::array<ADADtype,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
		const std::array<dealii::Tensor<1,dim,ADADtype>,nstate> &soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,ADADtype>,nstate> &target_soln_grad_at_q) override
	{
		return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, target_soln_at_q, soln_grad_at_q, target_soln_grad_at_q);
	}
};

template <int dim, int nstate>
dealii::Point<dim> warp (const dealii::Point<dim> &p)
{
    dealii::Point<dim> q = p;
    if (dim == 1) {
		q[dim-1] *= 1.5;
	} else if (dim == 2) {
		q[0] *= p[0]*std::sin(2.0*dealii::numbers::PI*p[1]);
	} else if (dim == 3) {
		q[0] *= p[0]*std::sin(2.0*dealii::numbers::PI*p[1]);
		q[1] *= p[0]*std::sin(2.0*dealii::numbers::PI*p[1]);
	}
    return q;
}
template <int dim, int nstate>
OptimizationInverseManufactured<dim,nstate>::OptimizationInverseManufactured(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
void initialize_perturbed_solution(PHiLiP::DGBase<dim,double> &dg, const PHiLiP::Physics::PhysicsBase<dim,nstate,double> &physics)
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg.locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg.dof_handler, *physics.manufactured_solution_function, solution_no_ghost);
    dg.solution = solution_no_ghost;
}

template<int dim, int nstate>
int OptimizationInverseManufactured<dim,nstate>
::run_test () const
{
	const double amplitude = 0.1;
    const int poly_degree = 1;
    int fail_bool = false;
	pcout << " Running optimization case... " << std::endl;

	// *****************************************************************************
	// Create target mesh
	// *****************************************************************************
	const unsigned int initial_n_cells = 10;
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    dealii::Triangulation<dim> grid(
        typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::smoothing_on_refinement |
        dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
	dealii::parallel::distributed::Triangulation<dim> grid( MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::smoothing_on_refinement |
        dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
	dealii::GridGenerator::subdivided_hyper_cube(grid, initial_n_cells);
	for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
		// Set a dummy boundary ID
		cell->set_material_id(9002);
		for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
			if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
		}
	}

	// Create DG from which we'll modify the HighOrderGrid
	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, &grid);
    dg->allocate_system ();

	HighOrderGrid<dim,double> &high_order_grid = dg->high_order_grid;
#if PHILIP_DIM!=1
	high_order_grid.prepare_for_coarsening_and_refinement();
	grid.repartition();
	high_order_grid.execute_coarsening_and_refinement();
	high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);
#endif

	// *****************************************************************************
	// Prescribe surface displacements
	// *****************************************************************************
	std::vector<dealii::Tensor<1,dim,double>> point_displacements(high_order_grid.locally_relevant_surface_points.size());
	const unsigned int n_locally_relevant_surface_nodes = dim * high_order_grid.locally_relevant_surface_points.size();
	std::vector<dealii::types::global_dof_index> surface_node_global_indices(n_locally_relevant_surface_nodes);
	std::vector<double> surface_node_displacements(n_locally_relevant_surface_nodes);
	{
		auto displacement = point_displacements.begin();
		auto point = high_order_grid.locally_relevant_surface_points.begin();
		auto point_end = high_order_grid.locally_relevant_surface_points.end();
		for (;point != point_end; ++point, ++displacement) {
			(*displacement)[0] = amplitude * (*point)[0];
			if(dim>=2) {
				(*displacement)[0] *= std::sin(2.0*dealii::numbers::PI*(*point)[1]);
			}
			if(dim>=3) {
				(*displacement)[0] *= std::sin(2.0*dealii::numbers::PI*(*point)[2]);
			}
		}
		int inode = 0;
		for (unsigned int ipoint=0; ipoint<point_displacements.size(); ++ipoint) {
			for (unsigned int d=0;d<dim;++d) {
				const std::pair<unsigned int, unsigned int> point_axis = std::make_pair(ipoint,d);
				const dealii::types::global_dof_index global_index = high_order_grid.point_and_axis_to_global_index[point_axis];
				surface_node_global_indices[inode] = global_index;
				surface_node_displacements[inode] = point_displacements[ipoint][d];
				inode++;
			}
		}
	}
	// *****************************************************************************
	// Perform mesh movement
	// *****************************************************************************
	using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
	MeshMover::LinearElasticity<dim, double, VectorType , dealii::DoFHandler<dim>> 
		meshmover(high_order_grid, surface_node_global_indices, surface_node_displacements);
	VectorType volume_displacements = meshmover.get_volume_displacements();

	high_order_grid.nodes += volume_displacements;
	high_order_grid.nodes.update_ghost_values();
    high_order_grid.update_surface_indices();
    high_order_grid.update_surface_nodes();
	high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

	// Get discrete solution on this target grid
	std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,double>> physics_double = PHiLiP::Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters);
	initialize_perturbed_solution(*dg, *physics_double);
	std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
	ode_solver->steady_state();

	// Save target solution and nodes
	const auto target_solution = dg->solution;
	const auto target_nodes    = high_order_grid.nodes;

	// *****************************************************************************
	// Get back our square mesh through mesh deformation
	// *****************************************************************************
	{
		auto displacement = point_displacements.begin();
		auto point = high_order_grid.locally_relevant_surface_points.begin();
		auto point_end = high_order_grid.locally_relevant_surface_points.end();
		for (;point != point_end; ++point, ++displacement) {
			if ((*point)[0] > 0.5 && (*point)[1] > 1e-10 && (*point)[1] < 1-1e-10) {
				const double final_location = 1.0;
				const double current_location = (*point)[0];
				(*displacement)[0] = final_location - current_location;
			}

			// (*displacement)[0] = (*point)[0] * 0.5 - (*point)[0];
			// (*displacement)[1] = (*point)[1] * 0.9 - (*point)[1];
		}
		int inode = 0;
		for (unsigned int ipoint=0; ipoint<point_displacements.size(); ++ipoint) {
			for (unsigned int d=0;d<dim;++d) {
				const std::pair<unsigned int, unsigned int> point_axis = std::make_pair(ipoint,d);
				const dealii::types::global_dof_index global_index = high_order_grid.point_and_axis_to_global_index[point_axis];
				surface_node_global_indices[inode] = global_index;
				surface_node_displacements[inode] = point_displacements[ipoint][d];
				inode++;
			}
		}
	}
	volume_displacements = meshmover.get_volume_displacements();

	high_order_grid.nodes += volume_displacements;
	high_order_grid.nodes.update_ghost_values();
    high_order_grid.update_surface_indices();
    high_order_grid.update_surface_nodes();
	dg->output_results_vtk(high_order_grid.nth_refinement);
	high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

	// Solve on this new grid
	ode_solver->steady_state();

	// Compute current error
	auto error_vector = dg->solution;
	error_vector -= target_solution;
	const double l2_vector_error = error_vector.l2_norm();

	InverseTarget<dim,nstate,double> inverse_target_functional(dg, target_solution, true, false);
	bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
    const double current_l2_error = inverse_target_functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
	pcout << "Vector l2_norm of the coefficients: " << l2_vector_error << std::endl;
	pcout << "Functional l2_norm : " << current_l2_error << std::endl;

	pcout << "*************************************************************" << std::endl;
	pcout << "Starting design... " << std::endl;
	pcout << "Number of state variables: " << dg->dof_handler.n_dofs() << std::endl;
	pcout << "Number of mesh nodes: " << high_order_grid.dof_handler_grid.n_dofs() << std::endl;
	pcout << "Number of constraints: " << dg->dof_handler.n_dofs() << std::endl;

	dealii::LinearAlgebra::distributed::BlockVector<double> kkt_rhs(3);
	kkt_rhs = evaluate_kkt_rhs(*dg, inverse_target_functional);
	dealii::LinearAlgebra::distributed::BlockVector<double> kkt_soln(3), p4inv_kkt_rhs(3);
	kkt_soln.reinit(kkt_rhs);
	p4inv_kkt_rhs.reinit(kkt_rhs);

	auto mesh_error = target_nodes;
	mesh_error -= high_order_grid.nodes;

	double current_functional = inverse_target_functional.current_functional_value;
	double current_kkt_norm = kkt_rhs.l2_norm();
	double current_constraint_satisfaction = kkt_rhs.block(2).l2_norm();
	double current_mesh_error = mesh_error.l2_norm();
	pcout << "*************************************************************" << std::endl;
	pcout << "Initial design " << std::endl;
	pcout << "*************************************************************" << std::endl;
	pcout << "Current functional: " << current_functional << std::endl;
	pcout << "Constraint satisfaction: " << current_constraint_satisfaction << std::endl;
	pcout << "l2norm(Current mesh - optimal mesh): " << current_mesh_error << std::endl;
	pcout << "Current KKT norm: " << current_kkt_norm << std::endl;

	const unsigned int n_max_design = 1000;
	const double kkt_tolerance = 1e-10;
	for (unsigned int i_design = 0; i_design < n_max_design && current_kkt_norm > kkt_tolerance; i_design++) {

		// Set the Lagrange multipliers to the current KKT solution
		dg->set_dual(kkt_soln.block(2));

		// Evaluate KKT right-hand side
		kkt_rhs = evaluate_kkt_rhs(*dg, inverse_target_functional);

		// Evaluate KKT system matrix
		pcout << "Evaluating dRdW..." << std::endl;
		bool compute_dRdW = true, compute_dRdX = false, compute_d2R = false;
		dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
		pcout << "Evaluating dRdX..." << std::endl;
		compute_dRdW = false; compute_dRdX = true, compute_d2R = false;
		dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
		pcout << "Evaluating residual 2nd order partials..." << std::endl;
		compute_dRdW = false; compute_dRdX = false, compute_d2R = true;
		dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

		// Build required operators
		dealii::TrilinosWrappers::SparsityPattern zero_sparsity_pattern(dg->locally_owned_dofs, MPI_COMM_WORLD, 0);
		zero_sparsity_pattern.compress();
		dealii::TrilinosWrappers::BlockSparseMatrix kkt_system_matrix;
		kkt_system_matrix.reinit(3,3);
		kkt_system_matrix.block(0, 0).copy_from( inverse_target_functional.d2IdWdW);
		kkt_system_matrix.block(0, 1).copy_from( inverse_target_functional.d2IdWdX);
		kkt_system_matrix.block(0, 2).copy_from( transpose_trilinos_matrix(dg->system_matrix));

		kkt_system_matrix.block(1, 0).copy_from( transpose_trilinos_matrix(inverse_target_functional.d2IdWdX));
		kkt_system_matrix.block(1, 1).copy_from( inverse_target_functional.d2IdXdX);
		kkt_system_matrix.block(1, 2).copy_from( transpose_trilinos_matrix(dg->dRdXv));

		kkt_system_matrix.block(2, 0).copy_from( dg->system_matrix);
		kkt_system_matrix.block(2, 1).copy_from( dg->dRdXv);
		kkt_system_matrix.block(2, 2).reinit(zero_sparsity_pattern);

		kkt_system_matrix.collect_sizes();

		Parameters::LinearSolverParam linear_solver_param = all_parameters->linear_solver_param;
		linear_solver_param.linear_residual = 1e-12;
		apply_P4 (
			kkt_system_matrix,  // A
			kkt_rhs,            // b
			p4inv_kkt_rhs,      // x
			linear_solver_param);

		kkt_soln = p4inv_kkt_rhs;
		const auto old_solution = dg->solution;
		const auto old_nodes = high_order_grid.nodes;
		const auto old_dual = dg->dual;
		auto dual_step = kkt_soln.block(2);
		dual_step -= old_dual;

		const auto old_functional = current_functional;

		// Line search
		double step_length = 1.0;
		const unsigned int max_linesearch = 40;
		unsigned int i_linesearch = 0;
		for (; i_linesearch < max_linesearch; i_linesearch++) {

			dg->solution.add(step_length, kkt_soln.block(0));
			high_order_grid.nodes.add(step_length, kkt_soln.block(1));

			current_functional = inverse_target_functional.evaluate_functional();

			pcout << "Linesearch " << i_linesearch+1 << " Old functional = " << old_functional << " New functional = " << current_functional << std::endl;
			if (current_functional < old_functional) {
				break;
			} else {
				dg->solution = old_solution;
				high_order_grid.nodes = old_nodes;
				dg->dual = old_dual;
				step_length *= 0.5;
			}
		}
		dg->dual.add(step_length, dual_step);
		if (i_linesearch == max_linesearch) return 1;

		high_order_grid.nodes.update_ghost_values();
		high_order_grid.update_surface_indices();
		high_order_grid.update_surface_nodes();


		dg->output_results_vtk(high_order_grid.nth_refinement);
		high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

		mesh_error = target_nodes;
		mesh_error -= high_order_grid.nodes;
		current_kkt_norm = kkt_rhs.l2_norm();
		current_constraint_satisfaction = dg->right_hand_side.l2_norm();
		current_mesh_error = mesh_error.l2_norm();
		pcout << "*************************************************************" << std::endl;
		pcout << "Design iteration " << i_design << std::endl;
		pcout << "Current functional: " << current_functional << std::endl;
		pcout << "Constraint satisfaction: " << current_constraint_satisfaction << std::endl;
		pcout << "l2norm(Current mesh - optimal mesh): " << current_mesh_error << std::endl;
		pcout << "Current KKT norm: " << current_kkt_norm << std::endl;

	}

	pcout << std::endl << std::endl << std::endl << std::endl;
	// Make sure that if the nodes are located at the target nodes, then we recover our target functional
	high_order_grid.nodes = target_nodes;
	high_order_grid.nodes.update_ghost_values();
    high_order_grid.update_surface_indices();
    high_order_grid.update_surface_nodes();
	// Solve on this new grid
	ode_solver->steady_state();
    const double zero_l2_error = inverse_target_functional.evaluate_functional();
	pcout << "Nodes at target nodes should have zero functional l2 error : " << zero_l2_error << std::endl;
	if (zero_l2_error > 1e-10) return 1;

	if (current_kkt_norm > kkt_tolerance) return 1;
    return fail_bool;
}

template class OptimizationInverseManufactured <PHILIP_DIM,1>;
template class OptimizationInverseManufactured <PHILIP_DIM,2>;
template class OptimizationInverseManufactured <PHILIP_DIM,3>;
template class OptimizationInverseManufactured <PHILIP_DIM,4>;
template class OptimizationInverseManufactured <PHILIP_DIM,5>;

} // Tests namespace
} // PHiLiP namespace



