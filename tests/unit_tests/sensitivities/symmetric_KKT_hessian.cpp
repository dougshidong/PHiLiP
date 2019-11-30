#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <Epetra_RowMatrixTransposer.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "physics/physics_factory.h"
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "functional/functional.h"

const double STEPSIZE = 1e-7;
const double TOLERANCE = 1e-6;

template <int dim, int nstate, typename real>
class L2_Norm_Functional : public PHiLiP::Functional<dim, nstate, real>
{
	public:
        /// Constructor
        L2_Norm_Functional(
            std::shared_ptr<PHiLiP::DGBase<dim,real>> dg_input,
            const bool uses_solution_values = true,
            const bool uses_solution_gradient = false)
        : PHiLiP::Functional<dim,nstate,real>(dg_input,uses_solution_values,uses_solution_gradient)
        {}

        template <typename real2>
		real2 evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
            const dealii::Point<dim,real2> &phys_coord,
            const std::array<real2,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/)
		{
			real2 l2error = 0;
			
			for (int istate=0; istate<nstate; ++istate) {
				const real2 uexact = physics.manufactured_solution_function->value(phys_coord, istate);
				l2error += std::pow(soln_at_q[istate] - uexact, 2);
			}

			return l2error;
		}

        template <typename real2>
        real2 evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/*physics*/,
            const unsigned int /*boundary_id*/,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            std::vector<real2> local_solution)
        {
            real2 boundary_integral = 0;
            const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
            const unsigned int n_quad = fe_values_boundary.n_quadrature_points;
            std::array<real2,nstate> soln_at_q;
            for (unsigned int iquad=0;iquad<n_quad;++iquad) {
                soln_at_q.fill(0.0);
                for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
                    const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
                    soln_at_q[istate]      += local_solution[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
                }
                for (int s=0;s<nstate;++s) {
                    boundary_integral += soln_at_q[s] * fe_values_boundary.JxW(iquad);
                }
            }
            return boundary_integral;
        }

        using ADtype = Sacado::Fad::DFad<double>;
        using ADADtype = Sacado::Fad::DFad<ADtype>;

        real evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
            const unsigned int boundary_id,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            std::vector<real> local_solution) override
        {
            return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);
        }


        ADADtype evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,ADADtype> &physics,
            const unsigned int boundary_id,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            std::vector<ADADtype> local_solution) override
        {
            return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);
        }

    	// non-template functions to override the template classes
		real evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
            const dealii::Point<dim,real> &phys_coord,
            const std::array<real,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) override
		{
			return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
		}
		ADADtype evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,ADADtype> &physics,
            const dealii::Point<dim,ADADtype> &phys_coord,
            const std::array<ADADtype,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,ADADtype>,nstate> &soln_grad_at_q) override
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
	const int n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
	dealii::ConditionalOStream pcout(std::cout, this_mpi_process==0);

	// Initializing parameter handling
	dealii::ParameterHandler parameter_handler;
	PHiLiP::Parameters::AllParameters::declare_parameters(parameter_handler);
	PHiLiP::Parameters::AllParameters all_parameters;
	all_parameters.parse_parameters(parameter_handler);

	// polynomial order and mesh size
	const unsigned poly_degree = 1;

	// creating the grid
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
	dealii::Triangulation<dim> grid(
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
	dealii::parallel::distributed::Triangulation<dim> grid(
		MPI_COMM_WORLD,
	 	typename dealii::Triangulation<dim>::MeshSmoothing(
	 		dealii::Triangulation<dim>::smoothing_on_refinement |
	 		dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

    const unsigned int n_refinements = 2;
	double left = 0.0;
	double right = 2.0;
	const bool colorize = true;

	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);
    grid.refine_global(n_refinements);
    const double random_factor = 0.2;
    const bool keep_boundary = false;
    if (random_factor > 0.0) dealii::GridTools::distort_random (random_factor, grid, keep_boundary);

	pcout << "Grid generated and refined" << std::endl;

	// creating the dg
	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, &grid);
	pcout << "dg created" << std::endl;

	dg->allocate_system();
	pcout << "dg allocated" << std::endl;

    const int n_refine = 2;
    for (int i=0; i<n_refine;i++) {
        dg->high_order_grid.prepare_for_coarsening_and_refinement();
        grid.prepare_coarsening_and_refinement();
        unsigned int icell = 0;
        for (auto cell = grid.begin_active(); cell!=grid.end(); ++cell) {
            icell++;
            if (!cell->is_locally_owned()) continue;
            if (icell < grid.n_global_active_cells()/2) {
                cell->set_refine_flag();
            }
        }
        grid.execute_coarsening_and_refinement();
        bool mesh_out = (i==n_refine-1);
        dg->high_order_grid.execute_coarsening_and_refinement(mesh_out);
    }
    dg->allocate_system ();

	// manufactured solution function
    using ADtype = Sacado::Fad::DFad<double>;
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
	// const unsigned int n_geom_var    = dg->high_order_grid.dof_handler_grid.n_dofs();
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

    if (n_mpi_processes == 1) {
        dealii::FullMatrix<double> fullA(kkt_hessian.m());
        fullA.copy_from(kkt_hessian);
        pcout<<"d2IdWdW:"<<std::endl;
        if (pcout.is_active()) fullA.print_formatted(pcout.get_stream(), 3, true, 10, "0", 1., 0.);
    }

    dealii::LinearAlgebra::distributed::BlockVector<double> block_vector(3);
	block_vector.block(0) = dg->solution;
	block_vector.block(1) = dg->high_order_grid.nodes;
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

	return fail_bool;
}

