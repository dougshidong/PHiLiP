#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

#include <exception>
#include <deal.II/fe/mapping.h> 
#include <deal.II/base/exceptions.h> // ExcTransformationFailed

#include <deal.II/fe/mapping_fe_field.h> 
#include <deal.II/fe/mapping_q.h> 

#include <Sacado.hpp>

#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/high_order_grid.h"
#include "ode_solver/ode_solver.h"
#include "dg/dg.h"
#include "functional/functional.h"

const double STEPSIZE = 1e-6;
const double TOLERANCE = 1e-6;

template <int dim, int nstate, typename real>
class L2_Norm_Functional : public PHiLiP::Functional<dim, nstate, real>
{
	public:
		template <typename real2>
		real2 evaluate_cell_volume(
			const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
			const dealii::FEValues<dim,dim> &fe_values_volume,
			std::vector<real2> local_solution)
		{
			unsigned int n_quad_pts = fe_values_volume.n_quadrature_points;

			std::array<real2,nstate> soln_at_q;

			real2 l2error = 0;

			// looping over the quadrature points
			for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
				std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
				for (unsigned int idof=0; idof<fe_values_volume.dofs_per_cell; ++idof) {
					const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
					soln_at_q[istate] += local_solution[idof] * fe_values_volume.shape_value_component(idof, iquad, istate);
				}
			
				const dealii::Point<dim> qpoint = (fe_values_volume.quadrature_point(iquad));

				for (int istate=0; istate<nstate; ++istate) {
					const real2 uexact = physics.manufactured_solution_function.value(qpoint, istate);
					l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_volume.JxW(iquad);
				}
			}

			return l2error;
		}

		// non-template functions to override the template classes
		real evaluate_cell_volume(
			const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
			const dealii::FEValues<dim,dim> &fe_values_volume,
			std::vector<real> local_solution) override
		{
			return evaluate_cell_volume<>(physics, fe_values_volume, local_solution);
		}
		Sacado::Fad::DFad<real> evaluate_cell_volume(
			const PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics,
			const dealii::FEValues<dim,dim> &fe_values_volume,
			std::vector<Sacado::Fad::DFad<real>> local_solution) override
		{
			return evaluate_cell_volume<>(physics, fe_values_volume, local_solution);
		}

		dealii::LinearAlgebra::distributed::Vector<real> evaluate_dIdw_finiteDifferences(
			PHiLiP::DGBase<dim,real> &dg, 
			const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
			const double STEPSIZE)
		{
			// for taking the local derivatives
			double local_sum_old;
			double local_sum_new;

			// vector for storing the derivatives with respect to each DOF
			dealii::LinearAlgebra::distributed::Vector<real> dIdw;
		
			// allocating the vector
			dealii::IndexSet locally_owned_dofs = dg.dof_handler.locally_owned_dofs();
			dIdw.reinit(locally_owned_dofs, MPI_COMM_WORLD);

			// setup it mostly the same as evaluating the value (with exception that local solution is also AD)
			const unsigned int max_dofs_per_cell = dg.dof_handler.get_fe_collection().max_dofs_per_cell();
			std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
			std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);
			std::vector<real> local_solution(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
			std::vector<real> local_dIdw(max_dofs_per_cell);

			const auto mapping = (*(dg.high_order_grid.mapping_fe_field));
			dealii::hp::MappingCollection<dim> mapping_collection(mapping);

			dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
			dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);

			dg.solution.update_ghost_values();
			for(auto cell = dg.dof_handler.begin_active(); cell != dg.dof_handler.end(); cell++){
				if(!cell->is_locally_owned()) continue;

				// setting up the volume integration
				const unsigned int mapping_index = 0; // *** ask doug if this will ever be 
				const unsigned int fe_index_curr_cell = cell->active_fe_index();
				const unsigned int quad_index = fe_index_curr_cell;
				const dealii::FESystem<dim,dim> &current_fe_ref = dg.fe_collection[fe_index_curr_cell];
				//const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
				const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

				// reinitialize the volume integration
				fe_values_collection_volume.reinit(cell, quad_index, mapping_index, fe_index_curr_cell);
				const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

				// getting the indices
				current_dofs_indices.resize(n_dofs_curr_cell);
				cell->get_dof_indices(current_dofs_indices);

				// copying values for initial solution
				local_solution.resize(n_dofs_curr_cell);
				for(unsigned int idof = 0; idof < n_dofs_curr_cell; idof++){
					local_solution[idof] = dg.solution[current_dofs_indices[idof]];
				}

				// adding the contribution from the current volume, also need to pass the solution vector on these points
				local_sum_old = this->evaluate_cell_volume(physics, fe_values_volume, local_solution);

				// now looping over all the DOFs in this cell and taking the FD
				local_dIdw.resize(n_dofs_curr_cell);
				for(unsigned int idof = 0; idof < n_dofs_curr_cell; idof++){
					// for each dof copying the solution
					for(unsigned int idof2 = 0; idof2 < n_dofs_curr_cell; idof2++){
						local_solution[idof2] = dg.solution[current_dofs_indices[idof2]];
					}
					local_solution[idof] += STEPSIZE;

					// then peturb the idof'th value
					local_sum_new = this->evaluate_cell_volume(physics, fe_values_volume, local_solution);
					local_dIdw[idof] = (local_sum_new-local_sum_old)/STEPSIZE;
				}

				dIdw.add(current_dofs_indices, local_dIdw);
			}
			// compress before the return
			dIdw.compress(dealii::VectorOperation::add);
			
			return dIdw;
		}

};


template <int dim, int nstate>
void initialize_perturbed_solution(PHiLiP::DGBase<dim,double> &dg, const PHiLiP::Physics::PhysicsBase<dim,nstate,double> &physics)
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg.locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg.dof_handler, physics.manufactured_solution_function, solution_no_ghost);
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
	int n_refinements = 5;

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

	double left = 0.0;
	double right = 2.0;
	const bool colorize = true;

	dealii::GridGenerator::hyper_cube(grid, left, right, colorize);
	grid.refine_global(n_refinements);
	pcout << "Grid generated and refined" << std::endl;

	// creating the dg
	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, &grid);
	pcout << "dg created" << std::endl;

	dg->allocate_system();
	pcout << "dg allocated" << std::endl;

	// manufactured solution function
	std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,double>> physics_double = PHiLiP::Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
	std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<double>>> physics_adtype = PHiLiP::Physics::PhysicsFactory<dim, nstate, Sacado::Fad::DFad<double>>::create_Physics(&all_parameters);
	pcout << "Physics created" << std::endl;
	
	// performing the interpolation for the intial conditions
	initialize_perturbed_solution(*dg, *physics_double);
	pcout << "solution initialized" << std::endl;

	L2_Norm_Functional<dim,nstate,double> l2norm;
	double l2error_mpi_sum2 = std::sqrt(l2norm.evaluate_function(*dg, *physics_double));
	pcout << std::endl << "Overall error: " << l2error_mpi_sum2 << std::endl;

	// evaluating the derivative (using SACADO)
	dealii::LinearAlgebra::distributed::Vector<double> dIdw = l2norm.evaluate_dIdw(*dg, *physics_adtype);
	// dIdw.print(std::cout);

	// evaluating the derivative (using finite differneces)
	dealii::LinearAlgebra::distributed::Vector<double> dIdw_FD = l2norm.evaluate_dIdw_finiteDifferences(*dg, *physics_double, STEPSIZE);
	// dIdw_FD.print(std::cout);

	// comparing the results and checking its within the specified tolerance
	dealii::LinearAlgebra::distributed::Vector<double> dIdw_differnece = dIdw;
	dIdw_differnece -= dIdw_FD;
	double difference_L2_norm = dIdw_differnece.l2_norm();
	pcout << "L2 norm of the difference is " << difference_L2_norm << std::endl;

	fail_bool = difference_L2_norm > TOLERANCE;
	return fail_bool;
}