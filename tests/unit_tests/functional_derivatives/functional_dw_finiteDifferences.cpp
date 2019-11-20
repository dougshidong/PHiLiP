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

    	// non-template functions to override the template classes
		real evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
            const dealii::Point<dim,real> &phys_coord,
            const std::array<real,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) override
		{
			return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
		}
        using ADtype = Sacado::Fad::DFad<real>;
		ADtype evaluate_volume_integrand(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,ADtype> &physics,
            const dealii::Point<dim,ADtype> &phys_coord,
            const std::array<ADtype,nstate> &soln_at_q,
            const std::array<dealii::Tensor<1,dim,ADtype>,nstate> &soln_grad_at_q) override
		{
			return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, soln_grad_at_q);
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
			std::vector<real> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
			std::vector<real> local_dIdw(max_dofs_per_cell);

			const auto mapping = (*(dg.high_order_grid.mapping_fe_field));
			dealii::hp::MappingCollection<dim> mapping_collection(mapping);

			dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
			dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);

			dg.solution.update_ghost_values();
            auto metric_cell = dg.high_order_grid.dof_handler_grid.begin_active();
            auto cell = dg.dof_handler.begin_active();
            for( ; cell != dg.dof_handler.end(); ++cell, ++metric_cell) {
				if(!cell->is_locally_owned()) continue;

				// // setting up the volume integration
				// const unsigned int mapping_index = 0; // *** ask doug if this will ever be 
				// const unsigned int fe_index_curr_cell = cell->active_fe_index();
				// const unsigned int quad_index = fe_index_curr_cell;
				// const dealii::FESystem<dim,dim> &current_fe_ref = dg.fe_collection[fe_index_curr_cell];
				// //const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
				// const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

				// // reinitialize the volume integration
				// fe_values_collection_volume.reinit(cell, quad_index, mapping_index, fe_index_curr_cell);
				// const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

				// // getting the indices
				// current_dofs_indices.resize(n_dofs_curr_cell);
				// cell->get_dof_indices(current_dofs_indices);

				// // copying values for initial solution
				// soln_coeff.resize(n_dofs_curr_cell);
				// for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof){
				// 	soln_coeff[idof] = dg.solution[current_dofs_indices[idof]];
				// }

                // setting up the volume integration
                const unsigned int i_fele = cell->active_fe_index();
                const unsigned int i_quad = i_fele;

                // Get solution coefficients
                const dealii::FESystem<dim,dim> &fe_solution = dg.fe_collection[i_fele];
                const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
                current_dofs_indices.resize(n_soln_dofs_cell);
                cell->get_dof_indices(current_dofs_indices);
                soln_coeff.resize(n_soln_dofs_cell);
                for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
                    soln_coeff[idof] = dg.solution[current_dofs_indices[idof]];
                }

                // Get metric coefficients
                const dealii::FESystem<dim,dim> &fe_metric = dg.high_order_grid.fe_system;
                const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
                std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);
                metric_cell->get_dof_indices (cell_metric_dofs_indices);
                std::vector<real> coords_coeff(n_metric_dofs_cell);
                for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
                    coords_coeff[idof] = dg.high_order_grid.nodes[cell_metric_dofs_indices[idof]];
                }

                const dealii::Quadrature<dim> &volume_quadrature = dg.volume_quadrature_collection[i_quad];

				// adding the contribution from the current volume, also need to pass the solution vector on these points
				//local_sum_old = this->evaluate_volume_integrand(physics, fe_values_volume, soln_coeff);
                local_sum_old = this->evaluate_volume_cell_functional(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

				// now looping over all the DOFs in this cell and taking the FD
				local_dIdw.resize(n_soln_dofs_cell);
				for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
					// for each dof copying the solution
					for(unsigned int idof2 = 0; idof2 < n_soln_dofs_cell; ++idof2){
						soln_coeff[idof2] = dg.solution[current_dofs_indices[idof2]];
					}
					soln_coeff[idof] += STEPSIZE;

					// then peturb the idof'th value
					// local_sum_new = this->evaluate_volume_integrand(physics, fe_values_volume, soln_coeff);
                    local_sum_new = this->evaluate_volume_cell_functional(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
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

	L2_Norm_Functional<dim,nstate,double> l2norm(dg,true,false);
	double l2error_mpi_sum2 = std::sqrt(l2norm.evaluate_functional(*physics_adtype,true,false));
	pcout << std::endl << "Overall error: " << l2error_mpi_sum2 << std::endl;

	// evaluating the derivative (using SACADO)
	pcout << std::endl << "Starting AD: " << std::endl;
	dealii::LinearAlgebra::distributed::Vector<double> dIdw = l2norm.dIdw;
	// dIdw.print(std::cout);

	// evaluating the derivative (using finite differneces)
	pcout << std::endl << "Starting FD: " << std::endl;
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
