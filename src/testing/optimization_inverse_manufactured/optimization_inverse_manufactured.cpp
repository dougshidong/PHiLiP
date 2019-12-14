#include <stdlib.h>
#include <iostream>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "optimization_inverse_manufactured.h"

#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "dg/high_order_grid.h"
#include "ode_solver/ode_solver.h"

#include "functional/functional.h"
#include "functional/adjoint.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate, typename real>
class VolumeL2normError : public Functional<dim, nstate, real>
{
public:
	/// Constructor
	VolumeL2normError(
		std::shared_ptr<PHiLiP::DGBase<dim,real>> dg_input,
		const bool uses_solution_values = true,
		const bool uses_solution_gradient = true)
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
	using ADADtype = Sacado::Fad::DFad<ADtype>;
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

	// Create target mesh
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

	HighOrderGrid<dim,double> high_order_grid(all_parameters, poly_degree, &grid);
#if PHILIP_DIM!=1
	high_order_grid.prepare_for_coarsening_and_refinement();
	grid.repartition();
	high_order_grid.execute_coarsening_and_refinement();
	high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);
#endif

	// Prescribe surface displacements
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
	// Perform mesh movement
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
	std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, &grid);
    dg->allocate_system ();
	std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,double>> physics_double = PHiLiP::Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters);
	initialize_perturbed_solution(*dg, *physics_double);
	std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
	ode_solver->steady_state();

	// Get back our square mesh through mesh deformation
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
	high_order_grid.output_results_vtk(high_order_grid.nth_refinement++);

    return fail_bool;
}

template class OptimizationInverseManufactured <PHILIP_DIM,1>;
template class OptimizationInverseManufactured <PHILIP_DIM,2>;
template class OptimizationInverseManufactured <PHILIP_DIM,3>;
template class OptimizationInverseManufactured <PHILIP_DIM,4>;
template class OptimizationInverseManufactured <PHILIP_DIM,5>;

} // Tests namespace
} // PHiLiP namespace



