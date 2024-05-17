#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <fenv.h>

#include <fstream>

#include "dg/dg_base.hpp"
#include "functional/lift_drag.hpp"
#include "mesh/gmsh_reader.hpp"
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "physics/physics_factory.h"
#include <Sacado.hpp>

#include "navier_stokes_naca0012.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_base.h"
#include "ode_solver/ode_solver_factory.h"
#include "ode_solver/runge_kutta_methods/rk_tableau_base.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "flow_solver/flow_solver_cases/naca0012.h"
#include "ode_solver/explicit_ode_solver.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
NavierStokesNACA0012<dim, nstate> :: NavierStokesNACA0012(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}


template <int dim, int nstate>
int NavierStokesNACA0012<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    const unsigned int p_start             = param.manufactured_convergence_study_param.degree_start;
    const unsigned int p_end               = param.manufactured_convergence_study_param.degree_end;
    const unsigned int n_grids_input       = param.manufactured_convergence_study_param.number_of_grids;

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {
        for (unsigned int igrid=0; igrid<n_grids_input; ++igrid) {
            //param.flow_solver_param.poly_degree = poly_degree;
            //param.flow_solver_param.max_poly_degree_for_adaptation = poly_degree;
            //param.flow_solver_param.number_of_mesh_refinements = igrid;
            //std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
            flow_solver->run(); // implicit run
        }
	}

	//unsigned int poly_degree = 1;
    //unsigned int grid_degree = 4;
	// template <int dim, int nstate>
    // std::shared_ptr<Triangulation> NACA0012<dim,nstate>::generate_grid() const
    // {

    //Dummy triangulation
    // using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    // std::shared_ptr<Triangulation> grid;
    // if constexpr(dim!=1) {
    //     std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
    // #if PHILIP_DIM!=1
    //             this->mpi_communicator
    // #endif
    //     );
    //     dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    //     dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
    //     grid->refine_global();
    //     //return grid;
    // }
    //else if constexpr(dim==3) {
    //    const std::string mesh_filename = this->all_parameters.flow_solver_param.input_mesh_filename+std::string(".msh");
    //    const bool use_mesh_smoothing = false;
    //    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename, this->all_parameters.do_renumber_dofs, 0, use_mesh_smoothing);
    //    return naca0012_mesh->triangulation;
    //}
    
    // // TO DO: Avoid reading the mesh twice (here and in set_high_order_grid -- need a default dummy triangulation)
    // }
	PHiLiP::Parameters::AllParameters param_new = *all_parameters;
	param_new.ode_solver_param.ode_solver_type  = Parameters::ODESolverParam::ODESolverEnum::runge_kutta_solver;
	param_new.ode_solver_param.initial_time_step = 0.001;
    param_new.flow_solver_param.poly_degree = 1;
    param_new.flow_solver_param.steady_state_polynomial_ramping = false;
    param_new.flow_solver_param.steady_state = false;
    param_new.manufactured_convergence_study_param.degree_end = 1;
    param_new.manufactured_convergence_study_param.degree_start = 1;

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid;
    //Create NACA0012 using Dealii
    if constexpr (dim == 2) {
            //grid->clear();
            dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
            airfoil_data.airfoil_type = "NACA";
            airfoil_data.naca_id      = "0012";
            airfoil_data.airfoil_length = 1.0;
            airfoil_data.height         = 150.0; // Farfield radius.
            airfoil_data.length_b2      = 150.0;
            airfoil_data.incline_factor = 0.0;
            airfoil_data.bias_factor    = 4.5;
            airfoil_data.refinements    = 0;

            airfoil_data.n_subdivision_x_0 = 15;
            airfoil_data.n_subdivision_x_1 = 15;
            airfoil_data.n_subdivision_x_2 = 15;
            airfoil_data.n_subdivision_y = 15;

            airfoil_data.airfoil_sampling_factor = 10000;

            std::vector<unsigned int> n_subdivisions(dim);
            n_subdivisions[0] = airfoil_data.n_subdivision_x_0 + airfoil_data.n_subdivision_x_1 + airfoil_data.n_subdivision_x_2;
            n_subdivisions[1] = airfoil_data.n_subdivision_y;
            Grids::naca_airfoil(*grid, airfoil_data);
            for (typename Triangulation::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
                for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                    if (cell->face(face)->at_boundary()) {
                        cell->face(face)->set_boundary_id (1004); // riemann
                        //cell->face(face)->set_boundary_id (1005); // farfield
                    }
                }
            }
    } //else {
        //create_curved_grid (grid, GridType::eccentric_hyper_shell);
    //}

    //using meshtype = dealii::parallel::distributed::Triangulation<dim>
    //Set the DG spatial sys
    //
    //std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&param_new,param_new.flow_solver_param.poly_degree, param_new.manufactured_convergence_study_param.degree_end, param_new.flow_solver_param.grid_degree, flow_solver->flow_solver_case->generate_grid());
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&param_new,param_new.flow_solver_param.poly_degree, param_new.manufactured_convergence_study_param.degree_end, param_new.flow_solver_param.grid_degree, grid);
    //dg->allocate_system (false,false,false);

	//std::cout << "Implement initial conditions" << std::endl;
	//Create initial condition function
	//std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
	//		InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&param_new); 
	//SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

	// Create ODE solver using the factory and providing the DG object
	//std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_explicit = PHiLiP::ODE::ODESolverFactory<dim, double>::create_RungeKuttaODESolver(flow_solver->dg);
	
    std::shared_ptr<PHiLiP::ODE::RKTableauBase<dim, double>> rk_tableau = PHiLiP::ODE::ODESolverFactory<dim, double>::create_RKTableau(flow_solver->dg);
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_explicit = std::make_shared<PHiLiP::ODE::RungeKuttaODESolver<dim,double,4,dealii::parallel::distributed::Triangulation<dim>>>(flow_solver->dg,rk_tableau);
    double finalTime = 0.01;

	double dt = param_new.ode_solver_param.initial_time_step;
	ode_solver_explicit->current_iteration = 0;
	for (int i = 0; i < std::ceil(finalTime/dt); ++ i){
		ode_solver_explicit->advance_solution_time(dt);
	}
	return 0;
}

#if PHILIP_DIM!=1
template class NavierStokesNACA0012 <PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
