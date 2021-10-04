// for the grid:
#include "grid_refinement_study.h"
// which includes:
// #include "tests.h"
// -- which includes:
// -- // #include "parameters/all_parameters.h"
// -- // #include <deal.II/grid/tria.h>
// -- // #include <deal.II/base/conditional_ostream.h>
// #include "dg/dg.h"
// #include "physics/physics.h"
// #include "parameters/all_parameters.h"
// #include "grid_refinement/gnu_out.h"

// for the actual test:
#include "flow_solver.h" // which includes all required for InitialConditionFunction
#include "initial_condition.h"

#include <deal.II/base/function.h>

// whats in grid_refinement_study.cpp:
#include <stdlib.h>     /* srand, rand */
#include <iostream>
// #include <chrono> // not needed?
// #include <type_traits> // not needed?

// #include <deal.II/base/convergence_table.h> // not needed?

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/tria.h>

// #include <deal.II/grid/tria.h> // included in tests.h
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/grid_refinement.h> // not needed?
// #include <deal.II/grid/grid_tools.h> // not needed?
// #include <deal.II/grid/grid_out.h> // not needed?
// #include <deal.II/grid/grid_in.h> // not needed?

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

// #include <Sacado.hpp> // not needed?

// #include "tests.h" // already included
// #include "grid_refinement_study.h" // already included

#include "physics/physics_factory.h"
// #include "physics/manufactured_solution.h" // not needed?

#include "dg/dg.h"
#include "dg/dg_factory.hpp"

#include "ode_solver/ode_solver.h"

// #include "functional/functional.h" // not needed?
// #include "functional/adjoint.h" // not needed?
// #include "grid_refinement/grid_refinement.h" // not needed?
// #include "grid_refinement/gmsh_out.h" // not needed?
// #include "grid_refinement/msh_out.h" // not needed?
// #include "grid_refinement/size_field.h" // not needed?
// #include "grid_refinement/gnu_out.h" // not needed?

namespace PHiLiP {

namespace Tests {
//=========================================================
// FLOW SOLVER TEST CASE -- What runs the test
//========================================================
template <int dim, int nstate>
FlowSolver<dim, nstate>::FlowSolver(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{
    // Assign initial condition function from the InitialConditionFunction
    // initial_condition_function = InitialConditionFactory_FlowSolver<dim,double>::create_InitialConditionFunction_FlowSolver(parameters_input, nstate);
}

// template <int dim, int nstate>
// void FlowSolver<dim,nstate>::get_grid() const
// {
//     // -- 
//     /** Triangulation to store the grid.
//      *  In 1D, dealii::Triangulation<dim> is used.
//      *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
//      */
//     // For 2D and 3D, the MeshType == Triangulation, is
//     using Triangulation = dealii::parallel::distributed::Triangulation<dim>; // Triangulation == MeshType
//     // create 'grid' pointer
//     std::shared_ptr<Triangulation> grid = MeshFactory<Triangulation>::create_MeshType(this->mpi_communicator);
//     // Create the grid refinement study object: grs
//     // std::shared_ptr<TestBase> grs = std::make_shared < TestBase::GridRefinementStudy<dim,nstate,Triangulation> >(param);
//     GridRefinementStudy<dim,nstate,Triangulation> grs = GridRefinementStudy(param);
//     // Generate the grid based on parameter file
//     grs.get_grid(grid, grs_param);
// }

// template <int dim, int nstate>
// void FlowSolver<dim,nstate>::initialize_solution(PHiLiP::DGBase<dim,double> &dg, const PHiLiP::Physics::PhysicsBase<dim,nstate,double> &physics) const
// {
//     dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
//     solution_no_ghost.reinit(dg.locally_owned_dofs, MPI_COMM_WORLD);
//     dealii::VectorTools::interpolate(dg.dof_handler, *initial_condition_function, solution_no_ghost);
//     dg.solution = solution_no_ghost;
// }

template <int dim, int nstate>
int FlowSolver<dim,nstate>::run_test() const
{
    pcout << " Running Flow Solver. " << std::endl;
    // TO DO: Display the flow case string
    //----------------------------------------------------
    // Parameters
    //----------------------------------------------------
    const Parameters::AllParameters param                = *(TestsBase::all_parameters);
    // const Parameters::GridRefinementStudyParam grs_param = param.grid_refinement_study_param;
    // //----------------------------------------------------
    // // Initialization
    // //----------------------------------------------------
    // const unsigned int poly_degree = grs_param.poly_degree;
    // // const unsigned int poly_degree_max  = grs_param.poly_degree_max;
    // // const unsigned int poly_degree_grid = grs_param.poly_degree_grid;
    // // const unsigned int num_refinements = grs_param.num_refinements;
    // //----------------------------------------------------
    // // Physics
    // //----------------------------------------------------
    // // creating the physics object
    // std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > physics_double = Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&param);
    // //----------------------------------------------------
    // // Grid -- (fixed for now)
    // //----------------------------------------------------
    // // TO DO: Move this to a member function
    // * Triangulation to store the grid.
    //  *  In 1D, dealii::Triangulation<dim> is used.
    //  *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     
    // // For 2D and 3D, the MeshType == Triangulation, is
    // using Triangulation = dealii::parallel::distributed::Triangulation<dim>; // Triangulation == MeshType
    // // create 'grid' pointer
    // std::shared_ptr<Triangulation> grid = MeshFactory<Triangulation>::create_MeshType(this->mpi_communicator);
    // // Create the grid refinement study object: grs
    // // std::shared_ptr<TestBase> grs = std::make_shared < TestBase::GridRefinementStudy<dim,nstate,Triangulation> >(param);
    // GridRefinementStudy<dim,nstate,Triangulation> grs = GridRefinementStudy(param);
    // // Generate the grid based on parameter file
    // grs.get_grid(grid, grs_param);
    // //----------------------------------------------------
    // // Discontinuous Galerkin
    // //----------------------------------------------------
    // // Create DG object using the factory
    // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, grid);
    // dg->allocate_system ();
    //----------------------------------------------------
    // Initialize the solution
    //----------------------------------------------------
    // Create initial condition function from InitialConditionFactory_FlowSolver
    // TO DO: Drop the "_FlowSolver"
    // std::shared_ptr< InitialConditionFunction_FlowSolver<dim,double> > initial_condition_function 
                // = InitialConditionFactory_FlowSolver<dim,double>::create_InitialConditionFunction_FlowSolver(&param, nstate);
    // TO DO: Move this to a member function
    std::cout << "Initializing solution with initial condition function." << std::endl;
    // dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    // solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    // dealii::VectorTools::interpolate(dg->dof_handler, *initial_condition_function, solution_no_ghost);
    // dg->solution = solution_no_ghost;

    return 0;
}

#if PHILIP_DIM==3    
    template class FlowSolver <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

