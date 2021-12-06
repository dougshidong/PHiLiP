#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <Sacado.hpp>

#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "ode_solver/ode_solver_factory.h"
#include "dg/dg_factory.hpp"
#include "functional/target_functional.h"

const double STEPSIZE = 1e-7;
const double TOLERANCE = 1e-4;

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate, typename real>
class L2_Norm_Functional : public PHiLiP::TargetFunctional<dim, nstate, real>
{
 public:
        /// Constructor
        L2_Norm_Functional(
            std::shared_ptr<PHiLiP::DGBase<dim,real>> dg_input,
            const bool uses_solution_values = true,
            const bool uses_solution_gradient = false)
        : PHiLiP::TargetFunctional<dim,nstate,real>(dg_input,uses_solution_values,uses_solution_gradient)
        {}

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
    using FadType = Sacado::Fad::DFad<double>;
 std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,double>> physics_double = PHiLiP::Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
 pcout << "Physics created" << std::endl;
 
 // performing the interpolation for the intial conditions
 initialize_perturbed_solution(*dg, *physics_double);
 pcout << "solution initialized" << std::endl;

 // evaluating the derivative (using SACADO)
 pcout << std::endl << "Starting AD... " << std::endl;
 L2_Norm_Functional<dim,nstate,double> l2norm(dg,true,false);
    dg->solution.add(1.0);
 double l2error_mpi_sum2 = std::sqrt(l2norm.evaluate_functional(true,true));

 dealii::LinearAlgebra::distributed::Vector<double> dIdw = l2norm.dIdw;
 dealii::LinearAlgebra::distributed::Vector<double> dIdX = l2norm.dIdX;

 pcout << std::endl << "Overall error (its ok since we added 1.0 to the target solution): " << l2error_mpi_sum2 << std::endl;

 // evaluating the derivative (using finite differences)
 pcout << std::endl << "Starting FD dIdW... " << std::endl;
 dealii::LinearAlgebra::distributed::Vector<double> dIdw_FD = l2norm.evaluate_dIdw_finiteDifferences(*dg, *physics_double, STEPSIZE);
 // dIdw_FD.print(std::cout);

 pcout << std::endl << "Starting FD dIdX... " << std::endl;
 dealii::LinearAlgebra::distributed::Vector<double> dIdX_FD = l2norm.evaluate_dIdX_finiteDifferences(*dg, *physics_double, STEPSIZE);

 // comparing the results and checking its within the specified tolerance
 dealii::LinearAlgebra::distributed::Vector<double> dIdw_difference = dIdw;
 dIdw_difference -= dIdw_FD;
 double dIdW_L2_diff = dIdw_difference.l2_norm();
 pcout << "L2 norm of FD-AD dIdW: " << dIdW_L2_diff << std::endl;

 // comparing the results and checking its within the specified tolerance
 dealii::LinearAlgebra::distributed::Vector<double> dIdX_difference = dIdX;
 dIdX_difference -= dIdX_FD;
 double dIdX_L2_diff = dIdX_difference.l2_norm();
 pcout << "L2 norm of FD-AD dIdX: " << dIdX_L2_diff << std::endl;

 fail_bool = dIdW_L2_diff > TOLERANCE || dIdX_L2_diff > TOLERANCE;
 return fail_bool;
}

