#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <Sacado.hpp>

#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "ode_solver/ode_solver.h"
#include "dg/dg_factory.hpp"
#include "functional/functional.h"

const double STEPSIZE = 1e-7;
const double TOLERANCE = 1e-6;

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

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

        /// Templated L2 norm integrand.
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

        /// Templated cell boundary integral.
        template <typename real2>
        real2 evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/*physics*/,
            const unsigned int /*boundary_id*/,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            const std::vector<real2> &local_solution) const
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

        using FadType = Sacado::Fad::DFad<double>; ///< Sacado AD type.
        using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type, allows second derivatives.

        /// Non-templated cell boundary integral.
        real evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
            const unsigned int boundary_id,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            const std::vector<real> &local_solution) const override
        {
            return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);
        }

        /// Non-templated cell boundary integral.
        FadFadType evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &physics,
            const unsigned int boundary_id,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            const std::vector<FadFadType> &local_solution) const override
        {
            return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);
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
        dg->high_order_grid.prepare_for_coarsening_and_refinement();
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
        dg->high_order_grid.execute_coarsening_and_refinement(mesh_out);
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
 double l2error_mpi_sum2 = std::sqrt(l2norm.evaluate_functional(true,true));

 dealii::LinearAlgebra::distributed::Vector<double> dIdw = l2norm.dIdw;
 dealii::LinearAlgebra::distributed::Vector<double> dIdX = l2norm.dIdX;

 pcout << std::endl << "Overall error (its ok that it's high since we have extraneous boundary terms): " << l2error_mpi_sum2 << std::endl;

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
