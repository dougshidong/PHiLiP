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

#include <deal.II/lac/full_matrix.h>

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
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"

#include "functional/target_functional.h"
#include "functional/adjoint.h"

#include "linear_solver/linear_solver.h"

#include "testing/full_space_optimization.h"

#include <list>

namespace PHiLiP {
namespace Tests {

#define AMPLITUDE_OPT 0.1
//#define HESSIAN_DIAG 1e7
#define HESSIAN_DIAG 1e0


template<int dim>
dealii::Point<dim> reverse_deformation(dealii::Point<dim> point) {
    dealii::Point<dim> new_point = point;
    const double amplitude = AMPLITUDE_OPT;
 double denom = amplitude;
    if(dim>=2) {
        denom *= std::sin(2.0*dealii::numbers::PI*point[1]);
    }
    if(dim>=3) {
        denom *= std::sin(2.0*dealii::numbers::PI*point[2]);
    }
 denom += 1.0;
 new_point[0] /= denom;
    return new_point;
}
template<int dim>
dealii::Point<dim> target_deformation(dealii::Point<dim> point) {
    const double amplitude = AMPLITUDE_OPT;
    dealii::Tensor<1,dim,double> disp;
    disp[0] = amplitude;
    disp[0] *= point[0];
    if(dim>=2) {
        disp[0] *= std::sin(2.0*dealii::numbers::PI*point[1]);
    }
    if(dim>=3) {
        disp[0] *= std::sin(2.0*dealii::numbers::PI*point[2]);
    }
    return point + disp;
}

// class SplineY
// {
// private:
//     double getbij(const double x, const unsigned int i, const unsigned int j) const
//     {
//         if (j==0) {
//             if (knots[i] <= x && x < knots[i+1]) return 1;
//             else return 0;
//         }
// 
//         double h = getbij(x, i,   j-1);
//         double k = getbij(x, i+1, j-1);
// 
//         double bij = 0;
// 
//         if (h!=0) bij += (x        - knots[i]) / (knots[i+j]   - knots[i]  ) * h;
//         if (k!=0) bij += (knots[i+j+1] - x   ) / (knots[i+j+1] - knots[i+1]) * k;
// 
//         return bij;
//     }
// public:
//     const unsigned int n_design;
//     const unsigned int spline_degree;
//     const unsigned int n_control_pts; // n_design + 2
//     const unsigned int n_knots; // n_control_pts + spline_degree + 1;
// 
//     const double x_start, x_end;
// 
//     std::vector<double> knots;
//     std::vector<double> control_pts;
// 
//     SplineY(const unsigned int n_design, const unsigned int spline_degree, const double x_start, const double x_end)
//         : n_design(n_design)
//         , spline_degree(spline_degree)
//         , n_control_pts(n_design+2)
//         , n_knots(n_control_pts + spline_degree + 1)
//         , x_start(x_start)
//         , x_end(x_end)
//     {
//         knots = getKnots(n_knots, spline_degree, x_start, x_end);
//     };
// 
//     template<typename real>
//     std::vector<real> evalSpline(const std::vector<real> &x) const
//     {
//         const unsigned int nx = x.size();
//         std::vector<real> value(nx, 0);
//         for (unsigned int ix = 0; ix < nx; ix++) {
//             for (unsigned int ictl = 0; ictl < n_control_pts; ictl++) {
//                 value[ix] += control_pts[ictl] * getbij(x[ix], ictl, spline_degree);
//             }
//         }
//         return value;
//     }
//     template<typename real, typename MatrixType>
//     MatrixType eval_dVal_dDesign(const std::vector<real> &x) const
//     {
//         const unsigned int nx = x.size();
// 
//         MatrixType dArea_dSpline = evalSplineDerivative(x);
// 
//         const unsigned int block_start_i = 0;
//         const unsigned int block_start_j = 0;
//         const unsigned int i_size = nx;
//         const unsigned int j_size = n_design;
//         return dArea_dSpline.block(block_start_i, block_start_j, i_size, j_size); // Do not return the endpoints
//     }
//     // now returns dSdCtl instead of dSdDesign (nctl = ndes+2)
//     template<typename real, typename MatrixType>
//     MatrixType evalSplineDerivative(const std::vector<real> &x) const
//     {
//         const int nx = x.size();
//         MatrixType dSdCtl(nx, n_control_pts);
//         dSdCtl.setZero();
// 
//         // Define Area at i+1/2
//         for (int Si = 0; Si < nx; Si++) {
//             //if (Si < n_elem) xh = x[Si] - dx[Si] / 2.0;
//             //else xh = 1.0;
//             const real xh = x[Si];
//             for (int ictl = 0; ictl < n_control_pts; ictl++) {// Not including the inlet/outlet
//                 dSdCtl(Si, ictl) += getbij(xh, ictl, spline_degree);
//             }
//         }
//         return dSdCtl;
//     }
// 
//     //std::vector<dreal> fit_bspline(
//     //    const std::vector<double> &x,
//     //    const std::vector<double> &dx,
//     //    const std::vector<dreal> &area,
//     //    const int n_control_pts,
//     //    const int spline_degree)
//     //{
//     //    int n_face = x.size();
//     //    // Get Knot Vector
//     //    int nknots = n_control_pts + spline_degree + 1;
//     //    std::vector<double> knots(nknots);
//     //    const double x_start = 0.0;
//     //    const double x_end = 1.0;
//     //    knots = getKnots(nknots, spline_degree, x_start, x_end);
// 
//     //    VectorXd ctl_pts(n_control_pts), s_eig(n_face);
//     //    MatrixXd A(n_face, n_control_pts);
//     //    A.setZero();
//     //    
//     //    //double xend = 1.0;
//     //    for (int iface = 0; iface < n_face; iface++) {
//     //        //if (iface < n_elem) xh = x[iface] - dx[iface] / 2.0;
//     //        //else xh = xend;
//     //        const double xh = x[iface];
//     //        s_eig(iface) = area[iface];
//     //        for (int ictl = 0; ictl < n_control_pts; ictl++)
//     //        {
//     //            A(iface, ictl) = getbij(xh, ictl, spline_degree, knots);
//     //        }
//     //    }
//     //    // Solve least square to fit bspline onto sine parametrization
//     //    ctl_pts = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(s_eig);
//     //    
//     //    std::vector<dreal> ctlpts(n_control_pts);
//     //    for (int ictl = 0; ictl < n_control_pts; ictl++)
//     //    {
//     //        ctlpts[ictl] = ctl_pts(ictl);
//     //    }
//     //    //ctlpts[0] = area[0];
//     //    //ctlpts[n_control_pts - 1] = area[n_elem];
// 
//     //    return ctlpts;
//     //}
// private:
//     std::vector<double> getKnots(
//         const unsigned int n_knots,
//         const unsigned int spline_degree,
//         const double grid_xstart,
//         const double grid_xend) const
//     {
//         std::vector<double> knots(n_knots);
//         const unsigned int nb_outer = 2 * (spline_degree + 1);
//         const unsigned int nb_inner = n_knots - nb_outer;
//         double eps = 2e-15; // Allow Spline Definition at End Point
//         // Clamped Open-Ended
//         for (unsigned int iknot = 0; iknot < spline_degree + 1; iknot++) {
//             knots[iknot] = grid_xstart;
//             knots[n_knots - iknot - 1] = grid_xend + eps;
//         }
//         // Uniform Knot Vector
//         double knot_dx = (grid_xend + eps - grid_xstart) / (nb_inner + 1);
//         for (unsigned int iknot = 1; iknot < nb_inner+1; iknot++) {
//             knots[iknot + spline_degree] = iknot * knot_dx;
//         }
//         return knots;
//     }
// };


/** Target boundary values.
 *  Simply zero out the default volume contribution.
 */
template <int dim, int nstate, typename real>
class BoundaryInverseTarget : public TargetFunctional<dim, nstate, real>
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real>::evaluate_volume_integrand;

public:
    /// Constructor
    BoundaryInverseTarget(
        std::shared_ptr<DGBase<dim,real>> dg_input,
  const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = true)
 : TargetFunctional<dim,nstate,real>(dg_input, target_solution, uses_solution_values, uses_solution_gradient)
 {}

    /// Zero out the default inverse target volume functional.
 template <typename real2>
 real2 evaluate_volume_integrand(
  const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/*physics*/,
  const dealii::Point<dim,real2> &/*phys_coord*/,
  const std::array<real2,nstate> &,//soln_at_q,
        const std::array<real,nstate> &,//target_soln_at_q,
  const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/,
  const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*target_soln_grad_at_q*/) const
 {
  real2 l2error = 0;
  
  return l2error;
 }

 /// non-template functions to override the template classes
 real evaluate_volume_integrand(
  const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
  const dealii::Point<dim,real> &phys_coord,
  const std::array<real,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
  const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q,
  const std::array<dealii::Tensor<1,dim,real>,nstate> &target_soln_grad_at_q) const override
 {
  return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, target_soln_at_q, soln_grad_at_q, target_soln_grad_at_q);
 }
 /// non-template functions to override the template classes
 FadFadType evaluate_volume_integrand(
  const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &physics,
  const dealii::Point<dim,FadFadType> &phys_coord,
  const std::array<FadFadType,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
  const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &target_soln_grad_at_q) const override
 {
  return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, target_soln_at_q, soln_grad_at_q, target_soln_grad_at_q);
 }
};

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
 const double amplitude = AMPLITUDE_OPT;
    const int poly_degree = 1;
    int fail_bool = false;
 pcout << " Running optimization case... " << std::endl;

 // *****************************************************************************
 // Create target mesh
 // *****************************************************************************
 //const unsigned int initial_n_cells = 10;
 const unsigned int initial_n_cells = 4;

#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<dim>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif

    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
        this->mpi_communicator,
#endif
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

 dealii::GridGenerator::subdivided_hyper_cube(*grid, initial_n_cells);
 for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
  // Set a dummy boundary ID
  cell->set_material_id(9002);
  for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
   if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
  }
 }

 // Create DG from which we'll modify the HighOrderGrid
 std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
    dg->allocate_system ();

 std::shared_ptr<HighOrderGrid<dim,double>> high_order_grid = dg->high_order_grid;
#if PHILIP_DIM!=1
 high_order_grid->prepare_for_coarsening_and_refinement();
 grid->repartition();
 high_order_grid->execute_coarsening_and_refinement();
 high_order_grid->output_results_vtk(high_order_grid->nth_refinement++);
#endif

 // *****************************************************************************
 // Prescribe surface displacements
 // *****************************************************************************
 std::vector<dealii::Tensor<1,dim,double>> point_displacements(high_order_grid->locally_relevant_surface_points.size());
 const unsigned int n_locally_relevant_surface_nodes = dim * high_order_grid->locally_relevant_surface_points.size();
 std::vector<dealii::types::global_dof_index> surface_node_global_indices(n_locally_relevant_surface_nodes);
 std::vector<double> surface_node_displacements(n_locally_relevant_surface_nodes);
 {
  auto displacement = point_displacements.begin();
  auto point = high_order_grid->locally_relevant_surface_points.begin();
  auto point_end = high_order_grid->locally_relevant_surface_points.end();
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
    const dealii::types::global_dof_index global_index = high_order_grid->point_and_axis_to_global_index[point_axis];
    surface_node_global_indices[inode] = global_index;
    surface_node_displacements[inode] = point_displacements[ipoint][d];
    inode++;
   }
  }
 }
 const auto initial_grid = high_order_grid->volume_nodes;
 // *****************************************************************************
 // Obtain target grid
 // *****************************************************************************
 using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
 std::function<dealii::Point<dim>(dealii::Point<dim>)> target_transformation = target_deformation<dim>;
 VectorType surface_node_displacements_vector = high_order_grid->transform_surface_nodes(target_transformation);
 surface_node_displacements_vector -= high_order_grid->surface_nodes;
 surface_node_displacements_vector.update_ghost_values();


    // const unsigned int n_design = 10;
    // const unsigned int spline_degree = 2;
    // const double y_start = 0.0, y_end = 0.0;
    // SplineY spline(n_design, spline_degree, y_start, y_end);
    // auto x_displacements = spline.evalSpline(y_locations);

 MeshMover::LinearElasticity<dim, double> meshmover(*high_order_grid, surface_node_displacements_vector);
 VectorType volume_displacements = meshmover.get_volume_displacements();

 high_order_grid->volume_nodes += volume_displacements;
 high_order_grid->volume_nodes.update_ghost_values();
    high_order_grid->update_surface_nodes();
 //{
 // std::function<dealii::Point<dim>(dealii::Point<dim>)> reverse_transformation = reverse_deformation<dim>;
 // surface_node_displacements_vector = high_order_grid->transform_surface_nodes(reverse_transformation);
 // surface_node_displacements_vector -= high_order_grid->surface_nodes;
 // surface_node_displacements_vector.update_ghost_values();
 // volume_displacements = meshmover.get_volume_displacements();
 //}
 //high_order_grid->volume_nodes += volume_displacements;
 //high_order_grid->volume_nodes.update_ghost_values();
    //high_order_grid->update_surface_nodes();

 // Get discrete solution on this target grid
 std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,double>> physics_double = PHiLiP::Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters);
 initialize_perturbed_solution(*dg, *physics_double);
 dg->output_results_vtk(999);
 high_order_grid->output_results_vtk(high_order_grid->nth_refinement++);
 std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
 ode_solver->steady_state();

 // Save target solution and volume_nodes
 const auto target_solution = dg->solution;
 const auto target_nodes    = high_order_grid->volume_nodes;

 pcout << "Target grid: " << std::endl;
 dg->output_results_vtk(9999);
 high_order_grid->output_results_vtk(9999);

 // *****************************************************************************
 // Get back our square mesh through mesh deformation
 // *****************************************************************************
 {
  auto displacement = point_displacements.begin();
  auto point = high_order_grid->locally_relevant_surface_points.begin();
  auto point_end = high_order_grid->locally_relevant_surface_points.end();
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
    const dealii::types::global_dof_index global_index = high_order_grid->point_and_axis_to_global_index[point_axis];
    surface_node_global_indices[inode] = global_index;
    surface_node_displacements[inode] = point_displacements[ipoint][d];
    inode++;
   }
  }
 }

 //{
 // std::function<dealii::Point<dim>(dealii::Point<dim>)> reverse_transformation = reverse_deformation<dim>;
 // surface_node_displacements_vector = high_order_grid->transform_surface_nodes(reverse_transformation);
 // surface_node_displacements_vector -= high_order_grid->surface_nodes;
 // surface_node_displacements_vector.update_ghost_values();
 // volume_displacements = meshmover.get_volume_displacements();
 //}
 //high_order_grid->volume_nodes += volume_displacements;
 //high_order_grid->volume_nodes.update_ghost_values();
    //high_order_grid->update_surface_nodes();
 
 high_order_grid->volume_nodes = initial_grid;
 high_order_grid->volume_nodes.update_ghost_values();
    high_order_grid->update_surface_nodes();
 pcout << "Initial grid: " << std::endl;
 dg->output_results_vtk(9998);
 high_order_grid->output_results_vtk(9998);

 // Solve on this new grid
 ode_solver->steady_state();

 // Compute current error
 auto error_vector = dg->solution;
 error_vector -= target_solution;
 const double l2_vector_error = error_vector.l2_norm();

 BoundaryInverseTarget<dim,nstate,double> inverse_target_functional(dg, target_solution, true, false);
 bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
    const double current_l2_error = inverse_target_functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
 pcout << "Vector l2_norm of the coefficients: " << l2_vector_error << std::endl;
 pcout << "Functional l2_norm : " << current_l2_error << std::endl;

 pcout << "*************************************************************" << std::endl;
 pcout << "Starting design... " << std::endl;
 pcout << "Number of state variables: " << dg->dof_handler.n_dofs() << std::endl;
 pcout << "Number of mesh volume_nodes: " << high_order_grid->dof_handler_grid.n_dofs() << std::endl;
 pcout << "Number of constraints: " << dg->dof_handler.n_dofs() << std::endl;

 const dealii::IndexSet surface_locally_owned_indexset = high_order_grid->surface_nodes.locally_owned_elements();
 dealii::TrilinosWrappers::SparseMatrix dRdXs;
 dealii::TrilinosWrappers::SparseMatrix d2LdWdXs;
 dealii::TrilinosWrappers::SparseMatrix d2LdXsdXs;
 // Analytical dXvdXs
 meshmover.evaluate_dXvdXs();

 VectorType dIdXs;
 dIdXs.reinit(dg->high_order_grid->surface_nodes);

 auto grad_lagrangian = dIdXs;

 // Assemble KKT rhs
 dealii::LinearAlgebra::distributed::BlockVector<double> kkt_rhs(3);
    pcout << "Evaluating KKT right-hand side: dIdW, dIdX, d2I, Residual..." << std::endl;
 compute_dIdW = true, compute_dIdX = true, compute_d2I = true;
 (void) inverse_target_functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
 bool compute_dRdW = false, compute_dRdX = false, compute_d2R = false;
 dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
 for (unsigned int isurf = 0; isurf < dg->high_order_grid->surface_nodes.size(); ++isurf) {
  const auto scalar_product = meshmover.dXvdXs[isurf] * inverse_target_functional.dIdX;
  if (dIdXs.locally_owned_elements().is_element(isurf)) {
   dIdXs[isurf] = scalar_product;
  }
 }
 dIdXs.update_ghost_values();

 kkt_rhs.block(0) = inverse_target_functional.dIdw;
 kkt_rhs.block(1) = dIdXs;
 kkt_rhs.block(2) = dg->right_hand_side;
 kkt_rhs *= -1.0;

 dealii::LinearAlgebra::distributed::BlockVector<double> kkt_soln(3), p4inv_kkt_rhs(3);
 kkt_soln.reinit(kkt_rhs);
 p4inv_kkt_rhs.reinit(kkt_rhs);

 auto mesh_error = target_nodes;
 mesh_error -= high_order_grid->volume_nodes;

 double current_functional = inverse_target_functional.current_functional_value;
 double current_kkt_norm = kkt_rhs.l2_norm();
 double current_constraint_satisfaction = kkt_rhs.block(2).l2_norm();
 double current_mesh_error = mesh_error.l2_norm();
 pcout << std::scientific << std::setprecision(5);
 pcout << "*************************************************************" << std::endl;
 pcout << "Initial design " << std::endl;
 pcout << "*************************************************************" << std::endl;
 pcout << "Current functional: " << current_functional << std::endl;
 pcout << "Constraint satisfaction: " << current_constraint_satisfaction << std::endl;
 pcout << "l2norm(Current mesh - optimal mesh): " << current_mesh_error << std::endl;
 pcout << "Current KKT norm: " << current_kkt_norm << std::endl;
 //pcout << std::fixed << std::setprecision(6);

 const unsigned int n_des_var = high_order_grid->surface_nodes.size();
 dealii::FullMatrix<double> Hessian_inverse = dealii::IdentityMatrix(n_des_var);
 const double diagonal_hessian = HESSIAN_DIAG;
 Hessian_inverse *= 1.0/diagonal_hessian;
 dealii::Vector<double> oldg(n_des_var), currentg(n_des_var), searchD(n_des_var);

 const bool use_BFGS = true;
 const unsigned int bfgs_max_history = n_des_var;
 const unsigned int n_max_design = 1000;
 const double kkt_tolerance = 1e-10;

 // Initialize Lagrange multipliers to zero
 dg->set_dual(kkt_soln.block(2));

    std::list<VectorType> bfgs_search_s;
    std::list<VectorType> bfgs_dgrad_y;
    std::list<double> bfgs_denom_rho;

 // Conventional design optimization
 {
  // Solve the flow
  ode_solver->steady_state();

  // Analytical dXvdXs
  meshmover.evaluate_dXvdXs();

  // Functional derivatives
  pcout << "Evaluating KKT right-hand side: dIdW, dIdX, Residual..." << std::endl;
  bool compute_dIdW = true, compute_dIdX = true, compute_d2I = false;
  (void) inverse_target_functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
  compute_dRdW = false, compute_dRdX = false, compute_d2R = false;
  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
  for (unsigned int isurf = 0; isurf < dg->high_order_grid->surface_nodes.size(); ++isurf) {

   const auto scalar_product = meshmover.dXvdXs[isurf] * inverse_target_functional.dIdX;
   if (dIdXs.locally_owned_elements().is_element(isurf)) {

    // Only do X-direction
    const unsigned int surface_index = high_order_grid->surface_to_volume_indices[isurf];
    const unsigned int component = high_order_grid->global_index_to_point_and_axis[surface_index].second;
    if (component != 0) dIdXs[isurf] = 0.0;
    else dIdXs[isurf] = scalar_product;
   }
  }
  dIdXs.update_ghost_values();

  // Residual derivatives
  pcout << "Evaluating dRdW..." << std::endl;
  compute_dRdW = true, compute_dRdX = false, compute_d2R = false;
  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
  pcout << "Evaluating dRdX..." << std::endl;
  compute_dRdW = false; compute_dRdX = true, compute_d2R = false;
  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

  {
   dRdXs.clear();
   dealii::SparsityPattern sparsity_pattern_dRdXs = dg->get_dRdXs_sparsity_pattern ();
   const dealii::IndexSet &row_parallel_partitioning_dRdXs = dg->locally_owned_dofs;
   const dealii::IndexSet &col_parallel_partitioning_dRdXs = surface_locally_owned_indexset;
   dRdXs.reinit(row_parallel_partitioning_dRdXs, col_parallel_partitioning_dRdXs, sparsity_pattern_dRdXs, mpi_communicator);
  }
  for (unsigned int isurf = 0; isurf < high_order_grid->surface_nodes.size(); ++isurf) {
   VectorType dRdXs_i(dg->solution);
   dg->dRdXv.vmult(dRdXs_i,meshmover.dXvdXs[isurf]);
   for (unsigned int irow = 0; irow < dg->dof_handler.n_dofs(); ++irow) {
    if (dg->locally_owned_dofs.is_element(irow)) {
     dRdXs.add(irow, isurf, dRdXs_i[irow]);
    }
   }
  }
        dRdXs.compress(dealii::VectorOperation::add);

  // Solve for the adjoint variable
  auto dRdW_T = transpose_trilinos_matrix(dg->system_matrix);

  Parameters::LinearSolverParam linear_solver_param = all_parameters->linear_solver_param;
  solve_linear (dRdW_T, inverse_target_functional.dIdw, dg->dual, linear_solver_param);

  grad_lagrangian = dIdXs;
  grad_lagrangian *= -1.0;
  dRdXs.Tvmult_add(grad_lagrangian, dg->dual);
  grad_lagrangian *= -1.0;
 }
 for (unsigned int i_design = 0; i_design < n_max_design && current_kkt_norm > kkt_tolerance; i_design++) {

  // Gradient descent
  auto search_direction = grad_lagrangian;
  if (use_BFGS && bfgs_search_s.size() > 0) {
   search_direction = LBFGS(grad_lagrangian, bfgs_search_s, bfgs_dgrad_y, bfgs_denom_rho);
   auto descent = search_direction * grad_lagrangian;
   if (descent < 0) {
    search_direction = grad_lagrangian;
    bfgs_search_s.pop_front();
    bfgs_dgrad_y.pop_front();
    bfgs_denom_rho.pop_front();
   }
  }
  search_direction *= -1.0;
  pcout << "Gradient magnitude = " << grad_lagrangian.l2_norm() << " Search direction magnitude = " << search_direction.l2_norm() << std::endl;
  for (unsigned int isurf = 0; isurf < dg->high_order_grid->surface_nodes.size(); ++isurf) {
   if (search_direction.locally_owned_elements().is_element(isurf)) {
    // Only do X-direction
    const unsigned int surface_index = high_order_grid->surface_to_volume_indices[isurf];
    const unsigned int component = high_order_grid->global_index_to_point_and_axis[surface_index].second;
    if (component != 0) search_direction[isurf] = 0.0;
   }
  }

  const auto old_solution = dg->solution;
  const auto old_volume_nodes = high_order_grid->volume_nodes;
  const auto old_surface_nodes = high_order_grid->surface_nodes;
  const auto old_functional = current_functional;

  // Line search
  double step_length = 1.0;
  const unsigned int max_linesearch = 80;
  unsigned int i_linesearch = 0;
  for (; i_linesearch < max_linesearch; i_linesearch++) {

   surface_node_displacements_vector = search_direction;
   surface_node_displacements_vector *= step_length;


   const auto disp_norm = surface_node_displacements_vector.l2_norm();
   if (disp_norm > 1e-2) {
    pcout << "Displacement of " << disp_norm << " too large. Reducing step length" << std::endl;

   } else {

    VectorType volume_displacements = meshmover.get_volume_displacements();
    high_order_grid->volume_nodes += volume_displacements;
    high_order_grid->volume_nodes.update_ghost_values();
    high_order_grid->update_surface_nodes();

    ode_solver->steady_state();

    current_functional = inverse_target_functional.evaluate_functional();

    pcout << "Linesearch " << i_linesearch+1
       << std::scientific << std::setprecision(15)
       << " step = " << step_length
       << " Old functional = " << old_functional
       << " New functional = " << current_functional << std::endl;
    pcout << std::fixed << std::setprecision(6);

    const double wolfe_c = 1e-4;
    double old_func_plus = search_direction*grad_lagrangian;
    old_func_plus *= wolfe_c * step_length;
    old_func_plus += old_functional;

    if (current_functional < old_func_plus) break;
   }

   dg->solution = old_solution;
   high_order_grid->volume_nodes = old_volume_nodes;
   high_order_grid->volume_nodes.update_ghost_values();
   high_order_grid->update_surface_nodes();
   step_length *= 0.5;
  }
  if (i_linesearch == max_linesearch) return 1;

  // Analytical dXvdXs
  meshmover.evaluate_dXvdXs();

  // Functional derivatives
  pcout << "Evaluating KKT right-hand side: dIdW, dIdX, Residual..." << std::endl;
  bool compute_dIdW = true, compute_dIdX = true, compute_d2I = false;
  (void) inverse_target_functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
  compute_dRdW = false, compute_dRdX = false, compute_d2R = false;
  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
  for (unsigned int isurf = 0; isurf < dg->high_order_grid->surface_nodes.size(); ++isurf) {
   const auto scalar_product = meshmover.dXvdXs[isurf] * inverse_target_functional.dIdX;
   if (dIdXs.locally_owned_elements().is_element(isurf)) {
    // Only do X-direction
    const unsigned int surface_index = high_order_grid->surface_to_volume_indices[isurf];
    const unsigned int component = high_order_grid->global_index_to_point_and_axis[surface_index].second;
    if (component != 0) dIdXs[isurf] = 0.0;
    else dIdXs[isurf] = scalar_product;
   }
  }
  dIdXs.update_ghost_values();

  // Residual derivatives
  pcout << "Evaluating dRdW..." << std::endl;
  compute_dRdW = true, compute_dRdX = false, compute_d2R = false;
  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
  pcout << "Evaluating dRdX..." << std::endl;
  compute_dRdW = false; compute_dRdX = true, compute_d2R = false;
  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

  {
   dRdXs.clear();
   dealii::SparsityPattern sparsity_pattern_dRdXs = dg->get_dRdXs_sparsity_pattern ();
   const dealii::IndexSet &row_parallel_partitioning_dRdXs = dg->locally_owned_dofs;
   const dealii::IndexSet &col_parallel_partitioning_dRdXs = surface_locally_owned_indexset;
   dRdXs.reinit(row_parallel_partitioning_dRdXs, col_parallel_partitioning_dRdXs, sparsity_pattern_dRdXs, mpi_communicator);
  }
  for (unsigned int isurf = 0; isurf < high_order_grid->surface_nodes.size(); ++isurf) {
   VectorType dRdXs_i(dg->solution);
   dg->dRdXv.vmult(dRdXs_i,meshmover.dXvdXs[isurf]);
   for (unsigned int irow = 0; irow < dg->dof_handler.n_dofs(); ++irow) {
    if (dg->locally_owned_dofs.is_element(irow)) {
     dRdXs.add(irow, isurf, dRdXs_i[irow]);
    }
   }
  }
        dRdXs.compress(dealii::VectorOperation::add);

  // Solve for the adjoint variable
  auto dRdW_T = transpose_trilinos_matrix(dg->system_matrix);

  Parameters::LinearSolverParam linear_solver_param = all_parameters->linear_solver_param;
  solve_linear (dRdW_T, inverse_target_functional.dIdw, dg->dual, linear_solver_param);

  const auto old_grad_lagrangian = grad_lagrangian;
  grad_lagrangian = dIdXs;
  grad_lagrangian *= -1.0;
  dRdXs.Tvmult_add(grad_lagrangian, dg->dual);
  grad_lagrangian *= -1.0;

  if (use_BFGS) {

   auto s = search_direction;
   s *= step_length;
   const auto y = grad_lagrangian - old_grad_lagrangian;
   const double curvature = s*y;
   if (curvature > 0 && bfgs_max_history > 0) {
    bfgs_denom_rho.push_front(1.0/curvature);
    bfgs_search_s.push_front(s);
    bfgs_dgrad_y.push_front(y);
   }

   if (bfgs_search_s.size() > bfgs_max_history) {
    bfgs_search_s.pop_back();
    bfgs_dgrad_y.pop_back();
    bfgs_denom_rho.pop_back();
   }
  }


  dg->output_results_vtk(high_order_grid->nth_refinement);
  high_order_grid->output_results_vtk(high_order_grid->nth_refinement++);

  mesh_error = target_nodes;
  mesh_error -= high_order_grid->volume_nodes;
  current_kkt_norm = grad_lagrangian.l2_norm();
  current_constraint_satisfaction = dg->right_hand_side.l2_norm();
  current_mesh_error = mesh_error.l2_norm();
  pcout << std::scientific << std::setprecision(5);
  pcout << "*************************************************************" << std::endl;
  pcout << "Design iteration " << i_design+1 << std::endl;
  pcout << "Current functional: " << current_functional << std::endl;
  pcout << "Constraint satisfaction: " << current_constraint_satisfaction << std::endl;
  pcout << "l2norm(Current mesh - optimal mesh): " << current_mesh_error << std::endl;
  pcout << "Current KKT norm: " << current_kkt_norm << std::endl;
  //pcout << std::fixed << std::setprecision(6);


 }
 // for (unsigned int i_design = 0; i_design < n_max_design && current_kkt_norm > kkt_tolerance; i_design++) {

 //  // Evaluate KKT right-hand side
 //  dealii::LinearAlgebra::distributed::BlockVector<double> kkt_rhs(3);
 //  pcout << "Evaluating KKT right-hand side: dIdW, dIdX, d2I, Residual..." << std::endl;
 //  bool compute_dIdW = true, compute_dIdX = true, compute_d2I = true;
 //  (void) inverse_target_functional.evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
 //  compute_dRdW = false, compute_dRdX = false, compute_d2R = false;
 //  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
 //  for (unsigned int isurf = 0; isurf < dg->high_order_grid->surface_nodes.size(); ++isurf) {
 //   const auto scalar_product = meshmover.dXvdXs[isurf] * inverse_target_functional.dIdX;
 //   if (dIdXs.locally_owned_elements().is_element(isurf)) {
 //    dIdXs[isurf] = scalar_product;
 //   }
 //  }
 //  dIdXs.update_ghost_values();

 //  kkt_rhs.block(0) = inverse_target_functional.dIdw;
 //  kkt_rhs.block(1) = dIdXs;
 //  kkt_rhs.block(2) = dg->right_hand_side;
 //  kkt_rhs *= -1.0;

 //  // Evaluate KKT system matrix
 //  pcout << "Evaluating dRdW..." << std::endl;
 //  compute_dRdW = true, compute_dRdX = false, compute_d2R = false;
 //  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
 //  pcout << "Evaluating dRdX..." << std::endl;
 //  compute_dRdW = false; compute_dRdX = true, compute_d2R = false;
 //  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
 //  pcout << "Evaluating residual 2nd order partials..." << std::endl;
 //  compute_dRdW = false; compute_dRdX = false, compute_d2R = true;
 //  dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

    //     std::vector<dealii::types::global_dof_index> surface_node_global_indices(dim*high_order_grid->locally_relevant_surface_points.size());
    //     std::vector<double> surface_node_displacements(dim*high_order_grid->locally_relevant_surface_points.size());
    //     {
    //         int inode = 0;
    //         for (unsigned int ipoint=0; ipoint<point_displacements.size(); ++ipoint) {
    //             for (unsigned int d=0;d<dim;++d) {
    //                 const std::pair<unsigned int, unsigned int> point_axis = std::make_pair(ipoint,d);
    //                 const dealii::types::global_dof_index global_index = high_order_grid->point_and_axis_to_global_index[point_axis];
    //                 surface_node_global_indices[inode] = global_index;
    //                 surface_node_displacements[inode] = point_displacements[ipoint][d];
    //                 inode++;
    //             }
    //         }
    //     }

 //  // Form Lagrangian Hessian
 //  inverse_target_functional.d2IdWdW.add(1.0,dg->d2RdWdW);
 //  inverse_target_functional.d2IdWdX.add(1.0,dg->d2RdWdX);
 //  inverse_target_functional.d2IdXdX.add(1.0,dg->d2RdXdX);

 //  // Analytical dXvdXs
 //  meshmover.evaluate_dXvdXs();

 //  {
 //   dRdXs.clear();
 //   dealii::SparsityPattern sparsity_pattern_dRdXs = dg->get_dRdXs_sparsity_pattern ();
 //   const dealii::IndexSet &row_parallel_partitioning_dRdXs = dg->locally_owned_dofs;
 //   const dealii::IndexSet &col_parallel_partitioning_dRdXs = surface_locally_owned_indexset;
 //   dRdXs.reinit(row_parallel_partitioning_dRdXs, col_parallel_partitioning_dRdXs, sparsity_pattern_dRdXs, mpi_communicator);
 //  }
 //  {
 //   d2LdWdXs.clear();
 //   dealii::SparsityPattern sparsity_pattern_d2LdWdXs = dg->get_d2RdWdXs_sparsity_pattern ();
 //   const dealii::IndexSet &row_parallel_partitioning_d2LdWdXs = dg->locally_owned_dofs;
 //   const dealii::IndexSet &col_parallel_partitioning_d2LdWdXs = surface_locally_owned_indexset;
 //   d2LdWdXs.reinit(row_parallel_partitioning_d2LdWdXs, col_parallel_partitioning_d2LdWdXs, sparsity_pattern_d2LdWdXs, mpi_communicator);
 //  }
 //  {
 //   d2LdXsdXs.clear();
 //   dealii::SparsityPattern sparsity_pattern_d2LdXsdXs = dg->get_d2RdXsdXs_sparsity_pattern ();
 //   const dealii::IndexSet &row_parallel_partitioning_d2LdXsdXs = surface_locally_owned_indexset;
 //   d2LdXsdXs.reinit(row_parallel_partitioning_d2LdXsdXs, sparsity_pattern_d2LdXsdXs, mpi_communicator);
 //  }
 //  for (unsigned int isurf = 0; isurf < high_order_grid->surface_nodes.size(); ++isurf) {
 //   VectorType dLdWdXs_i(dg->solution);
 //   inverse_target_functional.d2IdWdX.vmult(dLdWdXs_i,meshmover.dXvdXs[isurf]);
 //   for (unsigned int irow = 0; irow < dg->dof_handler.n_dofs(); ++irow) {
 //    if (dg->locally_owned_dofs.is_element(irow)) {
 //     d2LdWdXs.add(irow, isurf, dLdWdXs_i[irow]);
 //    }
 //   }

 //   VectorType dRdXs_i(dg->solution);
 //   dg->dRdXv.vmult(dRdXs_i,meshmover.dXvdXs[isurf]);
 //   for (unsigned int irow = 0; irow < dg->dof_handler.n_dofs(); ++irow) {
 //    if (dg->locally_owned_dofs.is_element(irow)) {
 //     dRdXs.add(irow, isurf, dRdXs_i[irow]);
 //    }
 //   }


 //   const dealii::IndexSet volume_locally_owned_indexset = high_order_grid->volume_nodes.locally_owned_elements();
 //   dealii::TrilinosWrappers::MPI::Vector dXvdXs_i(volume_locally_owned_indexset);
 //   dealii::LinearAlgebra::ReadWriteVector<double> dXvdXs_i_rwv(volume_locally_owned_indexset);
 //   dXvdXs_i_rwv.import(meshmover.dXvdXs[isurf], dealii::VectorOperation::insert);
 //   dXvdXs_i.import(dXvdXs_i_rwv, dealii::VectorOperation::insert);
 //   for (unsigned int jsurf = isurf; jsurf < high_order_grid->surface_nodes.size(); ++jsurf) {

 //    dealii::TrilinosWrappers::MPI::Vector dXvdXs_j(volume_locally_owned_indexset);
 //    dealii::LinearAlgebra::ReadWriteVector<double> dXvdXs_j_rwv(volume_locally_owned_indexset);
 //    dXvdXs_j_rwv.import(meshmover.dXvdXs[jsurf], dealii::VectorOperation::insert);
 //    dXvdXs_j.import(dXvdXs_j_rwv, dealii::VectorOperation::insert);

 //    const auto d2LdXsidXsj = inverse_target_functional.d2IdXdX.matrix_scalar_product(dXvdXs_i,dXvdXs_j);
 //    if (volume_locally_owned_indexset.is_element(isurf)) {
 //     d2LdXsdXs.add(isurf,jsurf,d2LdXsidXsj);
 //    }
 //    if (volume_locally_owned_indexset.is_element(jsurf)) {
 //     d2LdXsdXs.add(jsurf,isurf,d2LdXsidXsj);
 //    }
 //   }
 //  }
    //     d2LdWdXs.compress(dealii::VectorOperation::add);
    //     dRdXs.compress(dealii::VectorOperation::add);
    //     d2LdXsdXs.compress(dealii::VectorOperation::add);


 //  // Build required operators
 //  dealii::TrilinosWrappers::SparsityPattern zero_sparsity_pattern(dg->locally_owned_dofs, MPI_COMM_WORLD, 0);
 //  zero_sparsity_pattern.compress();
 //  dealii::TrilinosWrappers::BlockSparseMatrix kkt_system_matrix;
 //  kkt_system_matrix.reinit(3,3);
 //  kkt_system_matrix.block(0, 0).copy_from( inverse_target_functional.d2IdWdW);
 //  kkt_system_matrix.block(0, 1).copy_from( d2LdWdXs);
 //  kkt_system_matrix.block(0, 2).copy_from( transpose_trilinos_matrix(dg->system_matrix));

 //  kkt_system_matrix.block(1, 0).copy_from( transpose_trilinos_matrix(d2LdWdXs));
 //  kkt_system_matrix.block(1, 1).copy_from( d2LdXsdXs);
 //  kkt_system_matrix.block(1, 2).copy_from( transpose_trilinos_matrix(dRdXs));

 //  kkt_system_matrix.block(2, 0).copy_from( dg->system_matrix);
 //  kkt_system_matrix.block(2, 1).copy_from( dRdXs);
 //  kkt_system_matrix.block(2, 2).reinit(zero_sparsity_pattern);
 //  //kkt_system_matrix.block(0, 0).copy_from( inverse_target_functional.d2IdWdW);
 //  //kkt_system_matrix.block(0, 1).copy_from( inverse_target_functional.d2IdWdX);
 //  //kkt_system_matrix.block(0, 2).copy_from( transpose_trilinos_matrix(dg->system_matrix));

 //  //kkt_system_matrix.block(1, 0).copy_from( transpose_trilinos_matrix(inverse_target_functional.d2IdWdX));
 //  //kkt_system_matrix.block(1, 1).copy_from( inverse_target_functional.d2IdXdX);
 //  //kkt_system_matrix.block(1, 2).copy_from( transpose_trilinos_matrix(dg->dRdXv));

 //  //kkt_system_matrix.block(2, 0).copy_from( dg->system_matrix);
 //  //kkt_system_matrix.block(2, 1).copy_from( dg->dRdXv);
 //  //kkt_system_matrix.block(2, 2).reinit(zero_sparsity_pattern);

 //  kkt_system_matrix.collect_sizes();

 //  Parameters::LinearSolverParam linear_solver_param = all_parameters->linear_solver_param;
 //  linear_solver_param.linear_residual = 1e-12;
 //  pcout << "Applying P4" << std::endl;
 //  apply_P4 (
 //   kkt_system_matrix,  // A
 //   kkt_rhs,            // b
 //   p4inv_kkt_rhs,      // x
 //   linear_solver_param);

 //  kkt_soln = p4inv_kkt_rhs;
 //  const auto old_solution = dg->solution;
 //  const auto old_volume_nodes = high_order_grid->volume_nodes;
 //  const auto old_surface_nodes = high_order_grid->surface_nodes;
 //  const auto old_dual = dg->dual;
 //  auto dual_step = kkt_soln.block(2);
 //  dual_step -= old_dual;

 //  const auto old_functional = current_functional;

 //  // Line search
 //  double step_length = 1.0;
 //  const unsigned int max_linesearch = 40;
 //  unsigned int i_linesearch = 0;
 //  pcout << "l2norm of DW no linesearch " << kkt_soln.block(0).l2_norm() << std::endl;
 //  pcout << "l2norm of DX no linesearch " << kkt_soln.block(1).l2_norm() << std::endl;
 //  pcout << "l2norm of DL no linesearch " << dual_step.l2_norm() << std::endl;
 //  //if (i_design > 6) {
 //  // dg->solution.add(step_length, kkt_soln.block(0));
 //  //} else
 //  for (; i_linesearch < max_linesearch; i_linesearch++) {

 //   dg->solution.add(step_length, kkt_soln.block(0));
 //   //high_order_grid->surface_nodes.add(step_length, kkt_soln.block(1));

 //   surface_node_displacements_vector = kkt_soln.block(1);
 //   surface_node_displacements_vector *= step_length;
 //   VectorType volume_displacements = meshmover.get_volume_displacements();
 //   high_order_grid->volume_nodes += volume_displacements;
 //   high_order_grid->volume_nodes.update_ghost_values();
 //   high_order_grid->update_surface_nodes();

 //   current_functional = inverse_target_functional.evaluate_functional();

 //   pcout << "Linesearch " << i_linesearch+1
 //      << std::scientific << std::setprecision(15)
 //      << " step = " << step_length
 //      << " Old functional = " << old_functional
 //      << " New functional = " << current_functional << std::endl;
 //   pcout << std::fixed << std::setprecision(6);
 //   if (current_functional < old_functional) {
 //    break;
 //   } else {
 //    dg->solution = old_solution;
 //    high_order_grid->volume_nodes = old_volume_nodes;
 //    high_order_grid->volume_nodes.update_ghost_values();
 //    high_order_grid->update_surface_nodes();
 //    step_length *= 0.5;
 //   }
 //  }
 //  dg->dual.add(step_length, dual_step);
 //  if (i_linesearch == max_linesearch) return 1;

 //  high_order_grid->volume_nodes.update_ghost_values();
 //  high_order_grid->update_surface_nodes();


 //  dg->output_results_vtk(high_order_grid->nth_refinement);
 //  high_order_grid->output_results_vtk(high_order_grid->nth_refinement++);

 //  mesh_error = target_nodes;
 //  mesh_error -= high_order_grid->volume_nodes;
 //  current_kkt_norm = kkt_rhs.l2_norm();
 //  current_constraint_satisfaction = dg->right_hand_side.l2_norm();
 //  current_mesh_error = mesh_error.l2_norm();
 //  pcout << std::scientific << std::setprecision(5);
 //  pcout << "*************************************************************" << std::endl;
 //  pcout << "Design iteration " << i_design+1 << std::endl;
 //  pcout << "Current functional: " << current_functional << std::endl;
 //  pcout << "Constraint satisfaction: " << current_constraint_satisfaction << std::endl;
 //  pcout << "l2norm(Current mesh - optimal mesh): " << current_mesh_error << std::endl;
 //  pcout << "Current KKT norm: " << current_kkt_norm << std::endl;
 //  pcout << std::fixed << std::setprecision(6);


 // }

 pcout << std::endl << std::endl << std::endl << std::endl;
 // Make sure that if the volume_nodes are located at the target volume_nodes, then we recover our target functional
 high_order_grid->volume_nodes = target_nodes;
 high_order_grid->volume_nodes.update_ghost_values();
    high_order_grid->update_surface_nodes();
 // Solve on this new grid
 ode_solver->steady_state();
    const double zero_l2_error = inverse_target_functional.evaluate_functional();
 pcout << "Nodes at target volume_nodes should have zero functional l2 error : " << zero_l2_error << std::endl;
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



