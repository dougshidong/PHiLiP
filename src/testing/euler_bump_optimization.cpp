#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "euler_bump_optimization.h"

#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "functional/target_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"
#include "mesh/meshmover_linear_elasticity.hpp"

#include "linear_solver/linear_solver.h"

#include "parameters/all_parameters.h"

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_Objective_SimOpt.hpp"
#include "ROL_Objective.hpp"
#include "ROL_StatusTest.hpp"
#include <deal.II/optimization/rol/vector_adaptor.h>
#include "Teuchos_GlobalMPISession.hpp"

#include <Epetra_RowMatrixTransposer.h>

#include "optimization/flow_constraints.hpp"
namespace PHiLiP {
namespace Tests {

using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
using ROL_Vector = ROL::Vector<double>;
using AdaptVector = dealii::Rol::VectorAdaptor<VectorType>;

Teuchos::RCP<const VectorType> get_rcp_to_VectorType(const ROL_Vector &x)
{
    return (Teuchos::dyn_cast<const AdaptVector>(x)).getVector();
}

Teuchos::RCP<VectorType> get_rcp_to_VectorType(ROL_Vector &x)
{
    return (Teuchos::dyn_cast<AdaptVector>(x)).getVector();
}

const VectorType & get_ROLvec_to_VectorType(const ROL_Vector &x)
{
    return *(Teuchos::dyn_cast<const AdaptVector>(x)).getVector();
}

VectorType &get_ROLvec_to_VectorType(ROL_Vector &x)
{
    return *(Teuchos::dyn_cast<AdaptVector>(x)).getVector();
}


/** Target boundary values.
 *  Simply zero out the default volume contribution.
 */
template <int dim, int nstate, typename real>
class BoundaryInverseTarget1 : public TargetFunctional<dim, nstate, real>
{
    using ADType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using ADADType = Sacado::Fad::DFad<ADType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real>::evaluate_volume_integrand;

public:
    /// Constructor
    BoundaryInverseTarget1(
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
	ADADType evaluate_volume_integrand(
		const PHiLiP::Physics::PhysicsBase<dim,nstate,ADADType> &physics,
		const dealii::Point<dim,ADADType> &phys_coord,
		const std::array<ADADType,nstate> &soln_at_q,
        const std::array<real,nstate> &target_soln_at_q,
		const std::array<dealii::Tensor<1,dim,ADADType>,nstate> &soln_grad_at_q,
        const std::array<dealii::Tensor<1,dim,ADADType>,nstate> &target_soln_grad_at_q) const override
	{
		return evaluate_volume_integrand<>(physics, phys_coord, soln_at_q, target_soln_at_q, soln_grad_at_q, target_soln_grad_at_q);
	}
};

/// Interface between the ROL::Objective_SimOpt PHiLiP::Functional.
/** Uses FFD to parametrize the geometry.
 *  An update on the simulation variables updates the DGBase object within the Functional
 *  and an update on the control variables updates the FreeFormDeformation object, which in
 *  turn, updates the DGBase.HighOrderGrid.volume_nodes.
 */
template <int dim, int nstate>
class InverseObjective : public ROL::Objective_SimOpt<double> {
private:
    /// Functional to be evaluated
    Functional<dim,nstate,double> &functional;

    /// Free-form deformation used to parametrize the geometry.
    FreeFormDeformation<dim> ffd;

    /// List of FFD design variables and axes.
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;

    /// Design variables.
    dealii::LinearAlgebra::distributed::Vector<double> ffd_des_var;

public:

  /// Constructor.
  InverseObjective( Functional<dim,nstate,double> &_functional, 
                    const FreeFormDeformation<dim> &_ffd,
                    std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim)
      : functional(_functional)
      , ffd(_ffd)
      , ffd_design_variables_indices_dim(_ffd_design_variables_indices_dim)
  {
      ffd_des_var.reinit(ffd_design_variables_indices_dim.size());
      ffd.get_design_variables(ffd_design_variables_indices_dim, ffd_des_var);
  }

  using ROL::Objective_SimOpt<double>::value;
  using ROL::Objective_SimOpt<double>::update;

  /// Update the simulation and control variables.
  void update( const ROL_Vector &des_var_sim, const ROL_Vector &des_var_ctl, bool flag = true, int iter = -1) override
  {
      (void) flag; (void) iter;
      functional.set_state(get_ROLvec_to_VectorType(des_var_sim));

      ffd_des_var =  get_ROLvec_to_VectorType(des_var_ctl);
      ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);
      ffd.deform_mesh(functional.dg->high_order_grid);
  }


  /// Returns the value of the Functional object.
  double value( const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {
      update(des_var_sim, des_var_ctl);

      const bool compute_dIdW = false;
      const bool compute_dIdX = false;
      const bool compute_d2I = false;
      return functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
  }

  /// Returns the gradient w.\ r.\ t.\ the simulation variables of the Functional object.
  void gradient_1( ROL_Vector& g, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {
      update(des_var_sim, des_var_ctl);

      const bool compute_dIdW = true;
      const bool compute_dIdX = false;
      const bool compute_d2I = false;
      functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
      auto &dIdW = get_ROLvec_to_VectorType(g);
      dIdW = functional.dIdw;
  }

  /// Returns the gradient w.\ r.\ t.\ the control variables of the Functional object.
  void gradient_2( ROL_Vector& g, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {
      update(des_var_sim, des_var_ctl);

      const bool compute_dIdW = false;
      const bool compute_dIdX = true;
      const bool compute_d2I = false;
      functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

      auto &dIdXv = functional.dIdX;

      dealii::TrilinosWrappers::SparseMatrix dXvdXp;
      ffd.get_dXvdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

      auto &dIdXp = get_ROLvec_to_VectorType(g);
      dXvdXp.Tvmult(dIdXp, dIdXv);

  }

  /// Applies the functional Hessian w.\ r.\ t.\ the simulation variables onto a vector.
  /** More specifically, 
   *  \f[
   *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial w \partial w} \right)^T \mathbf{v}_{input}
   *  \f]
   */
  void hessVec_11( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
      update(des_var_sim, des_var_ctl);

      const bool compute_dIdW = false;
      const bool compute_dIdX = false;
      const bool compute_d2I = true;
      functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

      const auto &v = get_ROLvec_to_VectorType(input_vector);
      auto &hv = get_ROLvec_to_VectorType(output_vector);

      functional.d2IdWdW.vmult(hv, v);
  }

  /// Applies the functional Hessian w.\ r.\ t.\ the simulation and control variables onto a vector.
  /** More specifically, 
   *  \f[
   *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial w \partial x} \right)^T \mathbf{v}_{input}
   *  \f]
   */
  void hessVec_12( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
      update(des_var_sim, des_var_ctl);

      const bool compute_dIdW = false;
      const bool compute_dIdX = false;
      const bool compute_d2I = true;
      functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

      const auto &v = get_ROLvec_to_VectorType(input_vector);
      auto &hv = get_ROLvec_to_VectorType(output_vector);

      auto dXvdXp_input = functional.dg->high_order_grid.volume_nodes;

      dealii::TrilinosWrappers::SparseMatrix dXvdXp;
      ffd.get_dXvdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);
      dXvdXp.vmult(dXvdXp_input, v);
      functional.d2IdWdX.vmult(hv, dXvdXp_input);
  }

  /// Applies the functional Hessian w.\ r.\ t.\ the control and simulation variables onto a vector.
  /** More specifically, 
   *  \f[
   *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial x \partial w} \right)^T \mathbf{v}_{input}
   *  \f]
   */
  void hessVec_21( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
      update(des_var_sim, des_var_ctl);

      const bool compute_dIdW = false;
      const bool compute_dIdX = false;
      const bool compute_d2I = true;
      functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

      const auto &v = get_ROLvec_to_VectorType(input_vector);
      auto &hv = get_ROLvec_to_VectorType(output_vector);

      auto d2IdXdW_input = functional.dg->high_order_grid.volume_nodes;
      functional.d2IdWdX.Tvmult(d2IdXdW_input, v);

      dealii::TrilinosWrappers::SparseMatrix dXvdXp;
      ffd.get_dXvdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);
      dXvdXp.Tvmult(hv, d2IdXdW_input);
  }

  /// Applies the functional Hessian w.\ r.\ t.\ the control variables onto a vector.
  /** More specifically, 
   *  \f[
   *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial x \partial x} \right)^T \mathbf{v}_{input}
   *  \f]
   */
  void hessVec_21( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
  void hessVec_22( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
      update(des_var_sim, des_var_ctl);

      const bool compute_dIdW = false;
      const bool compute_dIdX = false;
      const bool compute_d2I = true;
      functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

      const auto &v = get_ROLvec_to_VectorType(input_vector);
      auto &hv = get_ROLvec_to_VectorType(output_vector);

      dealii::TrilinosWrappers::SparseMatrix dXvdXp;
      ffd.get_dXvdXp (functional.dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

      auto dXvdXp_input = functional.dg->high_order_grid.volume_nodes;
      dXvdXp.vmult(dXvdXp_input, v);

      auto d2IdXdX_dXvdXp_input = functional.dg->high_order_grid.volume_nodes;
      functional.d2IdXdX.vmult(d2IdXdX_dXvdXp_input, dXvdXp_input);

      dXvdXp.Tvmult(hv, d2IdXdX_dXvdXp_input);
  }
}; // InverseObjective


template <int dim, int nstate>
EulerBumpOptimization<dim,nstate>::EulerBumpOptimization(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
int EulerBumpOptimization<dim,nstate>
::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;


    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }

    int poly_degree = 1;

    //const int n_1d_cells = manu_grid_conv_param.initial_grid_size;

    std::vector<unsigned int> n_subdivisions(dim);
    //n_subdivisions[1] = n_1d_cells; // y-direction
    //n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction
    n_subdivisions[1] = 5; //20;// y-direction
    //n_subdivisions[1] = n_1d_cells; // y-direction
    n_subdivisions[0] = 9*n_subdivisions[1]; // x-direction
    dealii::parallel::distributed::Triangulation<dim> grid(this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    // VectorType target_solution;
    // double bump_height = 0.0625;
    // {
    //     const double channel_length = 3.0;
    //     const double channel_height = 0.8;
    //     Grids::gaussian_bump(grid, n_subdivisions, channel_length, channel_height, bump_height);
    //     // Create DG object
    //     std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);

    //     // Initialize coarse grid solution with free-stream
    //     dg->allocate_system ();
    //     dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    //     // Create ODE solver and ramp up the solution from p0
    //     std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    //     ode_solver->initialize_steady_polynomial_ramping (poly_degree);
    //     // Solve the steady state problem
    //     ode_solver->steady_state();
    //     // Output target solution
    //     dg->output_results_vtk(9998);

    //     target_solution = dg->solution;
    // }
    // grid.clear();
    // const double channel_length = 3.0;
    // const double channel_height = 0.8;
    // bump_height *= 0.5;
    // Grids::gaussian_bump(grid, n_subdivisions, channel_length, channel_height, bump_height);
    // // Create DG object
    // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);

    // // Initialize coarse grid solution with free-stream
    // dg->allocate_system ();
    // dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // // Create ODE solver and ramp up the solution from p0
    // std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    // ode_solver->initialize_steady_polynomial_ramping (poly_degree);
    // // Solve the steady state problem
    // ode_solver->steady_state();
    // // Output initial solution
    // dg->output_results_vtk(9999);

    const dealii::Point<dim> ffd_origin(-1.4,-0.1);
    const std::array<double,dim> ffd_rectangle_lengths = {2.8,0.6};
    const std::array<unsigned int,dim> ffd_ndim_control_pts = {10,2};
    FreeFormDeformation<dim> ffd( ffd_origin, ffd_rectangle_lengths, ffd_ndim_control_pts);

    unsigned int n_design_variables = 0;
    // Vector of ijk indices and dimension.
    // Each entry in the vector points to a design variable's ijk ctl point and its acting dimension.
    std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {

        const std::array<unsigned int,dim> ijk = ffd.global_to_grid ( i_ctl );
        for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {

            if (   ijk[0] == 0 // Constrain first column of FFD points.
                || ijk[0] == ffd_ndim_control_pts[0] - 1  // Constrain last column of FFD points.
                || d_ffd == 0 // Constrain x-direction of FFD points.
               ) {
                continue;
            }
            ++n_design_variables;
            ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
        }
    }

    //const std::vector<dealii::IndexSet> row_parts = dealii::Utilities::MPI::create_evenly_distributed_partitioning(this->mpi_communicator, n_design_variables);
    //const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator);
    //const dealii::IndexSet &row_part = row_parts[this_mpi_process];
    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_design_variables);

    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);

    VectorType ffd_design_variables(row_part,ghost_row_part,MPI_COMM_WORLD);

    ffd_design_variables.print(std::cout);


    ffd.get_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    const auto initial_design_variables = ffd_design_variables;
    ffd_design_variables[0] = -1.558e-01;
    ffd_design_variables[1] = -2.189e-01;
    ffd_design_variables[2] = -2.338e-01;
    ffd_design_variables[3] = -1.691e-01;
    ffd_design_variables[4] = -1.806e-01;
    ffd_design_variables[5] = -2.294e-01;
    ffd_design_variables[6] = -2.243e-01;
    ffd_design_variables[7] = -1.552e-01;
    ffd_design_variables[8] = 7.737e-01;
    ffd_design_variables[9] = 1.110e+00;
    ffd_design_variables[10] = 1.141e+00;
    ffd_design_variables[11] = 8.757e-01;
    ffd_design_variables[12] = 8.825e-01;
    ffd_design_variables[13] = 1.159e+00;
    ffd_design_variables[14] = 1.119e+00;
    ffd_design_variables[15] = 7.769e-01;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Create Target solution
    VectorType target_solution;
    const double bump_height = 0.0625;
    const double channel_length = 3.0;
    const double channel_height = 0.8;
    {
        grid.clear();
        Grids::gaussian_bump(grid, n_subdivisions, channel_length, channel_height, bump_height);
        // Create DG object
        std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);

        ffd.deform_mesh(dg->high_order_grid);

        // Initialize coarse grid solution with free-stream
        dg->allocate_system ();
        dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
        // Create ODE solver and ramp up the solution from p0
        std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver->initialize_steady_polynomial_ramping (poly_degree);
        // Solve the steady state problem
        ode_solver->steady_state();
        // Output target solution
        dg->output_results_vtk(9998);

        target_solution = dg->solution;
    }

    // Initial optimization point
    grid.clear();
    Grids::gaussian_bump(grid, n_subdivisions, channel_length, channel_height, bump_height);

    ffd_design_variables = initial_design_variables;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);

    // Initialize coarse grid solution with free-stream
    dg->allocate_system ();
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->initialize_steady_polynomial_ramping (poly_degree);
    // Solve the steady state problem
    ode_solver->steady_state();
    // Output initial solution
    dg->output_results_vtk(9999);

    BoundaryInverseTarget1<dim,nstate,double> functional(dg, target_solution, true, true);

    const bool has_ownership = false;
    auto des_var_sim = dg->solution;
    auto des_var_ctl = ffd_design_variables;
    auto des_var_adj = dg->dual;
    Teuchos::RCP<VectorType> des_var_sim_rcp = Teuchos::rcp(&des_var_sim, has_ownership);
    Teuchos::RCP<VectorType> des_var_ctl_rcp = Teuchos::rcp(&des_var_ctl, has_ownership);
    Teuchos::RCP<VectorType> des_var_adj_rcp = Teuchos::rcp(&des_var_adj, has_ownership);
    dg->set_dual(dg->solution);

    VectorType gradient_nodes(dg->high_order_grid.volume_nodes);
    {
        const bool compute_dIdW = true;
        const bool compute_dIdX = true;
        const bool compute_d2I = false;
        (void) functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        gradient_nodes = functional.dIdX;
        gradient_nodes.update_ghost_values();

        VectorType dIdW(functional.dIdw);

        {
            const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
            dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
        }
        {
            const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
            dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
        }

        dIdW *= -1.0;

        dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;
        Epetra_CrsMatrix *system_matrix_transpose_tril;

        Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&dg->system_matrix.trilinos_matrix()));
        epmt.CreateTranspose(false, system_matrix_transpose_tril);
        system_matrix_transpose.reinit(*system_matrix_transpose_tril);

        VectorType adjoint(functional.dIdw);
        Parameters::LinearSolverParam linear_solver_param;
        linear_solver_param.max_iterations = 1000;
        linear_solver_param.restart_number = 100;
        linear_solver_param.linear_residual = 1e-13;
        linear_solver_param.ilut_fill = 1;
        linear_solver_param.ilut_drop = 0.0;
        linear_solver_param.ilut_rtol = 1.0;
        linear_solver_param.ilut_atol = 0.0;
        linear_solver_param.linear_solver_output = Parameters::OutputEnum::quiet;
        linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;
        solve_linear(system_matrix_transpose, dIdW, adjoint, linear_solver_param);

        //dg->dRdXv.transpose();
        dg->dRdXv.Tvmult_add(gradient_nodes, adjoint);
        //gradient_nodes.print(std::cout);
    }

    AdaptVector des_var_sim_rol(des_var_sim_rcp);
    AdaptVector des_var_ctl_rol(des_var_ctl_rcp);
    AdaptVector des_var_adj_rol(des_var_adj_rcp);

    auto des_var_sim_rol_p = ROL::makePtr<AdaptVector>(des_var_sim_rol);
    auto des_var_ctl_rol_p = ROL::makePtr<AdaptVector>(des_var_ctl_rol);
    auto des_var_adj_rol_p = ROL::makePtr<AdaptVector>(des_var_adj_rol);


    // Output stream
    ROL::nullstream bhs; // outputs nothing
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization.log",std::ios::out);
    std::ostream ostr(&filebuffer);

    Teuchos::RCP<std::ostream> outStream;
    if (this->mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (this->mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    auto obj  = ROL::makePtr<InverseObjective<dim,nstate>>( functional, ffd, ffd_design_variables_indices_dim );
    auto con  = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);
    const bool storage = false;
    const bool useFDHessian = false;
    auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( obj, con, des_var_sim_rol_p, des_var_ctl_rol_p, des_var_adj_rol_p, storage, useFDHessian);
    //const bool full_space = true;
    const bool full_space = false;
    ROL::OptimizationProblem<double> opt;
    // Set parameters.
    Teuchos::ParameterList parlist;
    if (full_space) {
        // Full space problem
        auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(des_var_sim_rol_p, des_var_ctl_rol_p);
        auto dual_sim_p = des_var_sim_rol_p->clone();
        //auto dual_sim_p = ROL::makePtrFromRef(dual_sim);
        opt = ROL::OptimizationProblem<double> ( obj, des_var_p, con, dual_sim_p );
        ROL::EProblem problemType = opt.getProblemType();
        std::cout << ROL::EProblemToString(problemType) << std::endl;

        // Set parameters.
        //parlist.sublist("Secant").set("Use as Preconditioner", false);
        parlist.sublist("Status Test").set("Gradient Tolerance", 1e-14);
        parlist.sublist("Status Test").set("Iteration Limit", 5000);
        parlist.sublist("Step").set("Type","Line Search");
        parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
        parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);

        //parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Iteration Scaling");
        //parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Backtracking");
        parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Cubic Interpolation");

        //parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Null Curvature Condition");
        //parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Strong Wolfe Conditions");
        parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Goldstein Conditions");

        parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Quasi-Newton Method");
        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Steepest Descent");

        //parlist.sublist("Step").sublist("Interior Point").set("Initial Step Size",0.1);

        parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
        //parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
        parlist.sublist("General").sublist("Secant").set("Maximum Storage",5000);

    } else { 
        const auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(des_var_sim_rol_p, des_var_ctl_rol_p);
        // Reduced space problem
        opt = ROL::OptimizationProblem<double> ( robj, des_var_ctl_rol_p );
        ROL::EProblem problemType = opt.getProblemType();
        std::cout << ROL::EProblemToString(problemType) << std::endl;

        //{
        //    // Full space problem to check functions
        //    auto dual_sim_p = des_var_sim_rol_p->clone();
        //    ROL::OptimizationProblem<double> fullspaceopt = ROL::OptimizationProblem<double> ( obj, des_var_p, con, dual_sim_p );
        //    fullspaceopt.check(*outStream);
        //}
        {
            const auto u = des_var_sim_rol_p->clone();
            const auto z = des_var_ctl_rol_p->clone();
            const auto v = u->clone();
            const auto jv = v->clone();

            v->zero();
            v->setScalar(1.0);

            std::vector<double> steps;
            for (int i = -2; i > -9; i--) {
                steps.push_back(std::pow(10,i));
            }
            const int order = 2;
            *outStream << "con->checkApplyJacobian_1..." << std::endl;
            con->checkApplyJacobian_1(*u, *z, *v, *jv, steps, true, *outStream, order);

            *outStream << "con->checkInverseJacobian_1..." << std::endl;
            con->checkInverseJacobian_1(*jv, *v, *u, *z, true, *outStream);

            *outStream << "con->checkInverseAdjointJacobian_1..." << std::endl;
            con->checkInverseAdjointJacobian_1(*jv, *v, *u, *z, true, *outStream);

            const auto direction = des_var_p->clone();
            *outStream << "obj->checkGradient..." << std::endl;
            obj->checkGradient( *des_var_p, *direction, steps, true, *outStream, order);

            *outStream << "robj->checkGradient..." << std::endl;
            const auto direction_ctl = des_var_ctl_rol_p->clone();
            robj->checkGradient( *des_var_ctl_rol_p, *direction_ctl, steps, true, *outStream, order);

        }
        {
            const auto u = des_var_sim_rol_p->clone();
            const auto z = des_var_ctl_rol_p->clone();
            const auto v = z->clone();
            const auto jv = u->clone();

            std::vector<double> steps;
            for (int i = -2; i > -9; i--) {
                steps.push_back(std::pow(10,i));
            }
            const int order = 2;
            *outStream << "con->checkApplyJacobian_2..." << std::endl;
            con->checkApplyJacobian_2(*u, *z, *v, *jv, steps, true, *outStream, order);

        }
        // {
        //     *outStream << "con->checkApplyAdjointJacobian..." << std::endl;
        //     const auto v = des_var_sim_rol_p->clone();
        //     const auto c = des_var_sim_rol_p->clone();
        //     const auto ajv = des_var_p->clone();
        //     const int numSteps = 13;
        //     con->checkApplyAdjointJacobian(*des_var_p, *v, *c, *ajv, true, *outStream, numSteps);
        // }
        {
            auto dual = des_var_sim_rol_p->clone();
            dual->set(*des_var_sim_rol_p);
            const auto v = des_var_p->clone();
            v->set(*des_var_p);
            const auto hv = des_var_p->clone();

            std::vector<double> steps;
            for (int i = -1; i > -13; i--) {
                steps.push_back(std::pow(10,i));
            }
            const int order = 2; (void) order;

            *outStream << "con->checkApplyAdjointHessian..." << std::endl;
            (void) con->checkApplyAdjointHessian(*des_var_p, *dual, *v, *hv, steps, true, *outStream, order);

            const auto direction_1 = des_var_p->clone();
            auto direction_2 = des_var_p->clone();
            direction_2->scale(0.5);
            *outStream << "obj->checkHessVec..." << std::endl;
            obj->checkHessVec( *des_var_p, *direction_1, steps, true, *outStream, order);
            *outStream << "obj->checkHessSym..." << std::endl;
            obj->checkHessSym( *des_var_p, *direction_1, *direction_2, true, *outStream);

            const auto direction_ctl_1 = des_var_ctl_rol_p->clone();
            auto direction_ctl_2 = des_var_ctl_rol_p->clone();
            direction_ctl_2->scale(0.5);
            *outStream << "robj->checkHessVec..." << std::endl;
            robj->checkHessVec( *des_var_ctl_rol_p, *direction_ctl_1, steps, true, *outStream, order);

            *outStream << "Outputting Hessian..." << std::endl;
            dealii::FullMatrix<double> Hessian(n_design_variables, n_design_variables);
            for (unsigned int i=0; i<get_ROLvec_to_VectorType(*des_var_ctl_rol_p).size(); ++i) {
                auto direction_unit = des_var_ctl_rol_p->basis(i);
                auto hv = des_var_ctl_rol_p->clone();
                double tol = 1e-6;
                robj->hessVec( *hv, *direction_unit, *des_var_ctl_rol_p, tol );

                auto result = get_ROLvec_to_VectorType(*hv);
                result.update_ghost_values();

                for (unsigned int j=0; j<result.size(); ++j) {
                    Hessian[j][i] = result[j];
                }
            }
            if (mpi_rank == 0) Hessian.print_formatted(*outStream, 3, true, 10, "0", 1., 0.);

            *outStream << "robj->checkHessSym..." << std::endl;
            robj->checkHessSym( *des_var_ctl_rol_p, *direction_ctl_1, *direction_ctl_2, true, *outStream);

        }


        parlist.sublist("Status Test").set("Gradient Tolerance", 1e-10);
        parlist.sublist("Status Test").set("Iteration Limit", 5000);
        parlist.sublist("Step").set("Type","Line Search");
        parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
        parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);

        //parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Iteration Scaling");
        //parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Backtracking");
        parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Cubic Interpolation");

        //parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Null Curvature Condition");
        //parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Strong Wolfe Conditions");
        parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Goldstein Conditions");

        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Newton's Method");

        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Newton-Krylov");
        //parlist.sublist("General").sublist("Secant").set("Use as Preconditioner", false);

        parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Quasi-Newton Method");

        //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Steepest Descent");

        //parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
        //parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
        //parlist.sublist("General").sublist("Secant").set("Maximum Storage",5000);

    }

    *outStream << "Starting optimization..." << std::endl;
    ROL::OptimizationSolver<double> solver( opt, parlist );
    solver.solve( *outStream );

    ROL::Ptr< const ROL::AlgorithmState <double> > opt_state = solver.getAlgorithmState();

    ROL::EExitStatus opt_exit_state = opt_state->statusFlag;

    filebuffer.close();

    return opt_exit_state;
}


#if PHILIP_DIM==2
    template class EulerBumpOptimization <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

