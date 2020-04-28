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

template <int dim, int nstate>
class InverseObjective : public ROL::Objective_SimOpt<double> {
private:
    Functional<dim,nstate,double> &functional;

    FreeFormDeformation<dim> ffd;

    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    dealii::LinearAlgebra::distributed::Vector<double> ffd_des_var;

public:

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

  double value( const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {
      functional.set_state(get_ROLvec_to_VectorType(des_var_sim));
      // functional.set_geom(get_ROLvec_to_VectorType(des_var_ctl));

      ffd_des_var =  get_ROLvec_to_VectorType(des_var_ctl);
      ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);
      ffd.deform_mesh(functional.dg->high_order_grid);

      const bool compute_dIdW = false;
      const bool compute_dIdX = false;
      const bool compute_d2I = false;
      return functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
  }

  void gradient_1( ROL_Vector& g, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {
      functional.set_state(get_ROLvec_to_VectorType(des_var_sim));
      // functional.set_geom(get_ROLvec_to_VectorType(des_var_ctl));

      ffd_des_var =  get_ROLvec_to_VectorType(des_var_ctl);
      ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);
      ffd.deform_mesh(functional.dg->high_order_grid);

      const bool compute_dIdW = true;
      const bool compute_dIdX = false;
      const bool compute_d2I = false;
      functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
      auto &dIdW = get_ROLvec_to_VectorType(g);
      dIdW = functional.dIdw;
  }

  void gradient_2( ROL_Vector& g, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {
      functional.set_state(get_ROLvec_to_VectorType(des_var_sim));
      //functional.set_geom(get_ROLvec_to_VectorType(des_var_ctl));

      ffd_des_var =  get_ROLvec_to_VectorType(des_var_ctl);
      ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);
      ffd.deform_mesh(functional.dg->high_order_grid);

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

  // void hessVec_11( ROL_Vector& hv, const ROL_Vector& v, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
  //   hv.set(*w);  hv.scale(w->dot(v));
  // }

  // void hessVec_12( ROL_Vector& hv, const ROL_Vector& v, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
  //   hv.zero();
  // }

  // void hessVec_21( ROL_Vector& hv, const ROL_Vector& v, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
  //   hv.zero();
  // }

  // void hessVec_22( ROL_Vector& hv, const ROL_Vector& v, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) {
  //   hv.set(v);  hv.scale(alpha*dt);
  // }
}; // InverseObjective


template<int dim>
class FlowConstraint : public ROL::Constraint_SimOpt<double> {
private:
    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,double>> dg;
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver;

    FreeFormDeformation<dim> ffd;

    Parameters::LinearSolverParam linear_solver_param;
    dealii::ParameterHandler parameter_handler;

    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    dealii::LinearAlgebra::distributed::Vector<double> ffd_des_var;

    int i_out = 1000;
public:
    using ROL::Constraint_SimOpt<double>::value;
    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_2;

    // FlowConstraint(std::shared_ptr<DGBase<dim,double>> &_dg)
    // : dg(_dg)
    // , ode_solver(ODE::ODESolverFactory<dim, double>::create_ODESolver(dg))
    // {
    //     Parameters::LinearSolverParam::declare_parameters (parameter_handler);
    //     linear_solver_param.parse_parameters (parameter_handler);
    //     linear_solver_param.max_iterations = 1000;
    //     linear_solver_param.restart_number = 100;
    //     linear_solver_param.linear_residual = 1e-13;
    //     linear_solver_param.ilut_fill = 1;
    //     linear_solver_param.ilut_drop = 0.0;
    //     linear_solver_param.ilut_rtol = 1.0;
    //     linear_solver_param.ilut_atol = 0.0;
    //     linear_solver_param.linear_solver_output = Parameters::OutputEnum::quiet;
    //     linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;
    // };
    FlowConstraint(std::shared_ptr<DGBase<dim,double>> &_dg, 
                   const FreeFormDeformation<dim> &_ffd,
                   std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim)
    : dg(_dg)
    , ode_solver(ODE::ODESolverFactory<dim, double>::create_ODESolver(dg))
    , ffd(_ffd)
    , ffd_design_variables_indices_dim(_ffd_design_variables_indices_dim)
    {
        ffd_des_var.reinit(ffd_design_variables_indices_dim.size());
        ffd.get_design_variables(ffd_design_variables_indices_dim, ffd_des_var);

        Parameters::LinearSolverParam::declare_parameters (parameter_handler);
        linear_solver_param.parse_parameters (parameter_handler);
        linear_solver_param.max_iterations = 1000;
        linear_solver_param.restart_number = 100;
        linear_solver_param.linear_residual = 1e-13;
        linear_solver_param.ilut_fill = 1;
        linear_solver_param.ilut_drop = 0.0;
        linear_solver_param.ilut_rtol = 1.0;
        linear_solver_param.ilut_atol = 0.0;
        linear_solver_param.linear_solver_output = Parameters::OutputEnum::quiet;
        linear_solver_param.linear_solver_type = Parameters::LinearSolverParam::LinearSolverEnum::gmres;
    };

    // void update_1( const ROL_Vector& des_var_sim, bool flag = true, int iter = -1 ) override {
    //     (void) flag; (void) iter;
    //     dg->solution = get_ROLvec_to_VectorType(des_var_sim);
    //     dg->solution.update_ghost_values();
    // }
    // void update_2( const ROL_Vector& des_var_ctl, bool flag = true, int iter = -1 ) override {
    //     (void) flag; (void) iter;
    //     dg->high_order_grid.nodes = get_ROLvec_to_VectorType(des_var_ctl);
    //     dg->high_order_grid.nodes.update_ghost_values();
    // }

    void update_1( const ROL_Vector& des_var_sim, bool flag = true, int iter = -1 ) override {
        (void) flag; (void) iter;
        dg->solution = get_ROLvec_to_VectorType(des_var_sim);
        dg->solution.update_ghost_values();
    }
    void update_2( const ROL_Vector& des_var_ctl, bool flag = true, int iter = -1 ) override {
        (void) flag; (void) iter;
        ffd_des_var =  get_ROLvec_to_VectorType(des_var_ctl);
        ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);

        // std::vector<dealii::Tensor<1,dim,double>> point_displacements(dg->high_order_grid.locally_relevant_surface_points.size());
        // auto disp = point_displacements.begin();
        // auto point = dg->high_order_grid.locally_relevant_surface_points.begin();
        // auto point_end = dg->high_order_grid.locally_relevant_surface_points.end();
        // for (;point != point_end; ++point, ++disp) {
        //     (*disp) = get_displacement ( *point, ffd.control_pts);
        // }

        ffd.deform_mesh(dg->high_order_grid);
        // dealii::LinearAlgebra::distributed::Vector<double> surface_node_displacements(dg->high_order_grid.surface_nodes);
        // auto index = dg->high_order_grid.surface_indices.begin();
        // auto node = dg->high_order_grid.surface_nodes.begin();
        // auto new_node = surface_node_displacements.begin();
        // for (; index != dg->high_order_grid.surface_indices.end(); ++index, ++node, ++new_node) {
        //     const dealii::types::global_dof_index global_idof_index = *index;
        //     const std::pair<unsigned int, unsigned int> ipoint_component = dg->high_order_grid.global_index_to_point_and_axis.at(global_idof_index);
        //     const unsigned int ipoint = ipoint_component.first;
        //     const unsigned int component = ipoint_component.second;
        //     dealii::Point<dim> old_point;
        //     for (int d=0;d<dim;d++) {
        //         old_point[d] = dg->high_order_grid.locally_relevant_surface_points[ipoint][d];
        //     }
        //     const dealii::Point<dim> new_point = ffd.new_point_location(old_point);
        //     *new_node = new_point[component];
        // }
        // surface_node_displacements.update_ghost_values();
        // surface_node_displacements -= dg->high_order_grid.surface_nodes;
        // surface_node_displacements.update_ghost_values();

        // MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
        //     meshmover(dg->high_order_grid, surface_node_displacements);
        // dealii::LinearAlgebra::distributed::Vector<double> volume_displacements = meshmover.get_volume_displacements();
        // dg->high_order_grid.nodes += volume_displacements;
        // dg->high_order_grid.nodes.update_ghost_values();
    }

    void solve( ROL_Vector& constraint_values, ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {

        update_2(des_var_ctl);
        //ode_solver->steady_state();
        dg->output_results_vtk(i_out++);
        std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver_1 = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver_1->steady_state();

        dg->assemble_residual();
        auto constraint_p = get_rcp_to_VectorType(constraint_values);
        *constraint_p = dg->right_hand_side;
        auto des_var_sim_p = get_rcp_to_VectorType(des_var_sim);
        *des_var_sim_p = dg->solution;
    }

    void value( ROL_Vector& constraint_values, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double &/*tol*/ ) override {

        update_1(des_var_sim);
        update_2(des_var_ctl);

        dg->assemble_residual();
        auto constraint_p = get_rcp_to_VectorType(constraint_values);
        *constraint_p = dg->right_hand_side;
    }
    
    void applyJacobian_1( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {

        update_1(des_var_sim);
        update_2(des_var_ctl);

        const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

        const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
        auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);
        this->dg->system_matrix.vmult(output_vector_v, input_vector_v);
    }
    void applyInverseJacobian_1( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {

        update_1(des_var_sim);
        update_2(des_var_ctl);

        const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

        const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
        auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

        solve_linear_2 (
            this->dg->system_matrix,
            input_vector_v,
            output_vector_v,
            this->linear_solver_param);
    }

    void applyInverseAdjointJacobian_1( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {

        update_1(des_var_sim);
        update_2(des_var_ctl);

        const bool compute_dRdW=true; const bool compute_dRdX=false; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

        dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;
        Epetra_CrsMatrix *system_matrix_transpose_tril;
        Epetra_RowMatrixTransposer epmt( const_cast<Epetra_CrsMatrix *>( &( dg->system_matrix.trilinos_matrix() ) ) );
        epmt.CreateTranspose(false, system_matrix_transpose_tril);
        system_matrix_transpose.reinit(*system_matrix_transpose_tril);

        // Input vector is copied into temporary non-const vector.
        auto input_vector_v = get_ROLvec_to_VectorType(input_vector);
        auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

        solve_linear (system_matrix_transpose, input_vector_v, output_vector_v, this->linear_solver_param);

        // dg->system_matrix.transpose();
        // solve_linear (dg->system_matrix, input_vector_v, output_vector_v, this->linear_solver_param);
        // dg->system_matrix.transpose();
    }

    void applyJacobian_2( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {

        update_1(des_var_sim);
        update_2(des_var_ctl);

        const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);

        const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
        auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

        (void) input_vector_v;
        (void) output_vector_v;

        dealii::TrilinosWrappers::SparseMatrix dXvdXp;
        ffd.get_dXvdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

        auto dXvdXp_input = dg->high_order_grid.nodes;

        dXvdXp.vmult(dXvdXp_input, input_vector_v);
        dg->dRdXv.vmult(output_vector_v, dXvdXp_input);

        // dealii::LinearAlgebra::distributed::Vector<double> surface_node_displacements_vector = dg->high_order_grid.surface_nodes;
        // MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
        //     meshmover( 
        //       *(dg->high_order_grid.triangulation),
        //       dg->high_order_grid.initial_mapping_fe_field,
        //       dg->high_order_grid.dof_handler_grid,
        //       dg->high_order_grid.surface_indices,
        //       surface_node_displacements_vector);
        // meshmover.evaluate_dXvdXs();
        // dealii::TrilinosWrappers::SparseMatrix dRdXs;
		// {
		// 	dRdXs.clear();
		// 	dealii::SparsityPattern sparsity_pattern_dRdXs = dg->get_dRdXs_sparsity_pattern ();
		// 	const dealii::IndexSet &row_parallel_partitioning_dRdXs = dg->locally_owned_dofs;
        //     const dealii::IndexSet &surface_locally_owned_indexset = dg->high_order_grid.surface_nodes.locally_owned_elements();
		// 	const dealii::IndexSet &col_parallel_partitioning_dRdXs = surface_locally_owned_indexset;
		// 	dRdXs.reinit(row_parallel_partitioning_dRdXs, col_parallel_partitioning_dRdXs, sparsity_pattern_dRdXs, MPI_COMM_WORLD);
		// }
		// for (unsigned int isurf = 0; isurf < dg->high_order_grid.surface_nodes.size(); ++isurf) {
		// 	VectorType dRdXs_i(dg->solution);
		// 	dg->dRdXv.vmult(dRdXs_i,meshmover.dXvdXs[isurf]);
		// 	for (unsigned int irow = 0; irow < dg->dof_handler.n_dofs(); ++irow) {
		// 		if (dg->locally_owned_dofs.is_element(irow)) {
		// 			dRdXs.add(irow, isurf, dRdXs_i[irow]);
		// 		}
		// 	}
		// }
        // dRdXs.compress(dealii::VectorOperation::add);


        // dealii::TrilinosWrappers::SparseMatrix dRdXd;
        // dealii::SparsityPattern sparsity_pattern_dRdXs = dg->get_dRdXs_sparsity_pattern ();
        // const unsigned n_residuals = dof_handler.n_dofs();
        // const unsigned n_des = des_var_ctl.size();
        // const unsigned int n_rows = n_residuals;
        // const unsigned int n_cols = n_des;

        // dealii::DynamicSparsityPattern dsp(n_rows, n_cols);
    }

    void applyAdjointJacobian_2( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override {

        update_1(des_var_sim);
        update_2(des_var_ctl);

        const bool compute_dRdW=false; const bool compute_dRdX=true; const bool compute_d2R=false;
        dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
        //dealii::TrilinosWrappers::SparseMatrix dRdX_matrix_transpose;
        //Epetra_CrsMatrix *dRdX_transpose_tril;
        //Epetra_RowMatrixTransposer epmt( const_cast<Epetra_CrsMatrix *>( &( dg->dRdXv.trilinos_matrix() ) ) );
        //epmt.CreateTranspose(false, dRdX_transpose_tril);
        //dRdX_matrix_transpose.reinit(*dRdX_transpose_tril);

        const auto &input_vector_v = get_ROLvec_to_VectorType(input_vector);
        auto &output_vector_v = get_ROLvec_to_VectorType(output_vector);

        auto input_dRdXv = dg->high_order_grid.nodes;

        dg->dRdXv.Tvmult(input_dRdXv, input_vector_v);

        dealii::TrilinosWrappers::SparseMatrix dXvdXp;
        ffd.get_dXvdXp (dg->high_order_grid, ffd_design_variables_indices_dim, dXvdXp);

        dXvdXp.Tvmult(output_vector_v, input_dRdXv);
    }

};


template <typename VectorType, class Real = double, typename AdaptVector = dealii::Rol::VectorAdaptor<VectorType>>
class RosenbrockObjective : public ROL::Objective<Real>
{

public:
    Real value(const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType(x);
        // Rosenbrock function

        Real local_rosenbrock = 0.0;

        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i == (*xp).size() - 1) continue;
            const Real &x0 = (*xp)[i];
            const Real &x1 = (*xp)[i+1];
            local_rosenbrock += 100*(x1 - x0*x0)*(x1 - x0*x0) + (1.0-x0)*(1.0-x0);
        }
        // const double tpi = 2* std::atan(1.0)*4;
        // return std::sin((*xp)[0]*tpi) + std::sin((*xp)[1]*tpi);
        const Real rosenbrock = dealii::Utilities::MPI::sum(local_rosenbrock, MPI_COMM_WORLD);
        return rosenbrock;
    }

    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        using ADtype = Sacado::Fad::DFad<double>;


        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType(x);
        Teuchos::RCP<VectorType>       gp = get_rcp_to_VectorType(g);

        (*gp) *= 0.0;
        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i==(*xp).size()-1) continue;
            const Real &x1 = (*xp)[i];
            const Real &x2 = (*xp)[i+1];
            // https://www.wolframalpha.com/input/?i=f%28a%2Cb%29+%3D+100*%28b-a*a%29%5E2+%2B+%281-a%29%5E2%2C+df%2Fda
            const Real drosenbrock_dx1 = 2.0*(200*x1*x1*x1 - 200*x1*x2 + x1 - 1.0);
            (*gp)[i]  = drosenbrock_dx1;
        }
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i==0) continue;
            const Real &x1 = (*xp)[i-1];
            const Real &x2 = (*xp)[i];
            const Real drosenbrock_dx2 = 200.0*(x2-x1*x1);
            (*gp)[i] += drosenbrock_dx2;
        }
    }
};

template <typename VectorType, class Real = double, typename AdaptVector = dealii::Rol::VectorAdaptor<VectorType>>
class TargetPressure : public ROL::Objective<Real>
{
public:
    Real value(const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType(x);
        // Rosenbrock function

        Real local_rosenbrock = 0.0;

        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i == (*xp).size() - 1) continue;
            const Real &x0 = (*xp)[i];
            const Real &x1 = (*xp)[i+1];
            local_rosenbrock += 100*(x1 - x0*x0)*(x1 - x0*x0) + (1.0-x0)*(1.0-x0);
        }
        // const double tpi = 2* std::atan(1.0)*4;
        // return std::sin((*xp)[0]*tpi) + std::sin((*xp)[1]*tpi);
        const Real rosenbrock = dealii::Utilities::MPI::sum(local_rosenbrock, MPI_COMM_WORLD);
        return rosenbrock;
    }

    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real & /*tol*/)
    {
        using ADtype = Sacado::Fad::DFad<double>;


        Teuchos::RCP<const VectorType> xp = get_rcp_to_VectorType(x);
        Teuchos::RCP<VectorType>       gp = get_rcp_to_VectorType(g);

        (*gp) *= 0.0;
        const dealii::IndexSet &local_range = (*xp).locally_owned_elements ();
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i==(*xp).size()-1) continue;
            const Real &x1 = (*xp)[i];
            const Real &x2 = (*xp)[i+1];
            // https://www.wolframalpha.com/input/?i=f%28a%2Cb%29+%3D+100*%28b-a*a%29%5E2+%2B+%281-a%29%5E2%2C+df%2Fda
            const Real drosenbrock_dx1 = 2.0*(200*x1*x1*x1 - 200*x1*x2 + x1 - 1.0);
            (*gp)[i]  = drosenbrock_dx1;
        }
        for (auto ip = local_range.begin(); ip != local_range.end(); ++ip) {
            const auto i = *ip;
            if (i==0) continue;
            const Real &x1 = (*xp)[i-1];
            const Real &x2 = (*xp)[i];
            const Real drosenbrock_dx2 = 200.0*(x2-x1*x1);
            (*gp)[i] += drosenbrock_dx2;
        }
    }
};


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

    VectorType target_solution;
    double bump_height = 0.0625;
    {
        const double channel_length = 3.0;
        const double channel_height = 0.8;
        Grids::gaussian_bump(grid, n_subdivisions, channel_length, channel_height, bump_height);
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
        // Output target solution
        dg->output_results_vtk(9999);

        target_solution = dg->solution;
    }
    grid.clear();
    const double channel_length = 3.0;
    const double channel_height = 0.8;
    bump_height *= 0.5;
    Grids::gaussian_bump(grid, n_subdivisions, channel_length, channel_height, bump_height);
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
    // Output target solution
    dg->output_results_vtk(9998);

    BoundaryInverseTarget1<dim,nstate,double> functional(dg, target_solution, true, true);

    const dealii::Point<dim> ffd_origin(-1.0,-0.1);
    const std::array<double,dim> ffd_rectangle_lengths = {2,0.6};
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

    const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const unsigned int n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const unsigned int n_rows = n_design_variables;
    const unsigned int n_rows_per_cpu = n_rows / n_mpi_processes;
    const bool is_last_cpu = (this_mpi_process == n_mpi_processes - 1);
    const unsigned int row_start = this_mpi_process * n_rows_per_cpu;
    const unsigned int row_end = is_last_cpu ? n_rows : (this_mpi_process+1) * n_rows_per_cpu;

    dealii::IndexSet row_part(n_rows);
    dealii::IndexSet ghost_row_part(n_rows);
    row_part.add_range(row_start,row_end);
    ghost_row_part.add_range(0,n_design_variables);

    VectorType ffd_design_variables(row_part,ghost_row_part,MPI_COMM_WORLD);


    ffd.get_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    const bool has_ownership = false;
    Teuchos::RCP<VectorType> des_var_sim_rcp = Teuchos::rcp(&dg->solution, has_ownership);
    //Teuchos::RCP<VectorType> des_var_ctl_rcp = Teuchos::rcp(&dg->high_order_grid.nodes, has_ownership);
    Teuchos::RCP<VectorType> des_var_ctl_rcp = Teuchos::rcp(&ffd_design_variables, has_ownership);
    dg->set_dual(dg->solution);
    Teuchos::RCP<VectorType> des_var_adj_rcp = Teuchos::rcp(&dg->dual, has_ownership);

    VectorType gradient_nodes(dg->high_order_grid.nodes);
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
    Teuchos::RCP<std::ostream> outStream;
    if (this->mpi_rank == 0) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    // Run Algorithm
    //RosenbrockObjective<VectorType, double> rosenbrock_objective;
    //algo.run(x_rol, rosenbrock_objective, true, *outStream);

    auto obj  = ROL::makePtr<InverseObjective<dim,nstate>>( functional, ffd, ffd_design_variables_indices_dim );
    auto con  = ROL::makePtr<FlowConstraint<dim>>(dg,ffd,ffd_design_variables_indices_dim);
    auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( obj, con, des_var_sim_rol_p, des_var_ctl_rol_p, des_var_adj_rol_p );

    {
        const auto u = des_var_sim_rol_p->clone();
        const auto z = des_var_ctl_rol_p->clone();
        const auto v = u->clone();
        const auto jv = v->clone();

        std::vector<double> steps;
        for (int i = -4; i > -6; i--) {
            steps.push_back(std::pow(10,i));
        }
        const int order = 2;
        con->checkApplyJacobian_1(*u, *z, *v, *jv, steps, true, *outStream, order);
    }
    {
        const auto u = des_var_sim_rol_p->clone();
        const auto z = des_var_ctl_rol_p->clone();
        const auto v = z->clone();
        const auto jv = u->clone();

        std::vector<double> steps;
        for (int i = -4; i > -6; i--) {
            steps.push_back(std::pow(10,i));
        }
        const int order = 2;
        con->checkApplyJacobian_2(*u, *z, *v, *jv, steps, true, *outStream, order);
    }
    std::abort();

    // Full space problem
    ROL::OptimizationProblem<double> opt( robj, des_var_ctl_rol_p );
    opt.check(*outStream);
    ROL::EProblem problemType = opt.getProblemType();
    std::cout << ROL::EProblemToString(problemType) << std::endl;

    // Set parameters.
    Teuchos::ParameterList parlist;
    //parlist.sublist("Secant").set("Use as Preconditioner", false);
    parlist.sublist("Status Test").set("Gradient Tolerance", 1e-10);
    parlist.sublist("Status Test").set("Iteration Limit", 1000);
    parlist.sublist("Step").set("Type","Line Search");
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-3);
    parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    //parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Iteration Scaling");
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type","Backtracking");
    //parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Quasi-Newton Method");
    parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", "Steepest Descent");
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type","Null Curvature Condition");
    //parlist.sublist("Step").sublist("Interior Point").set("Initial Step Size",0.1);

    // Define algorithm.
    //ROL::Algorithm<double> algo("Line Search", parlist);
    ROL::OptimizationSolver<double> solver( opt, parlist );

    solver.solve( std::cout );
    
    int ifail = 1;
    return ifail;
}


#if PHILIP_DIM==2
    template class EulerBumpOptimization <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

