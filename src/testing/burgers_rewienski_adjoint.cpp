#include "burgers_rewienski_adjoint.h"
#include <fstream>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "burgers_rewienski_snapshot.h"
#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/pde_constraints.h"
#include "optimization/functional_objective.h"
#include "optimization/constraintfromobjective_simopt.hpp"

const double STEPSIZE = 1e-7;

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersRewienskiAdjoint<dim, nstate>::BurgersRewienskiAdjoint(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int BurgersRewienskiAdjoint<dim, nstate>::run_test() const
{
    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;

    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    pcout << "Running Burgers Rewienski with parameter a: "
          << param.reduced_order_param.rewienski_a
          << " and parameter b: "
          << param.reduced_order_param.rewienski_b
          << std::endl;

    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>();

    double left = param.grid_refinement_study_param.grid_left;
    double right = param.grid_refinement_study_param.grid_right;
    const bool colorize = true;
    int n_refinements = param.grid_refinement_study_param.num_refinements;
    unsigned int poly_degree = param.grid_refinement_study_param.poly_degree;
    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);

    grid->refine_global(n_refinements);
    pcout << "Grid generated and refined" << std::endl;

    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);

    // casting to dg state
    std::shared_ptr< DGBaseState<dim,nstate,double> > dg_state = std::dynamic_pointer_cast< DGBaseState<dim,nstate,double> >(dg);

    pcout << "dg created" <<std::endl;
    dg->allocate_system ();

    pcout << "Implement initial conditions" << std::endl;
    dealii::FunctionParser<1> initial_condition;
    std::string variables = "x";
    std::map<std::string,double> constants;
    constants["pi"] = dealii::numbers::PI;
    std::string expression = "1";
    initial_condition.initialize(variables, expression, constants);
    dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);

    // Create ODE solver using the factory and providing the DG object
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

    pcout << "Dimension: " << dim
          << "\t Polynomial degree p: " << poly_degree
          << std::endl
          << ". Number of active cells: " << grid->n_global_active_cells()
          << ". Number of degrees of freedom: " << dg->dof_handler.n_dofs()
          << std::endl;

    double finalTime = param.reduced_order_param.final_time;
    ode_solver->advance_solution_time(finalTime);

    pcout << "Computing functional ";

    // functional for computations
    auto burgers_functional = std::make_shared<BurgersRewienskiFunctional<dim,nstate,double>>(dg,dg_state->pde_physics_fad_fad,true,false);

    // evaluating functional
    double functional = burgers_functional->evaluate_functional(false,false);

    pcout << "Functional output ";
    pcout << functional;

    auto obj  = ROL::makePtr<FunctionalObjective<dim,nstate>>(*burgers_functional);
    auto con  = ROL::makePtr<PDEConstraints<dim>>(dg);

    DealiiVector des_var_sim = dg->solution;
    DealiiVector des_var_ctl = dg->high_order_grid->volume_nodes;
    DealiiVector des_var_adj = dg->dual;
    DealiiVector gradient_sim = dg->dual;
    DealiiVector des_var_adj_fd = dg->dual;

    const bool has_ownership = false;
    VectorAdaptor des_var_sim_rol(Teuchos::rcp(&des_var_sim, has_ownership));
    VectorAdaptor des_var_ctl_rol(Teuchos::rcp(&des_var_ctl, has_ownership));
    VectorAdaptor des_var_adj_rol(Teuchos::rcp(&des_var_adj, has_ownership));
    VectorAdaptor des_var_adj_fd_rol(Teuchos::rcp(&des_var_adj_fd, has_ownership));
    VectorAdaptor gradient_sim_rol(Teuchos::rcp(&gradient_sim, has_ownership));

    double empty = 0.0;
    obj->gradient_1(gradient_sim_rol, des_var_sim_rol , des_var_ctl_rol, empty);
    con->applyInverseAdjointJacobian_1(des_var_adj_rol, gradient_sim_rol, des_var_sim_rol, des_var_ctl_rol, empty);

    double adjoint_l2norm = des_var_adj.l2_norm();

    DealiiVector gradient_sim_fd = burgers_functional->evaluate_dIdw_finiteDifferences(*dg, *dg_state->pde_physics_double, STEPSIZE);
    VectorAdaptor gradient_sim_fd_rol(Teuchos::rcp(&gradient_sim_fd, has_ownership));
    con->applyInverseAdjointJacobian_1(des_var_adj_fd_rol, gradient_sim_fd_rol, des_var_sim_rol, des_var_ctl_rol, empty);

    double adjoint_l2norm_fd = des_var_adj_fd.l2_norm();

    pcout << "Difference: " << abs(adjoint_l2norm - adjoint_l2norm_fd) << std::endl;

    if (abs(adjoint_l2norm - adjoint_l2norm_fd) > 1E-04){
        pcout << "Fail!" <<std::endl;
        return -1;
    }else{
        pcout << "Pass!" << std::endl;
        return 0;
    }
}


template <int dim, int nstate, typename real>
template <typename real2>
real2 BurgersRewienskiFunctional<dim,nstate,real>::evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/*physics*/,
        const dealii::Point<dim,real2> &/*phys_coord*/,
        const std::array<real2,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/) const
{
    real2 val = 0;

    // integrating over the domain
    for (int istate=0; istate<nstate; ++istate) {
        val += soln_at_q[istate];
    }

    return val;
}

#if PHILIP_DIM==1
template class BurgersRewienskiAdjoint<PHILIP_DIM,PHILIP_DIM>;
template class BurgersRewienskiFunctional<PHILIP_DIM, PHILIP_DIM, double>;
#endif
} // Tests namespace
} // PHiLiP namespace
