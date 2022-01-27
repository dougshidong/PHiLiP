#include <fstream>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "reduced_order_pod_adaptation.h"
#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/pod_adaptation.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
ReducedOrderPODAdaptation<dim, nstate>::ReducedOrderPODAdaptation(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int ReducedOrderPODAdaptation<dim, nstate>::run_test() const
{
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
    pcout << "dg created" <<std::endl;
    dg->allocate_system ();

    // casting to dg state
    std::shared_ptr< DGBaseState<dim,nstate,double> > dg_state = std::dynamic_pointer_cast< DGBaseState<dim,nstate,double> >(dg);

    pcout << "Implement initial conditions" << std::endl;
    dealii::FunctionParser<1> initial_condition;
    std::string variables = "x";
    std::map<std::string,double> constants;
    constants["pi"] = dealii::numbers::PI;
    std::string expression = "1";
    initial_condition.initialize(variables, expression, constants);
    dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);

    pcout << "Dimension: " << dim
          << "\t Polynomial degree p: " << poly_degree
          << std::endl
          << ". Number of active cells: " << grid->n_global_active_cells()
          << ". Number of degrees of freedom: " << dg->dof_handler.n_dofs()
          << std::endl;

    // Create functional
    auto burgers_functional = BurgersRewienskiFunctional<dim,nstate,double>(dg,dg_state->pde_physics_fad_fad,true,false);

    //POD adaptation
    std::shared_ptr<ProperOrthogonalDecomposition::PODAdaptation<dim, nstate>> pod_adapt = std::make_shared<ProperOrthogonalDecomposition::PODAdaptation<dim, nstate>>(dg, burgers_functional);
    pod_adapt->progressivePODAdaptation();

    //Evaluate functional on fine space to compare
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg_fine = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
    std::shared_ptr< DGBaseState<dim,nstate,double> > dg_state_fine = std::dynamic_pointer_cast< DGBaseState<dim,nstate,double> >(dg_fine);
    dg_fine->allocate_system ();
    dealii::VectorTools::interpolate(dg_fine->dof_handler,initial_condition,dg_fine->solution);
    std::shared_ptr<ProperOrthogonalDecomposition::FinePOD<dim>> finePOD = std::make_shared<ProperOrthogonalDecomposition::FinePOD<dim>>(dg_fine);
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver_fine = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg_fine, finePOD);
    ode_solver_fine->steady_state();
    auto functional_fine = BurgersRewienskiFunctional<dim,nstate,double>(dg_fine,dg_state_fine->pde_physics_fad_fad,true,false);

    pcout << "Fine functional: " << std::setprecision(15)  << functional_fine.evaluate_functional(false,false) << std::setprecision(6) << std::endl;
    pcout << "Coarse functional: " << std::setprecision(15)  << pod_adapt->getCoarseFunctional() << std::setprecision(6) << std::endl;

    if(abs(pod_adapt->getCoarseFunctional() - functional_fine.evaluate_functional(false,false)) > all_parameters->reduced_order_param.adaptation_tolerance){
        pcout << "Adaptation tolerance not reached." << std::endl;
        return -1;
    }
    else{
        pcout << "Adaptation tolerance reached." << std::endl;
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
template class ReducedOrderPODAdaptation<PHILIP_DIM,PHILIP_DIM>;
template class BurgersRewienskiFunctional<PHILIP_DIM, PHILIP_DIM, double>;
#endif
} // Tests namespace
} // PHiLiP namespace
