#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/optimization/rol/vector_adaptor.h>

#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_BoundConstraint_SimOpt.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Reduced_Constraint_SimOpt.hpp"

#include "ROL_Constraint_Partitioned.hpp"

#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"

#include "ROL_SingletonVector.hpp"
#include <ROL_AugmentedLagrangian_SimOpt.hpp>

#include "euler_naca0012_optimization.hpp"

#include "physics/euler.h"
#include "physics/initial_conditions/initial_condition.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_base.h"
#include "ode_solver/ode_solver_factory.h"

#include "functional/target_boundary_functional.h"

#include "mesh/grids/gaussian_bump.h"
#include "mesh/free_form_deformation.h"

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"
#include "optimization/constraintfromobjective_simopt.hpp"

#include "optimization/primal_dual_active_set.hpp"
#include "optimization/full_space_step.hpp"
#include "optimization/sequential_quadratic_programming.hpp"

#include "mesh/gmsh_reader.hpp"
#include "functional/lift_drag.hpp"
#include "functional/geometric_volume.hpp"
#include "functional/target_wall_pressure.hpp"

#include "global_counter.hpp"

namespace {
enum class OptimizationAlgorithm { full_space_birosghattas, full_space_composite_step, reduced_space_bfgs, reduced_space_newton, reduced_sqp };
enum class Preconditioner { P2, P2A, P4, P4A, identity };

const std::vector<Preconditioner> precond_list { Preconditioner::P4 };
//const std::vector<Preconditioner> precond_list { Preconditioner::P2, Preconditioner::P2A, Preconditioner::P4, Preconditioner::P4A };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::full_space_birosghattas, OptimizationAlgorithm::reduced_space_bfgs, OptimizationAlgorithm::reduced_space_newton };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_bfgs };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_sqp };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_newton };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::full_space_birosghattas };
//const std::vector<OptimizationAlgorithm> opt_list { OptimizationAlgorithm::reduced_space_bfgs };
const std::vector<OptimizationAlgorithm> opt_list {
    //OptimizationAlgorithm::reduced_space_newton,
    OptimizationAlgorithm::full_space_birosghattas,
    OptimizationAlgorithm::reduced_space_bfgs,
    };

const unsigned int POLY_START = 0;
const unsigned int POLY_END = 1; // Can do until at least P2

const unsigned int n_des_var_start = 10;//20;
const unsigned int n_des_var_end   = 20;//100;
const unsigned int n_des_var_step  = 10;//20;

const int max_design_cycle = 1000;

const double FD_TOL = 1e-6;
const double CONSISTENCY_ABS_TOL = 1e-10;

bool USE_BFGS = false;
const int LINESEARCH_MAX_ITER = 10;
const double BACKTRACKING_RATE = 0.5;
const int PDAS_MAX_ITER = 1;

const std::string line_search_curvature = "Null Curvature Condition";
const std::string line_search_method = "Backtracking";
}

namespace PHiLiP {
namespace Tests {

namespace {
    double check_maximum_relative_error(std::vector<std::vector<double>> rol_check_results) {
        double max_rel_err = 999999;
        for (unsigned int i = 0; i < rol_check_results.size(); ++i) {
            const double abs_val_ad = std::abs(rol_check_results[i][1]);
            const double abs_val_fd = std::abs(rol_check_results[i][2]);
            const double abs_err    = std::abs(rol_check_results[i][3]);
            const double rel_err    = abs_err / std::max(abs_val_ad,abs_val_fd);
            max_rel_err = std::min(max_rel_err, rel_err);
        }
        return max_rel_err;
    }
}

template<int dim, int nstate>
int EulerNACADragOptimizationLiftConstrained<dim,nstate>
::check_flow_constraints(
    const unsigned int nx_ffd,
    ROL::Ptr<FlowConstraints<dim>> flow_constraints,
    ROL::Ptr<ROL::Vector<double>> design_simulation,
    ROL::Ptr<ROL::Vector<double>> design_control,
    ROL::Ptr<ROL::Vector<double>> dual_equality_state)
{
    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                1,
                1.4,
                0.8,
                1.25,
                0.0);
    FreeStreamInitialConditions<dim,nstate,double> initial_conditions(euler_physics_double);

    int test_error = 0;
    // Temporary vectors
    const auto temp_sim = design_simulation->clone();
    const auto temp_ctl = design_control->clone();
    const auto v1 = temp_sim->clone();
    const auto v2 = temp_ctl->clone();

    const auto jv1 = temp_sim->clone();
    const auto jv2 = temp_sim->clone();

    v1->zero();
    v1->setScalar(1.0);
    v2->zero();
    v2->setScalar(1.0);

    std::vector<double> steps;
    for (int i = -2; i > -12; i--) {
        steps.push_back(std::pow(10,i));
    }
    const int order = 2;

    const ROL::Ptr<ROL::Vector_SimOpt<double>> des_var_rol_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(design_simulation, design_control);

    Teuchos::RCP<std::ostream> outStream;
    ROL::nullstream bhs; // outputs nothing
    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    std::filebuf filebuffer;
    if (mpi_rank == 0) filebuffer.open ("flow_constraints_check"+std::to_string(nx_ffd)+".log",std::ios::out);
    std::ostream ostr(&filebuffer);
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    *outStream << "flow_constraints->checkApplyJacobian_1..." << std::endl;
    *outStream << "Checks dRdW * v1 against R(w+h*v1,x)/h  ..." << std::endl;
    {
        std::vector<std::vector<double>> results
            = flow_constraints->checkApplyJacobian_1(*temp_sim, *temp_ctl, *v1, *jv1, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkApplyJacobian_1..." << std::endl;
        }
    }

    *outStream << "flow_constraints->checkApplyJacobian_2..." << std::endl;
    *outStream << "Checks dRdX * v2 against R(w,x+h*v2)/h  ..." << std::endl;
    {
        std::vector<std::vector<double>> results
            = flow_constraints->checkApplyJacobian_2(*temp_sim, *temp_ctl, *v2, *jv2, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkApplyJacobian_2..." << std::endl;
        }
    }

    *outStream << "flow_constraints->checkInverseJacobian_1..." << std::endl;
    *outStream << "Checks || v - Jinv J v || == 0  ..." << std::endl;
    {
        const double v_minus_Jinv_J_v = flow_constraints->checkInverseJacobian_1(*jv1, *v1, *temp_sim, *temp_ctl, true, *outStream);
        const double normalized_v_minus_Jinv_J_v = v_minus_Jinv_J_v / v1->norm();
        if (normalized_v_minus_Jinv_J_v > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkInverseJacobian_1..." << std::endl;
        }
    }

    *outStream << "flow_constraints->checkInverseAdjointJacobian_1..." << std::endl;
    *outStream << "Checks || v - Jtinv Jt v || == 0  ..." << std::endl;
    {
        const double v_minus_Jinv_J_v = flow_constraints->checkInverseAdjointJacobian_1(*jv1, *v1, *temp_sim, *temp_ctl, true, *outStream);
        const double normalized_v_minus_Jinv_J_v = v_minus_Jinv_J_v / v1->norm();
        if (normalized_v_minus_Jinv_J_v > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkInverseAdjointJacobian_1..." << std::endl;
        }

    }

    *outStream << "flow_constraints->checkAdjointConsistencyJacobian..." << std::endl;
    *outStream << "Checks (w J v) versus (v Jt w)  ..." << std::endl;
    {
        const auto w = dual_equality_state->clone();
        const auto v = des_var_rol_p->clone();
        const auto x = des_var_rol_p->clone();
        const auto temp_Jv = dual_equality_state->clone();
        const auto temp_Jtw = des_var_rol_p->clone();
        const bool printToStream = true;
        const double wJv_minus_vJw = flow_constraints->checkAdjointConsistencyJacobian (*w, *v, *x, *temp_Jv, *temp_Jtw, printToStream, *outStream);
        if (wJv_minus_vJw > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkAdjointConsistencyJacobian..." << std::endl;
        }
    }

    *outStream << "flow_constraints->checkApplyAdjointHessian..." << std::endl;
    *outStream << "Checks (w H v) versus FD approximation  ..." << std::endl;
    {
        const auto dual = design_simulation->clone();
        const auto temp_sim_ctl = des_var_rol_p->clone();
        const auto v3 = des_var_rol_p->clone();
        const auto hv3 = des_var_rol_p->clone();

        std::vector<std::vector<double>> results
            = flow_constraints->checkApplyAdjointHessian(*des_var_rol_p, *dual, *v3, *hv3, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed flow_constraints->checkApplyAdjointHessian..." << std::endl;
        }
    }
    filebuffer.close();

    return test_error;
}

template<int dim, int nstate>
int EulerNACADragOptimizationLiftConstrained<dim,nstate>
::check_objective(
    ROL::Ptr<ROL::Objective_SimOpt<double>> objective_simopt,
    ROL::Ptr<FlowConstraints<dim>> flow_constraints,
    ROL::Ptr<ROL::Vector<double>> design_simulation,
    ROL::Ptr<ROL::Vector<double>> design_control,
    ROL::Ptr<ROL::Vector<double>> dual_equality_state)
{
    int test_error = 0;
    const bool storage = false;
    const bool useFDHessian = false;
    auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( objective_simopt, flow_constraints, design_simulation, design_control, dual_equality_state, storage, useFDHessian);

    auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(design_simulation, design_control);

    Teuchos::RCP<std::ostream> outStream;
    ROL::nullstream bhs; // outputs nothing
    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    std::filebuf filebuffer;
    static int objective_count = 0;
    if (mpi_rank == 0) filebuffer.open ("objective_simopt"+std::to_string(objective_count)+"_check"+std::to_string(999)+".log",std::ios::out);
    objective_count++;
    std::ostream ostr(&filebuffer);
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    std::vector<double> steps;
    for (int i = -2; i > -9; i--) {
        steps.push_back(std::pow(10,i));
    }
    const int order = 2;
    {
        const auto direction = des_var_p->clone();
        *outStream << "objective_simopt->checkGradient..." << std::endl;
        std::vector<std::vector<double>> results
            = objective_simopt->checkGradient( *des_var_p, *direction, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) test_error++;
    }
    {
        const auto direction_1 = des_var_p->clone();
        auto direction_2 = des_var_p->clone();
        direction_2->scale(0.5);
        *outStream << "objective_simopt->checkHessVec..." << std::endl;
        std::vector<std::vector<double>> results
            = objective_simopt->checkHessVec( *des_var_p, *direction_1, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) test_error++;

        *outStream << "objective_simopt->checkHessSym..." << std::endl;
        std::vector<double> results_HessSym = objective_simopt->checkHessSym( *des_var_p, *direction_1, *direction_2, true, *outStream);
        double wHv       = std::abs(results_HessSym[0]);
        double vHw       = std::abs(results_HessSym[1]);
        double abs_error = std::abs(wHv - vHw);
        double rel_error = abs_error / std::max(wHv, vHw);
        if (rel_error > FD_TOL) test_error++;
    }

    {
        const auto direction_ctl = design_control->clone();
        *outStream << "robj->checkGradient..." << std::endl;
        std::vector<std::vector<double>> results
            = robj->checkGradient( *design_control, *direction_ctl, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) test_error++;

    }
    filebuffer.close();

    return test_error;
}

template<int dim, int nstate>
int EulerNACADragOptimizationLiftConstrained<dim,nstate>
::check_reduced_constraint(
    const unsigned int nx_ffd,
    ROL::Ptr<ROL::Constraint<double>> reduced_constraint,
    ROL::Ptr<ROL::Vector<double>> control_variables,
    ROL::Ptr<ROL::Vector<double>> lift_residual_dual)
{
    int test_error = 0;

    std::vector<double> steps;
    for (int i = -2; i > -12; i--) {
        steps.push_back(std::pow(10,i));
    }
    const int order = 2;


    Teuchos::RCP<std::ostream> outStream;
    ROL::nullstream bhs; // outputs nothing
    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    std::filebuf filebuffer;
    if (mpi_rank == 0) filebuffer.open ("flow_constraints_check"+std::to_string(nx_ffd)+".log",std::ios::out);
    std::ostream ostr(&filebuffer);
    if (mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);

    *outStream << "reduced_constraint->checkApplyJacobian..." << std::endl;
    *outStream << "Checks dRdW * v1 against R(w+h*v1,x)/h  ..." << std::endl;
    {
        const auto temp_ctl = control_variables->clone();
        *outStream << "After temp_ctl declaration ..." << std::endl;
        const auto v1 = control_variables->clone();
        const auto jv1 = lift_residual_dual->clone();
        *outStream << "Right after v1->setScalar(1.0)  ..." << std::endl;
        v1->setScalar(1.0);
        jv1->setScalar(1.0);

        *outStream << "Right before checkApplyJac  ..." << std::endl;
        std::vector<std::vector<double>> results
            = reduced_constraint->checkApplyJacobian(*temp_ctl, *v1, *jv1, steps, true, *outStream, order);


        double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed reduced_constraint->checkApplyJacobian..." << std::endl;

        }

        jv1->setScalar(1.0);
        *outStream << "Right before checkApplyAdjointJac  ..." << std::endl;
        auto c_temp = lift_residual_dual->clone();
        results = reduced_constraint->checkApplyAdjointJacobian(*temp_ctl, *jv1, *c_temp, *v1, true, *outStream, 10);

        max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed reduced_constraint->checkApplyAdjointJacobian..." << std::endl;
        }
    }

    *outStream << "reduced_constraint->checkAdjointConsistencyJacobian..." << std::endl;
    *outStream << "Checks (w J v) versus (v Jt w)  ..." << std::endl;
    const ROL::Ptr<ROL::Vector<double>> des_var_rol_p = control_variables->clone();
    {
        const auto w = lift_residual_dual->clone(); w->setScalar(1.0);
        const auto v = des_var_rol_p->clone();
        const auto x = des_var_rol_p->clone();
        const auto temp_Jv = lift_residual_dual->clone(); temp_Jv->setScalar(1.0);
        const auto temp_Jtw = des_var_rol_p->clone();
        const bool printToStream = true;
        const double wJv_minus_vJw = reduced_constraint->checkAdjointConsistencyJacobian (*w, *v, *x, *temp_Jv, *temp_Jtw, printToStream, *outStream);
        if (wJv_minus_vJw > CONSISTENCY_ABS_TOL) {
            test_error++;
            *outStream << "Failed reduced_constraint->checkAdjointConsistencyJacobian..." << std::endl;
        }
    }

    *outStream << "reduced_constraint->checkApplyAdjointHessian..." << std::endl;
    *outStream << "Checks (w H v) versus FD approximation  ..." << std::endl;
    {
        const auto dual = lift_residual_dual->clone(); dual->setScalar(1.0);
        const auto temp_sim_ctl = des_var_rol_p->clone();
        const auto v3 = des_var_rol_p->clone();
        const auto hv3 = des_var_rol_p->clone();

        std::vector<std::vector<double>> results
            = reduced_constraint->checkApplyAdjointHessian(*des_var_rol_p, *dual, *v3, *hv3, steps, true, *outStream, order);

        const double max_rel_err = check_maximum_relative_error(results);
        if (max_rel_err > FD_TOL) {
            test_error++;
            *outStream << "Failed reduced_constraint->checkApplyAdjointHessian..." << std::endl;
        }
    }
    filebuffer.close();

    return test_error;
}


template <int dim, int nstate>
ROL::Ptr<ROL::Vector<double>> 
EulerNACADragOptimizationLiftConstrained<dim,nstate>::
getDesignVariables(
    ROL::Ptr<ROL::Vector<double>> simulation_variables,
    ROL::Ptr<ROL::Vector<double>> control_variables,
    const bool is_reduced_space) const
{
    if (is_reduced_space) {
        return control_variables;
    }
    ROL::Ptr<ROL::Vector<double>> design_variables_full = ROL::makePtr<ROL::Vector_SimOpt<double>>(simulation_variables, control_variables);
    return design_variables_full;
}


template <int dim, int nstate>
ROL::Ptr<ROL::Objective<double>> 
EulerNACADragOptimizationLiftConstrained<dim,nstate>::
getObjective(
    const ROL::Ptr<ROL::Objective_SimOpt<double>> objective,
    const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
    const ROL::Ptr<ROL::Vector<double>> simulation_variables,
    const ROL::Ptr<ROL::Vector<double>> control_variables,
    const bool is_reduced_space) const
{
    const auto state_constraints = ROL::makePtrFromRef<PHiLiP::FlowConstraints<PHILIP_DIM>>(dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*flow_constraints));
    ROL::Ptr<ROL::Vector<double>> drag_adjoint = simulation_variables->clone();
    // int objective_check_error = check_objective<PHILIP_DIM,PHILIP_DIM+2>( objective, state_constraints, simulation_variables, control_variables, drag_adjoint);
    // (void) objective_check_error;

    if (!is_reduced_space) return objective;

    const bool storage = true;
    const bool useFDHessian = false;

    return ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( objective, flow_constraints, simulation_variables, control_variables, drag_adjoint, storage, useFDHessian);
}

template <int dim, int nstate>
ROL::Ptr<ROL::BoundConstraint<double>>
EulerNACADragOptimizationLiftConstrained<dim,nstate>::
getDesignBoundConstraint(
    ROL::Ptr<ROL::Vector<double>> simulation_variables,
    ROL::Ptr<ROL::Vector<double>> control_variables,
    const bool is_reduced_space) const
{
    (void) simulation_variables;

    struct setUpper : public ROL::Elementwise::UnaryFunction<double> {
        private:
            double zero_;
        public:
            setUpper() : zero_(0) {}
            double apply(const double &x) const {
                if(x>zero_) { return 0.1; }//ROL::ROL_INF<double>(); }
                else { return zero_; }
            }
    } setupper;
    struct setLower : public ROL::Elementwise::UnaryFunction<double> {
        private:
            double zero_;
        public:
            setLower() : zero_(0) {}
            double apply(const double &x) const {
                if(x<zero_) { return -1.0*0.1; }//ROL::ROL_INF<double>(); }
                else { return zero_; }
            }
    } setlower;

    ROL::Ptr<ROL::Vector<double>> l = control_variables->clone();
    ROL::Ptr<ROL::Vector<double>> u = control_variables->clone();

    l->applyUnary(setlower);
    u->applyUnary(setupper);

    double scale = 1;
    double feasTol = 1e-8;
    ROL::Ptr<ROL::BoundConstraint<double>> control_bounds = ROL::makePtr<ROL::Bounds<double>>(l,u, scale, feasTol);

    if (is_reduced_space) return control_bounds;

    ROL::Ptr<ROL::BoundConstraint<double>> simulation_bounds = ROL::makePtr<ROL::BoundConstraint<double>>(*simulation_variables);
    simulation_bounds->deactivate();
    ROL::Ptr<ROL::BoundConstraint<double>> design_bounds = ROL::makePtr<ROL::BoundConstraint_SimOpt<double>> (simulation_bounds, control_bounds);

    return design_bounds;
}


template <int dim, int nstate>
ROL::Ptr<ROL::Constraint<double>>
EulerNACADragOptimizationLiftConstrained<dim,nstate>::getEqualityConstraint(void) const
{
    return ROL::nullPtr;
}

template <int dim, int nstate>
ROL::Ptr<ROL::Vector<double>> 
EulerNACADragOptimizationLiftConstrained<dim,nstate>::getEqualityMultiplier(void) const
{
    return ROL::nullPtr;
}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::Constraint<double>>>
EulerNACADragOptimizationLiftConstrained<dim,nstate>::
getInequalityConstraint(
    const std::vector<ROL::Ptr<ROL::Objective_SimOpt<double>>> constraints_as_objective,
    const ROL::Ptr<ROL::Constraint_SimOpt<double>> flow_constraints,
    const ROL::Ptr<ROL::Vector<double>> simulation_variables,
    const ROL::Ptr<ROL::Vector<double>> control_variables,
    const bool is_reduced_space
    ) const
{
    std::vector<ROL::Ptr<ROL::Constraint<double> > > cvec;
    if (is_reduced_space) {
		const bool storage = true;
		const bool useFDHessian = false;
		for (unsigned int i = 0; i < constraints_as_objective.size(); ++i) {
			ROL::Ptr<ROL::Vector<double>> adjoint = simulation_variables->clone();
			auto reduced_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>(
				constraints_as_objective[i], flow_constraints, simulation_variables, control_variables, adjoint, storage, useFDHessian);
			//const auto state_constraints = ROL::makePtrFromRef<PHiLiP::FlowConstraints<PHILIP_DIM>>(dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*flow_constraints));
			//int objective_check_error = check_objective<PHILIP_DIM,PHILIP_DIM+2>( constraints_as_objective[i], state_constraints, simulation_variables, control_variables, adjoint);
			//(void) objective_check_error;
			ROL::Ptr<ROL::Constraint<double>> reduced_constraint = ROL::makePtr<ROL::ConstraintFromObjective<double>> (reduced_objective, 0.0);

			cvec.push_back(reduced_constraint);
		}
    } else {
		for (unsigned int i = 0; i < constraints_as_objective.size(); ++i) {
			ROL::Ptr<ROL::Constraint<double>> constraint = ROL::makePtr<PHiLiP::ConstraintFromObjective_SimOpt<double>> (constraints_as_objective[i], 0.0);
			cvec.push_back(constraint);
		}
    }
	return cvec;


}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::Vector<double>>> 
EulerNACADragOptimizationLiftConstrained<dim,nstate>::
getInequalityMultiplier(const double volume_target) const
{
    const ROL::Ptr<ROL::SingletonVector<double>> lift_constraint_dual = ROL::makePtr<ROL::SingletonVector<double>> (1.0);
    std::vector<ROL::Ptr<ROL::Vector<double>>> emul;
    emul.push_back(lift_constraint_dual);
    (void) volume_target;
    //if (volume_target > 0) {
        const ROL::Ptr<ROL::SingletonVector<double>> volume_constraint_dual = ROL::makePtr<ROL::SingletonVector<double>> (1.0);
        emul.push_back(volume_constraint_dual);
    //}
    return emul;
}

template <int dim, int nstate>
std::vector<ROL::Ptr<ROL::BoundConstraint<double>>>
EulerNACADragOptimizationLiftConstrained<dim,nstate>::
getSlackBoundConstraint(const double lift_target, const double volume_target) const
{

    (void) lift_target;
    //const ROL::Ptr<ROL::SingletonVector<double>> lift_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (lift_target);
    // Constraint is already (lift - target_lift), therefore, the lower bound is 0.0.
    const ROL::Ptr<ROL::SingletonVector<double>> lift_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (lift_target+0.0);
    const ROL::Ptr<ROL::SingletonVector<double>> lift_upper_bound = ROL::makePtr<ROL::SingletonVector<double>> (lift_target+1.0);
    //const bool isLower = true;
    double scale = 1;
    double feasTol = 1e-4;
    auto lift_bounds = ROL::makePtr<ROL::Bounds<double>> (lift_lower_bound, lift_upper_bound, scale, feasTol);


    std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> bcon;
    bcon.push_back(lift_bounds);
    //if (volume_target > 0) {
        const ROL::Ptr<ROL::SingletonVector<double>> volume_lower_bound = ROL::makePtr<ROL::SingletonVector<double>> (volume_target-1e-4);
        const ROL::Ptr<ROL::SingletonVector<double>> volume_upper_bound = ROL::makePtr<ROL::SingletonVector<double>> (volume_target+1e+4);
        auto volume_bounds = ROL::makePtr<ROL::Bounds<double>> (volume_lower_bound, volume_upper_bound, scale, feasTol);
        bcon.push_back(volume_bounds);
    //}
    return bcon;
}



template <int dim, int nstate>
EulerNACADragOptimizationLiftConstrained<dim,nstate>::
EulerNACADragOptimizationLiftConstrained(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
int EulerNACADragOptimizationLiftConstrained<dim,nstate>
::run_test () const
{
    int test_error = 0;
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization.log", std::ios::out);
    if (this->mpi_rank == 0) filebuffer.close();

    for (unsigned int poly_degree = POLY_START; poly_degree <= POLY_END; ++poly_degree) {
        for (unsigned int n_des_var = n_des_var_start; n_des_var <= n_des_var_end; n_des_var += n_des_var_step) {
        //for (unsigned int n_des_var = n_des_var_start; n_des_var <= n_des_var_end; n_des_var *= 2) {
            // assert(n_des_var%2 == 0);
            // assert(n_des_var>=2);
            // const unsigned int nx_ffd = n_des_var / 2 + 2;
            const unsigned int nx_ffd = n_des_var + 2;
            test_error += optimize(nx_ffd, poly_degree);
        }
    }
    return test_error;
}

template<int dim, int nstate>
int EulerNACADragOptimizationLiftConstrained<dim,nstate>
::optimize (const unsigned int nx_ffd, const unsigned int poly_degree) const
{
    int test_error = 0;

    for (auto const opt_type : opt_list) {
    for (auto const precond_type : precond_list) {

    std::string opt_output_name = "";
    std::string descent_method = "";
    std::string preconditioner_string = "";
    switch(opt_type) {
        case OptimizationAlgorithm::full_space_birosghattas: {
            opt_output_name = "full_space";
            switch(precond_type) {
                case Preconditioner::P2: {
                    opt_output_name += "_p2";
                    preconditioner_string = "P2";
                    break;
                }
                case Preconditioner::P2A: {
                    opt_output_name += "_p2a";
                    preconditioner_string = "P2A";
                    break;
                }
                case Preconditioner::P4: {
                    opt_output_name += "_p4";
                    preconditioner_string = "P4";
                    break;
                }
                case Preconditioner::P4A: {
                    opt_output_name += "_p4a";
                    preconditioner_string = "P4A";
                    break;
                }
                case Preconditioner::identity: {
                    opt_output_name += "_identity";
                    preconditioner_string = "identity";
                    break;
                }
            }
            break;
        }
        case OptimizationAlgorithm::full_space_composite_step: {
            opt_output_name = "full_space_composite_step";
            break;
        }
        case OptimizationAlgorithm::reduced_space_bfgs: {
            opt_output_name = "reduced_space_bfgs";
            descent_method = "Quasi-Newton Method";
            break;
        }
        case OptimizationAlgorithm::reduced_sqp: {
        }
        case OptimizationAlgorithm::reduced_space_newton: {
            opt_output_name = "reduced_space_newton";
            descent_method = "Newton-Krylov";
            break;
        }
    }
    opt_output_name = opt_output_name + "_"
                      + "P" + std::to_string(poly_degree);

    // Output stream
    ROL::nullstream bhs; // outputs nothing
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization_"+opt_output_name+"_"+std::to_string(nx_ffd-2)+".log", std::ios::out);
    //if (this->mpi_rank == 0) filebuffer.open ("optimization.log", std::ios::out|std::ios::app);
    std::ostream ostr(&filebuffer);

    Teuchos::RCP<std::ostream> outStream;
    if (this->mpi_rank == 0) outStream = ROL::makePtrFromRef(ostr);
    else if (this->mpi_rank == 1) outStream = ROL::makePtrFromRef(std::cout);
    else outStream = ROL::makePtrFromRef(bhs);


    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(param.pde_type == param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;


    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    FreeStreamInitialConditions<dim,nstate,double> initial_conditions(euler_physics_double);

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation> (
        this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));


    const dealii::Point<dim> ffd_origin(0.0,-0.061);
    const std::array<double,dim> ffd_rectangle_lengths = {{0.999,0.122}};
    const std::array<unsigned int,dim> ffd_ndim_control_pts = {{nx_ffd,3}};
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
                || ijk[1] == 1 // Constrain middle row of FFD points.
                || d_ffd == 0 // Constrain x-direction of FFD points.
               ) {
                continue;
            }
            ++n_design_variables;
            ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
        }
    }

    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_design_variables);
    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);
    DealiiVector ffd_design_variables(row_part,ghost_row_part,MPI_COMM_WORLD);

    ffd.get_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    const auto initial_design_variables = ffd_design_variables;

    // Initial optimization point
    grid->clear();
    dealii::GridGenerator::hyper_cube(*grid);

    ffd_design_variables = initial_design_variables;
    ffd_design_variables.update_ghost_values();
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);

    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, grid);

    //naca0012_mesh->refine_global();
    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh <dim, dim> ("naca0012.msh");
    dg->set_high_order_grid(naca0012_mesh);

    dg->allocate_system ();
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);
    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    //param.ode_solver_param.nonlinear_steady_residual_tolerance = 1e-4;
    ode_solver->initialize_steady_polynomial_ramping (poly_degree);
    // // Solve the steady state problem
    ode_solver->steady_state();

    // Reset to initial_grid
    DealiiVector des_var_sim = dg->solution;
    DealiiVector des_var_ctl = initial_design_variables;
    DealiiVector des_var_adj = dg->dual;
    des_var_adj.add(0.1);

    const bool has_ownership = false;
    VectorAdaptor des_var_sim_rol(Teuchos::rcp(&des_var_sim, has_ownership));
    VectorAdaptor des_var_ctl_rol(Teuchos::rcp(&des_var_ctl, has_ownership));
    VectorAdaptor des_var_adj_rol(Teuchos::rcp(&des_var_adj, has_ownership));

    ROL::Ptr<ROL::Vector<double>> simulation_variables = ROL::makePtr<VectorAdaptor>(des_var_sim_rol);
    ROL::Ptr<ROL::Vector<double>> control_variables = ROL::makePtr<VectorAdaptor>(des_var_ctl_rol);
    auto des_var_p = ROL::makePtr<ROL::Vector_SimOpt<double>>(simulation_variables, control_variables);


    ROL::OptimizationProblem<double> opt;
    Teuchos::ParameterList parlist;

    LiftDragFunctional<dim,nstate,double> lift_functional( dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::lift );
    LiftDragFunctional<dim,nstate,double> drag_functional( dg, LiftDragFunctional<dim,dim+2,double>::Functional_types::drag );
    GeometricVolume<dim,nstate,double> volume_functional( dg );

    std::cout << " Current lift = " << lift_functional.evaluate_functional()
              << ". Current drag = " << drag_functional.evaluate_functional()
              << std::endl;

    double lift_target = lift_functional.evaluate_functional() * 1.0;
    double volume_target = volume_functional.evaluate_functional() * 1.0;


    ffd.output_ffd_vtu(8999);
    auto flow_constraints  = ROL::makePtr<FlowConstraints<dim>>(dg,ffd,ffd_design_variables_indices_dim);
    ROL::Ptr<ROL::Vector<double>> drag_adjoint = ROL::makePtr<VectorAdaptor>(des_var_adj_rol);
    // int flow_constraints_check_error
    //     = check_flow_constraints<dim,nstate>( nx_ffd,
    //                                            flow_constraints,
    //                                            simulation_variables,
    //                                            control_variables,
    //                                            drag_adjoint);
    // (void) flow_constraints_check_error;

    auto drag_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( drag_functional, ffd, ffd_design_variables_indices_dim, &(flow_constraints->dXvdXp) );
    auto lift_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( lift_functional, ffd, ffd_design_variables_indices_dim, &(flow_constraints->dXvdXp) );
    auto volume_objective = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>( volume_functional, ffd, ffd_design_variables_indices_dim, &(flow_constraints->dXvdXp) );

	auto objective = drag_objective;
	auto constraint1 = lift_objective;
	auto constraint2 = volume_objective;

    const unsigned int n_other_constraints = 1;
    const dealii::IndexSet constraint_row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_other_constraints);
    dealii::IndexSet constraint_ghost_row_part(n_other_constraints);
    constraint_ghost_row_part.add_range(0,n_other_constraints);

    double tol = 0.0;
    std::cout << "Drag value= " << objective->value(*simulation_variables, *control_variables, tol) << std::endl;

    dg->output_results_vtk(9999);

    double timing_start, timing_end;
    timing_start = MPI_Wtime();
    // Verbosity setting
    parlist.sublist("General").set("Print Verbosity", 1);

    //parlist.sublist("Status Test").set("Gradient Tolerance", 1e-9);
    parlist.sublist("Status Test").set("Gradient Tolerance", 1e-7);
    parlist.sublist("Status Test").set("Iteration Limit", max_design_cycle);

    parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
    parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit",LINESEARCH_MAX_ITER); // 0.5^30 ~  1e-10
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").get("Backtracking Rate", BACKTRACKING_RATE);
    parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


    parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
    //parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory SR1");
    //parlist.sublist("General").sublist("Secant").set("Maximum Storage",(int)n_design_variables);
    parlist.sublist("General").sublist("Secant").set("Maximum Storage", 100);
    parlist.sublist("General").sublist("Secant").set("Use as Hessian", USE_BFGS);

    parlist.sublist("Full Space").set("Preconditioner",preconditioner_string);

    ROL::Ptr< const ROL::AlgorithmState <double> > algo_state;
    n_vmult = 0;
    dRdW_form = 0;
    dRdW_mult = 0;
    dRdX_mult = 0;
    d2R_mult = 0;

    switch (opt_type) {
        case OptimizationAlgorithm::full_space_composite_step: {
            // Full space problem
            auto dual_sim_p = simulation_variables->clone();
            //opt = ROL::OptimizationProblem<double> ( objective, des_var_p, flow_constraints, dual_sim_p );
            opt = ROL::OptimizationProblem<double> ( objective, des_var_p, flow_constraints, dual_sim_p );

            // Set parameters.

            parlist.sublist("Step").set("Type","Composite Step");
            ROL::ParameterList& steplist = parlist.sublist("Step").sublist("Composite Step");
            steplist.set("Initial Radius", 1e2);
            steplist.set("Use Constraint Hessian", true); // default is true
            steplist.set("Output Level", 1);

            steplist.sublist("Optimality System Solver").set("Nominal Relative Tolerance", 1e-8); // default 1e-8
            steplist.sublist("Optimality System Solver").set("Fix Tolerance", true);
            const int cg_iteration_limit = 200;
            steplist.sublist("Tangential Subproblem Solver").set("Iteration Limit", cg_iteration_limit);
            steplist.sublist("Tangential Subproblem Solver").set("Relative Tolerance", 1e-2);

            *outStream << "Starting optimization with " << n_design_variables << "..." << std::endl;
            ROL::OptimizationSolver<double> solver( opt, parlist );
            solver.solve( *outStream );
            algo_state = solver.getAlgorithmState();

            break;
        }
        case OptimizationAlgorithm::reduced_space_bfgs:
            USE_BFGS = true;
            parlist.sublist("General").sublist("Secant").set("Use as Hessian", USE_BFGS);
            [[fallthrough]];
        case OptimizationAlgorithm::reduced_space_newton: {
            *outStream << "Starting optimization with " << n_design_variables << "..." << std::endl;

            const bool is_reduced_space = true;
            ROL::Ptr<ROL::Vector<double>>                       design_variables               = getDesignVariables(simulation_variables, control_variables, is_reduced_space);
            ROL::Ptr<ROL::BoundConstraint<double>>              design_bounds                  = getDesignBoundConstraint(simulation_variables, control_variables, is_reduced_space);
            ROL::Ptr<ROL::Objective<double>>                    reduced_drag_objective         = getObjective(objective, flow_constraints, simulation_variables, control_variables, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Constraint<double>>>      reduced_inequality_constraints = getInequalityConstraint({constraint1, constraint2}, flow_constraints, simulation_variables, control_variables, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Vector<double>>>          dual_inequality                = getInequalityMultiplier(volume_target);
            std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> inequality_bounds              = getSlackBoundConstraint(lift_target, volume_target);

            opt = ROL::OptimizationProblem<double> ( reduced_drag_objective, design_variables, design_bounds,
                                                     reduced_inequality_constraints, dual_inequality, inequality_bounds);
            ROL::EProblem problem_type_opt = opt.getProblemType();
            ROL::EProblem problem_type = ROL::TYPE_EB;
            if (problem_type_opt != problem_type) std::abort();

            parlist.sublist("Step").sublist("Primal Dual Active Set").set("Iteration Limit",PDAS_MAX_ITER);
            parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-10);
            parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-8);
            parlist.sublist("General").sublist("Krylov").set("Iteration Limit", 300);
            parlist.sublist("General").sublist("Krylov").set("Use Initial Guess", true);

            parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
            parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
            parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
            parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
            parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
            parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


            // This step transforms the inequality into equality + slack variables with box constraints.
            auto x      = opt.getSolutionVector();
            auto g      = x->dual().clone();
            auto l      = opt.getMultiplierVector();
            auto c      = l->dual().clone();
            auto obj    = opt.getObjective();
            auto con    = opt.getConstraint();
            auto bnd    = opt.getBoundConstraint();

            for (auto &constraint_dual : dual_inequality) {
                constraint_dual->zero();
            }

            auto pdas_step = ROL::makePtr<PHiLiP::PrimalDualActiveSetStep<double>>(parlist);
            auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
            const bool printHeader = true;

            const ROL::Ptr<ROL::Algorithm<double>> algorithm = ROL::makePtr<ROL::Algorithm<double>>( pdas_step, status_test, printHeader );
            algorithm->run(*x, *g, *l, *c, *obj, *con, *bnd, true, *outStream);
            algo_state = algorithm->getState();

            break;
        } case OptimizationAlgorithm::reduced_sqp: {
            [[fallthrough]];

        //     // Reduced space problem
        //     const bool storage = true;
        //     const bool useFDHessian = false;
        //     // Create reduced-objective by combining objective with PDE constraints.
        //     ROL::Ptr<ROL::Vector<double>> drag_adjoint = ROL::makePtr<VectorAdaptor>(des_var_adj_rol);
        //     auto reduced_drag_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( objective, flow_constraints, simulation_variables, control_variables, drag_adjoint, storage, useFDHessian);

        //     // Create reduced-constraint by combining lift-objective with PDE constraints.
        //     ROL::Ptr<ROL::SimController<double> > stateStore = ROL::makePtr<ROL::SimController<double>>();
        //     ROL::Ptr<ROL::Vector<double>> lift_adjoint = drag_adjoint->clone();
        //     ROL::Ptr<ROL::SingletonVector<double>> lift_constraint_residual_rol_p = ROL::makePtr<ROL::SingletonVector<double>> (0.0);
        //     //auto reduced_lift_constraint = ROL::makePtr<ROL::Reduced_Constraint_SimOpt<double>>(
        //     //    lift_constraint, flow_constraints, stateStore,
        //     //    simulation_variables, control_variables, lift_adjoint, lift_constraint_residual_rol_p,
        //     //    storage, useFDHessian);

        //     // Create reduced-objective by combining objective with PDE constraints.
        //     auto reduced_lift_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( constraint1, flow_constraints, simulation_variables, control_variables, lift_adjoint, storage, useFDHessian);
        //     std::cout << " Converting reduced lift objective into reduced_lift_constraint " << std::endl;
        //     ROL::Ptr<ROL::Constraint<double>> reduced_lift_constraint = ROL::makePtr<ROL::ConstraintFromObjective<double>> (reduced_lift_objective, lift_target);

        //     std::cout << " Starting check_reduced_constraint " << std::endl;
        //     lift_constraint_residual_rol_p->setScalar(1.0);
        //     //(void) check_reduced_constraint<dim,nstate>( nx_ffd, reduced_lift_constraint, control_variables, lift_constraint_residual_rol_p);

        //     // Run the algorithm
        //     parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
        //     //auto reduced_sqp_step = ROL::makePtr<ROL::SequentialQuadraticProgrammingStep<double>>(parlist);
        //     auto reduced_sqp_step = ROL::makePtr<ROL::InteriorPointStep<double>>(parlist);

        //     auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
        //     const bool printHeader = false;//true;
        //     ROL::Algorithm<double> algorithm(reduced_sqp_step, status_test, printHeader);
        //     algorithm.run(*control_variables, *lift_constraint_residual_rol_p, *reduced_drag_objective, *reduced_lift_constraint, false, *outStream);
        //     algo_state = algorithm.getState();
        //     break;
        } case OptimizationAlgorithm::full_space_birosghattas: {

            *outStream << "Starting optimization with " << n_design_variables << " control variables..." << std::endl;

            const bool is_reduced_space = false;
            ROL::Ptr<ROL::Vector<double>>                       design_variables               = getDesignVariables(simulation_variables, control_variables, is_reduced_space);
            ROL::Ptr<ROL::BoundConstraint<double>>              design_bounds                  = getDesignBoundConstraint(simulation_variables, control_variables, is_reduced_space);
            ROL::Ptr<ROL::Objective<double>>                    drag_objective_simopt          = getObjective(objective, flow_constraints, simulation_variables, control_variables, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Constraint<double>>>      inequality_constraints         = getInequalityConstraint({constraint1, constraint2}, flow_constraints, simulation_variables, control_variables, is_reduced_space);
            std::vector<ROL::Ptr<ROL::Vector<double>>>          dual_inequality                = getInequalityMultiplier(volume_target);
            std::vector<ROL::Ptr<ROL::BoundConstraint<double>>> inequality_bounds              = getSlackBoundConstraint(lift_target, volume_target);

            ROL::Ptr<ROL::Constraint<double>>                   equality_constraints           = flow_constraints;
            ROL::Ptr<ROL::Vector<double>>                       dual_equality                  = simulation_variables->clone();
            dual_equality->zero();

            opt = ROL::OptimizationProblem<double> ( drag_objective_simopt, design_variables, design_bounds,
                                                     equality_constraints, dual_equality,
                                                     inequality_constraints, dual_inequality, inequality_bounds);
            ROL::EProblem problem_type_opt = opt.getProblemType();
            ROL::EProblem problem_type = ROL::TYPE_EB;
            if (problem_type_opt != problem_type) std::abort();

            parlist.sublist("Step").sublist("Primal Dual Active Set").set("Iteration Limit",PDAS_MAX_ITER);
            parlist.sublist("General").sublist("Secant").set("Use as Preconditioner", true);
            parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-12);
            parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-4);
            parlist.sublist("General").sublist("Krylov").set("Iteration Limit", 400);
            parlist.sublist("General").sublist("Krylov").set("Use Initial Guess", true);

            parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
            parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",3e-1); // Might be needed for p2 BFGS
            parlist.sublist("Step").sublist("Line Search").set("Initial Step Size",1e-0);
            parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit",LINESEARCH_MAX_ITER); // 0.5^30 ~  1e-10
            parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);//false);
            parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type",line_search_method);
            parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type",line_search_curvature);


            // This step transforms the inequality into equality + slack variables with box constraints.
            auto x      = opt.getSolutionVector();
            auto g      = x->dual().clone();
            auto l      = opt.getMultiplierVector();
            auto c      = l->dual().clone();
            auto obj    = opt.getObjective();
            auto con    = opt.getConstraint();
            auto bnd    = opt.getBoundConstraint();

            for (auto &constraint_dual : dual_inequality) {
                constraint_dual->zero();
            }

            auto pdas_step = ROL::makePtr<PHiLiP::PrimalDualActiveSetStep<double>>(parlist);
            auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
            const bool printHeader = true;

            const ROL::Ptr<ROL::Algorithm<double>> algorithm = ROL::makePtr<ROL::Algorithm<double>>( pdas_step, status_test, printHeader );
            algorithm->run(*x, *g, *l, *c, *obj, *con, *bnd, true, *outStream);
            algo_state = algorithm->getState();

            break;
        }
    }
    std::cout << " Current lift = " << lift_functional.evaluate_functional()
              << ". Current drag = " << drag_functional.evaluate_functional()
              << ". Drag with quadratic lift penalty = " << objective->value(*simulation_variables, *control_variables, tol);
    static int resulting_optimization = 5000;
    std::cout << "Outputting final grid resulting_optimization: " << resulting_optimization << std::endl;
    dg->output_results_vtk(resulting_optimization++);


    timing_end = MPI_Wtime();
    *outStream << "The process took " << timing_end - timing_start << " seconds to run." << std::endl;

    *outStream << "Total n_vmult for algorithm " << n_vmult << std::endl;

    test_error += algo_state->statusFlag;

    filebuffer.close();

    if (opt_type != OptimizationAlgorithm::full_space_birosghattas) break;
    }
    }

    return test_error;
}


#if PHILIP_DIM==2
    template class EulerNACADragOptimizationLiftConstrained <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace



