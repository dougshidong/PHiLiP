// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

// Modified by Doug Shi-Dong
// Original source file from Trilinos/packages/rol/src/step/ROL_PrimalDualActiveSetStep.hpp
// Modified for the PHiLiP project

#include "ROL_LineSearch.hpp"
#include "ROL_AugmentedLagrangian.hpp"

#include "ROL_Constraint_Partitioned.hpp"
#include "ROL_Objective_SimOpt.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_SlacklessObjective.hpp"

#include <deal.II/lac/full_matrix.h>

#include "optimization/flow_constraints.hpp"
#include "optimization/pdas_kkt_system.hpp"
#include "optimization/pdas_preconditioner.hpp"
#include "optimization/primal_dual_active_set.hpp"
#include "optimization/dealii_solver_rol_vector.hpp"

#include "global_counter.hpp"

namespace PHiLiP {

class ApproximateJacobianFlowConstraints : public ROL::Constraint_SimOpt<double> {
private:
    const ROL::Ptr<FlowConstraints<PHILIP_DIM>> flow_constraints_;
public:
    ApproximateJacobianFlowConstraints(
        ROL::Ptr<FlowConstraints<PHILIP_DIM>> flow_constraints,
        const ROL::Ptr<const ROL::Vector<double>> des_var_sim,
        const ROL::Ptr<const ROL::Vector<double>> des_var_ctl)
    : flow_constraints_(flow_constraints)
    {
        flow_constraints_->construct_JacobianPreconditioner_1(*des_var_sim, *des_var_ctl);
        flow_constraints_->construct_AdjointJacobianPreconditioner_1(*des_var_sim, *des_var_ctl);
    }
    void update_1 ( const ROL::Vector<double>& des_var_sim, bool flag, int iter ) override { flow_constraints_->update_1(des_var_sim, flag, iter); }
    void update_2 ( const ROL::Vector<double>& des_var_ctl, bool flag, int iter ) override { flow_constraints_->update_2(des_var_ctl, flag, iter); }

    void solve (ROL::Vector<double>& constraint_values, ROL::Vector<double>& des_var_sim, const ROL::Vector<double>& des_var_ctl, double& tol) override
    {
        flow_constraints_->solve(constraint_values, des_var_sim, des_var_ctl, tol);
    }
    /// Avoid -Werror=overloaded-virtual.
    using ROL::Constraint_SimOpt<double>::value;
    /// Avoid -Werror=overloaded-virtual.
    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_1;
    /// Avoid -Werror=overloaded-virtual.
    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_2;
    void value(ROL::Vector<double>& constraint_values, const ROL::Vector<double>& des_var_sim, const ROL::Vector<double>& des_var_ctl, double &tol) override
    {
        flow_constraints_->value(constraint_values, des_var_sim, des_var_ctl, tol);
    }
    void applyJacobian_1(
        ROL::Vector<double>& output_vector,
        const ROL::Vector<double>& input_vector,
        const ROL::Vector<double>& des_var_sim,
        const ROL::Vector<double>& des_var_ctl,
        double& tol) override
    {
        flow_constraints_->applyJacobian_1( output_vector, input_vector, des_var_sim, des_var_ctl, tol );
    }
    void applyJacobian_2(
        ROL::Vector<double>& output_vector,
        const ROL::Vector<double>& input_vector,
        const ROL::Vector<double>& des_var_sim,
        const ROL::Vector<double>& des_var_ctl,
        double& tol) override
    {
        flow_constraints_->applyJacobian_2( output_vector, input_vector, des_var_sim, des_var_ctl, tol );
    }
    void applyAdjointJacobian_1(
        ROL::Vector<double>& output_vector,
        const ROL::Vector<double>& input_vector,
        const ROL::Vector<double>& des_var_sim,
        const ROL::Vector<double>& des_var_ctl,
        double& tol ) override
    {
        flow_constraints_->applyAdjointJacobian_1( output_vector, input_vector, des_var_sim, des_var_ctl, tol );
    }
    void applyAdjointJacobian_2(
        ROL::Vector<double>& output_vector,
        const ROL::Vector<double>& input_vector,
        const ROL::Vector<double>& des_var_sim,
        const ROL::Vector<double>& des_var_ctl,
        double& tol ) override
    {
        flow_constraints_->applyAdjointJacobian_2( output_vector, input_vector, des_var_sim, des_var_ctl, tol );
    }
    void applyAdjointHessian_11 ( ROL::Vector<double> &output_vector,
                                      const ROL::Vector<double> &dual,
                                      const ROL::Vector<double> &input_vector,
                                      const ROL::Vector<double> &des_var_sim,
                                      const ROL::Vector<double> &des_var_ctl,
                                      double &tol) override
    {
        flow_constraints_->applyAdjointHessian_11 ( output_vector, dual, input_vector, des_var_sim, des_var_ctl, tol);
    }
    void applyAdjointHessian_12 ( ROL::Vector<double> &output_vector,
                                      const ROL::Vector<double> &dual,
                                      const ROL::Vector<double> &input_vector,
                                      const ROL::Vector<double> &des_var_sim,
                                      const ROL::Vector<double> &des_var_ctl,
                                      double &tol) override
    {
        flow_constraints_->applyAdjointHessian_12 ( output_vector, dual, input_vector, des_var_sim, des_var_ctl, tol);
    }
    void applyAdjointHessian_21 ( ROL::Vector<double> &output_vector,
                                      const ROL::Vector<double> &dual,
                                      const ROL::Vector<double> &input_vector,
                                      const ROL::Vector<double> &des_var_sim,
                                      const ROL::Vector<double> &des_var_ctl,
                                      double &tol) override
    {
        flow_constraints_->applyAdjointHessian_21 ( output_vector, dual, input_vector, des_var_sim, des_var_ctl, tol);
    }
    void applyAdjointHessian_22 ( ROL::Vector<double> &output_vector,
                                      const ROL::Vector<double> &dual,
                                      const ROL::Vector<double> &input_vector,
                                      const ROL::Vector<double> &des_var_sim,
                                      const ROL::Vector<double> &des_var_ctl,
                                      double &tol) override
    {
        flow_constraints_->applyAdjointHessian_22 ( output_vector, dual, input_vector, des_var_sim, des_var_ctl, tol);
    }
    void applyInverseJacobian_1(
        ROL::Vector<double>& output_vector,
        const ROL::Vector<double>& input_vector,
        const ROL::Vector<double>& des_var_sim,
        const ROL::Vector<double>& des_var_ctl,
        double& tol ) override
    {
        flow_constraints_->applyInverseJacobianPreconditioner_1( output_vector, input_vector, des_var_sim, des_var_ctl, tol );
    }
    void applyInverseAdjointJacobian_1(
        ROL::Vector<double>& output_vector,
        const ROL::Vector<double>& input_vector,
        const ROL::Vector<double>& des_var_sim,
        const ROL::Vector<double>& des_var_ctl,
        double& tol ) override
    {
        flow_constraints_->applyInverseAdjointJacobianPreconditioner_1( output_vector, input_vector, des_var_sim, des_var_ctl, tol );
    }
};


template <typename Real>
Real PrimalDualActiveSetStep<Real>::computeCriticalityMeasure(
    Vector &design_variables,
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints,
    Real tol)
{
    Real one(1);
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    objective.gradient(*(step_state->gradientVec),design_variables,tol);
    desvar_tmp_->set(design_variables);
    desvar_tmp_->axpy(-one,(step_state->gradientVec)->dual());
    bound_constraints.project(*desvar_tmp_);
    desvar_tmp_->axpy(-one,design_variables);
    return desvar_tmp_->norm();
}

template <typename Real>
PrimalDualActiveSetStep<Real>::PrimalDualActiveSetStep( ROL::ParameterList &parlist ) 
    : ROL::Step<Real>::Step(),
      parlist_(parlist),
      krylov_(ROL::nullPtr),
      iter_Krylov_(0), flag_Krylov_(0), n_active_(0), itol_(0),
      maxit_(0), iter_PDAS_(0), flag_PDAS_(0), stol_(0), gtol_(0), scale_(0),
      //neps_(-ROL::ROL_EPSILON<Real>()), // Negative epsilon means that x = boundconstraint is INACTIVE when pruneActive occurs
      neps_(ROL::ROL_EPSILON<Real>()), // Positive epsilon means that x = boundconstraint is ACTIVE when pruneActive occurs
      feasible_(false),
      index_to_project_interior(0),
      dual_inequality_(ROL::nullPtr),
      des_plus_dual_(ROL::nullPtr),
      new_design_variables_(ROL::nullPtr),
      new_dual_equality_(ROL::nullPtr),
      search_temp_(ROL::nullPtr),
      search_direction_active_set_(ROL::nullPtr),
      desvar_tmp_(ROL::nullPtr),
      quadratic_residual_(ROL::nullPtr),
      gradient_active_set_(ROL::nullPtr),
      gradient_inactive_set_(ROL::nullPtr),
      old_gradient_(ROL::nullPtr),
      gradient_tmp1_(ROL::nullPtr),
      esec_(ROL::SECANT_LBFGS), objective_secant_(ROL::nullPtr), useSecantPrecond_(false),
      useSecantHessVec_(false),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    Real one(1), oem4(1.e-4), oem5(1.e-5);
    // ROL::Algorithmic parameters
    maxit_ = parlist_.sublist("Step").sublist("Primal Dual Active Set").get("Iteration Limit",10);
    stol_ = parlist_.sublist("Step").sublist("Primal Dual Active Set").get("Relative ROL::Step Tolerance",oem5);
    gtol_ = parlist_.sublist("Step").sublist("Primal Dual Active Set").get("Relative Gradient Tolerance",oem4);
    scale_ = parlist_.sublist("Step").sublist("Primal Dual Active Set").get("Dual Scaling", one);
    // Build secant object
    esec_ = ROL::StringToESecant(parlist_.sublist("General").sublist("Secant").get("Type","Limited-Memory BFGS"));
    //esec_ = ROL::StringToESecant(parlist_.sublist("General").sublist("Secant").get("Type","Limited-Memory SR1"));
    useSecantHessVec_ = parlist_.sublist("General").sublist("Secant").get("Use as Hessian", false); 

    useSecantPrecond_ = parlist_.sublist("General").sublist("Secant").get("Use as Preconditioner", false);

    //parlist_.sublist("General").sublist("Secant").set("Maximum Storage",1000);
    //if ( useSecantHessVec_ || useSecantPrecond_ ) {
      objective_secant_ = ROL::SecantFactory<Real>(parlist_);
      lagrangian_secant_ = ROL::SecantFactory<Real>(parlist_);
    //}
    // Build Krylov object
    krylov_ = ROL::KrylovFactory<Real>(parlist_);
    d_force_use_bfgs = false;
}

template<typename Real>
void PrimalDualActiveSetStep<Real>::initialize(
    Vector &design_variables,
    const Vector &gradient_vec_to_clone, 
    Vector &dual_equality,
    const Vector &constraint_vec_to_clone, 
    ROL::Objective<Real> &objective,
    ROL::Constraint<Real> &equality_constraints, 
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    initialize(design_variables, gradient_vec_to_clone, gradient_vec_to_clone, objective, bound_constraints, algo_state);

    (void) equality_constraints;
    try {
        auto& equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        auto& flow_constraints = dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equality_constraints_partitioned.get(0));
        (void) flow_constraints;
        flow_constraints.flow_CFL_ = 0.0;
        is_full_space_ = true;
        std::cout << "Found a FlowConstraints, full-space optimization occuring..." << std::endl;
    } catch (...) {
    }
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();

    step_state->constraintVec = constraint_vec_to_clone.clone();

    // Initialize equality dual variable
    Real one(1);
    //dual_equality_ = constraint_vec_to_clone.clone(); 
    //dual_equality_->set((step_state->constraintVec)->dual());
    //dual_equality_->scale(-one);

    dual_equality_ = dual_equality.clone();
    new_dual_equality_ = dual_equality.clone();
    new_dual_equality_->set(dual_equality);

    search_direction_dual_ = constraint_vec_to_clone.clone(); 
    search_direction_dual_->set((step_state->constraintVec)->dual());
    search_direction_dual_->scale(-one);

    index_to_project_interior = 0;

}

template<typename Real>
void PrimalDualActiveSetStep<Real>::initialize(
    Vector &design_variables,
    const Vector &search_direction_vec_to_clone,
    const Vector &gradient_vec_to_clone, 
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    is_full_space_ = false;

    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    Real zero(0);
    // Initialize state descent direction and gradient storage
    step_state->descentVec  = search_direction_vec_to_clone.clone();
    step_state->gradientVec = gradient_vec_to_clone.clone();
    step_state->searchSize  = zero;
    // Initialize additional storage
    des_plus_dual_ = design_variables.clone(); 
    search_temp_ = design_variables.clone();
    new_design_variables_ = design_variables.clone();
    search_direction_active_set_   = search_direction_vec_to_clone.clone(); 
    desvar_tmp_ = design_variables.clone(); 
    quadratic_residual_  = gradient_vec_to_clone.clone();
    gradient_active_set_   = gradient_vec_to_clone.clone(); 
    gradient_inactive_set_   = gradient_vec_to_clone.clone(); 
    gradient_tmp1_ = gradient_vec_to_clone.clone(); 
    old_gradient_ = gradient_vec_to_clone.clone(); 
    // Project design_variables onto constraint set
    //bound_constraints.project(design_variables);
    // Update objective function, get value, and get gradient
    Real tol = std::sqrt(ROL::ROL_EPSILON<Real>());
    objective.update(design_variables,true,algo_state.iter);
    algo_state.value = objective.value(design_variables,tol);
    algo_state.nfval++;
    algo_state.gnorm = computeCriticalityMeasure(design_variables,objective,bound_constraints,tol);
    algo_state.ngrad++;

    // Initialize inequality variable
    dual_inequality_ = search_direction_vec_to_clone.clone(); 
    //dual_inequality_->setScalar(0.01);
    dual_inequality_->setScalar(0.0);
    old_dual_inequality_ = dual_inequality_->clone(); 
    old_dual_inequality_->set(*dual_inequality_);
    //Real one(1);
    //dual_inequality_->set((step_state->gradientVec)->dual());
    //dual_inequality_->scale(-one);
}


template<typename Real>
void PrimalDualActiveSetStep<Real>::compute_PDAS_rhs(
    const Vector &old_design_variables,
    const Vector &new_design_variables,
    const Vector &new_dual_equality,
    const Vector &dual_inequality,
    const Vector &des_plus_dual,
    const Vector &old_objective_gradient,
    ROL::Objective<Real> &objective, 
    ROL::Constraint<Real> &equality_constraints, 
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::PartitionedVector<Real> &rhs_partitioned,
    const int objective_type, // 0 = nonlinear, 1 = linear, 2 = quadratic
    const int constraint_type, // 0 = nonlinear, 1 = linear
    const ROL::Secant<Real> &secant,
    const bool useSecant,
    const Real identity_factor_)
{
    //MPI_Barrier(MPI_COMM_WORLD);
    //static int ii = 0; (void) ii;
    //std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;
    // Define old_ as using variables that do not change over the PDAS iterations
    // For example, gradients applied onto a vector would use the old_design_variables.
    // However, evaluating the constraint itself would use new_design_variables.
    // Basically, anything involving linearization uses old_design_variables to ensure that
    // PDAS is solving a linear (quadratic objective) problem.
    Real one(1);
    Real tol = ROL::ROL_EPSILON<Real>();

    VectorPtr rhs_design          = rhs_partitioned.get(0);
    VectorPtr rhs_dual_equality   = rhs_partitioned.get(1);
    VectorPtr rhs_dual_inequality = rhs_partitioned.get(2);

    VectorPtr rhs_dual_inequality_temp = rhs_dual_inequality->clone();

    // ********************
    // Design RHS
    // Row 1-5
    // ********************
    VectorPtr rhs_design_temp = rhs_design->clone(); 
    // rhs_design = objective_gradient + constraint^T dualEquality + dualInequality
    (void) old_objective_gradient;
    (void) old_design_variables;
    //rhs_design->set( old_objective_gradient );
    if (objective_type == 2) {
        // Given a nonlinear objective I(x) w.r.t. design variables x, its gradient dI/dx is also nonlinear.
        // Assume a quadratic objectve J(x) that models I(x), with design variables x, matrix H, and vector f such that
        //     I(x) \approx J(x) = xAx + fx
        // Given a known matrix H, whether it is an exact Hessian of I(x) or a BFGS approximation
        // and the exact nonlinear gradient of the objective w.r.t. the design dI/dx
        //     dI/dx \approx (dJ/dx)_old = H x_old + f,
        // we can compute the constant vector (f) by subtracting H x_old from the nonlinear (dJ/dx)_old
        //     f = (dJ/dx)_old - H x_old
        // such that (dJ/dx)_new = H x_new + f = (dJ/dx)_old + H (x_new - x_old)
        objective.gradient(*rhs_design, old_design_variables, tol);

        VectorPtr dx_design = new_design_variables.clone(); 
        dx_design->set(new_design_variables);
        dx_design->axpy(-one, old_design_variables);
        if ( useSecant ) {
            rhs_design_temp->set(*dx_design);
            secant.applyB(*getCtlOpt(*rhs_design_temp),*getCtlOpt(*dx_design));
            //secant.applyB(*rhs_design_temp,*dx_design);
        } else {
            //objective.hessVec(*getCtlOpt(*rhs_design_temp),*getCtlOpt(*dx_design),old_design_variables,tol);
            objective.hessVec(*rhs_design_temp,*dx_design,old_design_variables,tol);
        }
        rhs_design->axpy(one, *rhs_design_temp);
    } else if (objective_type == 1) {
        objective.gradient(*rhs_design, old_design_variables, tol);
    } else if (objective_type == 0) {
        objective.gradient(*rhs_design, new_design_variables, tol);
    }
    if (identity_factor_) {
        rhs_design->axpy(identity_factor_, new_design_variables);
    }

    if (constraint_type == 1) {
        equality_constraints.applyAdjointJacobian(*rhs_design_temp, new_dual_equality, old_design_variables, tol);
    } else if (constraint_type == 0) {
        equality_constraints.applyAdjointJacobian(*rhs_design_temp, new_dual_equality, new_design_variables, tol);
    }
    rhs_design->axpy(one, *rhs_design_temp);
    rhs_design->axpy(one, dual_inequality);
    if (objective_type == 2) pcout << "quadratic objective_type ";
    if (objective_type == 1) pcout << "linear objective_type ";
    if (objective_type == 0) pcout << "nonlinear objective_type ";
    if (constraint_type == 1) pcout << "linear constraint_type ";
    if (constraint_type == 0) pcout << "nonlinear constraint_type ";
    pcout << std::endl;

    // ********************
    // Equality constraint RHS
    // Row 6-8
    // ********************
    // Dual equality RHS
    // dual_equality = equality_constraint_value - slacks   (which is already the case for Constraint_Partitioned)
    if (constraint_type == 1) {
        VectorPtr rhs_dual_equality_temp = rhs_dual_equality->clone();
        
        // temp = g_old
        equality_constraints.value(*rhs_dual_equality_temp, old_design_variables, tol);
        // rhs = A * x_old
        equality_constraints.applyJacobian(*rhs_dual_equality, old_design_variables, old_design_variables, tol);
        // temp = g_old - A * x_old
        rhs_dual_equality_temp->axpy(-one, *rhs_dual_equality);
        // rhs = A * x_new
        equality_constraints.applyJacobian(*rhs_dual_equality, new_design_variables, old_design_variables, tol);
        // rhs = g_old + A * (x_new - x_old)
        rhs_dual_equality->axpy(one,*rhs_dual_equality_temp);
    } else {
        equality_constraints.value(*rhs_dual_equality, new_design_variables, tol);
    }

    // Dual inequality RHS
    // Note that it should be equal to 
    // (        dual_inequality      )  on the inactive set
    // (  -c(design_variables - BOUND  )  on the active set
    // The term on the active set is associated with an identity matrix multiplied by -c.
    // Therefore, we will simply evaluate (BOUND - design_variables) on the right_hand_side, and ignore -c 
    // in the system matrix

    // ********************
    // Inequality constraint RHS
    // Row 9-12
    // ********************
    rhs_dual_inequality->zero();

    get_active_design_minus_bound(*rhs_dual_inequality, new_design_variables, des_plus_dual, bound_constraints);
    auto inactive_dual_inequality = dual_inequality.clone();
    inactive_dual_inequality->set(dual_inequality);
    bound_constraints.pruneActive(*inactive_dual_inequality,des_plus_dual,neps_);
    rhs_dual_inequality->plus(*inactive_dual_inequality);

    pcout << "RHS dual_inequality1 norm: " << rhs_dual_inequality->norm() << std::endl;
    if (SYMMETRIZE_MATRIX_) {// && !(objective_type==0 && constraint_type==0)) {
        rhs_dual_inequality_temp->set(*rhs_dual_inequality);
        bound_constraints.pruneActive(*rhs_dual_inequality_temp,des_plus_dual,neps_);
        rhs_design->axpy(-one, *rhs_dual_inequality_temp);

        bound_constraints.pruneInactive(*rhs_dual_inequality,des_plus_dual,neps_);

    }



    pcout << "RHS design : " << rhs_design->norm() << std::endl;
    pcout << "RHS dual_equality : " << rhs_dual_equality->norm() << std::endl;
    pcout << "RHS dual_inequality : " << rhs_dual_inequality->norm() << std::endl;

    auto dual_des_inequality = dynamic_cast<ROL::PartitionedVector<Real>&>(*rhs_dual_inequality).get(0);
    auto dual_slack_inequality = dynamic_cast<ROL::PartitionedVector<Real>&>(*rhs_dual_inequality).get(1);
    pcout << "RHS dual_des_inequality : " << dual_des_inequality->norm() << std::endl;
    pcout << "RHS dual_slack_inequality : " << dual_slack_inequality->norm() << std::endl;

    // Multiply result by -1
    rhs_partitioned.scale(-one);
    pcout << "RHS part : " << rhs_partitioned.norm() << std::endl;
}

template<typename Real>
void PrimalDualActiveSetStep<Real>::printDesignDual (
    const std::string &code_location,
    const Vector &design_variables,
    ROL::BoundConstraint<Real> &bound_constraints,
    const Vector &dual_inequalities,
    const Vector &designs_plus_duals,
    const Vector &dual_equalities) const
{
    constexpr int printWidth = 24;
    VectorPtr active_indices_one = designs_plus_duals.clone();
    bound_constraints.setActiveEntriesToOne( *active_indices_one, designs_plus_duals );

    ROL::Ptr<const Vector> lower_bounds = bound_constraints.getLowerBound();
    ROL::Ptr<const Vector> upper_bounds = bound_constraints.getUpperBound();

    std::streamsize ss = std::cout.precision();
    std::cout << std::scientific;
    std::cout << std::showpos;
    std::cout.precision(12); 
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0) {
        std::cout << code_location << std::endl;
        std::cout << std::setw(printWidth) << std::left << "Lower Bound <";
        std::cout << std::setw(printWidth) << std::left << "Design";
        std::cout << std::setw(printWidth) << std::left << "Des + factor*dual";
        std::cout << std::setw(printWidth) << std::left << "< Upper Bound";
        std::cout << std::setw(printWidth) << std::left << "Dual inequality";
        std::cout << std::setw(printWidth) << std::left << "Active Upp(1) Low(-1)";
        std::cout << std::endl;
    }

    int design_simulation_dimension = 0;
    if (is_full_space_) {
        ROL::Ptr<const Vector> design  = dynamic_cast<const ROL::PartitionedVector<Real>&>(design_variables).get(0);
        ROL::Ptr<const Vector> design_simulation = (dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*design)).get_1();
        design_simulation_dimension = design_simulation->dimension();
    }
    for (int i = 0; i < design_variables.dimension(); i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        // Avoid printing simulation variables
        if (is_full_space_ && i < design_simulation_dimension) continue;

        const std::optional<Real> low_bound = get_value(i, *lower_bounds);
        const std::optional<Real> design_variable = get_value(i, design_variables);
        const std::optional<Real> design_plus_dual = get_value(i, designs_plus_duals);
        const std::optional<Real> upp_bound = get_value(i, *upper_bounds);
        const std::optional<Real> dual_inequality = get_value(i, dual_inequalities);
        const std::optional<Real> active = get_value(i, *active_indices_one);
        const bool isAccessible = low_bound
                                  && design_variable
                                  && design_plus_dual
                                  && upp_bound
                                  && dual_inequality
                                  && active;
        if (isAccessible) {
            std::cout << std::setw(printWidth) << std::left << *low_bound;
            std::cout << std::setw(printWidth) << std::left << *design_variable;
            std::cout << std::setw(printWidth) << std::left << *design_plus_dual;
            std::cout << std::setw(printWidth) << std::left << *upp_bound;
            std::cout << std::setw(printWidth) << std::left << *dual_inequality;
            std::cout << std::setw(printWidth) << std::left << *active;
            std::cout << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    pcout << std::endl;
    pcout << std::endl;
    pcout << "Equality dual before KKT iteration" << std::endl;
    for (int i = 0; i < dual_equalities.dimension(); i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        // Avoid printing simulation variables
        if (is_full_space_ && i < design_simulation_dimension) continue;

        const std::optional<Real> dual_equality = get_value(i, dual_equalities);
        if (dual_equality) {
            std::cout << std::setw(printWidth) << std::left << *dual_equality;
            std::cout << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    pcout << std::endl;
    pcout << std::endl;

    std::cout.precision(ss);
    std::cout << std::scientific;
    std::cout << std::noshowpos;
}

template <typename Real>
void PrimalDualActiveSetStep<Real>::printSearchDirection (
    const std::string& vector_name,
    const Vector& search_design,
    const Vector& search_dual_equality,
    const Vector& search_dual_inequality) const
{
    pcout << std::endl;
    std::streamsize ss = std::cout.precision();
    std::cout << std::scientific;
    std::cout << std::showpos;
    std::cout.precision(12);

    constexpr int printWidth = 34;
    pcout << std::setw(printWidth) << std::left << vector_name + " Design";
    pcout << std::setw(printWidth) << std::left << vector_name + " Dual Ineq.";
    pcout << std::endl;

    int design_simulation_dimension = 0;
    if (is_full_space_) {
        ROL::Ptr<const Vector> design  = dynamic_cast<const ROL::PartitionedVector<Real>&>(search_design).get(0);
        ROL::Ptr<const Vector> design_simulation = (dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*design)).get_1();
        design_simulation_dimension = design_simulation->dimension();
    }
    for (int i = 0; i < search_design.dimension(); i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        // Avoid printing simulation variables
        if (is_full_space_ && i < design_simulation_dimension) continue;

        const std::optional<Real> design  = get_value(i, search_design);
        const std::optional<Real> dual_inequality  = get_value(i, search_dual_inequality);
        if (design && dual_inequality) {
            std::cout << std::setw(printWidth) << std::left << *design;
            std::cout << std::setw(printWidth) << std::left << *dual_inequality;
            std::cout << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    pcout << std::endl;

    pcout << std::setw(printWidth) << std::left << vector_name + " Dual Eq.";
    for (int i = 0; i < search_dual_equality.dimension(); i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        // Avoid printing simulation variables
        if (is_full_space_ && i < design_simulation_dimension) continue;

        const std::optional<Real> dual_equality = get_value(i, search_dual_equality);
        if (dual_equality) {
            std::cout << std::setw(printWidth) << std::left << *dual_equality;
            std::cout << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    pcout << std::endl;
    pcout << std::endl;

    std::cout.precision(ss);
    std::cout << std::scientific;
    std::cout << std::noshowpos;
}

template<typename Real>
void printKktSystem (const ROL::Vector<Real>& rhs,
                     const ROL::LinearOperator<Real>& hessian,
                     const ROL::LinearOperator<Real>& precond)
{
    if (1 != dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) return;

    ROL::Ptr<ROL::Vector<Real>> column_of_kkt_operator = rhs.clone();
    ROL::Ptr<ROL::Vector<Real>> column_of_precond_kkt_operator = rhs.clone();

    const int rhs_size = rhs.dimension();
    dealii::FullMatrix<double> fullA(rhs_size);

    for (int i = 0; i < rhs_size; ++i) {
        std::cout << "RHS NUMBER: " << i+1 << " OUT OF " << rhs_size << ": ";
        std::cout << *get_value<Real>(i, rhs) << std::endl;
    }
    Real tol = ROL::ROL_EPSILON<Real>();
    for (int i = 0; i < rhs_size; ++i) {
        ROL::Ptr<ROL::Vector<Real>> basis = rhs.basis(i);
        MPI_Barrier(MPI_COMM_WORLD);

        hessian.apply(*column_of_kkt_operator,*basis, tol);

        const bool print_preconditionned_system = true;
        if (print_preconditionned_system) {
            precond.applyInverse(*column_of_precond_kkt_operator,*column_of_kkt_operator, tol);
            for (int j = 0; j < rhs_size; ++j) {
                fullA[j][i] = *get_value(j,*column_of_precond_kkt_operator);
            }
        } else {
            // Print KKT system
            for (int j = 0; j < rhs_size; ++j) {
                fullA[j][i] = *get_value(j,*column_of_kkt_operator);
            }
        }
    }
    std::cout<<"Dense matrix:"<<std::endl;
    fullA.print_formatted(std::cout, 10, true, 18, "0", 1., 0.);
    //std::abort();
}

// Constrained PDAS
template<typename Real>
void PrimalDualActiveSetStep<Real>::compute(
    Vector &search_direction_design,
    const Vector &design_variables,
    const Vector &dual_equality,
    ROL::Objective<Real> &objective,
    ROL::Constraint<Real> &equality_constraints, 
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    const bool DO_LINE_SEARCH = false;
    const bool DO_PROJECT_DESIGN_FEASIBLE = false;

    const bool oldSecantHessVec = useSecantHessVec_;
    if (d_force_use_bfgs) {
        useSecantHessVec_ = true;
    }

    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    Real one(1); (void) one;
    search_direction_design.zero();
    quadratic_residual_->set(*(step_state->gradientVec));


    VectorPtr new_dual_equality = dual_equality.clone();
    new_dual_equality->set(dual_equality);

    VectorPtr old_dual_equality = dual_equality.clone();
    old_dual_equality->set(dual_equality);

    // PDAS iterates through 3 steps.
    // 1. Estimate active set
    // 2. Use active set to determine search direction of the active constraints
    // 3. Solve KKT system for remaining inactive constraints

    VectorPtr rhs_design = design_variables.clone();
    VectorPtr rhs_dual_equality = dual_equality.clone();
    VectorPtr rhs_dual_inequality = design_variables.clone();

    ROL::Ptr<ROL::PartitionedVector<Real>> rhs_partitioned = ROL::makePtr<ROL::PartitionedVector<Real>>(
            std::vector<VectorPtr >( {rhs_design, rhs_dual_equality, rhs_dual_inequality}));

    VectorPtr search_design = design_variables.clone();
    VectorPtr search_dual_equality = dual_equality.clone();
    VectorPtr search_dual_inequality = design_variables.clone();

    ROL::Ptr<ROL::PartitionedVector<Real>> search_partitioned = ROL::makePtr<ROL::PartitionedVector<Real>>(
            std::vector<VectorPtr >( {search_design, search_dual_equality, search_dual_inequality}));

    VectorPtr rhs_design_temp = rhs_design->clone(); 
    VectorPtr rhs_dual_inequality_temp = rhs_dual_inequality->clone();

    Real tol = ROL::ROL_EPSILON<Real>();
    VectorPtr objective_gradient = design_variables.clone();
    objective.gradient(*objective_gradient, design_variables, tol);

    iter_Krylov_ = 0;

    Real max_eig_estimate_ = 0.0;
    const bool power_method = false;
    if (is_full_space_ && power_method) {
        auto& equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        auto flow_constraints = ROL::dynamicPtrCast<PHiLiP::FlowConstraints<PHILIP_DIM>>(equality_constraints_partitioned.get(0));

        auto& slackless_objective = dynamic_cast<ROL::SlacklessObjective<Real>&>(objective);
        auto objective_simopt = ROL::dynamicPtrCast<ROL::Objective_SimOpt<Real>>(slackless_objective.getObjective());

        auto design_sim_ctl = (dynamic_cast<const ROL::PartitionedVector<Real>&>(design_variables)).get(0);
        auto dual_equality_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(dual_equality);
        auto design_simulation = (ROL::dynamicPtrCast<const ROL::Vector_SimOpt<Real>>(design_sim_ctl))->get_1();
        auto design_control    = (ROL::dynamicPtrCast<const ROL::Vector_SimOpt<Real>>(design_sim_ctl))->get_2();
        auto dual_state = dual_equality_partitioned.get(0);

        auto design_simulation_clone = design_simulation->clone(); design_simulation_clone->set(*design_simulation);
        auto design_control_clone = design_control->clone(); design_control_clone->set(*design_control);
        auto dual_state_clone = dual_state->clone(); dual_state_clone->set(*dual_state);

        auto approximate_flow_constraints = ROL::makePtr<ApproximateJacobianFlowConstraints>( flow_constraints, design_simulation, design_control);
        (void) approximate_flow_constraints;
        //auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( objective_simopt, flow_constraints, design_simulation, design_control, dual_state);
        auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>( objective_simopt, approximate_flow_constraints, design_simulation_clone, design_control_clone, dual_state_clone);

        auto oldVec = design_control->clone();
        auto newVec = design_control->clone();

        oldVec->setScalar(1.0);

        double vec_norm = oldVec->norm();
        oldVec->scale(1.0/vec_norm);
        newVec->set(*oldVec);

        for (int i = 0; i < 500; ++i) {
            robj->hessVec(*newVec,*oldVec,*design_control,tol);

            vec_norm = newVec->norm();

            max_eig_estimate_ = newVec->dot(*oldVec) / oldVec->dot(*oldVec);
            newVec->scale(1.0/vec_norm);

            oldVec->set(*newVec);

            pcout << "max_eig_estimate_ " << max_eig_estimate_ << std::endl;
        }
        const double largest_eig_estimate = max_eig_estimate_;
        pcout << "power_method: largest_eig_estimate_ " << max_eig_estimate_ << std::endl;
        max_eig_estimate_ = std::abs(max_eig_estimate_);
        if (max_eig_estimate_ < 0) {
            max_eig_estimate_ = std::abs(max_eig_estimate_);
        } else {

            const double first_largest_positive_eig = max_eig_estimate_;
            oldVec->setScalar(1.0);
            double vec_norm = oldVec->norm();
            oldVec->scale(1.0/vec_norm);
            newVec->set(*oldVec);

            double previous_eig;
            for (int i = 0; i < 500; ++i) {
                robj->hessVec(*newVec,*oldVec,*design_control,tol);
                newVec->axpy(-first_largest_positive_eig,*oldVec);

                vec_norm = newVec->norm();

                previous_eig = max_eig_estimate_;
                max_eig_estimate_ = newVec->dot(*oldVec) / oldVec->dot(*oldVec);
                newVec->scale(1.0/vec_norm);

                oldVec->set(*newVec);

                pcout << "deflated_max_eig_estimate_ " << max_eig_estimate_ << std::endl;
            }
            pcout << "power_method: largest_eig_estimate_ " << largest_eig_estimate << std::endl;
            pcout << "power_method: deflated_eig " << max_eig_estimate_ << std::endl;
            const double max_previous_eig = previous_eig + first_largest_positive_eig;
            const double max_negative_eig = max_eig_estimate_ + first_largest_positive_eig;
            //if (max_previous_eig > max_negative_eig) {
            (void) max_previous_eig;
            if (max_negative_eig < 0) {
                max_eig_estimate_ = std::abs(max_negative_eig);
            } else {
                max_eig_estimate_ = 0.0;
            }
            //add_identity_ = 0.0;
        }
        pcout << "power_method: max_negative_eig_estimate_ " << max_eig_estimate_ << std::endl;
    }
    // POWER METHOD to find largest??
    // https://scicomp.stackexchange.com/questions/10321/largest-negative-eigenvalue
    //const Real identity_factor_ = 10.0;
    //const Real identity_factor_ = std::sqrt(algo_state.gnorm) * 3.0;
    //const Real identity_factor_ = 1.0;//algo_state.gnorm * 3.0;
    //const Real identity_factor_ = 0.0;//std::min(algo_state.gnorm * algo_state.gnorm, 0.25) * 1000.0;
    //const Real identity_factor_ = std::min(algo_state.gnorm * algo_state.gnorm, 0.25) * 100.0;//1000.0;
    //const Real identity_factor_ = std::min(std::pow(algo_state.gnorm, 1.5), 0.25) * 100.0;//1000.0;
    //const Real identity_factor_ = std::min(std::pow(algo_state.gnorm, 1.5), 0.25) * 1000.0;//1000.0;
    //const Real identity_factor = std::min(std::pow(algo_state.gnorm, 1.0), 0.25) * 100.0;//1000.0;
    //const Real identity_factor_ = std::min(algo_state.gnorm, 0.25) * 300.0;//1000.0;
    Real identity_factor = std::min(std::pow(algo_state.gnorm, 1.0), 0.25) * 10.0;//1000.0;
    //if (!is_full_space_) identity_factor /= 10.0;
    identity_factor_ = identity_factor + max_eig_estimate_;
    pcout << "identity_factor = " << identity_factor << std::endl;
    pcout << "identity_factor_ + max_eig_estimate_ = " << identity_factor_ << std::endl;

    for ( iter_PDAS_ = 0; iter_PDAS_ < maxit_; iter_PDAS_++ ) {

        /********************************************************************/
        // Modify iterate vector to check active set
        /********************************************************************/
        des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
        // The larger the scale, the harder it will try and predict the next active set.
        // However, a large scale might result in bouncing back and forth between the active constraints.
        //const Real positive_scale = 0.00001;
        const Real positive_scale = 0.01;
        // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply
        des_plus_dual_->axpy(positive_scale,*(dual_inequality_));

        auto projected_design_plus_dual = des_plus_dual_->clone();
        projected_design_plus_dual->set(*des_plus_dual_);
        bound_constraints.pruneActive( *projected_design_plus_dual, *des_plus_dual_, neps_);
        if (projected_design_plus_dual->norm() < 1e-14) {
        //if (getCtlOpt(*projected_design_plus_dual)->norm() < 1e-14) {
            // If all the bounded constraints are estimated to be active,
            // rotate between the indices of the active constraints.

            // WARNING:
            // Need to be careful in full-space
            // Can't rotate through state variables.
            projected_design_plus_dual->set(*des_plus_dual_);
            bound_constraints.projectInterior( *projected_design_plus_dual );
            const unsigned int n = des_plus_dual_->dimension();
            //const unsigned int n = getCtlOpt(design_variables)->dimension();
            const unsigned int index = index_to_project_interior % n;

            std::cout << "ALL CONSTRAINTS ACTIVE, overdefined problem. Projecting index "<< index << " in feasible region..." << std::endl;
            std::cout << "projected_design_plus_dual norm: " << projected_design_plus_dual->norm() << std::endl;

            const std::optional<Real> val = get_value(index, *projected_design_plus_dual);
            set_value(index, *val, *des_plus_dual_);
            index_to_project_interior++;
        }

        auto active_indices_one = des_plus_dual_->clone();
        bound_constraints.setActiveEntriesToOne( *active_indices_one, *des_plus_dual_ );

        printDesignDual("Before KKT iteration", *new_design_variables_, bound_constraints, *dual_inequality_, *des_plus_dual_, *new_dual_equality);

        const int objective_type = 2; // 0 = nonlinear, 1 = linear, 2 = quadratic
        const int constraint_type = 1; // 0 = nonlinear, 1 = linear
        compute_PDAS_rhs(
            design_variables,
            *new_design_variables_,
            *new_dual_equality_,
            *dual_inequality_,
            *des_plus_dual_,
            *objective_gradient,
            objective, 
            equality_constraints, 
            bound_constraints, 
            *rhs_partitioned,
            objective_type,
            constraint_type,
            *objective_secant_,
            useSecantHessVec_);//, add_identity_factor);

        pcout << "RHS norm: " << rhs_partitioned->norm() << std::endl;

        if (rhs_partitioned->norm() < gtol_*algo_state.gnorm) {
            flag_PDAS_ = 0;
            pcout << "QP problem converged because RHS norm < 1e-4 * gnorm..." << std::endl;
            break;
        }
        search_partitioned->set(*rhs_partitioned);

        // Initialize Hessian and preconditioner
        const ROL::Ptr<ROL::Objective<Real> >       objective_ptr           = ROL::makePtrFromRef(objective);
        const ROL::Ptr<ROL::Constraint<Real> >      equality_constraint_ptr = ROL::makePtrFromRef(equality_constraints);
        const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraint_ptr    = ROL::makePtrFromRef(bound_constraints);
        const ROL::Ptr<const Vector >    old_design_var_ptr      = ROL::makePtrFromRef(design_variables);
        const VectorPtr    old_dual_equality_ptr   = ROL::makePtrFromRef(*old_dual_equality);
        pcout << "Building PDAS KKT system..." << std::endl;
        ROL::Ptr<ROL::LinearOperator<Real> > hessian = ROL::makePtr<PDAS_KKT_System<Real>>(objective_ptr,
                                                                                           equality_constraint_ptr, bound_constraint_ptr,
                                                                                           old_design_var_ptr, old_dual_equality_ptr,
                                                                                           des_plus_dual_,
                                                                                           identity_factor_, neps_,
                                                                                           //objective_secant_, useSecantHessVec_);
                                                                                           lagrangian_secant_, useSecantHessVec_);
        ROL::Ptr<ROL::LinearOperator<Real> > precond = ROL::makePtr<Identity_Preconditioner<Real>>();

        ROL::Ptr<ROL::Constraint_Partitioned<Real> >  other_equality_constraints_ptr;
        ROL::Ptr<const ROL::PartitionedVector<Real> >  other_dual_equality_partitioned_ptr;
        if (is_full_space_) {
            // Get equality FlowConstraint
            const ROL::Ptr<ROL::Constraint_Partitioned<Real> >  equality_constraint_partitioned_ptr = ROL::dynamicPtrCast<ROL::Constraint_Partitioned<Real>>(equality_constraint_ptr);
            const ROL::Ptr<ROL::Constraint<Real> >  state_constraints_ptr = equality_constraint_partitioned_ptr->get(0);

            pcout << "Build constraints..." << std::endl;
            // Get remaining constraints
            const unsigned int n_equality_constraints = equality_constraint_partitioned_ptr->get_n_constraints();
            std::vector<ROL::Ptr<ROL::Constraint<Real> >> remaining_equality_constraints(n_equality_constraints-1);
            std::vector<bool> remaining_is_inequality(n_equality_constraints-1);
            for (unsigned int i = 1; i < n_equality_constraints; ++i) {
                remaining_equality_constraints[i-1] = equality_constraint_partitioned_ptr->get(i);
                remaining_is_inequality[i-1] = equality_constraint_partitioned_ptr->isInequality_[i];
            }
            other_equality_constraints_ptr = ROL::makePtr<ROL::Constraint_Partitioned<Real>>(remaining_equality_constraints, remaining_is_inequality);

            pcout << "Build dual vectors..." << std::endl;
            const ROL::Ptr<ROL::PartitionedVector<Real> >  old_dual_equality_partitioned_ptr = ROL::dynamicPtrCast<ROL::PartitionedVector<Real>>(old_dual_equality_ptr);
            const ROL::Ptr<const Vector>   dual_state_ptr = old_dual_equality_partitioned_ptr->get(0);

            const unsigned int n_dual_vecs = old_dual_equality_partitioned_ptr->numVectors();
            std::vector<VectorPtr> other_dual_equality_vectors(n_dual_vecs-1);
            for (unsigned int i_vec = 1; i_vec < n_dual_vecs; ++i_vec) {
                other_dual_equality_vectors[i_vec-1] = old_dual_equality_partitioned_ptr->get(i_vec);
            }
            other_dual_equality_partitioned_ptr = ROL::makePtr<ROL::PartitionedVector<Real>>(other_dual_equality_vectors);

            auto &slackless_objective = dynamic_cast<ROL::SlacklessObjective<Real> &>(*objective_ptr);
            auto &objective_simopt = dynamic_cast<ROL::Objective_SimOpt<Real> &>(*(slackless_objective.getObjective()));
            pcout << "Build PDAS_P24_Constrained_Preconditioner..." << std::endl;
            precond = ROL::makePtr<PDAS_P24_Constrained_Preconditioner<Real>>(
                old_design_var_ptr,
                ROL::makePtrFromRef(objective_simopt),
                state_constraints_ptr,
                dual_state_ptr,
                other_equality_constraints_ptr,
                other_dual_equality_partitioned_ptr,
                bound_constraint_ptr,
                old_dual_inequality_,
                des_plus_dual_,
                neps_,
                objective_secant_
                );
        }

        pcout << "old_design_variables norm " << design_variables.norm() << std::endl;
        pcout << "new_design_variables_ norm " << new_design_variables_->norm() << std::endl;

        bool print_KKT = (algo_state.iter < 0);
        if (print_KKT) {
            printKktSystem(*search_partitioned, *hessian, *precond);
        }

        auto &gmres = dynamic_cast<ROL::GMRES<Real>&>(*krylov_);
        if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0) gmres.enableOutput(std::cout);
        int iter_Krylov_this_iteration = 0;
        pcout << "right before gmres.run " << std::endl;
        gmres.run(*search_partitioned,*hessian,*rhs_partitioned,*precond,iter_Krylov_this_iteration,flag_Krylov_);
        pcout << "right after gmres.run " << std::endl;
        iter_Krylov_ += iter_Krylov_this_iteration;

        if (SYMMETRIZE_MATRIX_) {
            rhs_dual_inequality_temp->set(*dual_inequality_);
            bound_constraints.pruneActive(*rhs_dual_inequality_temp, *des_plus_dual_, tol);

            bound_constraints.pruneInactive(*search_dual_inequality, *des_plus_dual_, tol);
            search_dual_inequality->axpy(-one, *rhs_dual_inequality_temp);

            auto search_design_temp = rhs_dual_inequality->clone();
            search_design_temp->set(*rhs_dual_inequality);
            bound_constraints.pruneInactive(*search_design_temp, *des_plus_dual_, tol);
            bound_constraints.pruneActive(*search_design, *des_plus_dual_, tol);
            search_design->axpy(one, *search_design_temp);
        }
        pcout << "QP Search direction: " << std::endl;
        printSearchDirection("Search", *search_design, *search_dual_equality, *search_dual_inequality);

        pcout << "Search norm design: " << search_design->norm() << std::endl;
        pcout << "Search norm equality: " << search_dual_equality->norm() << std::endl;
        pcout << "Search norm inequality: " << search_dual_inequality->norm() << std::endl;
        pcout << "Search norm: " << search_partitioned->norm() << std::endl;

        // Check that inactive dual inequality equal to 0
        {
            rhs_dual_inequality_temp->set(*search_dual_inequality);
            rhs_dual_inequality_temp->axpy(one,*dual_inequality_);
            bound_constraints.pruneActive(*rhs_dual_inequality_temp,*des_plus_dual_,tol);
            Real inactive_dual_inequality_norm = rhs_dual_inequality_temp->norm();
            if (inactive_dual_inequality_norm != 0.0) {
                std::cout << "Inactive dual_inequality is not zero" << std::endl;
                std::abort();
            }
        }

        if (DO_LINE_SEARCH) {
            /********************************************************************/
            // UPDATE STEP 
            /********************************************************************/
            // NOTE: Using a linesearch here ruins the 1 KKT iteration of the Newton-based solver
            // on the the quadratic objective linear constraint test case.
            int max_linesearches = 100;
            double steplength = 1.0;
            double linesearch_factor = 0.8;
            VectorPtr design_copy = new_design_variables_->clone();
            design_copy->set(*new_design_variables_);
            for (int i_linesearch = 0; i_linesearch < max_linesearches; ++i_linesearch) {
                if (i_linesearch > 0) steplength *= linesearch_factor;

                new_design_variables_->set(*design_copy);
                new_design_variables_->axpy(steplength, *search_design);

                rhs_design_temp->set(*new_design_variables_);
                //printVec(*new_design_variables_);
                bound_constraints.project(*new_design_variables_);
                //printVec(*new_design_variables_);

                rhs_design_temp->axpy(-one, *new_design_variables_);

                // x <- f(x) = { 0      if x == 0
                //             { 1      otherwise
                class SetNonZeroToOne : public ROL::Elementwise::UnaryFunction<Real> {
                public:
                Real apply( const Real &x ) const {
                    const Real zero(0);
                    return (x == zero) ? 0 : 1.0;
                }
                }; // class SetNonZeroToOne
                SetNonZeroToOne unary;
                rhs_design_temp->applyUnary(unary);

                Real difference_norm = rhs_design_temp->norm();
                int number_of_violating_constraints = std::round(difference_norm * difference_norm);
                pcout << i_linesearch << " n_constraint violated = " << number_of_violating_constraints << std::endl;
                if (number_of_violating_constraints <= 2) break;
                //if (rhs_design_temp->norm() == 0.0) break;
            }
            // new_dual_equality_->axpy(steplength, *search_dual_equality);
            // dual_inequality_->axpy(steplength, *search_dual_inequality);
            pcout << "QP Linesearch resulting in steplength of " << steplength << std::endl;
        } else {
            new_design_variables_->plus(*search_design);
        }
        new_dual_equality_->plus(*search_dual_equality);
        dual_inequality_->plus(*search_dual_inequality);

        if ( DO_PROJECT_DESIGN_FEASIBLE && bound_constraints.isActivated() ) {
            bound_constraints.project(*new_design_variables_);
        }

        //const bool DO_CHECK_ACTIVE_DUAL = true;
        //if (DO_CHECK_ACTIVE_DUAL) {
        //    auto old_dual_inequality = dual_inequality_->clone();
        //    old_dual_inequality->set(*dual_inequality_);
        //    old_dual_inequality->axpy(-one, search_dual_inequality_);


        //    auto dual_inequality_temp = dual_inequality_->clone();
        //    dual_inequality_temp->set(
        //}

        //quadratic_residual_->set(*(step_state->gradientVec));
        //quadratic_residual_->plus(*rhs_design_temp);

        //  // Double check that the result matches
        //  search_direction_active_set_->zero();                                     // active_set_search_direction   = 0
        //  search_temp_->set(*bound_constraints.getUpperBound());                    // search_tmp = upper_bound
        //  search_temp_->axpy(-one,*new_design_variables_);                          // search_tmp = upper_bound - design_variables
        //  desvar_tmp_->set(*search_temp_);                                          // tmp        = upper_bound - design_variables
        //  bound_constraints.pruneUpperActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = (upper_bound - (upper_bound - design_variables + c*dual_variables)) < 0 ? 0 : upper_bound - design_variables
        //  search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(upper_bound - design_variables)
        //  search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(upper_bound - design_variables)
        //  search_temp_->set(*bound_constraints.getLowerBound());                    // search_tmp = lower_bound
        //  search_temp_->axpy(-one,*new_design_variables_);                          // search_tmp = lower_bound - design_variables
        //  desvar_tmp_->set(*search_temp_);                                          // tmp        = lower_bound - design_variables
        //  bound_constraints.pruneLowerActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = INACTIVE(lower_bound - design_variables)
        //  search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(lower_bound - design_variables)
        //  search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(lower_bound - design_variables)
        //  if (search_direction_active_set_->norm() > 1e-10) {
        //      pcout << " des - bound is not 0 on the active set " << search_direction_active_set_->norm() << std::endl;
        //      pcout << "search_direction_active_set_: " << std::endl;
        //      for (int i = 0; i < search_direction_active_set_->dimension(); ++i) {
        //          pcout << get_value<Real>(i, *search_direction_active_set_) << std::endl;
        //      }
        //      std::abort();
        //  }


        if ( search_partitioned->norm() < gtol_*algo_state.gnorm ) {
            flag_PDAS_ = 0;
            pcout << "QP problem converged because RHS norm < 1e-4 * gnorm..." << std::endl;
            break;
        }
        //if ( search_partitioned.norm() < stol_*design_variables.norm() ) {
        //    flag_PDAS_ = 2;
        //    break;
        //} 

        des_plus_dual_->set(*new_design_variables_);
        des_plus_dual_->axpy(positive_scale,*(dual_inequality_));

        // If active set didn't change, exit QP.
        auto new_active_indices_one = des_plus_dual_->clone();
        bound_constraints.setActiveEntriesToOne( *new_active_indices_one, *des_plus_dual_ );
        new_active_indices_one->axpy(-one, *active_indices_one);
        if ( new_active_indices_one->norm() < 1e-10 ) {
            flag_PDAS_ = 0;
            pcout << "QP problem converged active set did not change..." << std::endl;
            break;
        }

    }
    if ( iter_PDAS_ == maxit_ ) {
        flag_PDAS_ = 1;
    } else {
        iter_PDAS_++;
    }
    if (d_force_use_bfgs) {
        useSecantHessVec_ = oldSecantHessVec;
    }
}

template<typename Real>
void PrimalDualActiveSetStep<Real>::compute(
    Vector &search_direction_design,
    const Vector &design_variables,
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    (void) search_direction_design;
    (void) design_variables;
    (void) objective;
    (void) bound_constraints;
    (void) algo_state;
}
  
template<typename Real>
void PrimalDualActiveSetStep<Real>::update(
    Vector &design_variables,
    const Vector &search_direction_design,
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints,
    ROL::AlgorithmState<Real> &algo_state )
{
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    step_state->SPiter = (maxit_ > 1) ? iter_PDAS_ : iter_Krylov_;
    step_state->SPflag = (maxit_ > 1) ? flag_PDAS_ : flag_Krylov_;

    design_variables.plus(search_direction_design);
    feasible_ = bound_constraints.isFeasible(design_variables);
    algo_state.snorm = search_direction_design.norm();
    algo_state.iter++;
    //if (algo_state.iter > 5) {
    //  useSecantHessVec_ = false;
    //}
    Real tol = std::sqrt(ROL::ROL_EPSILON<Real>());
    objective.update(design_variables,true,algo_state.iter);
    algo_state.value = objective.value(design_variables,tol);
    algo_state.nfval++;
    
    if ( objective_secant_ != ROL::nullPtr ) {
      gradient_tmp1_->set(*(step_state->gradientVec));
    }
    algo_state.gnorm = computeCriticalityMeasure(design_variables,objective,bound_constraints,tol);
    algo_state.ngrad++;

    if ( objective_secant_ != ROL::nullPtr ) {
      objective_secant_->updateStorage(design_variables,*(step_state->gradientVec),*gradient_tmp1_,search_direction_design,algo_state.snorm,algo_state.iter+1);
    }
    (algo_state.iterateVec)->set(design_variables);
}

template<typename Real>
class PDAS_Lagrangian: public ROL::AugmentedLagrangian<Real>
{
    using Vector = ROL::Vector<Real>;
    using VectorPtr = ROL::Ptr<Vector>;
    VectorPtr active_des_minus_bnd;
    VectorPtr des_plus_dual_;
    const Vector &dual_inequality;
    ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;

    private:
        Real penaltyParameter_;

public:
    PDAS_Lagrangian(const ROL::Ptr<ROL::Objective<Real> > &obj,
                    const ROL::Ptr<ROL::Constraint<Real> > &con,
                    const Vector &multiplier,
                    const Real penaltyParameter,
                    const Vector &optVec,
                    const Vector &conVec,
                    ROL::ParameterList &parlist,
                    const Vector &inequality_multiplier,
                    const ROL::Ptr<ROL::BoundConstraint<Real>> &bound_constraints)
    : ROL::AugmentedLagrangian<Real>(obj, con, multiplier, penaltyParameter, optVec, conVec, parlist)
    , dual_inequality(inequality_multiplier)
    , bound_constraints_(bound_constraints)
    , penaltyParameter_(penaltyParameter)
    {
        active_des_minus_bnd = optVec.clone();
        des_plus_dual_ = optVec.clone();
    }
    virtual Real value( const Vector &x, Real &tol ) override
    {
        Real val = ROL::AugmentedLagrangian<Real>::value(x,tol);
        std::cout << " AugmentedLagrangian::value()  " << val << std::endl;
        const Real positive_scale = 0.01;
        des_plus_dual_->set(x);
        des_plus_dual_->axpy(positive_scale,dual_inequality);
        get_active_design_minus_bound(*active_des_minus_bnd, x, *des_plus_dual_, *bound_constraints_);
        //get_active_design_minus_bound(*active_des_minus_bnd, x, x, *bound_constraints_);
        val += dual_inequality.dot(*active_des_minus_bnd);
        std::cout << " dual_inequality.dot(*active_des_minus_bnd) " << dual_inequality.dot(*active_des_minus_bnd) << std::endl;
        std::cout << " AugmentedLagrangian::value() + dual_inequality*(des-bound)" << val << std::endl;
        val += 0.5 * penaltyParameter_ * active_des_minus_bnd->dot(*active_des_minus_bnd);
        std::cout << " AugmentedLagrangian::value() + dual_inequality*(des-bound) + 0.5*pen*(des-bound)" << val << std::endl;
        std::cout << "0.5 * penaltyParameter_ * active_des_minus_bnd->dot(*active_des_minus_bnd) " << 0.5 * penaltyParameter_ * active_des_minus_bnd->dot(*active_des_minus_bnd) << std::endl;
        std::cout << " PDAS_Lagrangian::value()  " << val << std::endl;
        return val;
    }
    virtual void gradient( Vector &g, const Vector &x, Real &tol ) override
    {
        ROL::AugmentedLagrangian<Real>::gradient( g, x, tol );

        auto temp_dual_inequality = dual_inequality.clone();
        temp_dual_inequality->set(dual_inequality);
        
        const Real neps = ROL::ROL_EPSILON<Real>();
        bound_constraints_->pruneInactive(*temp_dual_inequality,x,neps);

        const Real positive_scale = 0.01;
        des_plus_dual_->set(x);
        des_plus_dual_->axpy(positive_scale,dual_inequality);
        get_active_design_minus_bound(*active_des_minus_bnd, x, *des_plus_dual_, *bound_constraints_);
        //get_active_design_minus_bound(*active_des_minus_bnd, x, x, *bound_constraints_);
        temp_dual_inequality->axpy(penaltyParameter_, *active_des_minus_bnd);

        g.plus(*temp_dual_inequality);
    }
    // Reset with upated penalty parameter
    virtual void reset(const Vector &multiplier, const Real penaltyParameter) {
        ROL::AugmentedLagrangian<Real>::reset(multiplier, penaltyParameter);
        penaltyParameter_ = penaltyParameter;
    }
};

template<typename Real>
void PrimalDualActiveSetStep<Real>::update(
    Vector &design_variables,
    Vector &dual_equality,
    const Vector &search_direction_design,
    ROL::Objective<Real> &objective,
    ROL::Constraint<Real> &equality_constraints,
    ROL::BoundConstraint<Real> &bound_constraints,
    ROL::AlgorithmState<Real> &algo_state )
{
    //const Real positive_scale = 0.0001;
    const Real positive_scale = 0.01;
    des_plus_dual_->set(design_variables);
    des_plus_dual_->axpy(positive_scale,*(dual_inequality_));
    printDesignDual("Before update", design_variables, bound_constraints, *old_dual_inequality_, *des_plus_dual_, dual_equality);

    (void) search_direction_design;
    const double one = 1;
    Real tol = ROL::ROL_EPSILON<Real>();
    Real sqrttol = std::sqrt(tol);

    const bool DO_DUAL_UPDATE_FIRST = true;
    const bool DO_PROJECT_DESIGN_FEASIBLE = true;
    const bool DO_EVALUATE_NONLINEAR_SLACKS = true;
    const bool DO_SCALE_DUAL_SEARCH_STEP = false;
    const bool DO_ZERO_MASS_LINESEARCH = true;

    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    step_state->SPiter = (maxit_ > 1) ? iter_PDAS_ : iter_Krylov_;
    step_state->SPflag = (maxit_ > 1) ? flag_PDAS_ : flag_Krylov_;

    VectorPtr dual_times_equality_jacobian = design_variables.clone(); 

    VectorPtr old_design = design_variables.clone();
    old_design->set(design_variables);

    auto search_design = new_design_variables_->clone();
    search_design->set(*new_design_variables_);
    search_design->axpy(-one, design_variables);

    auto search_dual_equality = new_dual_equality_->clone();
    search_dual_equality->set(*new_dual_equality_);
    search_dual_equality->axpy(-one, dual_equality);

    auto search_dual_inequality = dual_inequality_->clone();
    search_dual_inequality->set(*dual_inequality_);
    search_dual_inequality->axpy(-one, *old_dual_inequality_);
    dual_inequality_->set(*old_dual_inequality_);

    if (DO_DUAL_UPDATE_FIRST) {
        dual_equality.plus(*search_dual_equality);
        dual_inequality_->plus(*search_dual_inequality);

        // Save current gradient as previous gradient.
        // WARNING: NEEDS TO BE DONE AFTER DUAL UPDATE, BUT BEFORE DESIGN UPDATE
        objective.gradient(*old_gradient_,design_variables,sqrttol);
        equality_constraints.applyAdjointJacobian(*dual_times_equality_jacobian, dual_equality, design_variables, sqrttol);
        old_gradient_->axpy(one, *dual_times_equality_jacobian);
        //old_gradient_->axpy(one, *dual_inequality_);
    }

    double old_CFL = 0.0;
    if (DO_ZERO_MASS_LINESEARCH && is_full_space_) {
        auto& equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        auto& flow_constraints = dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equality_constraints_partitioned.get(0));
        old_CFL = flow_constraints.flow_CFL_;
        flow_constraints.flow_CFL_ = 0.0;
    }

    bool linesearch_success = false;
    Real fold = 0.0;
    int n_searches = 0;
    const int max_n_searches = 0;
    Real merit_function_value = 0.0;
    //double penalty_value_ = 10.0/algo_state.gnorm;
    //double penalty_value_ = 10.0/(algo_state.gnorm);
    //double penalty_value_ = 0.0;
    //double penalty_value_ = 1.0;
    double penalty_value_ = 1.0e-1;

    auto merit_function = ROL::makePtr<PHiLiP::PDAS_Lagrangian<Real>> (
            ROL::makePtrFromRef<ROL::Objective<Real>>(objective),
            ROL::makePtrFromRef<ROL::Constraint<Real>>(equality_constraints),
            dual_equality,
            penalty_value_,
            design_variables,
            *(step_state->constraintVec),
            parlist_,
            *dual_inequality_,
            ROL::makePtrFromRef<ROL::BoundConstraint<Real>>(bound_constraints));
    (void) merit_function;

    const bool is_changed_design_variables = true;
    merit_function->update(design_variables, is_changed_design_variables, algo_state.iter);
    merit_function->reset(dual_equality, penalty_value_);
    merit_function_value = merit_function->value(design_variables, tol );
    std::cout << "old_lagrangian_value: " << merit_function_value << std::endl;

    //merit_function->reset(dual_equality, penalty_value_);
    //merit_function->reset(dual_equality, 0.0);
    //merit_function->update(design_variables, is_changed_design_variables, algo_state.iter);

    auto lineSearch_ = ROL::LineSearchFactory<Real>(parlist_);
    lineSearch_->initialize(design_variables, *search_design, *(step_state->gradientVec), *merit_function, bound_constraints);

    VectorPtr merit_function_gradient = design_variables.clone();
    kkt_linesearches_ = 0;
    pcout << "Nonlinear Search direction: " << std::endl;
    printSearchDirection("Before Line Search", *search_design, *search_dual_equality, *search_dual_inequality);
    while (!linesearch_success) {

        merit_function->reset(dual_equality, penalty_value_);
        merit_function->update(design_variables, is_changed_design_variables, algo_state.iter);
        merit_function_value = merit_function->value(design_variables, tol );

        merit_function->gradient( *merit_function_gradient, design_variables, tol );

        fold = merit_function_value;
        Real directional_derivative_step = merit_function_gradient->dot(*search_design);
        directional_derivative_step += step_state->constraintVec->dot(*search_dual_equality);

        /* Perform line-search */
        pcout
            << "Performing line search..."
            << " Initial merit function value = " << merit_function_value
            << std::endl;
        Real neps = ROL::ROL_EPSILON<Real>();
        lineSearch_->setData(neps,*merit_function_gradient);
        //lineSearch_->setData(algo_state.gnorm,*merit_function_gradient);

        int n_linesearches = 0;
        pcout << "step_state->searchSize " << step_state->searchSize << std::endl;
        pcout << "merit_function_value " << merit_function_value << std::endl;
        pcout << "n_linesearches " << n_linesearches << std::endl;
        pcout << "directional_derivative_step " << directional_derivative_step << std::endl;
        pcout << "search_design->norm() " << search_design->norm() << std::endl;
        pcout << "design_variables.norm() " << design_variables.norm() << std::endl;
        lineSearch_->run(step_state->searchSize,
                         merit_function_value,
                         n_linesearches,
                         step_state->ngrad,
                         directional_derivative_step,
                         *search_design,
                         design_variables,
                         *merit_function,
                         bound_constraints);
        //bound_constraints.activate();
        step_state->nfval += n_linesearches;
        const int max_line_searches = parlist_.sublist("Step").sublist("Line Search").get("Function Evaluation Limit",20);
        if (n_linesearches < max_line_searches) {
            linesearch_success = true;
            pcout
                << "End of line search... searchSize is..."
                << step_state->searchSize
                << " and number of function evaluations: "
                << step_state->nfval
                << " and n_linesearches: "
                << n_linesearches
                << " Max linesearches : " << max_line_searches
                << " Final merit function value = " << merit_function_value
                << std::endl;
        } else {
            n_searches++;
            Real penalty_reduction = 0.1;
            pcout
                << " Max linesearches achieved: " << max_line_searches
                << " Current merit_function_value value = " << merit_function_value
                << " Reducing penalty value from " << penalty_value_
                << " to " << penalty_value_ * penalty_reduction
                << std::endl;
            penalty_value_ = penalty_value_ * penalty_reduction;
            //penalty_value_ = std::sqrt(penalty_value_);

            //linesearch_success = true;
            if (n_searches > max_n_searches) {

                linesearch_success = true;
            }
        }
        kkt_linesearches_ += n_linesearches;
    }
    if (n_searches > max_n_searches) {
        // Unsuccessful line search
        if (d_force_use_bfgs) {
            // Unsuccessful backup BFGS
            d_force_use_bfgs = false;
            (void) fold;
            lineSearch_->setMaxitUpdate(step_state->searchSize, merit_function_value, fold);
            std::cout << "Failed the linesearch with the backup BFGS step. Taking minimizing step of " << step_state->searchSize << std::endl;
            ROL::Ptr<ROL::SecantState<Real>> secant_state = lagrangian_secant_->get_state();
            secant_state->iterDiff.clear();
            secant_state->gradDiff.clear();
            secant_state->product.clear();
            secant_state->current = -1;
            std::cout << "Also resetting BFGS approximation" << std::endl;
            //step_state->searchSize = 0.1;
            //std::cout << "Failed the linesearch with the BFGS step. Taking a constant step of 0.1" << std::endl;
        } else if (useSecantHessVec_) {
            // Unsuccessful BFGS
            d_force_use_bfgs = false;
            (void) fold;
            lineSearch_->setMaxitUpdate(step_state->searchSize, merit_function_value, fold);

            std::cout << "Failed the linesearch with the BFGS step. Taking minimizing step of " << step_state->searchSize << std::endl;

            //ROL::Ptr<ROL::SecantState<Real>> secant_state = lagrangian_secant_->get_state();
            //secant_state->iterDiff.clear();
            //secant_state->gradDiff.clear();
            //secant_state->product.clear();
            //secant_state->current = -1;
            //std::cout << "Also resetting BFGS approximation" << std::endl;
        } else {
            // Unsuccessful BFGS
            // Unsuccessful Newton
            std::cout << "Failed the linesearch with the Newton step. Trying again with a BFGS step" << std::endl;
            d_force_use_bfgs = true;
            step_state->searchSize = 0.0;
            if (DO_DUAL_UPDATE_FIRST) {
                dual_equality.axpy(-one, *search_dual_equality);
                dual_inequality_->axpy(-one, *search_dual_inequality);
            }
            search_dual_equality->scale(step_state->searchSize);
            search_dual_inequality->scale(step_state->searchSize);
        }
    } else {
        if (d_force_use_bfgs) {
            // Successful BFGS
            d_force_use_bfgs = false;
        } else {
            // Successful Newton
        }
    }

    if (DO_ZERO_MASS_LINESEARCH && is_full_space_) {
        auto& equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        auto& flow_constraints = dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equality_constraints_partitioned.get(0));
        flow_constraints.flow_CFL_ = old_CFL;
    }

    auto s_design = dynamic_cast<ROL::PartitionedVector<Real>&>(*search_design).get(0);
    auto s_slack  = dynamic_cast<ROL::PartitionedVector<Real>&>(*search_design).get(1);
    //s_design->scale(step_state->searchSize);
    search_design->scale(step_state->searchSize);
    if (DO_SCALE_DUAL_SEARCH_STEP) {
        search_dual_equality->scale(step_state->searchSize);
        search_dual_inequality->scale(step_state->searchSize);
    }
    pcout << "searchSize " << step_state->searchSize << std::endl;
    pcout << "search_design.norm() " << search_design->norm() << std::endl;
    pcout << "search_dual_equality.norm() " << search_dual_equality->norm() << std::endl;
    pcout << "search_dual_inequality.norm() " << search_dual_inequality->norm() << std::endl;

    if ( bound_constraints.isActivated() && DO_PROJECT_DESIGN_FEASIBLE) {
        search_design->plus(design_variables);
        bound_constraints.project(*search_design);
        search_design->axpy(static_cast<Real>(-1),design_variables);
    }

    if (!DO_DUAL_UPDATE_FIRST) {
        dual_equality.plus(*search_dual_equality);
        dual_inequality_->plus(*search_dual_inequality);
        // Save current gradient as previous gradient.
        // WARNING: NEEDS TO BE DONE AFTER DUAL UPDATE, BUT BEFORE DESIGN UPDATE
        objective.gradient(*old_gradient_,design_variables,sqrttol);
        equality_constraints.applyAdjointJacobian(*dual_times_equality_jacobian, dual_equality, design_variables, sqrttol);
        old_gradient_->axpy(one, *dual_times_equality_jacobian);
        //old_gradient_->axpy(one, *dual_inequality_);
    }
    design_variables.plus(*search_design);

    pcout << "Nonlinear Search direction: " << std::endl;
    printSearchDirection("After Line Search", *search_design, *search_dual_equality, *search_dual_inequality);

    //merit_function->reset(dual_equality, 0.0);
    merit_function->update(design_variables, is_changed_design_variables, algo_state.iter);
    merit_function_value = merit_function->value(design_variables, tol );
    pcout << "new_lagrangian_value: " << merit_function_value << std::endl;

    //design_variables.plus(*search_design);
    const bool localFeasible = bound_constraints.isFeasible(design_variables);
    MPI_Allreduce(&localFeasible, &feasible_, 1, MPI::BOOL, MPI::LAND, MPI_COMM_WORLD);
    algo_state.snorm = search_design->norm();
    algo_state.snorm += search_dual_equality->norm();
    algo_state.snorm += search_dual_inequality->norm();
    algo_state.iter++;
    if (d_force_use_bfgs) {
        algo_state.snorm = 1e99;
    }
    

    // Update objective
    objective.update(design_variables,true,algo_state.iter);
    algo_state.value = objective.value(design_variables,sqrttol);
    algo_state.nfval++;

    // Update constraints
    equality_constraints.update(design_variables,true,algo_state.iter);
    equality_constraints.value(*(step_state->constraintVec),design_variables,sqrttol);

    if (is_full_space_) {
        ROL::Constraint_Partitioned<Real> &equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        PHiLiP::FlowConstraints<PHILIP_DIM> &flow_constraints = dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equality_constraints_partitioned.get(0));
        VectorPtr input_design_sim_ctl = (dynamic_cast<ROL::PartitionedVector<Real>&>(design_variables)).get(0);
        VectorPtr input_design_simulation = (dynamic_cast<ROL::Vector_SimOpt<Real>&>(*input_design_sim_ctl)).get_1();
        VectorPtr input_design_control    = (dynamic_cast<ROL::Vector_SimOpt<Real>&>(*input_design_sim_ctl)).get_2();
        flowcnorm_ = flow_constraints.dg_l2_norm(*input_design_simulation, *input_design_control);
        //flowcnorm_ = dynamic_cast<ROL::PartitionedVector<Real>&>(*(step_state->constraintVec)).get(0)->norm();
        if (flowcnorm_ > 1e-7) {
            flow_constraints.solve( *((dynamic_cast<ROL::PartitionedVector<Real>&>(*(step_state->constraintVec))).get(0)),
                                    *input_design_simulation,
                                    *input_design_control,
                                    flowcnorm_);
        }

    } else {
        flowcnorm_ = sqrttol;
    }

    if (DO_EVALUATE_NONLINEAR_SLACKS) {

        ROL::PartitionedVector<Real>& design_partitioned  = dynamic_cast<ROL::PartitionedVector<Real>&>(design_variables);

        VectorPtr design  = design_partitioned.get(0);
        const VectorPtr current_design = design->clone();
        current_design->set(*design);

        const ROL::PartitionedVector<Real>& nonlinear_equality_constraints = dynamic_cast<ROL::PartitionedVector<Real>&>(*(step_state->constraintVec));

        if (is_full_space_) {
            for (unsigned int i = 1; i < nonlinear_equality_constraints.numVectors(); ++i) {
                ROL::Ptr<const Vector> constraint = nonlinear_equality_constraints.get(i); // constraint vec starts at 1, because cvec[0] is FlowConstraints
                VectorPtr slack  = design_partitioned.get(i); // design_partitioned = [design, slack0, slack1, ...]
                slack->axpy(one,*constraint); // s = s + (g-s)
            }
        } else {
            for (unsigned int i = 0; i < nonlinear_equality_constraints.numVectors(); ++i) {
                ROL::Ptr<const Vector> constraint = nonlinear_equality_constraints.get(i); // constraint vec starts at 0
                VectorPtr slack  = design_partitioned.get(i+1); // design_partitioned = [design, slack0, slack1, ...]
                slack->axpy(one,*constraint); // s = s + (g-s)
            }
        }

        // Project all variables within bound, but reset the nonslack variables to their original value such that they are not projected into the bounds.
        // This effectively only projects the slack variables.
        bound_constraints.project(design_variables);
        design->set(*current_design);
    }
    equality_constraints.value(*(step_state->constraintVec),design_variables,sqrttol);

    // Evaluate constraint norm
    if (is_full_space_) {
        const ROL::PartitionedVector<Real>& nonlinear_equality_constraints = dynamic_cast<ROL::PartitionedVector<Real>&>(*(step_state->constraintVec));
        ecnorm_ = flowcnorm_ * flowcnorm_;
        for (unsigned int i = 1; i < nonlinear_equality_constraints.numVectors(); ++i) {
            ROL::Ptr<const Vector> constraint = nonlinear_equality_constraints.get(i); // constraint vec starts at 1, because cvec[0] is FlowConstraints
            const Real constraint_norm = constraint->norm();
            ecnorm_ += constraint_norm * constraint_norm;
        }
        ecnorm_ = sqrt(ecnorm_);
    } else {
        ecnorm_ = (step_state->constraintVec)->norm();
    }
    algo_state.cnorm = ecnorm_;

    auto active_set_des_min_bnd = design_variables.clone();
    get_active_design_minus_bound(*active_set_des_min_bnd, design_variables, design_variables, bound_constraints);
    icnorm_ = active_set_des_min_bnd->norm();
    algo_state.cnorm += icnorm_;
    algo_state.ncval++;

    des_plus_dual_->set(design_variables);
    des_plus_dual_->axpy(positive_scale,*(dual_inequality_));

    VectorPtr rhs_design = design_variables.clone();
    VectorPtr rhs_dual_equality = dual_equality.clone();
    VectorPtr rhs_dual_inequality = design_variables.clone();

    ROL::Ptr<ROL::PartitionedVector<Real>> rhs_partitioned
        = ROL::makePtr<ROL::PartitionedVector<Real>>(
            std::vector<VectorPtr >(
                {rhs_design, rhs_dual_equality, rhs_dual_inequality}
            )
          );
    const int objective_type = 0; // 0 = nonlinear, 1 = linear, 2 = quadratic
    const int constraint_type = 0; // 0 = nonlinear, 1 = linear

    if (is_full_space_) {
        auto& equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        auto& flow_constraints = dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equality_constraints_partitioned.get(0));
        old_CFL = flow_constraints.flow_CFL_;
        flow_constraints.flow_CFL_ = 0.0;
    }
    compute_PDAS_rhs(
        design_variables,
        design_variables,
        dual_equality,
        *dual_inequality_,
        *des_plus_dual_,
        *(step_state->gradientVec),
        objective, 
        equality_constraints, 
        bound_constraints, 
        *rhs_partitioned,
        objective_type,
        constraint_type,
        *objective_secant_,
        useSecantHessVec_);
    printSearchDirection("RHS", *rhs_design, *rhs_dual_equality, *rhs_dual_inequality);

    {
        class SetNonZeroToOne : public ROL::Elementwise::UnaryFunction<Real> {
        public:
            Real apply( const Real &x ) const {
                const Real zero(0);
                return (x == zero) ? 0 : 1.0;
            }
        }; // class SetNonZeroToOne
        SetNonZeroToOne unary;
        auto temp_design = design_variables.clone();
        temp_design->set(design_variables);
        temp_design->applyUnary(unary);
        bound_constraints.pruneInactive(*temp_design, *des_plus_dual_, tol);
        Real difference_norm = temp_design->norm();
        n_active_ = std::round(difference_norm * difference_norm);
    }

    if (is_full_space_) {
        auto& equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        auto& flow_constraints = dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equality_constraints_partitioned.get(0));
        flow_constraints.flow_CFL_ = old_CFL;
    }

    algo_state.gnorm = rhs_partitioned->norm();

    if (is_full_space_) {
        ROL::Constraint_Partitioned<Real> &equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        PHiLiP::FlowConstraints<PHILIP_DIM> &flow_constraints = dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equality_constraints_partitioned.get(0));
        //flow_constraints.flow_CFL_ = -10.0*std::max(1.0, 1.0/std::pow(algo_state.cnorm, 1.50));
        //flow_constraints.flow_CFL_ = -10*std::max(1.0, 1.0/std::pow(algo_state.cnorm, 2.00));
        flow_constraints.flow_CFL_ = -10000*std::max(1.0, 1.0/std::pow(algo_state.cnorm, 2.00));
        flow_constraints.flow_CFL_ = -std::max(1.0, 1.0/std::pow(flowcnorm_, 1.25)) / 10000.0;
        flow_constraints.flow_CFL_ = -std::max(1.0, 1.0/std::pow(flowcnorm_, 1.00)) / 100.0;
        flow_constraints.flow_CFL_ = -std::max(1.0, 1.0/std::pow(flowcnorm_, 1.00)) / algo_state.gnorm * 100.0;
        const double factor = std::pow(flowcnorm_, 1.00) * std::pow(algo_state.gnorm, 1.0) * 1.0;
        flow_constraints.flow_CFL_ = 1.0/std::min(factor, 0.25) * 10.0;//1000.0;
        flow_cfl_ = flow_constraints.flow_CFL_;
    }

    // Update Secant
    // Let gradientVec be the gradient of the Lagrangian.
    objective.gradient(*(step_state->gradientVec),design_variables,sqrttol);
    equality_constraints.applyAdjointJacobian(*dual_times_equality_jacobian, dual_equality, design_variables, sqrttol);
    step_state->gradientVec->axpy(one, *dual_times_equality_jacobian);
    //step_state->gradientVec->axpy(one, *dual_inequality_);

    printSearchDirection("Old Gradient", *(old_gradient_), *search_dual_equality, *search_dual_inequality);
    printSearchDirection("Gradient", *(step_state->gradientVec), *search_dual_equality, *search_dual_inequality);
    if ( lagrangian_secant_ != ROL::nullPtr && !d_force_use_bfgs) {
        search_design->set(design_variables);
        search_design->axpy(-one, *old_design);

        Real design_snorm = getCtlOpt(*search_design)->norm();
        const Real sy2 = getCtlOpt(*(step_state->gradientVec))->dot(*getCtlOpt(*search_design));
        auto Bs = getCtlOpt(*search_design)->clone();
        lagrangian_secant_->applyB(*Bs, *getCtlOpt(*search_design));
        Real sBs = Bs->dot(*getCtlOpt(*search_design));
        sBs *= 1e-2;
        //if (sy2 <= sBs || step_state->searchSize < 1e-1) {
        //if (std::abs(sy2) < 1e-12) {
            pcout << "Not updating Lagrangian BFGS since sy <= 1e-15..." << sy2 << std::endl;
        //} else {
            pcout << "Updating Lagrangian BFGS..." << std::endl;
            lagrangian_secant_->updateStorage(*getCtlOpt(design_variables),*getCtlOpt(*(step_state->gradientVec)),*getCtlOpt(*old_gradient_),*getCtlOpt(*search_design),design_snorm,algo_state.iter+1);
        //}
    }
    n_design_iterations = algo_state.iter;

    printDesignDual("After Update", design_variables, bound_constraints, *dual_inequality_, *des_plus_dual_, dual_equality);

    algo_state.ngrad++;

    (algo_state.iterateVec)->set(design_variables);
    (algo_state.lagmultVec)->set(dual_equality);

    old_dual_inequality_->set(*dual_inequality_);
    new_design_variables_->set(design_variables);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if( comm_size == 1) std::cout << print( algo_state, true);
}


  
template<typename Real>
std::string PrimalDualActiveSetStep<Real>::printHeader( void ) const 
{
    std::stringstream hist;
    hist << "  ";
    hist << std::setw(6) << std::left << "iter";
    hist << std::setw(15) << std::left << "value";
    hist << std::setw(15) << std::left << "gnorm";
    hist << std::setw(15) << std::left << "cnorm";
    hist << std::setw(15) << std::left << "ecnorm";
    hist << std::setw(15) << std::left << "flowcnorm";
    if (is_full_space_)
        hist << std::setw(15) << std::left << "flow_cfl";
    hist << std::setw(15) << std::left << "identity";
    hist << std::setw(15) << std::left << "icnorm";
    hist << std::setw(15) << std::left << "snorm";
    hist << std::setw(11) << std::left << "#fval";
    hist << std::setw(11) << std::left << "#grad";
    hist << std::setw(11) << std::left << "#linesear";
    hist << std::setw(11) << std::left << "iterPDAS";
    hist << std::setw(11) << std::left << "flagPDAS";
    hist << std::setw(11) << std::left << "iterGMRES";
    hist << std::setw(11) << std::left << "flagGMRES";
    hist << std::setw(11) << std::left << "feasible";
    hist << std::setw(11) << std::left << "n_active_";
    hist << std::setw(18) << std::left << "n_vmult";
    hist << std::setw(18) << std::left << "dRdW_form";
    hist << std::setw(18) << std::left << "dRdW_mult";
    hist << std::setw(18) << std::left << "dRdX_mult";
    hist << std::setw(18) << std::left << "d2R_mult";
    hist << "\n";
    return hist.str();
}
  
template<typename Real>
std::string PrimalDualActiveSetStep<Real>::printName( void ) const
{
    std::stringstream hist;
    hist << "\nPrimal Dual Active Set Method\n";
    if (is_full_space_) {
        hist << "\n Full-Space Newton Method\n";
    } else {
        hist << "\n Reduced-Space ";
        if (useSecantHessVec_) {
            hist << " Quasi-Newton\n";
        } else {
            hist << " Newton\n";
        }
    }
    return hist.str();
}
  
template<typename Real>
std::string PrimalDualActiveSetStep<Real>::print( ROL::AlgorithmState<Real> &algo_state, bool print_header ) const
{
    std::stringstream hist;
    hist << std::scientific << std::setprecision(6);
    if ( algo_state.iter == 0 ) hist << printName();
    if ( print_header ) hist << printHeader();
    if ( algo_state.iter == 0 ) {
        hist << "  ";
        hist << std::setw(6) << std::left << algo_state.iter;
        hist << std::setw(15) << std::left << algo_state.value;
        hist << std::setw(15) << std::left << algo_state.gnorm;
        hist << "\n";
    } else {
        hist << "  ";
        hist << std::setw(6) << std::left << algo_state.iter;
        hist << std::setw(15) << std::left << algo_state.value;
        hist << std::setw(15) << std::left << algo_state.gnorm;
        hist << std::setw(15) << std::left << algo_state.cnorm;
        hist << std::setw(15) << std::left << ecnorm_;
        hist << std::setw(15) << std::left << flowcnorm_;
        if (is_full_space_)
            hist << std::setw(15) << std::left << flow_cfl_;
        hist << std::setw(15) << std::left << identity_factor_;
        hist << std::setw(15) << std::left << icnorm_;
        hist << std::setw(15) << std::left << algo_state.snorm;
        hist << std::setw(11) << std::left << algo_state.nfval;
        hist << std::setw(11) << std::left << algo_state.ngrad;
        hist << std::setw(11) << std::left << kkt_linesearches_;
        hist << std::setw(11) << std::left << iter_PDAS_;
        hist << std::setw(11) << std::left << flag_PDAS_;
        hist << std::setw(11) << std::left << iter_Krylov_;
        hist << std::setw(11) << std::left << flag_Krylov_;
        if ( feasible_ ) {
            hist << std::setw(10) << std::left << "YES";
        } else {
            hist << std::setw(10) << std::left << "NO";
        }
        hist << std::setw(11) << std::left << n_active_;
        hist << std::setw(18) << std::left << n_vmult;
        hist << std::setw(18) << std::left << dRdW_form;
        hist << std::setw(18) << std::left << dRdW_mult;
        hist << std::setw(18) << std::left << dRdX_mult;
        hist << std::setw(18) << std::left << d2R_mult;
        hist << "\n";
    }
    return hist.str();
}
  
template class PrimalDualActiveSetStep <double>;
} // namespace PHiLiP

//  template<typename Real>
//  void PrimalDualActiveSetStep<Real>::compute2(
//      Vector &search_direction_design,
//      const Vector &design_variables,
//      const Vector &dual_equality,
//      ROL::Objective<Real> &objective,
//      ROL::Constraint<Real> &equality_constraints, 
//      ROL::BoundConstraint<Real> &bound_constraints, 
//      ROL::AlgorithmState<Real> &algo_state )
//  {
//      ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
//      Real zero(0), one(1);
//      search_direction_design.zero();
//      new_dual_equality_->set(dual_equality);
//      quadratic_residual_->set(*(step_state->gradientVec));
//  
//      new_design_variables_->set(design_variables);
//  
//      // Temporary variables
//      VectorPtr inactive_dual_equality_ = dual_equality.clone();
//      VectorPtr active_dual_equality_ = dual_equality.clone();
//  
//      VectorPtr rhs_design_temp = design_variables.clone(); 
//  
//      // PDAS iterates through 3 steps.
//      // 1. Estimate active set
//      // 2. Use active set to determine search direction of the active constraints
//      // 3. Solve KKT system for remaining inactive constraints
//      for ( iter_PDAS_ = 0; iter_PDAS_ < maxit_; iter_PDAS_++ ) {
//  
//          /********************************************************************/
//          // Modify iterate vector to check active set
//          /********************************************************************/
//          des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
//          des_plus_dual_->axpy(scale_,*(dual_inequality_));    // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply
//  
//          /********************************************************************/
//          // Project design_variables onto primal dual feasible set
//          // Using approximation of the active set, obtain the search direction since
//          // we know that step will be constrained.
//          /********************************************************************/
//          if (false) {
//              search_direction_active_set_->zero();                                     // active_set_search_direction   = 0
//          
//              search_temp_->set(*bound_constraints.getUpperBound());                    // search_tmp = upper_bound
//              search_temp_->axpy(-one,design_variables);                                // search_tmp = upper_bound - design_variables
//              desvar_tmp_->set(*search_temp_);                                          // tmp        = upper_bound - design_variables
//              bound_constraints.pruneUpperActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = (upper_bound - (upper_bound - design_variables + c*dual_variables)) < 0 ? 0 : upper_bound - design_variables
//              search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(upper_bound - design_variables)
//  
//              search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(upper_bound - design_variables)
//        
//              search_temp_->set(*bound_constraints.getLowerBound());                    // search_tmp = lower_bound
//              search_temp_->axpy(-one,design_variables);                                // search_tmp = lower_bound - design_variables
//              desvar_tmp_->set(*search_temp_);                                          // tmp        = lower_bound - design_variables
//              bound_constraints.pruneLowerActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = INACTIVE(lower_bound - design_variables)
//              search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(lower_bound - design_variables)
//              search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(lower_bound - design_variables)
//          } else { 
//              get_active_design_minus_bound(*search_direction_active_set_, design_variables, *des_plus_dual_, bound_constraints);
//              search_direction_active_set_->scale(-1.0);
//          }
//  
//          inactive_dual_equality_ = new_dual_equality_->clone();
//          active_dual_equality_ = new_dual_equality_->clone();
//  
//          /********************************************************************/
//          // Apply Hessian to active components of search_direction_design and remove inactive
//          /********************************************************************/
//          itol_ = std::sqrt(ROL::ROL_EPSILON<Real>());
//          // INACTIVE(H)*active_set_search_direction = H*active_set_search_direction
//          // INACTIVE(H)*active_set_search_direction = INACTIVE(H*active_set_search_direction)
//          gradient_tmp1_->zero();
//          if ( useSecantHessVec_ && objective_secant_ != ROL::nullPtr ) {
//              objective_secant_->applyB(*getOpt(*gradient_tmp1_),*getOpt(*search_direction_active_set_));
//          } else {
//              objective.hessVec(*gradient_tmp1_,*search_direction_active_set_,design_variables,itol_);
//              equality_constraints.applyAdjointHessian(*old_gradient_, dual_equality, *getOpt(*search_direction_active_set_), *getOpt(design_variables), itol_);
//              gradient_tmp1_->axpy(one, *old_gradient_);
//          }
//          bound_constraints.pruneActive(*gradient_tmp1_,*des_plus_dual_,neps_);
//          /********************************************************************/
//          // SEPARATE ACTIVE AND INACTIVE COMPONENTS OF THE GRADIENT
//          /********************************************************************/
//          // Inactive components
//          gradient_inactive_set_->set(*(step_state->gradientVec));
//          bound_constraints.pruneActive(*gradient_inactive_set_,*des_plus_dual_,neps_);
//          // Active components
//          gradient_active_set_->set(*(step_state->gradientVec));
//          gradient_active_set_->axpy(-one,*gradient_inactive_set_);
//          /********************************************************************/
//          // SOLVE REDUCED NEWTON SYSTEM 
//          /********************************************************************/
//  
//          // rhs_design_temp = -(INACTIVE(gradient) + INACTIVE(H*active_set_search_direction))
//          rhs_design_temp->set(*gradient_inactive_set_);
//          rhs_design_temp->plus(*gradient_tmp1_);
//          rhs_design_temp->scale(-one);
//  
//          search_direction_design.zero();
//          if ( rhs_design_temp->norm() > zero ) {             
//              // Initialize Hessian and preconditioner
//              ROL::Ptr<ROL::Objective<Real> >       objective_ptr  = ROL::makePtrFromRef(objective);
//              ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraint_ptr = ROL::makePtrFromRef(bound_constraints);
//              ROL::Ptr<ROL::Constraint<Real> > equality_constraints_ptr = ROL::makePtrFromRef(equality_constraints);
//  
//              VectorPtr search_direction_des_ptr = ROL::makePtrFromRef(search_direction_design);
//              VectorPtr search_direction_dual_ptr = search_direction_dual_;
//              ROL::Ptr<ROL::PartitionedVector<Real>> search_partitioned = ROL::makePtr<ROL::PartitionedVector<Real>>(
//                      std::vector<VectorPtr >({search_direction_des_ptr, search_direction_dual_ptr})
//                  );
//  
//              VectorPtr rhs_des_ptr = rhs_design_temp;
//              VectorPtr rhs_dual_ptr = (step_state->constraintVec)->clone();
//              rhs_dual_ptr->scale(-one);
//              ROL::Ptr<ROL::PartitionedVector<Real>> rhs_partitioned = ROL::makePtr<ROL::PartitionedVector<Real>>(std::vector<VectorPtr >({rhs_des_ptr, rhs_dual_ptr}));
//  
//              ROL::Ptr<ROL::LinearOperator<Real> > hessian = ROL::makePtr<InactiveConstrainedHessian>(objective_ptr, equality_constraints_ptr, bound_constraint_ptr, algo_state.iterateVec, algo_state.lagmultVec, des_plus_dual_, neps_, objective_secant_, useSecantHessVec_);
//              ROL::Ptr<ROL::LinearOperator<Real> > precond = ROL::makePtr<InactiveConstrainedHessianPreconditioner>(objective_ptr, bound_constraint_ptr, algo_state.iterateVec, des_plus_dual_, neps_, objective_secant_, useSecantPrecond_);
//  
//              krylov_->run(*search_partitioned, *hessian, *rhs_partitioned, *precond, iter_Krylov_, flag_Krylov_);
//  
//              bound_constraints.pruneActive(search_direction_design,*des_plus_dual_,neps_);        // search_direction_design <- inactive_search_direction
//          }
//          search_direction_design.plus(*search_direction_active_set_);                             // search_direction_design = inactive_search_direction + active_set_search_direction
//          /********************************************************************/
//          // UPDATE MULTIPLIER 
//          /********************************************************************/
//          rhs_design_temp->zero();
//          if ( useSecantHessVec_ && objective_secant_ != ROL::nullPtr ) {
//              objective_secant_->applyB(*getOpt(*rhs_design_temp),*getOpt(search_direction_design));
//          } else {
//              objective.hessVec(*rhs_design_temp,search_direction_design,design_variables,itol_);
//          }
//          gradient_tmp1_->set(*rhs_design_temp);
//          bound_constraints.pruneActive(*gradient_tmp1_,*des_plus_dual_,neps_);
//  
//          // dual^{k+1} = - ( ACTIVE(H * search_direction_design) + gradient_active_set_ )
//          dual_inequality_->set(*rhs_design_temp);
//          dual_inequality_->axpy(-one,*gradient_tmp1_);
//          dual_inequality_->plus(*gradient_active_set_);
//          dual_inequality_->scale(-one);
//          dual_inequality_->zero();
//          /********************************************************************/
//          // UPDATE STEP 
//          /********************************************************************/
//          new_design_variables_->set(design_variables);
//          new_design_variables_->plus(search_direction_design);
//          quadratic_residual_->set(*(step_state->gradientVec));
//          quadratic_residual_->plus(*rhs_design_temp);
//          // Compute criticality measure  
//          desvar_tmp_->set(*new_design_variables_);
//          desvar_tmp_->axpy(-one,quadratic_residual_->dual());
//          bound_constraints.project(*desvar_tmp_);
//          desvar_tmp_->axpy(-one,*new_design_variables_);
//          pcout << "des_var_temp " << desvar_tmp_->norm() << std::endl;
//          pcout << "gtol gnorm " << gtol_*algo_state.gnorm << std::endl;
//          pcout << "rhs_design_temp.norm() " << rhs_design_temp->norm() << std::endl;
//          if ( desvar_tmp_->norm() < gtol_*algo_state.gnorm ) {
//              flag_PDAS_ = 0;
//              break;
//          }
//          if ( search_direction_design.norm() < stol_*design_variables.norm() ) {
//              flag_PDAS_ = 2;
//              break;
//          } 
//      }
//      if ( iter_PDAS_ == maxit_ ) {
//          flag_PDAS_ = 1;
//      } else {
//          iter_PDAS_++;
//      }
//  }



// Unconstrained PDAS
// template<typename Real>
// void PrimalDualActiveSetStep<Real>::compute(
//     Vector &search_direction_design,
//     const Vector &design_variables,
//     ROL::Objective<Real> &objective,
//     ROL::BoundConstraint<Real> &bound_constraints, 
//     ROL::AlgorithmState<Real> &algo_state )
// {
//     ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
//     Real zero(0), one(1);
//     search_direction_design.zero();
//     quadratic_residual_->set(*(step_state->gradientVec));
// 
//     new_design_variables_->set(design_variables);
//     // PDAS iterates through 3 steps.
//     // 1. Estimate active set
//     // 2. Use active set to determine search direction of the active constraints
//     // 3. Solve KKT system for remaining inactive constraints
//     VectorPtr rhs_design_temp = design_variables.clone(); 
//     for ( iter_PDAS_ = 0; iter_PDAS_ < maxit_; iter_PDAS_++ ) {
//         /********************************************************************/
//         // Modify iterate vector to check active set
//         /********************************************************************/
//         des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
//         des_plus_dual_->axpy(scale_,*(dual_inequality_));    // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply
//         /********************************************************************/
//         // Project design_variables onto primal dual feasible set
//         // Using approximation of the active set, obtain the search direction since
//         // we know that step will be constrained.
//         /********************************************************************/
//         search_direction_active_set_->zero();                                     // active_set_search_direction   = 0
//     
//         search_temp_->set(*bound_constraints.getUpperBound());                    // search_tmp = upper_bound
//         search_temp_->axpy(-one,design_variables);                                // search_tmp = upper_bound - design_variables
//         desvar_tmp_->set(*search_temp_);                                          // tmp        = upper_bound - design_variables
//         bound_constraints.pruneUpperActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = (upper_bound - (upper_bound - design_variables + c*dual_variables)) < 0 ? 0 : upper_bound - design_variables
//         search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(upper_bound - design_variables)
// 
//         search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(upper_bound - design_variables)
//   
//         search_temp_->set(*bound_constraints.getLowerBound());                    // search_tmp = lower_bound
//         search_temp_->axpy(-one,design_variables);                                // search_tmp = lower_bound - design_variables
//         desvar_tmp_->set(*search_temp_);                                          // tmp        = lower_bound - design_variables
//         bound_constraints.pruneLowerActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = INACTIVE(lower_bound - design_variables)
//         search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(lower_bound - design_variables)
//         search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(lower_bound - design_variables)
//         /********************************************************************/
//         // Apply Hessian to active components of search_direction_design and remove inactive
//         /********************************************************************/
//         itol_ = std::sqrt(ROL::ROL_EPSILON<Real>());
//         // INACTIVE(H)*active_set_search_direction = H*active_set_search_direction
//         // INACTIVE(H)*active_set_search_direction = INACTIVE(H*active_set_search_direction)
//         gradient_tmp1_->zero();
//         if ( useSecantHessVec_ && lagrangian_secant_ != ROL::nullPtr ) {
//             //secant_->applyB(*getOpt(*gradient_tmp1_),*getOpt(*search_direction_active_set_));
//             lagrangian_secant_->applyB(*getOpt(*gradient_tmp1_),*getOpt(*search_direction_active_set_));
//         } else {
//             objective.hessVec(*gradient_tmp1_,*search_direction_active_set_,design_variables,itol_);
//             //gradient_tmp1_->axpy(10.0, *search_direction_active_set_);
//         }
//         bound_constraints.pruneActive(*gradient_tmp1_,*des_plus_dual_,neps_);
//         /********************************************************************/
//         // SEPARATE ACTIVE AND INACTIVE COMPONENTS OF THE GRADIENT
//         /********************************************************************/
//         // Inactive components
//         gradient_inactive_set_->set(*(step_state->gradientVec));
//         bound_constraints.pruneActive(*gradient_inactive_set_,*des_plus_dual_,neps_);
//         // Active components
//         gradient_active_set_->set(*(step_state->gradientVec));
//         gradient_active_set_->axpy(-one,*gradient_inactive_set_);
//         /********************************************************************/
//         // SOLVE REDUCED NEWTON SYSTEM 
//         /********************************************************************/
// 
//         // rhs_design_temp = -(INACTIVE(gradient) + INACTIVE(H*active_set_search_direction))
//         rhs_design_temp->set(*gradient_inactive_set_);
//         rhs_design_temp->plus(*gradient_tmp1_);
//         rhs_design_temp->scale(-one);
// 
//         search_direction_design.zero();
//         if ( rhs_design_temp->norm() > zero ) {             
//             // Initialize Hessian and preconditioner
//             ROL::Ptr<ROL::Objective<Real> >       objective_ptr  = ROL::makePtrFromRef(objective);
//             ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraint_ptr = ROL::makePtrFromRef(bound_constraints);
//             ROL::Ptr<ROL::LinearOperator<Real> > hessian = ROL::makePtr<InactiveHessian<Real>>(objective_ptr, bound_constraint_ptr, algo_state.iterateVec, des_plus_dual_, neps_, lagrangian_secant_, useSecantHessVec_);
//             ROL::Ptr<ROL::LinearOperator<Real> > precond = ROL::makePtr<InactiveHessianPreconditioner<Real>>(objective_ptr, bound_constraint_ptr, algo_state.iterateVec, des_plus_dual_, neps_, lagrangian_secant_, useSecantPrecond_);
//             krylov_->run(search_direction_design,*hessian,*rhs_design_temp,*precond,iter_Krylov_,flag_Krylov_);
//             bound_constraints.pruneActive(search_direction_design,*des_plus_dual_,neps_);        // search_direction_design <- inactive_search_direction
//         }
//         search_direction_design.plus(*search_direction_active_set_);                             // search_direction_design = inactive_search_direction + active_set_search_direction
//         /********************************************************************/
//         // UPDATE MULTIPLIER 
//         /********************************************************************/
//         rhs_design_temp->zero();
//         if ( useSecantHessVec_ && lagrangian_secant_ != ROL::nullPtr ) {
//             lagrangian_secant_->applyB(*getOpt(*rhs_design_temp),*getOpt(search_direction_design));
//         } else {
//             objective.hessVec(*rhs_design_temp,search_direction_design,design_variables,itol_);
//             //rhs_design_temp->axpy(10.0, search_direction_design);
//         }
//         gradient_tmp1_->set(*rhs_design_temp);
//         bound_constraints.pruneActive(*gradient_tmp1_,*des_plus_dual_,neps_);
// 
//         // dual^{k+1} = - ( ACTIVE(H * search_direction_design) + gradient_active_set_ )
//         dual_inequality_->set(*rhs_design_temp);
//         dual_inequality_->axpy(-one,*gradient_tmp1_);
//         dual_inequality_->plus(*gradient_active_set_);
//         dual_inequality_->scale(-one);
//         /********************************************************************/
//         // UPDATE STEP 
//         /********************************************************************/
//         new_design_variables_->set(design_variables);
//         new_design_variables_->plus(search_direction_design);
//         //quadratic_residual_->set(*(step_state->gradientVec));
//         //quadratic_residual_->plus(*rhs_design_temp);
//         quadratic_residual_->set(*rhs_design_temp);
//         // Compute criticality measure  
//         desvar_tmp_->set(*new_design_variables_);
//         desvar_tmp_->axpy(-one,quadratic_residual_->dual());
//         bound_constraints.project(*desvar_tmp_);
//         desvar_tmp_->axpy(-one,*new_design_variables_);
//         std::cout << "des_var_temp " << desvar_tmp_->norm() << std::endl;
//         std::cout << "gtol gnorm " << gtol_*algo_state.gnorm << std::endl;
//         std::cout << "rhs_design_temp.norm() " << rhs_design_temp->norm() << std::endl;
// 
//         if ( desvar_tmp_->norm() < gtol_*algo_state.gnorm ) {
//             flag_PDAS_ = 0;
//             break;
//         }
//         if ( search_direction_design.norm() < stol_*design_variables.norm() ) {
//             flag_PDAS_ = 2;
//             break;
//         } 
//     }
//     if ( iter_PDAS_ == maxit_ ) {
//         flag_PDAS_ = 1;
//     } else {
//         iter_PDAS_++;
//     }
// }
  
