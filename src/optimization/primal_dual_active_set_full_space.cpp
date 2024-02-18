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
#include <deal.II/lac/full_matrix.h>

#include "primal_dual_active_set_full_space.hpp"
#include "optimization/dealii_solver_rol_vector.hpp"

namespace PHiLiP {

template<typename Real>
void get_active_design_minus_bound(
    ROL::Vector<Real> &active_design_minus_bound,
    const ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &predicted_design_variables,
    ROL::BoundConstraint<Real> &bound_constraints)
{
    const Real one(1);
    const Real neps = -ROL::ROL_EPSILON<Real>();
    active_design_minus_bound.zero();
    auto temp = design_variables.clone();
    // Evaluate active (design - upper_bound)
    temp->set(*bound_constraints.getUpperBound());                               // temp = upper_bound
    temp->axpy(-one,design_variables);                                           // temp = upper_bound - design_variables
    temp->scale(-one);                                                           // temp = design_variables - upper_bound
    bound_constraints.pruneUpperInactive(*temp,predicted_design_variables,neps); // temp = (predicted_design_variables) <= upper_bound ? 0 : design_variables - upper_bound 
    // Store active (design - upper_bound)
    active_design_minus_bound.axpy(one,*temp);

    // Evaluate active (design - lower_bound)
    temp->set(*bound_constraints.getLowerBound());                               // temp = lower_bound
    temp->axpy(-one,design_variables);                                           // temp = lower_bound - design_variables
    temp->scale(-one);                                                           // temp = design_variables - lower_bound
    bound_constraints.pruneLowerInactive(*temp,predicted_design_variables,neps); // temp = (predicted_design_variables) <= lower_bound ? 0 : design_variables - lower_bound 
    // Store active (design - lower_bound)
    active_design_minus_bound.axpy(one,*temp);
}


template <typename Real>
FullSpacePrimalDualActiveSetStep<Real>::PDAS_KKT_System::
PDAS_KKT_System( const ROL::Ptr<ROL::Objective<Real> > &objective,
                 const ROL::Ptr<PHiLiP::FlowConstraints<PHILIP_DIM> > &flow_constraints,
                 const ROL::Ptr<ROL::Constraint<Real> > &equality_constraints,
                 const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                 const ROL::Ptr<const ROL::Vector<Real> > &design_variables,
                 const ROL::Ptr<const ROL::Vector<Real> > &dual_equality,
                 const ROL::Ptr<const ROL::Vector<Real> > &des_plus_dual,
                 const Real constraint_tolerance,
                 const ROL::Ptr<ROL::Secant<Real> > &secant,
                 const bool useSecant)
    : objective_(objective)
    , flow_constraints_(flow_constraints),
    , equality_constraints_(equality_constraints)
    , bound_constraints_(bound_constraints)
    , design_variables_(design_variables)
    , dual_equality_(dual_equality)
    , des_plus_dual_(des_plus_dual)
    , bounded_constraint_tolerance_(constraint_tolerance)
    , secant_(secant)
    , useSecant_(useSecant)
{
    temp_design_          = design_variables_->clone();
    temp_dual_equality_   = dual_equality_->clone();
    temp_dual_inequality_ = design_variables_->clone();
    if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
}

template <typename Real>
void FullSpacePrimalDualActiveSetStep<Real>::PDAS_KKT_System::
apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
{
    Real one(1);

    ROL::PartitionedVector<Real> &output_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(Hv);
    const ROL::PartitionedVector<Real> &input_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(v);

    const ROL::Ptr< const ROL::Vector<Real> > input_design           = input_partitioned.get(0);
    const ROL::Ptr< const ROL::Vector<Real> > input_dual_equality    = input_partitioned.get(1);
    const ROL::Ptr< const ROL::Vector<Real> > input_dual_inequality  = input_partitioned.get(2);

    // std::cout << "input_design " << input_design->norm() << std::endl;
    // std::cout << "input_dual_equality " << input_dual_equality->norm() << std::endl;
    // std::cout << "input_dual_inequality " << input_dual_inequality->norm() << std::endl;

    const ROL::Ptr<       ROL::Vector<Real> > output_design          = output_partitioned.get(0);
    const ROL::Ptr<       ROL::Vector<Real> > output_dual_equality   = output_partitioned.get(1);
    const ROL::Ptr<       ROL::Vector<Real> > output_dual_inequality = output_partitioned.get(2);

    output_design->zero();
    output_dual_equality->zero();
    output_dual_inequality->zero();

    // Rows 1-4: inactive design, active design, inactive slacks, active slacks
    //// Columns 1-4
    if ( useSecant_ ) {
        secant_->applyB(*getOpt(*output_design),*getOpt(*input_design));
    } else {
        objective_->hessVec(*output_design,*input_design,*design_variables_,tol);

        equality_constraints_->applyAdjointHessian(*temp_design_, *dual_equality_, *input_design, *design_variables_, tol);
        output_design->axpy(one,*temp_design_);

    }
    //objective_->hessVec(*output_design,*input_design,*design_variables_, tol);
    //secant_->applyB(*output_design,*input_design);
    double add_identity = 0.0; // 10.0
    getOpt(*output_design)->axpy(add_identity,*getOpt(*input_design));
    //std::cout << "Adding identity matrix of " << add_identity << std::endl;

    //objective_->hessVec(*output_design,*input_design,*design_variables_,tol);
    //equality_constraints_->applyAdjointHessian(*temp_design_, *dual_equality_, *input_design, *design_variables_, tol);
    //output_design->axpy(one,*temp_design_);

    //output_design->zero();
    //double add_identity = 10.0; // 10.0
    //output_design->axpy(add_identity,*input_design);
    //std::cout << "output_design1 " << output_design->norm() << std::endl;

    //// Columns 5-6
    equality_constraints_->applyAdjointJacobian(*temp_design_, *input_dual_equality, *design_variables_, tol);
    output_design->axpy(one,*temp_design_);
    //std::cout << "output_design2 " << temp_design_->norm() << std::endl;

    //// Columns 7-10
    output_design->axpy(one,*input_dual_inequality);
    //std::cout << "output_design3 " << input_dual_inequality->norm() << std::endl;

    // Rows 5-6: inactive dual_equality, active dual_equality
    //// Columns 1-4
    equality_constraints_->applyJacobian(*output_dual_equality, *input_design, *design_variables_, tol);
    //std::cout << "output_dual_equality " << output_dual_equality->norm() << std::endl;

    // Rows 7-10: inactive dual_inequality, active dual_inequality
    //// Rows 7 & 9
    temp_dual_inequality_->set(*input_dual_inequality);
    bound_constraints_->pruneActive(*temp_dual_inequality_,*des_plus_dual_,bounded_constraint_tolerance_);
    output_dual_inequality->set(*temp_dual_inequality_);
    //std::cout << "output_dual_inequality_inactive " << temp_dual_inequality_->norm() << std::endl;

    //// Rows 8 & 10
    temp_dual_inequality_->set(*input_design);
    bound_constraints_->pruneInactive(*temp_dual_inequality_,*des_plus_dual_,bounded_constraint_tolerance_);
    output_dual_inequality->axpy(one,*temp_dual_inequality_);
    //std::cout << "output_dual_inequality_active " << temp_dual_inequality_->norm() << std::endl;

}

template <typename Real>
Real FullSpacePrimalDualActiveSetStep<Real>::computeCriticalityMeasure(
    ROL::Vector<Real> &design_variables,
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
FullSpacePrimalDualActiveSetStep<Real>::FullSpacePrimalDualActiveSetStep( ROL::ParameterList &parlist ) 
    : ROL::Step<Real>::Step(),
      parlist_(parlist),
      krylov_(ROL::nullPtr),
      iter_Krylov_(0), flag_Krylov_(0), itol_(0),
      maxit_(0), iter_PDAS_(0), flag_PDAS_(0), stol_(0), gtol_(0), scale_(0),
      //neps_(-ROL::ROL_EPSILON<Real>()), // Negative epsilon means that x = boundconstraint is INACTIVE when pruneActive occurs
      neps_(ROL::ROL_EPSILON<Real>()), // Positive epsilon means that x = boundconstraint is ACTIVE when pruneActive occurs
      feasible_(false),
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
      gradient_tmp1_(ROL::nullPtr),
      gradient_tmp2_(ROL::nullPtr),
      esec_(ROL::SECANT_LBFGS), secant_(ROL::nullPtr), useSecantPrecond_(false),
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
    useSecantHessVec_ = parlist_.sublist("General").sublist("Secant").get("Use as Hessian", false); 

    useSecantPrecond_ = parlist_.sublist("General").sublist("Secant").get("Use as Preconditioner", false);

    parlist_.sublist("General").sublist("Secant").set("Maximum Storage",1000);
    if ( useSecantHessVec_ || useSecantPrecond_ ) {
      secant_ = ROL::SecantFactory<Real>(parlist_);
    }
    // Build Krylov object
    krylov_ = ROL::KrylovFactory<Real>(parlist_);
}

template<typename Real>
void FullSpacePrimalDualActiveSetStep<Real>::initialize(
    ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &gradient_vec_to_clone, 
    ROL::Vector<Real> &dual_equality,
    const ROL::Vector<Real> &constraint_vec_to_clone, 
    ROL::Objective<Real> &objective,
    PHiLiP::FlowConstraints<PHILIP_DIM> &flow_constraints, 
    ROL::Constraint<Real> &equality_constraints, 
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    (void) flow_constraints;
    (void) equality_constraints;
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


    initialize(design_variables, gradient_vec_to_clone, gradient_vec_to_clone, objective, bound_constraints, algo_state);
}

template<typename Real>
void FullSpacePrimalDualActiveSetStep<Real>::initialize(
    ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &search_direction_vec_to_clone,
    const ROL::Vector<Real> &gradient_vec_to_clone, 
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
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
    gradient_tmp2_ = gradient_vec_to_clone.clone(); 
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
    //Real one(1);
    //dual_inequality_->set((step_state->gradientVec)->dual());
    //dual_inequality_->scale(-one);
}

template<typename Real>
void split_design_into_control_slacks(
    const ROL::Vector<Real> &design_variables,
    ROL::Vector<Real> &control_variables,
    ROL::Vector<Real> &slack_variables
    )
{
    const ROL::PartitionedVector<Real> &design_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(design_variables);
    const unsigned int n_vec = design_partitioned.numVectors();

    ROL::Ptr<ROL::Objective<Real> > control_variables_ptr = ROL::makePtrFromRef(control_variables);
    control_variables_ptr = design_partitioned[0].clone();
    control_variables_ptr->set( *(design_partitioned.get(0)) );
    std::vector<ROL::Ptr<ROL::Vector<Real>>> slack_vecs;
    for (unsigned int i_vec = 1; i_vec < n_vec; ++i_vec) {
        slack_vecs._push_back( design_partitioned.get(i_vec)->clone() );
        slack_vecs[i_vec-1].set( *(design_partitioned.get(i_vec)) );
    }
    slack_variables = ROL::PartitionedVector<Real> (slack_vecs);
}

template<typename Real>
void FullSpacePrimalDualActiveSetStep<Real>::compute_PDAS_rhs(
    const ROL::Vector<Real> &old_design_variables,
    const ROL::Vector<Real> &new_design_variables,
    const ROL::Vector<Real> &new_dual_equality,
    const ROL::Vector<Real> &dual_inequality,
    const ROL::Vector<Real> &des_plus_dual,
    const ROL::Vector<Real> &old_objective_gradient,
    ROL::Objective<Real> &objective, 
    ROL::Constraint<Real> &equality_constraints, 
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::PartitionedVector<Real> &rhs_partitioned)
{
    // Define old_ as using variables that do not change over the PDAS iterations
    // For example, gradients applied onto a vector would use the old_design_variables.
    // However, evaluating the constraint itself would use new_design_variables.
    // Basically, anything involving linearization uses old_design_variables to ensure that
    // PDAS is solving a linear (quadratic objective) problem.
    Real one(1);
    Real tol = ROL::ROL_EPSILON<Real>();

    ROL::Ptr<ROL::Vector<Real>> rhs_design          = rhs_partitioned.get(0);
    ROL::Ptr<ROL::Vector<Real>> rhs_dual_equality   = rhs_partitioned.get(1);
    ROL::Ptr<ROL::Vector<Real>> rhs_dual_inequality = rhs_partitioned.get(2);

    ROL::Ptr<ROL::Vector<Real>> rhs_dual_inequality_temp = rhs_dual_inequality->clone();

    // Design RHS
    // rhs_design = objective_gradient + constraint^T dualEquality + dualInequality
    (void) old_objective_gradient;
    (void) old_design_variables;
    //rhs_design->set( old_objective_gradient );
    //objective.gradient(*rhs_design, old_design_variables, tol);
    objective.gradient(*rhs_design, new_design_variables, tol);

    ROL::Ptr<ROL::Vector<Real>> rhs_design_temp = rhs_design->clone(); 
    //equality_constraints.applyAdjointJacobian(*rhs_design_temp, new_dual_equality, old_design_variables, tol);
    equality_constraints.applyAdjointJacobian(*rhs_design_temp, new_dual_equality, new_design_variables, tol);
    rhs_design->axpy(one, *rhs_design_temp);
    rhs_design->axpy(one, dual_inequality);
    pcout << "RHS design : " << rhs_design->norm() << std::endl;

    // Dual equality RHS
    // dual_equality = equality_constraint_value - slacks   (which is already the case for Constraint_Partitioned)
    equality_constraints.value(*rhs_dual_equality, new_design_variables, tol);
    pcout << "RHS dual_equality : " << rhs_dual_equality->norm() << std::endl;

    // Dual inequality RHS
    // Note that it should be equal to 
    // (        dual_inequality      )  on the inactive set
    // (  -c(design_variables - BOUND  )  on the active set
    // The term on the active set is associated with an identity matrix multiplied by -c.
    // Therefore, we will simply evaluate (BOUND - design_variables) on the right_hand_side, and ignore -c 
    // in the system matrix

    // // Set RHS = inactivedual_inequality)
    rhs_dual_inequality->zero();
    if (false) {
        pcout << "dual_inequality : " << rhs_dual_inequality->norm() << std::endl;
        rhs_dual_inequality->set(dual_inequality);
        bound_constraints.pruneActive(*rhs_dual_inequality,des_plus_dual,neps_);
        pcout << "inactive dual_inequality : " << rhs_dual_inequality->norm() << std::endl;

        // // Evaluate active (design - upper_bound)
        rhs_dual_inequality_temp->set(*bound_constraints.getUpperBound());                       // rhs_dual_inequality_temp = upper_bound
        rhs_dual_inequality_temp->axpy(-one,new_design_variables);                               // rhs_dual_inequality_temp = upper_bound - design_variables
        rhs_dual_inequality_temp->scale(-one);                                                   // rhs_dual_inequality_temp = design_variables - upper_bound
        bound_constraints.pruneUpperInactive(*rhs_dual_inequality_temp,des_plus_dual,neps_);   // rhs_dual_inequality_temp = (design_variables + c*dual_variables)) <= upper_bound ? 0 : design_variables - upper_bound 

        // // Store active (design - upper_bound)
        rhs_dual_inequality->axpy(one,*rhs_dual_inequality_temp);
        pcout << "rhs_dual_inequality_temp_upper : " << rhs_dual_inequality_temp->norm() << std::endl;

        // // Evaluate active (design - lower_bound)
        rhs_dual_inequality_temp->set(*bound_constraints.getLowerBound());                       // rhs_dual_inequality_temp = lower_bound
        rhs_dual_inequality_temp->axpy(-one,new_design_variables);                               // rhs_dual_inequality_temp = lower_bound - design_variables
        rhs_dual_inequality_temp->scale(-one);                                                   // rhs_dual_inequality_temp = design_variables - lower_bound
        bound_constraints.pruneLowerInactive(*rhs_dual_inequality_temp,des_plus_dual,neps_);   // rhs_dual_inequality_temp = (design_variables + c*dual_variables)) <= lower_bound ? 0 : design_variables - lower_bound 

        // // Store active (design - lower_bound)
        rhs_dual_inequality->axpy(one,*rhs_dual_inequality_temp);
        pcout << "rhs_dual_inequality_temp_lower: " << rhs_dual_inequality_temp->norm() << std::endl;
    } else {

        get_active_design_minus_bound(*rhs_dual_inequality, new_design_variables, des_plus_dual, bound_constraints);

        auto inactive_dual_inequality = dual_inequality.clone();
        inactive_dual_inequality->set(dual_inequality);
        bound_constraints.pruneActive(*inactive_dual_inequality,des_plus_dual,neps_);
        rhs_dual_inequality->plus(*inactive_dual_inequality);
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
const Real get_value(unsigned int i, const ROL::Vector<Real> &vec) {

    try {
        /// Base case 1
        /// We have a VectorAdapter from deal.II which can return a value (if single processor).
        return PHiLiP::ROL_vector_to_dealii_vector_reference(vec)[i];
    } catch (...) {
        try {
            /// Base case 2
            /// We have a Singleton, which can return a value
            const auto &vec_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(vec);
            return vec_singleton.getValue();
        } catch (const std::bad_cast& e) {

            try {
                /// Try to convert into Vector_SimOpt
                const auto &vec_simopt = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(vec);

                const unsigned int size_1 = vec_simopt.get_1()->dimension();

                if (i < size_1) {
                    return get_value(i, *(vec_simopt.get_1()));
                } else {
                    return get_value(i-size_1, *(vec_simopt.get_2()));
                }
                return -99999;
            } catch (const std::bad_cast& e) {
                /// Try to convert into PartitionedVector
                const auto &vec_part = dynamic_cast<const ROL::PartitionedVector<Real>&>(vec);

                const unsigned int numVec = vec_part.numVectors();

                unsigned int start_index = 0;
                unsigned int end_index = 0;
                for (unsigned int i_vec = 0; i_vec < numVec; ++i_vec) {
                    start_index = end_index;
                    end_index += vec_part[i_vec].dimension();
                    if (i < end_index) {
                        return get_value(i-start_index, vec_part[i_vec]);
                    }
                }
                return -99999999;
            }
        }
    }

}
template<typename Real>
void set_value(unsigned int i, const Real value, ROL::Vector<Real> &vec) {

    try {
        /// Base case 1
        /// We have a VectorAdapter from deal.II which can return a value (if single processor).
        PHiLiP::ROL_vector_to_dealii_vector_reference(vec)[i] = value;
        return;
    } catch (...) {
        try {
            /// Base case 2
            /// We have a Singleton, which can return a value
            auto &vec_singleton = dynamic_cast<ROL::SingletonVector<Real>&>(vec);
            vec_singleton.setValue(value);
            return;
        } catch (const std::bad_cast& e) {

            try {
                /// Try to convert into Vector_SimOpt
                auto &vec_simopt = dynamic_cast<ROL::Vector_SimOpt<Real>&>(vec);

                const unsigned int size_1 = vec_simopt.get_1()->dimension();

                if (i < size_1) {
                    set_value(i, value, *(vec_simopt.get_1()));
                } else {
                    set_value(i-size_1, value, *(vec_simopt.get_2()));
                }
                return;
            } catch (const std::bad_cast& e) {
                /// Try to convert into PartitionedVector
                auto &vec_part = dynamic_cast<ROL::PartitionedVector<Real>&>(vec);

                const unsigned int numVec = vec_part.numVectors();

                unsigned int start_index = 0;
                unsigned int end_index = 0;
                for (unsigned int i_vec = 0; i_vec < numVec; ++i_vec) {
                    start_index = end_index;
                    end_index += vec_part[i_vec].dimension();
                    if (i < end_index) {
                        set_value(i-start_index, value, vec_part[i_vec]);
                        return;
                    }
                }
            }
        }
    }
    throw;
}

template<typename Real>
void FullSpacePrimalDualActiveSetStep<Real>::compute(
    ROL::Vector<Real> &search_direction_design,
    const ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &dual_equality,
    ROL::Objective<Real> &objective,
    ROL::Constraint<Real> &equality_constraints, 
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    Real one(1); (void) one;
    search_direction_design.zero();
    quadratic_residual_->set(*(step_state->gradientVec));


    ROL::Ptr<ROL::Vector<Real>> new_dual_equality = dual_equality.clone();
    new_dual_equality->set(dual_equality);

    // PDAS iterates through 3 steps.
    // 1. Estimate active set
    // 2. Use active set to determine search direction of the active constraints
    // 3. Solve KKT system for remaining inactive constraints

    ROL::Ptr<ROL::Vector<Real>> rhs_design = design_variables.clone();
    ROL::Ptr<ROL::Vector<Real>> rhs_dual_equality = dual_equality.clone();
    ROL::Ptr<ROL::Vector<Real>> rhs_dual_inequality = design_variables.clone();

    ROL::Ptr<ROL::PartitionedVector<Real>> rhs_partitioned
        = ROL::makePtr<ROL::PartitionedVector<Real>>(
            std::vector<ROL::Ptr<ROL::Vector<Real>> >(
                {rhs_design, rhs_dual_equality, rhs_dual_inequality}
            )
          );

    ROL::Ptr<ROL::Vector<Real>> search_design = design_variables.clone();
    ROL::Ptr<ROL::Vector<Real>> search_dual_equality = dual_equality.clone();
    ROL::Ptr<ROL::Vector<Real>> search_dual_inequality = design_variables.clone();

    ROL::Ptr<ROL::PartitionedVector<Real>> search_partitioned 
        = ROL::makePtr<ROL::PartitionedVector<Real>>(
            std::vector<ROL::Ptr<ROL::Vector<Real>> >(
                {search_design, search_dual_equality, search_dual_inequality}
            )
          );
    //const ROL::Ptr<ROL::Vector<Real>> search_design          = search_partitioned->get(0);
    //const ROL::Ptr<ROL::Vector<Real>> search_dual_equality   = search_partitioned->get(1);
    //const ROL::Ptr<ROL::Vector<Real>> search_dual_inequality = search_partitioned->get(2);

    ROL::Ptr<ROL::Vector<Real>> rhs_design_temp = rhs_design->clone(); 
    ROL::Ptr<ROL::Vector<Real>> rhs_dual_inequality_temp = rhs_dual_inequality->clone();

    Real tol = ROL::ROL_EPSILON<Real>();
    ROL::Ptr<ROL::Vector<Real>> objective_gradient = design_variables.clone();
    objective.gradient(*objective_gradient, design_variables, tol);

    pcout << "Old design: " << std::endl;
    for (int i = 0; i < design_variables.dimension(); ++i) {
        pcout << get_value<Real>(i, design_variables) << std::endl;
    }
    pcout << "Old dual equality: " << std::endl;
    for (int i = 0; i < new_dual_equality_->dimension(); ++i) {
        pcout << get_value<Real>(i, *new_dual_equality_) << std::endl;
    }
    pcout << "Old dual inequality: " << std::endl;
    for (int i = 0; i < dual_inequality_->dimension(); ++i) {
        pcout << get_value<Real>(i, *dual_inequality_) << std::endl;
    }

    old_dual_inequality_->set(*dual_inequality_);
    new_design_variables_->set(design_variables);
    for ( iter_PDAS_ = 0; iter_PDAS_ < maxit_; iter_PDAS_++ ) {

        /********************************************************************/
        // Modify iterate vector to check active set
        /********************************************************************/
        des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
        const Real positive_scale = 0.001;
        des_plus_dual_->axpy(positive_scale,*(dual_inequality_));    // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply

        pcout << "des_plus_dual: " << std::endl;
        for (int i = 0; i < des_plus_dual_->dimension(); ++i) {
            pcout << get_value<Real>(i, *des_plus_dual_) << std::endl;
        }
        auto des_plus_dual_clone = des_plus_dual_->clone();
        des_plus_dual_clone->set(*des_plus_dual_);
        bound_constraints.pruneActive( *des_plus_dual_clone, *des_plus_dual_, neps_);
        std::cout << "des_plus_dual_clone norm: " << des_plus_dual_clone->norm() << std::endl;
        static unsigned int index_to_project_interior = 0;

        if (des_plus_dual_clone->norm() < 1e-14) {

            des_plus_dual_clone->set(*des_plus_dual_);
            bound_constraints.projectInterior( *des_plus_dual_clone );
            const unsigned int n = des_plus_dual_->dimension();


            const unsigned int index = index_to_project_interior % n;

            for (int i = 0; i < des_plus_dual_->dimension(); ++i) {
                pcout << get_value<Real>(i, *des_plus_dual_) << std::endl;
            }

            set_value(index, get_value(index, *des_plus_dual_clone), *des_plus_dual_);
            index_to_project_interior++;

            for (int i = 0; i < des_plus_dual_->dimension(); ++i) {
                pcout << get_value<Real>(i, *des_plus_dual_) << std::endl;
            }
            

            //bound_constraints.projectInterior( *new_design_variables_ );
            //dual_inequality_->zero();
            //des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
            //des_plus_dual_->axpy(positive_scale,*(dual_inequality_));    // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply
        }

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
            *rhs_partitioned);

        //if (rhs_partitioned->norm() < gtol_*algo_state.gnorm) {
        pcout << "RHS norm: " << rhs_partitioned->norm() << std::endl;
        /********************************************************************/
        const unsigned int rhs_size = rhs_partitioned->dimension();
        pcout << "RHS: " << std::endl;
        for (unsigned int i = 0; i < rhs_size; ++i) {
            pcout << get_value<Real>(i, *rhs_partitioned) << std::endl;
        }
        if (rhs_partitioned->norm() < gtol_*algo_state.gnorm) {
            flag_PDAS_ = 0;
            break;
        }
        search_partitioned->set(*rhs_partitioned);

        // Initialize Hessian and preconditioner
        const ROL::Ptr<ROL::Objective<Real> >       objective_ptr           = ROL::makePtrFromRef(objective);
        const ROL::Ptr<ROL::Constraint<Real> >      equality_constraint_ptr = ROL::makePtrFromRef(equality_constraints);
        const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraint_ptr    = ROL::makePtrFromRef(bound_constraints);
        const ROL::Ptr<const ROL::Vector<Real> >    old_design_var_ptr      = ROL::makePtrFromRef(design_variables);
        const ROL::Ptr<const ROL::Vector<Real> >    old_dual_equality_ptr   = ROL::makePtrFromRef(dual_equality);
        ROL::Ptr<ROL::LinearOperator<Real> > hessian = ROL::makePtr<PDAS_KKT_System>(objective_ptr, equality_constraint_ptr, bound_constraint_ptr, old_design_var_ptr, old_dual_equality_ptr, des_plus_dual_, neps_, secant_, useSecantHessVec_);
        ROL::Ptr<ROL::LinearOperator<Real> > precond = ROL::makePtr<Identity_Preconditioner>();

        pcout << "old_design_variables norm " << design_variables.norm() << std::endl;
        pcout << "new_design_variables_ norm " << new_design_variables_->norm() << std::endl;

        //std::abort();
        bool print_KKT = true;
        if (print_KKT) {
            const int do_full_matrix = (1 == dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
            //pcout << "do_full_matrix: " << do_full_matrix << std::endl;
            if (do_full_matrix) {

                auto column_of_kkt_operator = search_partitioned->clone();
                auto column_of_precond_kkt_operator = search_partitioned->clone();

                const int rhs_size = rhs_partitioned->dimension();
                dealii::FullMatrix<double> fullA(rhs_size);

                for (int i = 0; i < rhs_size; ++i) {
                    pcout << "COLUMN NUMBER: " << i+1 << " OUT OF " << rhs_size << std::endl;
                    auto basis = rhs_partitioned->basis(i);
                    MPI_Barrier(MPI_COMM_WORLD);
                    {
                        hessian->apply(*column_of_kkt_operator,*basis, tol);
                        precond->applyInverse(*column_of_precond_kkt_operator,*column_of_kkt_operator, tol);
                    }
                    //preconditioner.vmult(column_of_precond_kkt_operator,*basis);
                    if (do_full_matrix) {
                        for (int j = 0; j < rhs_size; ++j) {
                            //fullA[j][i] = column_of_precond_kkt_operator[j];
                            fullA[j][i] = get_value(j,*column_of_kkt_operator);
                            //pcout<< get_value(j,*basis) << " ";
                        }
                        //pcout << std::endl;
                    }
                }
                pcout<<"Dense matrix:"<<std::endl;
                fullA.print_formatted(std::cout, 10, true, 18, "0", 1., 0.);
                //std::abort();
            }
        }

        auto &gmres = dynamic_cast<ROL::GMRES<Real>&>(*krylov_);
        if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0) gmres.enableOutput(std::cout);
        gmres.run(*search_partitioned,*hessian,*rhs_partitioned,*precond,iter_Krylov_,flag_Krylov_);

        pcout << "Search norm: " << search_partitioned->norm() << std::endl;
        pcout << "Search norm design: " << search_design->norm() << std::endl;
        pcout << "Search norm equality: " << search_dual_equality->norm() << std::endl;
        pcout << "Search norm inequality: " << search_dual_inequality->norm() << std::endl;
        if (search_partitioned->norm() > 1e10) {

            auto temp = search_dual_inequality->clone();

            temp->set(*search_dual_inequality);
            bound_constraints.pruneActive(*temp,*des_plus_dual_,tol);
            pcout << "Search norm inactive inequality: " << temp->norm() << std::endl;

            temp->set(*search_dual_inequality);
            bound_constraints.pruneInactive(*temp,*des_plus_dual_,tol);
            pcout << "Search norm active inequality: " << temp->norm() << std::endl;
        }

        // Check that inactive dual inequality equal to 0
        rhs_dual_inequality_temp->set(*search_dual_inequality);
        rhs_dual_inequality_temp->axpy(one,*dual_inequality_);
        bound_constraints.pruneActive(*rhs_dual_inequality_temp,*des_plus_dual_,tol);
        Real inactive_dual_inequality_norm = rhs_dual_inequality_temp->norm();
        pcout << "Inactive dual inequality norm: " << inactive_dual_inequality_norm << std::endl;

        /********************************************************************/
        // Double check some values
        const unsigned int sea_size = search_partitioned->dimension();
        pcout << "Search: " << std::endl;
        for (unsigned int i = 0; i < sea_size; ++i) {
            pcout << get_value<Real>(i, *search_partitioned) << std::endl;
        }

        /********************************************************************/
        // UPDATE STEP 
        /********************************************************************/
        new_design_variables_->plus(*search_design);
        new_dual_equality_->plus(*search_dual_equality);
        dual_inequality_->plus(*search_dual_inequality);

        if ( bound_constraints.isActivated() ) {
            bound_constraints.project(*new_design_variables_);
        }

        pcout << "new design: " << std::endl;
        for (int i = 0; i < new_design_variables_->dimension(); ++i) {
            pcout << get_value<Real>(i, *new_design_variables_) << std::endl;
        }
        pcout << "new dual equality: " << std::endl;
        for (int i = 0; i < new_dual_equality_->dimension(); ++i) {
            pcout << get_value<Real>(i, *new_dual_equality_) << std::endl;
        }
        pcout << "new dual inequality: " << std::endl;
        for (int i = 0; i < dual_inequality_->dimension(); ++i) {
            pcout << get_value<Real>(i, *dual_inequality_) << std::endl;
        }

        //quadratic_residual_->set(*(step_state->gradientVec));
        //quadratic_residual_->plus(*rhs_design_temp);

        //// Compute criticality measure  
        //desvar_tmp_->set(*new_design_variables_);
        //desvar_tmp_->axpy(-one,quadratic_residual_->dual());
        //bound_constraints.project(*desvar_tmp_);
        //desvar_tmp_->axpy(-one,*new_design_variables_);
        //std::cout << "des_var_temp " << desvar_tmp_->norm() << std::endl;
        //std::cout << "gtol gnorm " << gtol_*algo_state.gnorm << std::endl;
        //std::cout << "rhs_design_temp.norm() " << rhs_design_temp->norm() << std::endl;

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
            break;
        }
        //if ( search_partitioned.norm() < stol_*design_variables.norm() ) {
        //    flag_PDAS_ = 2;
        //    break;
        //} 
    }
    if ( iter_PDAS_ == maxit_ ) {
        flag_PDAS_ = 1;
    } else {
        iter_PDAS_++;
    }
}
  
template<typename Real>
class PDAS_Lagrangian: public ROL::AugmentedLagrangian<Real>
{
    ROL::Ptr<ROL::Vector<Real> > active_des_minus_bnd;
    const ROL::Vector<Real> &dual_inequality;
    ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;

    private:
        Real penaltyParameter_;

public:
    PDAS_Lagrangian(const ROL::Ptr<ROL::Objective<Real> > &obj,
                    const ROL::Ptr<PHiLiP::FlowConstraints<PHILIP_DIM> > &flow_constraints,
                    const ROL::Ptr<ROL::Constraint<Real> > &con,
                    const ROL::Vector<Real> &multiplier,
                    const Real penaltyParameter,
                    const ROL::Vector<Real> &optVec,
                    const ROL::Vector<Real> &conVec,
                    ROL::ParameterList &parlist,
                    const ROL::Vector<Real> &inequality_multiplier,
                    const ROL::Ptr<ROL::BoundConstraint<Real>> &bound_constraints)
    : ROL::AugmentedLagrangian<Real>(obj, con, multiplier, penaltyParameter, optVec, conVec, parlist)
    , dual_inequality(inequality_multiplier)
    , bound_constraints_(bound_constraints)
    , penaltyParameter_(penaltyParameter)
    {
        active_des_minus_bnd = optVec.clone();
    }
    virtual Real value( const ROL::Vector<Real> &x, Real &tol ) override
    {
        Real val = ROL::AugmentedLagrangian<Real>::value(x,tol);
        get_active_design_minus_bound(*active_des_minus_bnd, x, x, *bound_constraints_);
        val += dual_inequality.dot(*active_des_minus_bnd);
        val += penaltyParameter_ * active_des_minus_bnd->dot(*active_des_minus_bnd);
        return val;
    }
    // Reset with upated penalty parameter
    virtual void reset(const ROL::Vector<Real> &multiplier, const Real penaltyParameter) {
        ROL::AugmentedLagrangian<Real>::reset(multiplier, penaltyParameter);
        penaltyParameter_ = penaltyParameter;
    }
};

template<typename Real>
void FullSpacePrimalDualActiveSetStep<Real>::update(
    ROL::Vector<Real> &design_variables,
    ROL::Vector<Real> &dual_equality,
    const ROL::Vector<Real> &search_direction_design,
    ROL::Objective<Real> &objective,
    PHiLiP::FlowConstraints<PHILIP_DIM> &flow_constraints,
    ROL::Constraint<Real> &equality_constraints,
    ROL::BoundConstraint<Real> &bound_constraints,
    ROL::AlgorithmState<Real> &algo_state )
{
    (void) search_direction_design;
    const double one = 1;

    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    step_state->SPiter = (maxit_ > 1) ? iter_PDAS_ : iter_Krylov_;
    step_state->SPflag = (maxit_ > 1) ? flag_PDAS_ : flag_Krylov_;


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

    //const double step_length = 0.1;
    //auto old_des = design_variables.clone();
    //old_des->set(design_variables);
    //new_design_variables_->axpy(-one, *old_des);
    //design_variables.axpy(step_length, *new_design_variables_);

    //std::cout << "new_dual_equality_.snorm" << new_dual_equality_->norm() << std::endl;
    //std::cout << "dual_equality_.snorm" << dual_equality.norm() << std::endl;
    //auto old_dual = dual_equality.clone();
    //old_dual->set(dual_equality);
    //new_dual_equality_->axpy(-one,*old_dual);
    //dual_equality.axpy(step_length,*new_dual_equality_);

    bool linesearch_success = false;
    Real fold = 0.0;
    int n_searches = 0;
    Real merit_function_value = 0.0;

    // Create a merit function based on the Augmented Lagrangian
    double penalty_value_ = 10.0;
    //auto dual_equality_zero = dual_equality.clone();
    //dual_equality_zero->zero();
    //auto merit_function = ROL::makePtr<ROL::AugmentedLagrangian<Real>> (
    //        ROL::makePtrFromRef<ROL::Objective<Real>>(objective),
    //        ROL::makePtrFromRef<ROL::Constraint<Real>>(equality_constraints),
    //        dual_equality,
    //        penalty_value_,
    //        design_variables,
    //        *(step_state->constraintVec),
    //        parlist_);
    auto merit_function = ROL::makePtr<PHiLiP::PDAS_Lagrangian<Real>> (
            ROL::makePtrFromRef<ROL::Objective<Real>>(objective),
            ROL::makePtrFromRef<PHiLiP::FlowConstraints<PHILIP_DIM>>(flow_constraints),
            ROL::makePtrFromRef<ROL::Constraint<Real>>(equality_constraints),
            dual_equality,
            penalty_value_,
            design_variables,
            *(step_state->constraintVec),
            parlist_,
            *dual_inequality_,
            ROL::makePtrFromRef<ROL::BoundConstraint<Real>>(bound_constraints));
    (void) merit_function;

    const bool changed_design_variables = true;
    merit_function->reset(dual_equality, penalty_value_);
    merit_function->update(design_variables, changed_design_variables, algo_state.iter);

    auto lineSearch_ = ROL::LineSearchFactory<Real>(parlist_);
    lineSearch_->initialize(design_variables, *search_design, *(step_state->gradientVec), *merit_function, bound_constraints);

    ROL::Ptr<ROL::Vector<Real> > merit_function_gradient = design_variables.clone();
    kkt_linesearches_ = 0;
    while (!linesearch_success) {

        merit_function->reset(dual_equality, penalty_value_);
        merit_function->update(design_variables, changed_design_variables, algo_state.iter);

        Real tol = ROL::ROL_EPSILON<Real>();
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
        lineSearch_->setData(algo_state.gnorm,*merit_function_gradient);

        int n_linesearches = 0;
        //bound_constraints.deactivate();
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

            //linesearch_success = true;
            if (n_searches > 1) {
                linesearch_success = true;
                //pcout << " Linesearch failed, searching other direction " << std::endl;
                //search_design->scale(-1.0);
                //penalty_value_ = std::max(1e-0/step_state->gradientVec->norm(), 1.0);
            }
            if (n_searches > 2) {
                pcout << " Linesearch failed in other direction... ending " << std::endl;
                linesearch_success = true;
                //std::abort();
            }
        }
        kkt_linesearches_ += n_linesearches;
        lineSearch_->setMaxitUpdate(step_state->searchSize, merit_function_value, fold);
    }
    //if (n_searches > 0) {
    //    step_state->searchSize = -0.001;
    //}
    search_design->scale(step_state->searchSize);
    //search_dual_equality->scale(step_state->searchSize);
    //search_dual_inequality->scale(step_state->searchSize);
    std::cout << "searchSize " << step_state->searchSize << std::endl;
    std::cout << "search_design.norm() " << search_design->norm() << std::endl;
    std::cout << "search_dual_equality.norm() " << search_dual_equality->norm() << std::endl;
    std::cout << "search_dual_inequality.norm() " << search_dual_inequality->norm() << std::endl;
    if ( bound_constraints.isActivated() ) {
        search_design->plus(design_variables);
        //bound_constraints.project(*search_design);
        search_design->axpy(static_cast<Real>(-1),design_variables);
    }
    design_variables.plus(*search_design);
    dual_equality.plus(*search_dual_equality);
    dual_inequality_->plus(*search_dual_inequality);

    //design_variables.plus(*search_design);
    feasible_ = bound_constraints.isFeasible(design_variables);
    algo_state.snorm = search_design->norm();
    algo_state.snorm += search_dual_equality->norm();
    algo_state.snorm += search_dual_inequality->norm();
    algo_state.iter++;
    //if (algo_state.iter > 5) {
    //    useSecantHessVec_ = false;
    //}
    Real tol = std::sqrt(ROL::ROL_EPSILON<Real>());
    objective.update(design_variables,true,algo_state.iter);
    algo_state.value = objective.value(design_variables,tol);
    algo_state.nfval++;
    
    equality_constraints.update(design_variables,true,algo_state.iter);
    equality_constraints.value(*(step_state->constraintVec),design_variables,tol);
    algo_state.cnorm = (step_state->constraintVec)->norm();

    auto active_set_des_min_bnd = design_variables.clone();
    get_active_design_minus_bound(*active_set_des_min_bnd, design_variables, design_variables, bound_constraints);
    algo_state.cnorm += active_set_des_min_bnd->norm();
    algo_state.ncval++;

    
    if ( secant_ != ROL::nullPtr ) {
        // Save current gradient as previous gradient.
        gradient_tmp1_->set(*(step_state->gradientVec));
    }

    objective.gradient(*(step_state->gradientVec),design_variables,tol);
    ROL::Ptr<ROL::Vector<Real>> rhs_design_temp = design_variables.clone(); 
    equality_constraints.applyAdjointJacobian(*rhs_design_temp, dual_equality, design_variables, tol);
    step_state->gradientVec->axpy(one, *rhs_design_temp);
    //step_state->gradientVec->axpy(one, *dual_inequality_);

    des_plus_dual_->set(design_variables);
    des_plus_dual_->plus(*dual_inequality_);

    ROL::Ptr<ROL::Vector<Real>> rhs_design = design_variables.clone();
    ROL::Ptr<ROL::Vector<Real>> rhs_dual_equality = dual_equality.clone();
    ROL::Ptr<ROL::Vector<Real>> rhs_dual_inequality = design_variables.clone();

    ROL::Ptr<ROL::PartitionedVector<Real>> rhs_partitioned
        = ROL::makePtr<ROL::PartitionedVector<Real>>(
            std::vector<ROL::Ptr<ROL::Vector<Real>> >(
                {rhs_design, rhs_dual_equality, rhs_dual_inequality}
            )
          );
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
        *rhs_partitioned);
    //algo_state.gnorm = step_state->gradientVec->norm();
    algo_state.gnorm = rhs_partitioned->norm();

    if ( secant_ != ROL::nullPtr ) {
        Real design_snorm = getOpt(*search_design)->norm();
        secant_->updateStorage(*getOpt(design_variables),*getOpt(*(step_state->gradientVec)),*getOpt(*gradient_tmp1_),*getOpt(*search_design),design_snorm,algo_state.iter+1);

        pcout << "new gradient: " << std::endl;
        for (int i = 0; i < step_state->gradientVec->dimension(); ++i) {
            pcout << get_value<Real>(i, *step_state->gradientVec) << std::endl;
        }
        pcout << "old gradient: " << std::endl;
        for (int i = 0; i < step_state->gradientVec->dimension(); ++i) {
            pcout << get_value<Real>(i, *gradient_tmp1_) << std::endl;
        }
    }

    algo_state.ngrad++;

    (algo_state.iterateVec)->set(design_variables);
    (algo_state.lagmultVec)->set(dual_equality);
}

  
template<typename Real>
std::string FullSpacePrimalDualActiveSetStep<Real>::printHeader( void ) const 
{
    std::stringstream hist;
    hist << "  ";
    hist << std::setw(6) << std::left << "iter";
    hist << std::setw(15) << std::left << "value";
    hist << std::setw(15) << std::left << "gnorm";
    hist << std::setw(15) << std::left << "cnorm";
    hist << std::setw(15) << std::left << "snorm";
    hist << std::setw(11) << std::left << "#fval";
    hist << std::setw(11) << std::left << "#grad";
    hist << std::setw(11) << std::left << "#linesear";
    hist << std::setw(11) << std::left << "iterPDAS";
    hist << std::setw(11) << std::left << "flagPDAS";
    hist << std::setw(11) << std::left << "iterGMRES";
    hist << std::setw(11) << std::left << "flagGMRES";
    hist << std::setw(11) << std::left << "feasible";
    hist << "\n";
    return hist.str();
}
  
template<typename Real>
std::string FullSpacePrimalDualActiveSetStep<Real>::printName( void ) const
{
    std::stringstream hist;
    hist << "\nPrimal Dual Active Set Newton's Method\n";
    return hist.str();
}
  
template<typename Real>
std::string FullSpacePrimalDualActiveSetStep<Real>::print( ROL::AlgorithmState<Real> &algo_state, bool print_header ) const
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
        hist << "\n";
    }
    return hist.str();
}
  
template class FullSpacePrimalDualActiveSetStep <double>;
} // namespace ROL

