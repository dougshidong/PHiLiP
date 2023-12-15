#ifndef PHILIP_OPTIMIZATIONUTILS_HPP
#define PHILIP_OPTIMIZATIONUTILS_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/full_matrix.h>

#include "ROL_AugmentedLagrangian.hpp"
#include "ROL_Constraint_Partitioned.hpp"
#include "ROL_Objective_SimOpt.hpp"
#include "ROL_SlacklessObjective.hpp"

#include "ROL_Step.hpp"
#include "ROL_Vector.hpp"
#include "ROL_Vector_SimOpt.hpp"
#include "ROL_KrylovFactory.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"
#include "ROL_Types.hpp"
#include "ROL_Secant.hpp"
#include "ROL_PartitionedVector.hpp"
#include "ROL_ParameterList.hpp"

#include "optimization/dealii_solver_rol_vector.hpp"


#define SYMMETRIZE_MATRIX_ true
//#define SYMMETRIZE_MATRIX_ false

namespace PHiLiP {

template<typename Real>
const std::optional<Real> get_value(unsigned int i, const ROL::Vector<Real> &vec) {

    try {
        /// Base case 1
        /// We have a VectorAdapter from deal.II which can return a value (if single processor).
        const dealii::LinearAlgebra::distributed::Vector<double> &vecdealii = PHiLiP::ROL_vector_to_dealii_vector_reference(vec);
        if (vecdealii.in_local_range(i)) {
            return vecdealii[i];
        } else {
            return std::nullopt;
        }
    } catch (...) {
        try {
            /// Base case 2
            /// We have a Singleton, which can return a value
            const auto &vec_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(vec);
            int my_rank;
            MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
            if (my_rank==0) {
                return vec_singleton.getValue();
            } else {
                return std::nullopt;
            }
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
                return std::nullopt;
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
                return std::nullopt;
            }
        }
    }

}
template<typename Real>
void set_value(unsigned int i, const Real value, ROL::Vector<Real> &vec) {

    try {
        /// Base case 1
        /// We have a VectorAdapter from deal.II which can return a value (if single processor).
        dealii::LinearAlgebra::distributed::Vector<double> &vecdealii = PHiLiP::ROL_vector_to_dealii_vector_reference(vec);
        if (vecdealii.in_local_range(i)) {
            vecdealii[i] = value;
        }
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

template <typename Real>
class Identity_Preconditioner : public ROL::LinearOperator<Real> {
    public:
      void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
      {
          (void) v;
          (void) tol;
          Hv.set(v);
      }
      void applyInverse( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
      {
          (void) v;
          (void) tol;
          Hv.set(v);
      }
};


template <typename Real>
class InactiveHessian : public ROL::LinearOperator<Real> {
    private:
        const ROL::Ptr<ROL::Objective<Real> > objective_;
        const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
        const ROL::Ptr<ROL::Vector<Real> > design_variables_;
        const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;
        ROL::Ptr<ROL::Vector<Real> > v_;
        Real bounded_constraint_tolerance_;
        const ROL::Ptr<ROL::Secant<Real> > secant_;
        bool useSecant_;
    public:
      InactiveHessian(const ROL::Ptr<ROL::Objective<Real> > &objective,
                const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                const Real constraint_tolerance = 0,
                const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                const bool useSecant = false )
        : objective_(objective)
        , bound_constraints_(bound_constraints)
        , design_variables_(design_variables)
        , des_plus_dual_(des_plus_dual)
        , bounded_constraint_tolerance_(constraint_tolerance)
        , secant_(secant)
        , useSecant_(useSecant)
      {
          v_ = design_variables_->clone();
          if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
      }
      void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
      {
          v_->set(v);
          bound_constraints_->pruneActive(*v_,*des_plus_dual_,bounded_constraint_tolerance_);
          if ( useSecant_ ) {
              secant_->applyB(Hv,*v_);
          } else {
              objective_->hessVec(Hv,*v_,*design_variables_,tol);
              //Hv.axpy(10.0,*v_);
          }
          bound_constraints_->pruneActive(Hv,*des_plus_dual_,bounded_constraint_tolerance_);
      }
};
  
template <typename Real>
class InactiveHessianPreconditioner : public ROL::LinearOperator<Real> {
    private:

        const ROL::Ptr<ROL::Objective<Real> > objective_;
        const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
        const ROL::Ptr<ROL::Vector<Real> > design_variables_;
        const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;
        ROL::Ptr<ROL::Vector<Real> > v_;
        Real bounded_constraint_tolerance_;
        const ROL::Ptr<ROL::Secant<Real> > secant_;
        bool useSecant_;

    public:
        InactiveHessianPreconditioner(const ROL::Ptr<ROL::Objective<Real> > &objective,
                  const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                  const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                  const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                  const Real constraint_tolerance = 0,
                  const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                  const bool useSecant = false )
          : objective_(objective)
          , bound_constraints_(bound_constraints)
          , design_variables_(design_variables)
          , des_plus_dual_(des_plus_dual)
          , bounded_constraint_tolerance_(constraint_tolerance)
          , secant_(secant)
          , useSecant_(useSecant)
        {
            v_ = design_variables_->dual().clone();
            if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
        }
        void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &/*tol*/ ) const
        {
            Hv.set(v.dual());
        }
        void applyInverse( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
        {
            v_->set(v);
            bound_constraints_->pruneActive(*v_,*des_plus_dual_,bounded_constraint_tolerance_);
            if ( useSecant_ ) {
                secant_->applyH(Hv,*v_);
            } else {
                objective_->precond(Hv,*v_,*design_variables_,tol);
            }
            bound_constraints_->pruneActive(Hv,*des_plus_dual_,bounded_constraint_tolerance_);
        }
};

/// Used to prune active or inactive constraints values or dual values using the same
/// BoundConstraint_Partitioned class.
template <typename Real>
ROL::PartitionedVector<Real> augment_constraint_to_design_and_constraint(
    const ROL::Ptr<ROL::Vector<Real>> vector_of_design_size,
    const ROL::Ptr<ROL::Vector<Real>> vector_of_constraint_size)
{
    std::vector<ROL::Ptr<ROL::Vector<Real>>> vec_of_vec { vector_of_design_size, vector_of_constraint_size };
    ROL::PartitionedVector<Real> partitioned_vector( vec_of_vec );

    partitioned_vector[0].set(*vector_of_design_size);
    partitioned_vector[1].set(*vector_of_constraint_size);

    return partitioned_vector;
}
template <typename Real>
void prune_active_constraints(const ROL::Ptr<ROL::Vector<Real>> constraint_vector_to_prune,
                              const ROL::Ptr<const ROL::Vector<Real>> ref_controlslacks_vector,
                              const ROL::Ptr<ROL::BoundConstraint<Real>> bound_constraints,
                              Real /*eps*/)
{
    const ROL::PartitionedVector<Real> &ref_controlslacks_vector_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*ref_controlslacks_vector);
    ROL::Ptr<ROL::Vector<Real>> design_vector = ref_controlslacks_vector_partitioned.get(0)->clone();

    ROL::PartitionedVector<Real> dummydes_constraint = augment_constraint_to_design_and_constraint( design_vector, constraint_vector_to_prune );
    
    bound_constraints->pruneActive(dummydes_constraint, ref_controlslacks_vector_partitioned);

    constraint_vector_to_prune->set(dummydes_constraint[1]);
}
template <typename Real>
void prune_inactive_constraints(const ROL::Ptr<ROL::Vector<Real>> constraint_vector_to_prune,
                              const ROL::Ptr<const ROL::Vector<Real>> ref_controlslacks_vector,
                              const ROL::Ptr<ROL::BoundConstraint<Real>> bound_constraints)
{
    const ROL::PartitionedVector<Real> &ref_controlslacks_vector_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*ref_controlslacks_vector);
    ROL::Ptr<ROL::Vector<Real>> design_vector = ref_controlslacks_vector_partitioned.get(0)->clone();

    ROL::PartitionedVector<Real> dummydes_constraint = augment_constraint_to_design_and_constraint( design_vector, constraint_vector_to_prune );
    
    bound_constraints->pruneInactive(dummydes_constraint, ref_controlslacks_vector_partitioned);

    constraint_vector_to_prune->set(dummydes_constraint[1]);
}
template <typename Real>
ROL::Ptr<ROL::Vector<Real>> getOpt( ROL::Vector<Real> &xs )
{
    return dynamic_cast<ROL::PartitionedVector<Real>&>(xs).get(0);
}
template <typename Real>
ROL::Ptr<const ROL::Vector<Real>> getOpt( const ROL::Vector<Real> &xs )
{
    return dynamic_cast<const ROL::PartitionedVector<Real>&>(xs).get(0);
}
template <typename Real>
ROL::Ptr<ROL::Vector<Real>> getCtlOpt( ROL::Vector<Real> &xs )
{
    ROL::Ptr<ROL::Vector<Real>> xopt = getOpt( xs );
    try {
        ROL::Vector_SimOpt<Real> &xopt_simopt = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*xopt);
        return xopt_simopt.get_2();
    } catch (...) {
    }
    return xopt;
}
template <typename Real>
ROL::Ptr<const ROL::Vector<Real>> getCtlOpt( const ROL::Vector<Real> &xs )
{
    ROL::Ptr<const ROL::Vector<Real>> xopt = getOpt( xs );
    try {
        const ROL::Vector_SimOpt<Real> &xopt_simopt = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*xopt);
        return xopt_simopt.get_2();
    } catch (...) {
    }
    return xopt;
}


template<typename Real>
class InactiveConstrainedHessian : public ROL::LinearOperator<Real> {
    private:
        const ROL::Ptr<ROL::Objective<Real> > objective_;
        const ROL::Ptr<ROL::Constraint<Real> > equality_constraints_;
        const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
        const ROL::Ptr<ROL::Vector<Real> > design_variables_;
        const ROL::Ptr<ROL::Vector<Real> > dual_equality_;
        const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;

        ROL::Ptr<ROL::Vector<Real> > temp_des_;
        ROL::Ptr<ROL::Vector<Real> > temp_dual_;

        ROL::Ptr<ROL::Vector<Real> > v_;
        const ROL::Ptr<ROL::Vector<Real> > inactive_input_des_;
        const ROL::Ptr<ROL::Vector<Real> > active_input_dual_;

        Real bounded_constraint_tolerance_;
        const ROL::Ptr<ROL::Secant<Real> > secant_;
        bool useSecant_;
    public:
      InactiveConstrainedHessian(
                const ROL::Ptr<ROL::Objective<Real> > &objective,
                const ROL::Ptr<ROL::Constraint<Real> > &equality_constraints,
                const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                const ROL::Ptr<ROL::Vector<Real> > &dual_equality,
                const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                const Real constraint_tolerance = 0,
                const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                const bool useSecant = false )
        : objective_(objective)
        , equality_constraints_(equality_constraints)
        , bound_constraints_(bound_constraints)
        , design_variables_(design_variables)
        , dual_equality_(dual_equality)
        , des_plus_dual_(des_plus_dual)
        , inactive_input_des_(design_variables_->clone())
        , active_input_dual_(dual_equality_->clone())
        , bounded_constraint_tolerance_(constraint_tolerance)
        , secant_(secant)
        , useSecant_(useSecant)
      {

          temp_des_ = design_variables_->clone();
          temp_dual_ = dual_equality_->clone();
          if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
      }
      void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
      {

          ROL::PartitionedVector<Real> &output_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(Hv);
          const ROL::PartitionedVector<Real> &input_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(v);

          const ROL::Ptr< const ROL::Vector< Real > > input_des = input_partitioned.get(0);
          const ROL::Ptr< const ROL::Vector< Real > > input_dual = input_partitioned.get(1);
          const ROL::Ptr< ROL::Vector< Real > > output_des = output_partitioned.get(0);
          const ROL::Ptr< ROL::Vector< Real > > output_dual = output_partitioned.get(1);

          inactive_input_des_->set(*input_des);
          bound_constraints_->pruneActive(*inactive_input_des_,*des_plus_dual_,bounded_constraint_tolerance_);

          //const ROL::PartitionedVector<Real> &inactive_input_des_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(inactive_input_des_);
          //inactive_input_des_partitioned.get(0);

          Real one(1);
          if ( useSecant_ ) {
              secant_->applyB(*output_des,*inactive_input_des_);
          } else {
              // Hv1 = H11 * v1
              objective_->hessVec(*output_des,*inactive_input_des_,*design_variables_,tol);
              equality_constraints_->applyAdjointHessian(*temp_des_, *dual_equality_, *inactive_input_des_, *design_variables_, tol);
              output_des->axpy(one,*temp_des_);
              //*output_des.axpy(10.0,*inactive_input_des_);
          }
          // Hv1 += H12 * v2

          active_input_dual_->set(*input_dual);
          prune_inactive_constraints( active_input_dual_, des_plus_dual_, bound_constraints_);
          equality_constraints_->applyAdjointJacobian(*temp_des_, *active_input_dual_, *design_variables_, tol);
          output_des->axpy(one,*temp_des_);

          bound_constraints_->pruneActive(*output_des,*des_plus_dual_,bounded_constraint_tolerance_);

          equality_constraints_->applyJacobian(*output_dual, *input_des, *design_variables_, tol);
      }
};
  
template<typename Real>
class InactiveConstrainedHessianPreconditioner : public ROL::LinearOperator<Real> {
    private:

        const ROL::Ptr<ROL::Objective<Real> > objective_;
        const ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraints_;
        const ROL::Ptr<ROL::Vector<Real> > design_variables_;
        const ROL::Ptr<ROL::Vector<Real> > des_plus_dual_;
        ROL::Ptr<ROL::Vector<Real> > v_;
        Real bounded_constraint_tolerance_;
        const ROL::Ptr<ROL::Secant<Real> > secant_;
        bool useSecant_;

    public:
        InactiveConstrainedHessianPreconditioner(const ROL::Ptr<ROL::Objective<Real> > &objective,
                  const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                  const ROL::Ptr<ROL::Vector<Real> > &design_variables,
                  const ROL::Ptr<ROL::Vector<Real> > &des_plus_dual,
                  const Real constraint_tolerance = 0,
                  const ROL::Ptr<ROL::Secant<Real> > &secant = ROL::nullPtr,
                  const bool useSecant = false )
          : objective_(objective)
          , bound_constraints_(bound_constraints)
          , design_variables_(design_variables)
          , des_plus_dual_(des_plus_dual)
          , bounded_constraint_tolerance_(constraint_tolerance)
          , secant_(secant)
          , useSecant_(useSecant)
        {
            v_ = design_variables_->dual().clone();
            if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
        }
        void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &/*tol*/ ) const
        {
            Hv.set(v.dual());
        }
        void applyInverse( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
        {
            //v_->set(v);
            //bound_constraints_->pruneActive(*v_,*des_plus_dual_,bounded_constraint_tolerance_);
            //if ( useSecant_ ) {
            //    secant_->applyH(Hv,*v_);
            //} else {
            //    objective_->precond(Hv,*v_,*design_variables_,tol);
            //}
            //bound_constraints_->pruneActive(Hv,*des_plus_dual_,bounded_constraint_tolerance_);
            (void) tol;
            Hv.set(v.dual());
        }
};


template<typename Real>
void printVec(const ROL::Vector<Real>& vec)
{
    for (int i = 0; i < vec.dimension(); i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        const std::optional<Real> value = PHiLiP::get_value(i, vec);
        if (value) {
            std::cout << *value << std::endl;
        }
    }
}
  

template <typename Real>
void setActiveEntriesToOne(ROL::BoundConstraint<Real> &bound_constraint, ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real eps = 0 ) {
    if (bound_constraint.isActivated()) {
        Real one(1);

        v.setScalar(one);
        bound_constraint.pruneInactive(v, x, eps);
        bound_constraint.pruneLowerActive(v, x, eps);

        ROL::Ptr<ROL::Vector<Real> > tmp = v.clone();
        tmp->setScalar(-one);
        bound_constraint.pruneInactive(*tmp, x, eps);
        bound_constraint.pruneUpperActive(*tmp, x, eps);
        v.axpy(one,*tmp);
    }
}
  
template <typename Real>
void setInactiveEntriesToOne(ROL::BoundConstraint<Real> &bound_constraint, ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real eps = 0 ) {
    if (bound_constraint.isActivated()) {
        Real one(1);
        v.setScalar(one);
        bound_constraint.pruneActive(v, x, eps);
    }
}

template <typename Real>
unsigned int count_constraint_partitioned_size(const ROL::Constraint_Partitioned<Real> &constraint)
{
    unsigned int partitioned_size = 0;
    try {
        while (true) {
            constraint.get(partitioned_size);
            partitioned_size++;
        }
    } catch (const std::exception& e) {
    }
    return partitioned_size;
}

} // namespace PHiLiP

#endif

