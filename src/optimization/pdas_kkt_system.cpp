#include <deal.II/base/conditional_ostream.h>

#include "optimization_utils.hpp"
#include "pdas_kkt_system.hpp"
#include "optimization/flow_constraints.hpp"

namespace PHiLiP {

template <typename Real>
PDAS_KKT_System<Real>::
PDAS_KKT_System( const ROL::Ptr<ROL::Objective<Real> > &objective,
                 const ROL::Ptr<ROL::Constraint<Real> > &equality_constraints,
                 const ROL::Ptr<ROL::BoundConstraint<Real> > &bound_constraints,
                 const ROL::Ptr<const ROL::Vector<Real> > &design_variables,
                 const ROL::Ptr<const ROL::Vector<Real> > &dual_equality,
                 const ROL::Ptr<const ROL::Vector<Real> > &des_plus_dual,
                 const Real add_identity,
                 const Real constraint_tolerance,
                 const ROL::Ptr<ROL::Secant<Real> > &secant,
                 const bool useSecant)
    : objective_(objective)
    , equality_constraints_(equality_constraints)
    , bound_constraints_(bound_constraints)
    , design_variables_(design_variables)
    , dual_equality_(dual_equality)
    , des_plus_dual_(des_plus_dual)
    , add_identity_(add_identity)
    , bounded_constraint_tolerance_(constraint_tolerance)
    , secant_(secant)
    , useSecant_(useSecant)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    temp_design_          = design_variables_->clone();
    temp_dual_equality_   = dual_equality_->clone();
    temp_dual_inequality_ = design_variables_->clone();
    if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;

    max_eig_estimate_ = 0.0;
    //const bool power_method = true;
    //bool is_full_space_ = false;
    //try {
    //    auto& equality_constraints_partitioned = dynamic_cast<ROL::Constraint_Partitioned<Real>&>(*equality_constraints_);
    //    auto flow_constraints = ROL::dynamicPtrCast<PHiLiP::FlowConstraints<PHILIP_DIM>>(equality_constraints_partitioned.get(0));

    //    auto slackless_objective = ROL::dynamicPtrCast<ROL::SlacklessObjective<Real>(objective_);
    //    auto objective_simopt = ROL::dynamicPtrCast<ROL::Objective_SimOpt<Real>>(slackless_objective.getObjective());

    //    VectorPtr design_sim_ctl = (dynamic_cast<ROL::PartitionedVector<Real>&>(design_variables_)).get(0);
    //    VectorPtr design_simulation = (ROL::dynamicPtrCast<ROL::Vector_SimOpt<Real>>(design_sim_ctl))->get_1();
    //    VectorPtr design_control    = (ROL::dynamicPtrCast<ROL::Vector_SimOpt<Real>>(design_sim_ctl))->get_2();
    //    ApproximateJacobianFlowConstraints( flow_constraints, design_simulation, design_control);

    //    is_full_space_ = true;
    //} catch (...) {
    //}
    //Real tol = ROL::ROL_EPSILON<Real>();
    //const Real one = 1.0;
    //if (power_method && is_full_space_) {
    //    auto temp_design1 = design_variables_->clone();
    //    auto temp_design2 = design_variables_->clone();
    //    temp_design_->setScalar(0.0);

    //    const ROL::Ptr<ROL::Vector<Real>> newVec = getCtlOpt(*temp_design1);
    //    const ROL::Ptr<ROL::Vector<Real>> tmpVec = getCtlOpt(*temp_design2);
    //    const ROL::Ptr<ROL::Vector<Real>> oldVec = getCtlOpt(*temp_design_);

    //    oldVec->setScalar(1.0);

    //    double vec_norm = temp_design_->norm();
    //    temp_design_->scale(1.0/vec_norm);

    //    for (int i = 0; i < 50; ++i) {
    //        objective_->hessVec(*temp_design1,*temp_design_,*design_variables_,tol);
    //        equality_constraints_->applyAdjointHessian(*temp_design2, *dual_equality_, *temp_design_, *design_variables_, tol);

    //        newVec->axpy(one, *tmpVec);
    //        vec_norm = newVec->norm();

    //        max_eig_estimate_ = newVec->dot(*oldVec) / oldVec->dot(*oldVec);
    //        newVec->scale(1.0/vec_norm);

    //        oldVec->set(*newVec);

    //        std::cout << "max_eig_estimate_ " << max_eig_estimate_ << std::endl;
    //    }
    //    std::cout << "resulting_max_eig_estimate_ " << max_eig_estimate_ << std::endl;
    //    max_eig_estimate_ = std::abs(max_eig_estimate_);
    //    //if (max_eig_estimate_ < 0) {
    //    //    max_eig_estimate_ = std::abs(max_eig_estimate_);
    //    //} else {
    //    //    max_eig_estimate_ = 0.0;
    //    //    //add_identity_ = 0.0;
    //    //}
    //}
}

template <typename Real>
void PDAS_KKT_System<Real>::apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
{
    Real one(1);

    ROL::PartitionedVector<Real> &output_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(Hv);
    const ROL::PartitionedVector<Real> &input_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(v);

    const ROL::Ptr< const ROL::Vector<Real> > input_design           = input_partitioned.get(0);
    const ROL::Ptr< const ROL::Vector<Real> > input_dual_equality    = input_partitioned.get(1);
    const ROL::Ptr< const ROL::Vector<Real> > input_dual_inequality  = input_partitioned.get(2);

    //MPI_Barrier(MPI_COMM_WORLD);
    static int ii = 0; (void) ii;
    //std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;
    //std::cout << "input_design " << input_design->norm() << std::endl;
    //std::cout << "input_dual_equality " << input_dual_equality->norm() << std::endl;
    //std::cout << "input_dual_inequality " << input_dual_inequality->norm() << std::endl;

    const ROL::Ptr<       ROL::Vector<Real> > output_design          = output_partitioned.get(0);
    const ROL::Ptr<       ROL::Vector<Real> > output_dual_equality   = output_partitioned.get(1);
    const ROL::Ptr<       ROL::Vector<Real> > output_dual_inequality = output_partitioned.get(2);

    output_design->zero();
    output_dual_equality->zero();
    output_dual_inequality->zero();

    // Rows 1-4: inactive design, active design, inactive slacks, active slacks
    //// Columns 1-4
    if ( useSecant_ ) {
        //output_design->set(*input_design);
        //getCtlOpt(*output_design)->set(*getCtlOpt(*input_design));
        //getCtlOpt(*output_design)->scale(10);

        secant_->applyB(*getCtlOpt(*output_design),*getCtlOpt(*input_design));
        if (add_identity_ != 0.0) {
            //pcout << "Adding identity matrix of " << add_identity_ << std::endl;
            getCtlOpt(*output_design)->axpy(add_identity_,*getCtlOpt(*input_design));
        }
        //secant_->applyB(*getCtlOpt(*output_design),*getCtlOpt(*input_design));
        //getCtlOpt(*output_design)->set(*getCtlOpt(*input_design));

        //secant_->applyB((*output_design),(*input_design));
        //output_design->set(*input_design);
        //getCtlOpt(*output_design)->axpy(add_identity_,*getCtlOpt(*input_design));
        //getCtlOpt(*output_design)->set(*getCtlOpt(*input_design));
        //output_design->scale(1000);
    } else {
        objective_->hessVec(*output_design,*input_design,*design_variables_,tol);
        equality_constraints_->applyAdjointHessian(*temp_design_, *dual_equality_, *input_design, *design_variables_, tol);
        output_design->axpy(one,*temp_design_);

        const Real secantFactor = 0.00;
        if (secantFactor) {
            auto temp_design = output_design->clone();
            secant_->applyB(*getCtlOpt(*temp_design),*getCtlOpt(*input_design));
            getCtlOpt(*output_design)->scale(one-secantFactor);
            getCtlOpt(*output_design)->axpy(secantFactor,*getCtlOpt(*temp_design));
        }

        //pcout << "Adding max eig estimate identity matrix of " << max_eig_estimate_ << std::endl;

        const double identity_plus_eig = add_identity_ + max_eig_estimate_;

        // secant_->applyB(*getCtlOpt(*temp_design_),*getCtlOpt(*input_design));
        // output_design->axpy(one,*temp_design_);
        // output_design->scale(one/2.0);
        if (identity_plus_eig != 0.0) {
            pcout << "Adding identity matrix of " << identity_plus_eig << std::endl;
            getCtlOpt(*output_design)->axpy(identity_plus_eig,*getCtlOpt(*input_design));
        }

        //getCtlOpt(*output_design)->set(*getCtlOpt(*input_design));
        //output_design->scale(1000);
    }
    //objective_->hessVec(*output_design,*input_design,*design_variables_, tol);
    //secant_->applyB(*output_design,*input_design);
    //double add_identity = 1.0; // 10.0
    //getOpt(*output_design)->axpy(add_identity_,*getOpt(*input_design));

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
    if (SYMMETRIZE_MATRIX_) {
        temp_dual_inequality_->set(*input_dual_inequality);
        bound_constraints_->pruneInactive(*temp_dual_inequality_,*des_plus_dual_,bounded_constraint_tolerance_);
        output_design->axpy(one,*temp_dual_inequality_);
    } else {
        output_design->axpy(one,*input_dual_inequality);
    }

    // Rows 5-6: inactive dual_equality, active dual_equality
    //// Columns 1-4
    equality_constraints_->applyJacobian(*output_dual_equality, *input_design, *design_variables_, tol);

    // Rows 7-10: inactive dual_inequality, active dual_inequality
    //// Rows 7 & 9
    output_dual_inequality->zero();
    if (SYMMETRIZE_MATRIX_) {
        // Zeros-out identtity in bottom right corner.
        //temp_dual_inequality_->set(*input_dual_inequality);
        //bound_constraints_->pruneActive(*temp_dual_inequality_,*des_plus_dual_,bounded_constraint_tolerance_);
        //output_dual_inequality->axpy(one,*temp_dual_inequality_);
    } else {
        temp_dual_inequality_->set(*input_dual_inequality);
        bound_constraints_->pruneActive(*temp_dual_inequality_,*des_plus_dual_,bounded_constraint_tolerance_);
        output_dual_inequality->axpy(one,*temp_dual_inequality_);
    }

    //// Rows 8 & 10
    temp_dual_inequality_->set(*input_design);
    bound_constraints_->pruneInactive(*temp_dual_inequality_,*des_plus_dual_,bounded_constraint_tolerance_);
    output_dual_inequality->axpy(one,*temp_dual_inequality_);

}
template class PDAS_KKT_System<double>;

}
