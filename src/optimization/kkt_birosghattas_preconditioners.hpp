#ifndef __KKT_BIROSGHATTAS_PRECONDITIONERS_H__
#define __KKT_BIROSGHATTAS_PRECONDITIONERS_H__

#include "ROL_SecantStep.hpp"

#include "ROL_Objective_SimOpt.hpp"
#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_Vector_SimOpt.hpp"

/// P2 preconditioner from Biros & Ghattas 2005.
/** Second order terms of the Lagrangian Hessian are ignored
 *  and exact inverses of the Jacobian (transpose) are used.
 */
template<typename Real = double>
class KKT_P2_Preconditioner
{
    /// Objective function.
    const ROL::Ptr<ROL::Objective_SimOpt<Real>> objective_;
    /// Equality constraints.
    const ROL::Ptr<ROL::Constraint_SimOpt<Real>> equal_constraints_;

    /// Design variables.
    const ROL::Ptr<const ROL::Vector_SimOpt<Real>> design_variables_;
    /// Lagrange multipliers.
    const ROL::Ptr<const ROL::Vector<Real>> lagrange_mult_;

    /// Simulation design variables.
    const ROL::Ptr<const ROL::Vector<Real>> simulation_variables_;

    /// Control design variables.
    const ROL::Ptr<const ROL::Vector<Real>> control_variables_;

    /// Secant method used to precondition the reduced Hessian.
    const ROL::Ptr<ROL::Secant<Real> > secant_;

public:
    /// Constructor.
    KKT_P2_Preconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> equal_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant)
        : objective_
            (ROL::makePtrFromRef<ROL::Objective_SimOpt<Real>>(dynamic_cast<ROL::Objective_SimOpt<Real>&>(*objective)))
        , equal_constraints_
            (ROL::makePtrFromRef<ROL::Constraint_SimOpt<Real>>(dynamic_cast<ROL::Constraint_SimOpt<Real>&>(*equal_constraints)))
        , design_variables_
            (ROL::makePtrFromRef<const ROL::Vector_SimOpt<Real>>(dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*design_variables)))
        , lagrange_mult_(lagrange_mult)
        , simulation_variables_(design_variables_->get_1())
        , control_variables_(design_variables_->get_2())
        , secant_(secant)
    { };

    /// Application of KKT preconditionner on vector src outputted into dst.
    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        static int number_of_times = 0;
        number_of_times++;
        std::cout << "Number of P2_KKT vmult = " << number_of_times << std::endl;
        Real tol = 1e-15;
        //const Real one = 1.0;

        ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();
        auto &dst_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_rol);
        ROL::Ptr<ROL::Vector<Real>> dst_design = dst_split.get_1();
        auto &dst_design_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_design);

        ROL::Ptr<ROL::Vector<Real>> dst_1 = dst_design_split.get_1();
        ROL::Ptr<ROL::Vector<Real>> dst_2 = dst_design_split.get_2();
        ROL::Ptr<ROL::Vector<Real>> dst_3 = dst_split.get_2();

        const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();
        const auto &src_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_rol);
        const ROL::Ptr<const ROL::Vector<Real>> src_design = src_split.get_1();
        const auto &src_design_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_design);

        const ROL::Ptr<const ROL::Vector<Real>> src_1 = src_design_split.get_1();
        const ROL::Ptr<const ROL::Vector<Real>> src_2 = src_design_split.get_2();
        const ROL::Ptr<const ROL::Vector<Real>> src_3 = src_split.get_2();

        ROL::Ptr<ROL::Vector<Real>> rhs_1 = src_1->clone();
        ROL::Ptr<ROL::Vector<Real>> rhs_2 = src_2->clone();
        ROL::Ptr<ROL::Vector<Real>> rhs_3 = src_3->clone();
        equal_constraints_->applyInverseAdjointJacobian_1(*dst_3, *rhs_1, *simulation_variables_, *control_variables_, tol);

        equal_constraints_->applyAdjointJacobian_2(*rhs_2, *dst_3, *simulation_variables_, *control_variables_, tol);
        rhs_2->scale(-1.0);
        rhs_2->plus(*src_2);
        // Need to apply Hessian inverse on dst_2
        //secant_->applyH( *dst_2, *rhs_2);
        // Identity
        dst_2->set(*rhs_2);

        equal_constraints_->applyJacobian_2(*rhs_3, *dst_2, *simulation_variables_, *control_variables_, tol);
        rhs_3->scale(-1.0);
        rhs_3->plus(*src_3);
        equal_constraints_->applyInverseJacobian_1(*dst_1, *rhs_3, *simulation_variables_, *control_variables_, tol);

        // Identity Preconditioner
        // dst_1->set(*src_1);
        // dst_2->set(*src_2);
        // dst_3->set(*src_3);
        dealii::deallog.depth_console(99);

    }

    /// Application of transposed KKT preconditioner on vector src outputted into dst.
    /** Same as vmult since this KKT preconditioner is symmetric.
     */
    void Tvmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                 const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        vmult(dst, src);
    }
};

/// P4 preconditioner from Biros & Ghattas 2005.
/** Second order terms of the Lagrangian Hessian are ignored
 *  and exact inverses of the Jacobian (transpose) are used.
 */
template<typename Real = double>
class KKT_P4_Preconditioner
{
    /// Objective function.
    const ROL::Ptr<ROL::Objective_SimOpt<Real>> objective_;
    /// Equality constraints.
    const ROL::Ptr<ROL::Constraint_SimOpt<Real>> equal_constraints_;

    /// Design variables.
    const ROL::Ptr<const ROL::Vector_SimOpt<Real>> design_variables_;
    /// Lagrange multipliers.
    const ROL::Ptr<const ROL::Vector<Real>> lagrange_mult_;

    /// Simulation design variables.
    const ROL::Ptr<const ROL::Vector<Real>> simulation_variables_;

    /// Control design variables.
    const ROL::Ptr<const ROL::Vector<Real>> control_variables_;

    /// Secant method used to precondition the reduced Hessian.
    const ROL::Ptr<ROL::Secant<Real> > secant_;

public:
    /// Constructor.
    KKT_P4_Preconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> equal_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant)
        : objective_
            (ROL::makePtrFromRef<ROL::Objective_SimOpt<Real>>(dynamic_cast<ROL::Objective_SimOpt<Real>&>(*objective)))
        , equal_constraints_
            (ROL::makePtrFromRef<ROL::Constraint_SimOpt<Real>>(dynamic_cast<ROL::Constraint_SimOpt<Real>&>(*equal_constraints)))
        , design_variables_
            (ROL::makePtrFromRef<const ROL::Vector_SimOpt<Real>>(dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*design_variables)))
        , lagrange_mult_(lagrange_mult)
        , simulation_variables_(design_variables_->get_1())
        , control_variables_(design_variables_->get_2())
        , secant_(secant)
    { };

    /// Application of KKT preconditionner on vector src outputted into dst.
    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        static int number_of_times = 0;
        number_of_times++;
        std::cout << "Number of P4_KKT vmult = " << number_of_times << std::endl;
        Real tol = 1e-15;
        //const Real one = 1.0;

        ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();
        auto &dst_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_rol);
        ROL::Ptr<ROL::Vector<Real>> dst_design = dst_split.get_1();
        auto &dst_design_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_design);

        ROL::Ptr<ROL::Vector<Real>> dst_1 = dst_design_split.get_1();
        ROL::Ptr<ROL::Vector<Real>> dst_2 = dst_design_split.get_2();
        ROL::Ptr<ROL::Vector<Real>> dst_3 = dst_split.get_2();

        const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();
        const auto &src_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_rol);
        const ROL::Ptr<const ROL::Vector<Real>> src_design = src_split.get_1();
        const auto &src_design_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_design);

        const ROL::Ptr<const ROL::Vector<Real>> src_1 = src_design_split.get_1();
        const ROL::Ptr<const ROL::Vector<Real>> src_2 = src_design_split.get_2();
        const ROL::Ptr<const ROL::Vector<Real>> src_3 = src_split.get_2();

        ROL::Ptr<ROL::Vector<Real>> rhs_1 = src_1->clone();
        ROL::Ptr<ROL::Vector<Real>> rhs_2 = src_2->clone();
        ROL::Ptr<ROL::Vector<Real>> rhs_3 = src_3->clone();

        ROL::Ptr<ROL::Vector<Real>> y_1 = src_3->clone();
        ROL::Ptr<ROL::Vector<Real>> y_2 = src_2->clone();
        ROL::Ptr<ROL::Vector<Real>> y_3 = src_3->clone();

        y_1->set(*src_3);

        auto Asinv_y_1 = y_1->clone();
        auto temp_1 = y_1->clone();
        equal_constraints_->applyInverseJacobian_1(*Asinv_y_1, *y_1, *simulation_variables_, *control_variables_, tol);

        y_3->set(*src_1);
        equal_constraints_->applyAdjointHessian_11 (*temp_1, *lagrange_mult_, *Asinv_y_1, *simulation_variables_, *control_variables_, tol);
        y_3->axpy(-1.0, *temp_1);
        objective_->hessVec_11(*temp_1, *Asinv_y_1, *simulation_variables_, *control_variables_, tol);
        y_3->axpy(-1.0, *temp_1);

        auto AsTinv_y_3 = y_3->clone();
        equal_constraints_->applyInverseAdjointJacobian_1(*AsTinv_y_3, *y_3, *simulation_variables_, *control_variables_, tol);
        equal_constraints_->applyAdjointJacobian_2(*y_2, *AsTinv_y_3, *simulation_variables_, *control_variables_, tol);
        y_2->scale(-1.0);
        y_2->plus(*src_2);

        auto temp_2 = y_2->clone();
        equal_constraints_->applyAdjointHessian_12(*temp_2, *lagrange_mult_, *Asinv_y_1, *simulation_variables_, *control_variables_, tol);
        y_2->axpy(-1.0,*temp_2);
        objective_->hessVec_21(*temp_2, *Asinv_y_1, *simulation_variables_, *control_variables_, tol);
        y_2->axpy(-1.0,*temp_2);

        //secant_->applyH( *dst_2, *y_2);
        dst_2->set(*y_2);

        auto Asinv_Ad_dst_2 = y_1->clone();
        equal_constraints_->applyJacobian_2(*temp_1, *dst_2, *simulation_variables_, *control_variables_, tol);
        equal_constraints_->applyInverseJacobian_1(*Asinv_Ad_dst_2, *temp_1, *simulation_variables_, *control_variables_, tol);

        dst_1->set(*Asinv_y_1);
        dst_1->axpy(-1.0, *Asinv_Ad_dst_2);

        auto dst_3_rhs = y_3->clone();
        equal_constraints_->applyAdjointHessian_11 (*temp_1, *lagrange_mult_, *Asinv_Ad_dst_2, *simulation_variables_, *control_variables_, tol);
        dst_3_rhs->axpy(1.0, *temp_1);
        objective_->hessVec_11(*temp_1, *Asinv_Ad_dst_2, *simulation_variables_, *control_variables_, tol);
        dst_3_rhs->axpy(1.0, *temp_1);

        equal_constraints_->applyAdjointHessian_21 (*temp_1, *lagrange_mult_, *dst_2, *simulation_variables_, *control_variables_, tol);
        dst_3_rhs->axpy(-1.0, *temp_1);
        objective_->hessVec_12(*temp_1, *dst_2, *simulation_variables_, *control_variables_, tol);
        dst_3_rhs->axpy(-1.0, *temp_1);

        equal_constraints_->applyInverseAdjointJacobian_1(*dst_3, *dst_3_rhs, *simulation_variables_, *control_variables_, tol);

        // Identity Preconditioner
        // dst_1->set(*src_1);
        // dst_2->set(*src_2);
        // dst_3->set(*src_3);
        dealii::deallog.depth_console(99);

    }

    /// Application of transposed KKT preconditioner on vector src outputted into dst.
    /** Same as vmult since this KKT preconditioner is symmetric.
     */
    void Tvmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                 const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        vmult(dst, src);
    }
};

#endif
