#ifndef __KKT_BIROSGHATTAS_PRECONDITIONERS_H__
#define __KKT_BIROSGHATTAS_PRECONDITIONERS_H__

#include "ROL_SecantStep.hpp"

#include "ROL_Objective_SimOpt.hpp"
#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_Vector_SimOpt.hpp"

#include "flow_constraints.hpp"

/// Full-space system preconditioner based on the reduced-space.
/** See Biros and Ghattas' 2005 paper.
 */
template<typename Real = double>
class BirosGhattasPreconditioner
{
protected:
    /// Objective function.
    const ROL::Ptr<ROL::Objective_SimOpt<Real>> objective_;
    /// Equality constraints.
    const ROL::Ptr<PHiLiP::FlowConstraints<PHILIP_DIM>> equal_constraints_;

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

    /// Use an approximate inverse of the Jacobian and Jacobian transpose using
    /// the preconditioner to obtain the "tilde" operator version of Biros and Ghattas.
    const bool use_approximate_preconditioner_;

protected:
    const unsigned int mpi_rank; ///< MPI rank used to reset the deallog depth
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

public:

    /// Constructor.
    BirosGhattasPreconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> state_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> state_lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant,
        const bool use_approximate_preconditioner = false)
        : objective_
            (ROL::makePtrFromRef<ROL::Objective_SimOpt<Real>>(dynamic_cast<ROL::Objective_SimOpt<Real>&>(*objective)))
        , equal_constraints_
            (ROL::makePtrFromRef<PHiLiP::FlowConstraints<PHILIP_DIM>>(dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*state_constraints)))
        , design_variables_
            (ROL::makePtrFromRef<const ROL::Vector_SimOpt<Real>>(dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*design_variables)))
        , lagrange_mult_(state_lagrange_mult)
        , simulation_variables_(design_variables_->get_1())
        , control_variables_(design_variables_->get_2())
        , secant_(secant)
        , use_approximate_preconditioner_(use_approximate_preconditioner)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
    {
        if (use_approximate_preconditioner_) {
            const int error_precond1 = equal_constraints_->construct_JacobianPreconditioner_1(*simulation_variables_, *control_variables_);
            const int error_precond2 = equal_constraints_->construct_AdjointJacobianPreconditioner_1(*simulation_variables_, *control_variables_);
            assert(error_precond1 == 0);
            assert(error_precond2 == 0);
            (void) error_precond1;
            (void) error_precond2;
        }
    }
    ~BirosGhattasPreconditioner()
    {
        if (use_approximate_preconditioner_) {
            equal_constraints_->destroy_JacobianPreconditioner_1();
            equal_constraints_->destroy_AdjointJacobianPreconditioner_1();
        }
    };

    /// Application of KKT preconditionner on vector src outputted into dst.
    virtual void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        // Identity Preconditioner
        dst.equ(1.0, src);
    }

    /// Application of transposed KKT preconditioner on vector src outputted into dst.
    /** Same as vmult since this KKT preconditioner is symmetric.
     */
    virtual void Tvmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                 const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        vmult(dst, src);
    }
};
/// Full-space system preconditioner based on the reduced-space, with additional constraints.
/** See Biros and Ghattas' 2005 paper.
 */
template<typename Real = double>
class BirosGhattasConstrainedPreconditioner : public BirosGhattasPreconditioner<Real>
{
    /// Constructor.
    BirosGhattasConstrainedPreconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> state_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> state_lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant,
        const bool use_approximate_preconditioner = false)
        : BirosGhattasPreconditioner<Real>(objective, state_constraints, design_variables, state_lagrange_mult, secant, use_approximate_preconditioner)
    { }
};

/// P2 preconditioner from Biros & Ghattas 2005.
/** Second order terms of the Lagrangian Hessian are ignored
 *  and exact inverses of the Jacobian (transpose) are used.
 */
template<typename Real = double>
class KKT_P2_Preconditioner: public BirosGhattasPreconditioner<Real>
{
protected:
    /// Objective function.
    using BirosGhattasPreconditioner<Real>::objective_;
    /// Equality constraints.
    using BirosGhattasPreconditioner<Real>::equal_constraints_;

    /// Design variables.
    using BirosGhattasPreconditioner<Real>::design_variables_;
    /// Lagrange multipliers.
    using BirosGhattasPreconditioner<Real>::lagrange_mult_;

    /// Simulation design variables.
    using BirosGhattasPreconditioner<Real>::simulation_variables_;

    /// Control design variables.
    using BirosGhattasPreconditioner<Real>::control_variables_;

    /// Secant method used to precondition the reduced Hessian.
    using BirosGhattasPreconditioner<Real>::secant_;

    /// Use an approximate inverse of the Jacobian and Jacobian transpose using
    /// the preconditioner to obtain the "tilde" operator version of Biros and Ghattas.
    using BirosGhattasPreconditioner<Real>::use_approximate_preconditioner_;

protected:
    using BirosGhattasPreconditioner<Real>::mpi_rank; ///< MPI rank used to reset the deallog depth
    using BirosGhattasPreconditioner<Real>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

public:
    /// Constructor.
    KKT_P2_Preconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> state_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> state_lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant,
        const bool use_approximate_preconditioner = false)
        : BirosGhattasPreconditioner<Real>(
            objective,
            state_constraints,
            design_variables,
            state_lagrange_mult,
            secant,
            use_approximate_preconditioner)
    { }

    /// Application of KKT preconditionner on vector src outputted into dst.
    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const override
    {
        static int number_of_times = 0;
        number_of_times++;
        pcout << "Number of P2_KKT vmult = " << number_of_times << std::endl;
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
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseAdjointJacobianPreconditioner_1(*dst_3, *rhs_1, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseAdjointJacobian_1(*dst_3, *rhs_1, *simulation_variables_, *control_variables_, tol);
        }

        equal_constraints_->applyAdjointJacobian_2(*rhs_2, *dst_3, *simulation_variables_, *control_variables_, tol);
        rhs_2->scale(-1.0);
        rhs_2->plus(*src_2);
        // Need to apply Hessian inverse on dst_2
        secant_->applyH( *dst_2, *rhs_2);
        // Identity
        //dst_2->set(*rhs_2);

        equal_constraints_->applyJacobian_2(*rhs_3, *dst_2, *simulation_variables_, *control_variables_, tol);
        rhs_3->scale(-1.0);
        rhs_3->plus(*src_3);
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseJacobianPreconditioner_1(*dst_1, *rhs_3, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseJacobian_1(*dst_1, *rhs_3, *simulation_variables_, *control_variables_, tol);
        }

        if (mpi_rank == 0) {
            dealii::deallog.depth_console(99);
        } else {
            dealii::deallog.depth_console(0);
        }

    }

};

/// P4 preconditioner from Biros & Ghattas 2005.
/** Second order terms of the Lagrangian Hessian are ignored
 *  and exact inverses of the Jacobian (transpose) are used.
 */
template<typename Real = double>
class KKT_P4_Preconditioner: public BirosGhattasPreconditioner<Real>
{
protected:
    /// Objective function.
    using BirosGhattasPreconditioner<Real>::objective_;
    /// Equality constraints.
    using BirosGhattasPreconditioner<Real>::equal_constraints_;

    /// Design variables.
    using BirosGhattasPreconditioner<Real>::design_variables_;
    /// Lagrange multipliers.
    using BirosGhattasPreconditioner<Real>::lagrange_mult_;

    /// Simulation design variables.
    using BirosGhattasPreconditioner<Real>::simulation_variables_;

    /// Control design variables.
    using BirosGhattasPreconditioner<Real>::control_variables_;

    /// Secant method used to precondition the reduced Hessian.
    using BirosGhattasPreconditioner<Real>::secant_;

    /// Use an approximate inverse of the Jacobian and Jacobian transpose using
    /// the preconditioner to obtain the "tilde" operator version of Biros and Ghattas.
    using BirosGhattasPreconditioner<Real>::use_approximate_preconditioner_;

protected:
    using BirosGhattasPreconditioner<Real>::mpi_rank; ///< MPI rank used to reset the deallog depth
    using BirosGhattasPreconditioner<Real>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

public:
    /// Constructor.
    KKT_P4_Preconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> state_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> state_lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant,
        const bool use_approximate_preconditioner = false)
        : BirosGhattasPreconditioner<Real>(objective, state_constraints, design_variables, state_lagrange_mult, secant, use_approximate_preconditioner)
    { };

    /// Application of KKT preconditionner on vector src outputted into dst.
    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const override
    {
        static int number_of_times = 0;
        number_of_times++;
        pcout << "Number of P4_KKT vmult = " << number_of_times << std::endl;
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
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseJacobianPreconditioner_1(*Asinv_y_1, *y_1, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseJacobian_1(*Asinv_y_1, *y_1, *simulation_variables_, *control_variables_, tol);
        }

        y_3->set(*src_1);
        equal_constraints_->applyAdjointHessian_11 (*temp_1, *lagrange_mult_, *Asinv_y_1, *simulation_variables_, *control_variables_, tol);
        y_3->axpy(-1.0, *temp_1);
        objective_->hessVec_11(*temp_1, *Asinv_y_1, *simulation_variables_, *control_variables_, tol);
        y_3->axpy(-1.0, *temp_1);

        auto AsTinv_y_3 = y_3->clone();
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseAdjointJacobianPreconditioner_1(*AsTinv_y_3, *y_3, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseAdjointJacobian_1(*AsTinv_y_3, *y_3, *simulation_variables_, *control_variables_, tol);
        }
        equal_constraints_->applyAdjointJacobian_2(*y_2, *AsTinv_y_3, *simulation_variables_, *control_variables_, tol);
        y_2->scale(-1.0);
        y_2->plus(*src_2);

        auto temp_2 = y_2->clone();
        equal_constraints_->applyAdjointHessian_12(*temp_2, *lagrange_mult_, *Asinv_y_1, *simulation_variables_, *control_variables_, tol);
        y_2->axpy(-1.0,*temp_2);
        objective_->hessVec_21(*temp_2, *Asinv_y_1, *simulation_variables_, *control_variables_, tol);
        y_2->axpy(-1.0,*temp_2);

        secant_->applyH( *dst_2, *y_2);
        //dst_2->set(*y_2);

        auto Asinv_Ad_dst_2 = y_1->clone();
        equal_constraints_->applyJacobian_2(*temp_1, *dst_2, *simulation_variables_, *control_variables_, tol);
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseJacobianPreconditioner_1(*Asinv_Ad_dst_2, *temp_1, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseJacobian_1(*Asinv_Ad_dst_2, *temp_1, *simulation_variables_, *control_variables_, tol);
        }

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

        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseAdjointJacobianPreconditioner_1(*dst_3, *dst_3_rhs, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseAdjointJacobian_1(*dst_3, *dst_3_rhs, *simulation_variables_, *control_variables_, tol);
        }

        if (mpi_rank == 0) {
            dealii::deallog.depth_console(99);
        } else {
            dealii::deallog.depth_console(0);
        }

    }

};
/// Preconditioners from Biros & Ghattas 2005.
/** Option to use or ignore second-order term to obtain P4 or P2 preconditioners
 *  Option to use approximate or exact inverses of the Jacobian (transpose) to obtain the Tilde version.
 */
template<typename Real = double>
class KKT_P24_Constrained_Preconditioner: public BirosGhattasPreconditioner<Real>
{
protected:
    /// Objective function.
    using BirosGhattasPreconditioner<Real>::objective_;
    /// State equality constraints.
    using BirosGhattasPreconditioner<Real>::equal_constraints_;

    /// Additional equality constraints.
    const ROL::Ptr<ROL::Constraint_SimOpt<Real>> other_equal_constraints_;

    /// Design variables.
    using BirosGhattasPreconditioner<Real>::design_variables_;
    /// Lagrange multipliers.
    using BirosGhattasPreconditioner<Real>::lagrange_mult_;

    /// Simulation design variables.
    using BirosGhattasPreconditioner<Real>::simulation_variables_;

    /// Control design variables.
    using BirosGhattasPreconditioner<Real>::control_variables_;

    /// Secant method used to precondition the reduced Hessian.
    using BirosGhattasPreconditioner<Real>::secant_;

    /// Use an approximate inverse of the Jacobian and Jacobian transpose using
    /// the preconditioner to obtain the "tilde" operator version of Biros and Ghattas.
    using BirosGhattasPreconditioner<Real>::use_approximate_preconditioner_;

protected:
    using BirosGhattasPreconditioner<Real>::mpi_rank; ///< MPI rank used to reset the deallog depth
    using BirosGhattasPreconditioner<Real>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

    ROL::Ptr<ROL::Vector<Real>> y1;
    ROL::Ptr<ROL::Vector<Real>> y2;
    ROL::Ptr<ROL::Vector<Real>> y3;

    ROL::Ptr<ROL::Vector<Real>> temp_1;
    ROL::Ptr<ROL::Vector<Real>> Lxs_Rsinv_y1;
    ROL::Ptr<ROL::Vector<Real>> Rsinv_y1;

    bool use_second_order_terms;
public:
    /// Constructor.
    KKT_P24_Constrained_Preconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> state_constraints,
        const ROL::Ptr<ROL::Constraint<Real>> other_equal_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> state_lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant,
        const bool use_second_order_terms,
        const bool use_approximate_preconditioner = false)
        : BirosGhattasPreconditioner<Real>(objective,
                                           state_constraints,
                                           design_variables,
                                           state_lagrange_mult,
                                           secant,
                                           use_approximate_preconditioner)
        , other_equal_constraints_
            (ROL::makePtrFromRef<ROL::Constraint_SimOpt<Real>>(dynamic_cast<ROL::Constraint_SimOpt<Real>&>(*other_equal_constraints_)))
        , use_second_order_terms(use_second_order_terms)
    {
        const auto &design_simopt = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*design_variables);

        const ROL::Ptr<const ROL::Vector<Real>> input_1 = design_simopt.get_1();
        const ROL::Ptr<const ROL::Vector<Real>> input_2 = design_simopt.get_2();
        const ROL::Ptr<const ROL::Vector<Real>> input_3 = state_lagrange_mult;

        y1 = input_1->clone();
        y2 = input_2->clone();
        y3 = input_3->clone();

        temp_1 = input_1->clone();
        if (use_second_order_terms) {
            Rsinv_y1 = input_1->clone();
            Lxs_Rsinv_y1 = input_2->clone();
        }
    };

    /// Application of KKT preconditionner on vector src outputted into dst.
    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const override
    {
        static int number_of_times = 0;
        number_of_times++;
        pcout << "Number of P4_KKT vmult = " << number_of_times << std::endl;
        Real tol = 1e-15;
        //const Real one = 1.0;

        ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();
        auto &dst_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_rol);
        ROL::Ptr<ROL::Vector<Real>> dst_design = dst_split.get_1();
        auto &dst_design_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_design);

        ROL::Ptr<ROL::Vector<Real>> x1 = dst_design_split.get_1();
        ROL::Ptr<ROL::Vector<Real>> x2 = dst_design_split.get_2();
        ROL::Ptr<ROL::Vector<Real>> x3 = dst_split.get_2();

        const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();
        const auto &src_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_rol);
        const ROL::Ptr<const ROL::Vector<Real>> src_design = src_split.get_1();
        const auto &src_design_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_design);

        const ROL::Ptr<const ROL::Vector<Real>> z1 = src_design_split.get_1();
        const ROL::Ptr<const ROL::Vector<Real>> z2 = src_design_split.get_2();
        const ROL::Ptr<const ROL::Vector<Real>> z3 = src_split.get_2();


        // Evaluate y ********************

        // Evaluate y1 = z3
        y1->set(*z3);

        // Evaluate y3 = z1 - Lss Rs^{-1} y1
        y3->set(*z1);

        if (use_second_order_terms) {
            if (use_approximate_preconditioner_) {
                equal_constraints_->applyInverseJacobianPreconditioner_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
            } else {
                equal_constraints_->applyInverseJacobian_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
            }

            equal_constraints_->applyAdjointHessian_11 (*temp_1, *lagrange_mult_, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
            y3->axpy(-1.0, *temp_1);
            objective_->hessVec_11(*temp_1, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
            y3->axpy(-1.0, *temp_1);
        }

        // Evaluate y2 = z2 - Rx^{T} Rs^{-T} y3 - Lxs Rs^{-1} y1  
        auto RsTinv_y3 = y3->clone();
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseAdjointJacobianPreconditioner_1(*RsTinv_y3, *y3, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseAdjointJacobian_1(*RsTinv_y3, *y3, *simulation_variables_, *control_variables_, tol);
        }
        equal_constraints_->applyAdjointJacobian_2(*y2, *RsTinv_y3, *simulation_variables_, *control_variables_, tol);
        y2->scale(-1.0);
        y2->plus(*z2);

        if (use_second_order_terms) {
            equal_constraints_->applyAdjointHessian_12(*Lxs_Rsinv_y1, *lagrange_mult_, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
            y2->axpy(-1.0,*Lxs_Rsinv_y1);
            objective_->hessVec_21(*Lxs_Rsinv_y1, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
            y2->axpy(-1.0,*Lxs_Rsinv_y1);
        }

        // Evaluate x ********************

        // x2 = Lzz^{-1} y2
        const bool use_secant = true;
        if (use_secant) {
            secant_->applyH( *x2, *y2);
        } else {
            x2->set(*y2);
        }

        // x1 = Rs^{-1} (y1 - Ad x2)

        // temp1 = y1 - Ad x2
        equal_constraints_->applyJacobian_2(*temp_1, *x2, *simulation_variables_, *control_variables_, tol);
        temp_1->scale(-1.0);
        temp_1->axpy(1.0, *y1);

        auto Rsinv_y1_minus_Ad_x2 = y1->clone();
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseJacobianPreconditioner_1(*x1, *temp_1, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseJacobian_1(*x1, *temp_1, *simulation_variables_, *control_variables_, tol);
        }

        // x3 = Rs^{-T} x3_rhs
        // x3_rhs  = y3
        auto x3_rhs = y3->clone();

        if (use_second_order_terms) {

            // x3_rhs += -(Lsx - Lss Rs^{-1} Rx x2)

            auto negative_Rsinv_Ad_x2 = Rsinv_y1_minus_Ad_x2;
            negative_Rsinv_Ad_x2->axpy(-1.0, *Rsinv_y1);

            equal_constraints_->applyAdjointHessian_11 (*temp_1, *lagrange_mult_, *negative_Rsinv_Ad_x2, *simulation_variables_, *control_variables_, tol);
            x3_rhs->axpy(-1.0, *temp_1);
            objective_->hessVec_11(*temp_1, *negative_Rsinv_Ad_x2, *simulation_variables_, *control_variables_, tol);
            x3_rhs->axpy(-1.0, *temp_1);

            equal_constraints_->applyAdjointHessian_21 (*temp_1, *lagrange_mult_, *x2, *simulation_variables_, *control_variables_, tol);
            x3_rhs->axpy(-1.0, *temp_1);
            objective_->hessVec_12(*temp_1, *x2, *simulation_variables_, *control_variables_, tol);
            x3_rhs->axpy(-1.0, *temp_1);
        }

        // x3 = Rs^{-T} x3_rhs
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseAdjointJacobianPreconditioner_1(*x3, *x3_rhs, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseAdjointJacobian_1(*x3, *x3_rhs, *simulation_variables_, *control_variables_, tol);
        }

        if (mpi_rank == 0) {
            dealii::deallog.depth_console(99);
        } else {
            dealii::deallog.depth_console(0);
        }

    }

};


/// Preconditioners from Biros & Ghattas 2005.
/** Option to use or ignore second-order term to obtain P4 or P2 preconditioners
 *  Option to use approximate or exact inverses of the Jacobian (transpose) to obtain the Tilde version.
 */
template<typename Real = double>
class KKT_P24_Preconditioner: public BirosGhattasPreconditioner<Real>
{
protected:
    /// Objective function.
    using BirosGhattasPreconditioner<Real>::objective_;
    /// Equality constraints.
    using BirosGhattasPreconditioner<Real>::equal_constraints_;

    /// Design variables.
    using BirosGhattasPreconditioner<Real>::design_variables_;
    /// Lagrange multipliers.
    using BirosGhattasPreconditioner<Real>::lagrange_mult_;

    /// Simulation design variables.
    using BirosGhattasPreconditioner<Real>::simulation_variables_;

    /// Control design variables.
    using BirosGhattasPreconditioner<Real>::control_variables_;

    /// Secant method used to precondition the reduced Hessian.
    using BirosGhattasPreconditioner<Real>::secant_;

    /// Use an approximate inverse of the Jacobian and Jacobian transpose using
    /// the preconditioner to obtain the "tilde" operator version of Biros and Ghattas.
    using BirosGhattasPreconditioner<Real>::use_approximate_preconditioner_;

protected:
    using BirosGhattasPreconditioner<Real>::mpi_rank; ///< MPI rank used to reset the deallog depth
    using BirosGhattasPreconditioner<Real>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

    ROL::Ptr<ROL::Vector<Real>> y1;
    ROL::Ptr<ROL::Vector<Real>> y2;
    ROL::Ptr<ROL::Vector<Real>> y3;

    ROL::Ptr<ROL::Vector<Real>> temp_1;
    ROL::Ptr<ROL::Vector<Real>> Lxs_Rsinv_y1;
    ROL::Ptr<ROL::Vector<Real>> Rsinv_y1;

    bool use_second_order_terms;
public:
    /// Constructor.
    KKT_P24_Preconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> state_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> state_lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant,
        const bool use_second_order_terms,
        const bool use_approximate_preconditioner = false)
        : BirosGhattasPreconditioner<Real>(objective,
                                           state_constraints,
                                           design_variables,
                                           state_lagrange_mult,
                                           secant,
                                           use_approximate_preconditioner)
        , use_second_order_terms(use_second_order_terms)
    {
        const auto &design_simopt = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*design_variables);

        const ROL::Ptr<const ROL::Vector<Real>> input_1 = design_simopt.get_1();
        const ROL::Ptr<const ROL::Vector<Real>> input_2 = design_simopt.get_2();
        const ROL::Ptr<const ROL::Vector<Real>> input_3 = state_lagrange_mult;

        y1 = input_1->clone();
        y2 = input_2->clone();
        y3 = input_3->clone();

        temp_1 = input_1->clone();
        if (use_second_order_terms) {
            Rsinv_y1 = input_1->clone();
            Lxs_Rsinv_y1 = input_2->clone();
        }
    };

    /// Application of KKT preconditionner on vector src outputted into dst.
    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const override
    {
        static int number_of_times = 0;
        number_of_times++;
        pcout << "Number of P4_KKT vmult = " << number_of_times << std::endl;
        Real tol = 1e-15;
        //const Real one = 1.0;

        ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();
        auto &dst_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_rol);
        ROL::Ptr<ROL::Vector<Real>> dst_design = dst_split.get_1();
        auto &dst_design_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_design);

        ROL::Ptr<ROL::Vector<Real>> x1 = dst_design_split.get_1();
        ROL::Ptr<ROL::Vector<Real>> x2 = dst_design_split.get_2();
        ROL::Ptr<ROL::Vector<Real>> x3 = dst_split.get_2();

        const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();
        const auto &src_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_rol);
        const ROL::Ptr<const ROL::Vector<Real>> src_design = src_split.get_1();
        const auto &src_design_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_design);

        const ROL::Ptr<const ROL::Vector<Real>> z1 = src_design_split.get_1();
        const ROL::Ptr<const ROL::Vector<Real>> z2 = src_design_split.get_2();
        const ROL::Ptr<const ROL::Vector<Real>> z3 = src_split.get_2();


        // Evaluate y ********************

        // Evaluate y1 = z3
        y1->set(*z3);

        // Evaluate y3 = z1 - Lss Rs^{-1} y1
        y3->set(*z1);

        if (use_second_order_terms) {
            if (use_approximate_preconditioner_) {
                equal_constraints_->applyInverseJacobianPreconditioner_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
            } else {
                equal_constraints_->applyInverseJacobian_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
            }

            equal_constraints_->applyAdjointHessian_11 (*temp_1, *lagrange_mult_, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
            y3->axpy(-1.0, *temp_1);
            objective_->hessVec_11(*temp_1, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
            y3->axpy(-1.0, *temp_1);
        }

        // Evaluate y2 = z2 - Rx^{T} Rs^{-T} y3 - Lxs Rs^{-1} y1  
        auto RsTinv_y3 = y3->clone();
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseAdjointJacobianPreconditioner_1(*RsTinv_y3, *y3, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseAdjointJacobian_1(*RsTinv_y3, *y3, *simulation_variables_, *control_variables_, tol);
        }
        equal_constraints_->applyAdjointJacobian_2(*y2, *RsTinv_y3, *simulation_variables_, *control_variables_, tol);
        y2->scale(-1.0);
        y2->plus(*z2);

        if (use_second_order_terms) {
            equal_constraints_->applyAdjointHessian_12(*Lxs_Rsinv_y1, *lagrange_mult_, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
            y2->axpy(-1.0,*Lxs_Rsinv_y1);
            objective_->hessVec_21(*Lxs_Rsinv_y1, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
            y2->axpy(-1.0,*Lxs_Rsinv_y1);
        }

        // Evaluate x ********************

        // x2 = Lzz^{-1} y2
        const bool use_secant = true;
        if (use_secant) {
            secant_->applyH( *x2, *y2);
        } else {
            x2->set(*y2);
        }

        // x1 = Rs^{-1} (y1 - Ad x2)

        // temp1 = y1 - Ad x2
        equal_constraints_->applyJacobian_2(*temp_1, *x2, *simulation_variables_, *control_variables_, tol);
        temp_1->scale(-1.0);
        temp_1->axpy(1.0, *y1);

        auto Rsinv_y1_minus_Ad_x2 = y1->clone();
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseJacobianPreconditioner_1(*x1, *temp_1, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseJacobian_1(*x1, *temp_1, *simulation_variables_, *control_variables_, tol);
        }

        // x3 = Rs^{-T} x3_rhs
        // x3_rhs  = y3
        auto x3_rhs = y3->clone();

        if (use_second_order_terms) {

            // x3_rhs += -(Lsx - Lss Rs^{-1} Rx x2)

            auto negative_Rsinv_Ad_x2 = Rsinv_y1_minus_Ad_x2;
            negative_Rsinv_Ad_x2->axpy(-1.0, *Rsinv_y1);

            equal_constraints_->applyAdjointHessian_11 (*temp_1, *lagrange_mult_, *negative_Rsinv_Ad_x2, *simulation_variables_, *control_variables_, tol);
            x3_rhs->axpy(-1.0, *temp_1);
            objective_->hessVec_11(*temp_1, *negative_Rsinv_Ad_x2, *simulation_variables_, *control_variables_, tol);
            x3_rhs->axpy(-1.0, *temp_1);

            equal_constraints_->applyAdjointHessian_21 (*temp_1, *lagrange_mult_, *x2, *simulation_variables_, *control_variables_, tol);
            x3_rhs->axpy(-1.0, *temp_1);
            objective_->hessVec_12(*temp_1, *x2, *simulation_variables_, *control_variables_, tol);
            x3_rhs->axpy(-1.0, *temp_1);
        }

        // x3 = Rs^{-T} x3_rhs
        if (use_approximate_preconditioner_) {
            equal_constraints_->applyInverseAdjointJacobianPreconditioner_1(*x3, *x3_rhs, *simulation_variables_, *control_variables_, tol);
        } else {
            equal_constraints_->applyInverseAdjointJacobian_1(*x3, *x3_rhs, *simulation_variables_, *control_variables_, tol);
        }

        if (mpi_rank == 0) {
            dealii::deallog.depth_console(99);
        } else {
            dealii::deallog.depth_console(0);
        }

    }

};

/// Identity preconditioner.
template<typename Real = double>
class KKT_Identity_Preconditioner: public BirosGhattasPreconditioner<Real>
{
public:
    /// Constructor.
    KKT_Identity_Preconditioner(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> state_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> state_lagrange_mult,
        const ROL::Ptr<ROL::Secant<Real> > secant,
        const bool use_approximate_preconditioner = false)
        : BirosGhattasPreconditioner<Real>(objective, state_constraints, design_variables, state_lagrange_mult, secant, use_approximate_preconditioner)
    { }

    /// Application of KKT preconditionner on vector src outputted into dst.
    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const override
    {
        dst.equ(1.0,src);
    }

};

/// Full-space preconditioner from factory.
template<typename Real = double>
class BirosGhattasPreconditionerFactory
{
public:
    /// Creates a derived preconditioner object, but returns it as BirosGhattasPreconditioner.
    static std::shared_ptr< BirosGhattasPreconditioner<Real> >
        create_KKT_preconditioner( ROL::ParameterList &parlist,
                                   ROL::Objective<Real> &objective,
                                   ROL::Constraint<Real> &state_constraints,
                                   const ROL::Vector<Real> &design_variables,
                                   const ROL::Vector<Real> &state_lagrange_mult,
                                   const ROL::Ptr< ROL::Secant<Real> > secant_)
    {
        const std::string preconditioner_name_ = parlist.sublist("Full Space").get("Preconditioner","Identity"); 
        const bool use_approximate_full_space_preconditioner_ = (preconditioner_name_ == "P2A" || preconditioner_name_ == "P4A");

        if (preconditioner_name_ == "P2" || preconditioner_name_ == "P2A") {
            return std::make_shared<KKT_P2_Preconditioner<Real>> (
                ROL::makePtrFromRef<ROL::Objective<Real>>(objective),
                ROL::makePtrFromRef<ROL::Constraint<Real>>(state_constraints),
                ROL::makePtrFromRef<const ROL::Vector<Real>>(design_variables),
                ROL::makePtrFromRef<const ROL::Vector<Real>>(state_lagrange_mult),
                secant_,
                use_approximate_full_space_preconditioner_);
        } else if (preconditioner_name_ == "P4" || preconditioner_name_ == "P4A") {
            return std::make_shared<KKT_P4_Preconditioner<Real>> (
                ROL::makePtrFromRef<ROL::Objective<Real>>(objective),
                ROL::makePtrFromRef<ROL::Constraint<Real>>(state_constraints),
                ROL::makePtrFromRef<const ROL::Vector<Real>>(design_variables),
                ROL::makePtrFromRef<const ROL::Vector<Real>>(state_lagrange_mult),
                secant_,
                use_approximate_full_space_preconditioner_);
        } else {
            return std::make_shared<KKT_Identity_Preconditioner<Real>> (
                ROL::makePtrFromRef<ROL::Objective<Real>>(objective),
                ROL::makePtrFromRef<ROL::Constraint<Real>>(state_constraints),
                ROL::makePtrFromRef<const ROL::Vector<Real>>(design_variables),
                ROL::makePtrFromRef<const ROL::Vector<Real>>(state_lagrange_mult),
                secant_,
                false);
        }
    }
};

#endif
