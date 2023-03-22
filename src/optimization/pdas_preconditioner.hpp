#ifndef PHILIP_PDAS_PRECONDITIONER_HPP
#define PHILIP_PDAS_PRECONDITIONER_HPP

#include "ROL_LineSearch.hpp"
#include "ROL_AugmentedLagrangian.hpp"

#include "ROL_Constraint_Partitioned.hpp"
#include "ROL_Objective_SimOpt.hpp"
#include "ROL_SlacklessObjective.hpp"
#include "ROL_KrylovFactory.hpp"

#include <deal.II/lac/full_matrix.h>

#include "optimization/flow_constraints.hpp"
#include "primal_dual_active_set.hpp"
#include "optimization/dealii_solver_rol_vector.hpp"

//#define REPLACE_INVERSE_WITH_IDENTITY
namespace PHiLiP {

template<typename Real = double>
class Dealii_LinearOperator_From_ROL_LinearOperator
{
    ROL::Ptr<ROL::LinearOperator<Real>> rol_linear_operator;
public:
    Dealii_LinearOperator_From_ROL_LinearOperator(ROL::Ptr<ROL::LinearOperator<Real>> rol_linear_operator)
    : rol_linear_operator(rol_linear_operator)
    { };

    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();
        ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();

        double tol = 1e-15;
        rol_linear_operator->apply(*dst_rol, *src_rol, tol);
    }

    
};

template<typename Real = double>
class Dealii_Preconditioner_From_ROL_LinearOperator
{
    ROL::Ptr<ROL::LinearOperator<Real>> rol_linear_operator;
public:
    Dealii_Preconditioner_From_ROL_LinearOperator(ROL::Ptr<ROL::LinearOperator<Real>> rol_linear_operator)
    : rol_linear_operator(rol_linear_operator)
    { };

    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();
        ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();

        double tol = 1e-15;
        rol_linear_operator->applyInverse(*dst_rol, *src_rol, tol);
    }

    
};

/// Preconditioners from Biros & Ghattas 2005 with additional constraints.
/** Option to use or ignore second-order term to obtain P4 or P2 preconditioners
 *  Option to use approximate or exact inverses of the Jacobian (transpose) to obtain the Tilde version.
 */
template<typename Real = double>
class PDAS_P24_Constrained_Preconditioner: public ROL::LinearOperator<Real>
{

protected:
    const ROL::Ptr<const ROL::PartitionedVector<Real>>  design_variables_;     ///< Design variables.

    const ROL::Ptr<ROL::Objective_SimOpt<Real>>         objective_;            ///< Objective function.

    const ROL::Ptr<PHiLiP::FlowConstraints<PHILIP_DIM>> state_constraints_; ///< Equality constraints.
    const ROL::Ptr<const ROL::Vector<Real>>             dual_state_;        ///< Lagrange multipliers associated with state coonstraints.

    const ROL::Ptr<ROL::Constraint_Partitioned<Real>>        equality_constraints_simopt_;  ///< Equality constraints.
    const unsigned int n_equality_constraints_;
    const ROL::Ptr<const ROL::PartitionedVector<Real>>             dual_equality_;        ///< Lagrange multipliers associated with other equality coonstraints.

    const ROL::Ptr<ROL::BoundConstraint<Real>>          bound_constraints_;  ///< Equality constraints.
    const ROL::Ptr<const ROL::Vector<Real>>             dual_inequality_;        ///< Lagrange multipliers associated with box-bound constraints.
    const ROL::Ptr<const ROL::Vector<Real> > des_plus_dual_;            ///< Container for primal plus dual variables
    Real bounded_constraint_tolerance_;

    ROL::Ptr<const ROL::Vector_SimOpt<Real>>             simulation_and_control_variables_; ///< Simulation and control design variables.
    ROL::Ptr<const ROL::Vector<Real>>             simulation_variables_; ///< Simulation design variables.
    ROL::Ptr<const ROL::Vector<Real>>             control_variables_;    ///< Control design variables.
    ROL::Ptr<const ROL::Vector<Real>>             slack_variables_;    ///< Slack variables emanating from bounded constraints.
    const ROL::Ptr<ROL::Secant<Real> >                  secant_;               ///< Secant method used to precondition the reduced Hessian.

    /// Use an approximate inverse of the Jacobian and Jacobian transpose using
    /// the preconditioner to obtain the "tilde" operator version of Biros and Ghattas.

protected:
    ROL::Ptr<ROL::Vector<Real>> y1;
    ROL::Ptr<ROL::Vector<Real>> y2;
    ROL::Ptr<ROL::Vector<Real>> y3;
    ROL::Ptr<ROL::Vector<Real>> y4;

    ROL::Ptr<ROL::Vector<Real>> temp_1;
    ROL::Ptr<ROL::Vector<Real>> Lxs_Rsinv_y1;
    ROL::Ptr<ROL::Vector<Real>> Rsinv_y1;

    bool use_second_order_terms_;
    const bool use_approximate_preconditioner_;

    const unsigned int mpi_rank; ///< MPI rank used to reset the deallog depth
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
private:
    std::vector<ROL::Ptr<ROL::Vector<Real>>> cx;
    std::vector<ROL::Ptr<ROL::Vector<Real>>> cs;
    std::vector<ROL::Ptr<ROL::Vector<Real>>> cs_Rsinv;
    std::vector<ROL::Ptr<ROL::Vector<Real>>> cs_Rsinv_Rx;
    std::vector<ROL::Ptr<ROL::Vector<Real>>> C;
    std::vector<ROL::Ptr<ROL::Vector<Real>>> C_Lzz;
    //std::vector<ROL::Ptr<ROL::Vector<Real>>> C_Lzz_Ct;
    dealii::FullMatrix<double> C_Lzz_Ct;
    dealii::FullMatrix<double> C_Lzz_Ct_inv;

    const Real one = 1.0;

public:

    class CLzzC_block : public ROL::LinearOperator<Real> {
        private:
            const std::vector<ROL::Ptr<ROL::Vector<Real>>> &C;
            const ROL::Ptr<ROL::Secant<Real> > secant_;
            const ROL::Ptr<ROL::BoundConstraint<Real>> bound_constraints_;
            const ROL::Ptr<const ROL::Vector<Real> > des_plus_dual_;
            Real bounded_constraint_tolerance_;
        public:
            CLzzC_block(
              const std::vector<ROL::Ptr<ROL::Vector<Real>>> &C,
              const ROL::Ptr<ROL::Secant<Real> > secant,
              const ROL::Ptr<ROL::BoundConstraint<Real>> bound_constraints,
              const ROL::Ptr<const ROL::Vector<Real> > des_plus_dual,
              const Real constraint_tolerance)
              : C(C)
              , secant_(secant)
              , bound_constraints_(bound_constraints)
              , des_plus_dual_(des_plus_dual)
              , bounded_constraint_tolerance_(constraint_tolerance)
            { }

            void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
            {
                (void) v;
                (void) tol;
                MPI_Barrier(MPI_COMM_WORLD);
                static int ii = 0; (void) ii;
                //std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;

                const unsigned int neq = C.size();

                const auto &v_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(v);

                // Break down x2
                const ROL::Ptr< const ROL::Vector<Real> > x2 = v_partitioned.get(0);
                const auto &x2_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*x2);

                ROL::Ptr< const ROL::Vector<Real> > x2_ctl;
                std::vector<ROL::Ptr< const ROL::Vector<Real> >> x2_slk_vec(neq);

                unsigned int ith_x2_partition = 0;
                x2_ctl = x2_partitioned.get(ith_x2_partition++);
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    x2_slk_vec[ieq] = x2_partitioned.get(ith_x2_partition++);
                }

                // Break down x4
                const ROL::Ptr< const ROL::Vector<Real> > x4 = v_partitioned.get(1);
                const auto &x4_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*x4);

                unsigned int ith_x4_partition = 0;

                std::vector<ROL::Ptr< const ROL::Vector<Real> >> x4_adj_equ_var(neq);
                ROL::Ptr< const ROL::Vector<Real> > x4_adj_sim_ctl_bnd;
                std::vector<ROL::Ptr< const ROL::Vector<Real> >> x4_adj_slk_bnd(neq);

                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    x4_adj_equ_var[ieq] = x4_partitioned.get(ith_x4_partition++);
                    //std::cout << "x4_adj_equ_var " << ieq << " norm: " << x4_adj_equ_var[ieq]->norm() << std::endl;
                }
                x4_adj_sim_ctl_bnd = x4_partitioned.get(ith_x4_partition++);
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    x4_adj_slk_bnd[ieq] = x4_partitioned.get(ith_x4_partition++);
                }

                // std::cout << " x2->dimension() " << x2->dimension() << std::endl;
                // std::cout << " x2_partitioned->numVectors() " << x2_partitioned.numVectors() << std::endl;
                // std::cout << " x4->dimension() " << x4->dimension() << std::endl;
                // std::cout << " x4_partitioned->numVectors() " << x4_partitioned.numVectors() << std::endl;

                // std::cout << " x2_ctl->dimension() " << x2_ctl->dimension() << std::endl;
                // std::cout << " x2_slk_vec.size() " << x2_slk_vec.size() << std::endl;
                // for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                //     std::cout << " x2_slk_vec["<<ieq<<"]->dimension() " << x2_slk_vec[ieq]->dimension() << std::endl;
                // }

                // std::cout << " x4_adj_equ_var.size() " << x4_adj_equ_var.size() << std::endl;
                // for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                //     std::cout << " x4_adj_equ_var["<<ieq<<"]->dimension() " << x4_adj_equ_var[ieq]->dimension() << std::endl;
                // }
                // std::cout << " x4_adj_sim_ctl_bnd->dimension() " << x4_adj_sim_ctl_bnd->dimension() << std::endl;
                // std::cout << " x4_adj_slk_bnd.size() " << x4_adj_slk_bnd.size() << std::endl;
                // for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                //     std::cout << " x4_adj_slk_bnd["<<ieq<<"]->dimension() " << x4_adj_slk_bnd[ieq]->dimension() << std::endl;
                // }


                auto &Hv_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(Hv);

                // Break down y2
                const ROL::Ptr< ROL::Vector<Real> > y2 = Hv_partitioned.get(0);
                auto &y2_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(*y2);

                unsigned int ith_y2_partition = 0;

                std::vector<ROL::Ptr< ROL::Vector<Real> >> y2_adj_equ_var(neq);
                ROL::Ptr< ROL::Vector<Real> > y2_adj_sim_ctl_bnd;
                std::vector<ROL::Ptr< ROL::Vector<Real> >> y2_adj_slk_bnd(neq);

                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    y2_adj_equ_var[ieq] = y2_partitioned.get(ith_y2_partition++);
                }
                y2_adj_sim_ctl_bnd = y2_partitioned.get(ith_y2_partition++);
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    y2_adj_slk_bnd[ieq] = y2_partitioned.get(ith_y2_partition++);
                }

                // Break down y3
                const ROL::Ptr< ROL::Vector<Real> > y3 = Hv_partitioned.get(1);
                auto &y3_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(*y3);

                ROL::Ptr< ROL::Vector<Real> > y3_ctl;
                std::vector<ROL::Ptr< ROL::Vector<Real> >> y3_slk_vec(neq);

                unsigned int ith_y3_partition = 0;
                y3_ctl = y3_partitioned.get(ith_y3_partition++);
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    y3_slk_vec[ieq] = y3_partitioned.get(ith_y3_partition++);
                }



                // std::cout << " y3->dimension() " << y3->dimension() << std::endl;
                // std::cout << " y3_partitioned->numVectors() " << y3_partitioned.numVectors() << std::endl;
                // std::cout << " y2->dimension() " << y2->dimension() << std::endl;
                // std::cout << " y2_partitioned->numVectors() " << y2_partitioned.numVectors() << std::endl;

                // std::cout << " y3_ctl->dimension() " << y3_ctl->dimension() << std::endl;
                // std::cout << " y3_slk_vec.size() " << y3_slk_vec.size() << std::endl;
                // for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                //     std::cout << " y3_slk_vec["<<ieq<<"]->dimension() " << y3_slk_vec[ieq]->dimension() << std::endl;
                // }

                // std::cout << " y2_adj_equ_var.size() " << y2_adj_equ_var.size() << std::endl;
                // for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                //     std::cout << " y2_adj_equ_var["<<ieq<<"]->dimension() " << y2_adj_equ_var[ieq]->dimension() << std::endl;
                // }
                // std::cout << " y2_adj_sim_ctl_bnd->dimension() " << y2_adj_sim_ctl_bnd->dimension() << std::endl;
                // std::cout << " y2_adj_slk_bnd.size() " << y2_adj_slk_bnd.size() << std::endl;
                // for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                //     std::cout << " y2_adj_slk_bnd["<<ieq<<"]->dimension() " << y2_adj_slk_bnd[ieq]->dimension() << std::endl;
                // }

                // std::cout << " y2_adj_sim_ctl_bnd->dimension() " << y2_adj_sim_ctl_bnd->dimension() << std::endl;
                // std::cout << " y2_adj_slk_bnd.size() " << y2_adj_slk_bnd.size() << std::endl;
                // for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                //     std::cout << " y2_adj_slk_bnd["<<ieq<<"]->dimension() " << y2_adj_slk_bnd[ieq]->dimension() << std::endl;
                // }

                const Real one = 1.0;
                // Compute y2
                y2->zero();
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    const Real cx_dot_dx = C[ieq]->dot(*x2_ctl);
                    y2_adj_equ_var[ieq]->setScalar(cx_dot_dx);
                    y2_adj_equ_var[ieq]->axpy(-one, *x2_slk_vec[ieq]);
                }

                const auto &des_plus_dual_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*des_plus_dual_);

                //std::cout
                //<< " des_plus_dual_->dimension() "
                //<< des_plus_dual_->dimension()
                //<< std::endl;
                //std::cout
                //<< " des_plus_dual_partitioned.numVectors() "
                //<< des_plus_dual_partitioned.numVectors()
                //<< std::endl;
                //for (unsigned int ivec = 0; ivec < des_plus_dual_partitioned.numVectors(); ++ivec) {
                //    std::cout << ivec 
                //    << " des_plus_dual_->get(ivec)->dimension() "
                //    << des_plus_dual_partitioned.get(ivec)->dimension()
                //    << std::endl;
                //}

                ROL::Ptr<ROL::Vector<Real>> x2_adj_sim_ctl_bnd = des_plus_dual_partitioned.get(0)->clone();
                ROL::Vector_SimOpt<Real> &x2_adj_sim_ctl_bnd_simopt = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*x2_adj_sim_ctl_bnd);
                ROL::Ptr<ROL::Vector<Real>> x2_adj_sim_bnd = x2_adj_sim_ctl_bnd_simopt.get_1();
                ROL::Ptr<ROL::Vector<Real>> x2_adj_ctl_bnd = x2_adj_sim_ctl_bnd_simopt.get_2();

                x2_adj_sim_bnd->zero();
                x2_adj_ctl_bnd->set(*x2_ctl);
                

                std::vector<ROL::Ptr<ROL::Vector<Real>>> y2_dual_inequality_stdvec(des_plus_dual_partitioned.numVectors());
                y2_dual_inequality_stdvec[0] = x2_adj_sim_ctl_bnd->clone();
                y2_dual_inequality_stdvec[0]->set( *x2_adj_sim_ctl_bnd );
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    y2_dual_inequality_stdvec[ieq+1] = x2_slk_vec[ieq]->clone();
                    y2_dual_inequality_stdvec[ieq+1]->set( *(x2_slk_vec[ieq]) );
                }
                ROL::PartitionedVector<Real> y2_dual_inequality_partitioned(y2_dual_inequality_stdvec);

                Real bnd_tol = bounded_constraint_tolerance_;
                bound_constraints_->pruneInactive(y2_dual_inequality_partitioned, *des_plus_dual_, bnd_tol);

                y2_adj_sim_ctl_bnd->set(*(y2_dual_inequality_partitioned.get(0)));
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    y2_adj_slk_bnd[ieq]->set(*(y2_dual_inequality_partitioned.get(ieq+1)));
                }

                std::vector<ROL::Ptr<ROL::Vector<Real>>> x4_des_dual_vec(des_plus_dual_partitioned.numVectors());
                x4_des_dual_vec[0] = x4_adj_sim_ctl_bnd->clone();
                x4_des_dual_vec[0]->set(*x4_adj_sim_ctl_bnd);
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    x4_des_dual_vec[ieq+1] = x4_adj_slk_bnd[ieq]->clone();
                    x4_des_dual_vec[ieq+1]->set(*(x4_adj_slk_bnd[ieq]));
                }
                ROL::PartitionedVector<Real> x4_active_des_dual_inequality(x4_des_dual_vec);
                bound_constraints_->pruneInactive(x4_active_des_dual_inequality, *des_plus_dual_, bnd_tol);
                ROL::Vector<Real> &x4_active_ctl_sim_dual_inequality = *(x4_active_des_dual_inequality.get(0));
                ROL::Vector_SimOpt<Real> &x4_active_des_dual_inequality_simopt = dynamic_cast<ROL::Vector_SimOpt<Real>&>(x4_active_ctl_sim_dual_inequality);
                ROL::Vector<Real> &x4_active_ctl_dual_inequality = *(x4_active_des_dual_inequality_simopt.get_2());

                secant_->applyB( *y3_ctl, *x2_ctl);
                y3_ctl->set(*x2_ctl);

                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    const ROL::SingletonVector<Real> &x4_adj_equ_var_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(*x4_adj_equ_var[ieq]);
                    y3_ctl->axpy(x4_adj_equ_var_singleton.getValue(), *C[ieq]);
                }
                y3_ctl->axpy(one, x4_active_ctl_dual_inequality);
                (void)x4_active_ctl_dual_inequality;

                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    y3_slk_vec[ieq]->set(*x4_adj_equ_var[ieq]);
                    y3_slk_vec[ieq]->scale(-one);
                    y3_slk_vec[ieq]->axpy(one, *x4_active_des_dual_inequality.get(ieq+1));
                }
                //y3->set(*x2);
                //MPI_Barrier(MPI_COMM_WORLD);
                //std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;

                //std::cout << "Done applying CLzzC_block..." << std::endl;
                //std::abort();
            }
    };
    class Identity_Preconditioner_FlipVectors : public ROL::LinearOperator<Real> {
        public:
          void apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
          {
              applyInverse( Hv, v, tol );
          }
          void applyInverse( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
          {
              MPI_Barrier(MPI_COMM_WORLD);
              (void) v;
              (void) tol;
              //static int ii = 0;
              //std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;

              const auto &v_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(v);
              const ROL::Ptr< const ROL::Vector<Real> > x2 = v_partitioned.get(0);
              const ROL::Ptr< const ROL::Vector<Real> > x4 = v_partitioned.get(1);

              auto &Hv_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(Hv);
              const ROL::Ptr< ROL::Vector<Real> > y2 = Hv_partitioned.get(0);
              const ROL::Ptr< ROL::Vector<Real> > y3 = Hv_partitioned.get(1);

              // std::cout << " x2->dimension() " << x2->dimension() << std::endl;
              // std::cout << " x4->dimension() " << x4->dimension() << std::endl;

              // std::cout << " y2->dimension() " << y2->dimension() << std::endl;
              // std::cout << " y3->dimension() " << y3->dimension() << std::endl;

              const auto &x2_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*x2); (void) x2_partitioned;
              const auto &x4_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*x4); (void) x4_partitioned;

              auto &y2_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(*y2); (void) y2_partitioned;
              auto &y3_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(*y3); (void) y3_partitioned;
              // std::cout << " x2_partitioned->numVectors() " << x2_partitioned.numVectors() << std::endl;
              // std::cout << " x4_partitioned->numVectors() " << x4_partitioned.numVectors() << std::endl;
              // std::cout << " y2_partitioned->numVectors() " << y2_partitioned.numVectors() << std::endl;
              // std::cout << " y3_partitioned->numVectors() " << y3_partitioned.numVectors() << std::endl;


              // MPI_Barrier(MPI_COMM_WORLD);
              // std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;

              y2->set(*x4);
              y3->set(*x2);
          }
    };

    ROL::Ptr<const ROL::Vector<Real>> extract_simulation_variables( const ROL::PartitionedVector<Real> &design_variables )
    {
        const unsigned int nvecs = design_variables.numVectors();
        if (nvecs < 2) std::abort();

        ROL::Ptr<const ROL::Vector<Real>> non_slack_variables = design_variables[0];
        ROL::Ptr<const ROL::Vector<Real>> non_slack_variables_simopt = ROL::makePtrFromRef<const ROL::Vector_SimOpt<Real>>(dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*non_slack_variables));
        return non_slack_variables_simopt->get_1();
    }
    ROL::Ptr<const ROL::Vector<Real>> extract_control_variables( const ROL::PartitionedVector<Real> &design_variables ) const
    {
        MPI_Barrier(MPI_COMM_WORLD);
        const unsigned int nvecs = design_variables.numVectors();
        if (nvecs < 2) std::abort();

        ROL::Ptr<const ROL::Vector<Real>> non_slack_variables = design_variables.get(0);
        ROL::Ptr<const ROL::Vector_SimOpt<Real>> non_slack_variables_simopt = ROL::makePtrFromRef<const ROL::Vector_SimOpt<Real>>(dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*non_slack_variables));
        return non_slack_variables_simopt->get_2();
    }

    ROL::Ptr<const ROL::PartitionedVector<Real>> extract_slacks( const ROL::PartitionedVector<Real> &design_variables ) const
    {
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<ROL::Ptr<ROL::Vector<Real>>> slack_vecs;
        const unsigned int n_vec = design_variables.numVectors();
        for (unsigned int i_vec = 1; i_vec < n_vec; ++i_vec) {
            slack_vecs.push_back( design_variables.get(i_vec)->clone() );
            slack_vecs[i_vec-1]->set( *(design_variables.get(i_vec)) );
        }
        ROL::Ptr<const ROL::PartitionedVector<Real>> slack_variables_partitioned = ROL::makePtr<const ROL::PartitionedVector<Real>> (slack_vecs);
        return slack_variables_partitioned;
    }

    std::vector<ROL::Ptr<ROL::Vector<Real>>> extract_nonsim_dual_equality( const ROL::PartitionedVector<Real> &dual_equality ) const
    {
        std::vector<ROL::Ptr<ROL::Vector<Real>>> nonsim_dual_equality;
        const unsigned int n_vec = dual_equality.numVectors();
        for (unsigned int i_vec = 1; i_vec < n_vec; ++i_vec) {
            nonsim_dual_equality.push_back( dual_equality.get(i_vec)->clone() );
            nonsim_dual_equality[i_vec-1]->set( *(dual_equality.get(i_vec)) );
        }
        return nonsim_dual_equality;
    }

    ROL::Ptr<const ROL::PartitionedVector<Real>> extract_control_and_slacks( const ROL::PartitionedVector<Real> &design_variables ) const
    {
        MPI_Barrier(MPI_COMM_WORLD);
        ROL::Ptr<const ROL::Vector<Real>> ctl_variables = extract_control_variables( design_variables );

        std::vector<ROL::Ptr<ROL::Vector<Real>>> slack_vecs;

        slack_vecs.push_back( ctl_variables->clone() );
        slack_vecs[0]->set( *ctl_variables );

        const unsigned int n_vec = design_variables.numVectors();
        for (unsigned int i_vec = 1; i_vec < n_vec; ++i_vec) {
            slack_vecs.push_back( design_variables.get(i_vec)->clone() );
            slack_vecs[i_vec]->set( *(design_variables.get(i_vec)) );
        }
        ROL::Ptr<const ROL::PartitionedVector<Real>> slack_variables_partitioned = ROL::makePtr<const ROL::PartitionedVector<Real>> (slack_vecs);
        return slack_variables_partitioned;
    }

    // Non-const version
    ROL::Ptr<ROL::Vector<Real>> extract_control_variables( ROL::PartitionedVector<Real> &design_variables ) const
    {
        MPI_Barrier(MPI_COMM_WORLD);
        const unsigned int nvecs = design_variables.numVectors();
        if (nvecs < 2) std::abort();

        ROL::Ptr<ROL::Vector<Real>> non_slack_variables = design_variables.get(0);
        ROL::Ptr<ROL::Vector_SimOpt<Real>> non_slack_variables_simopt = ROL::makePtrFromRef<ROL::Vector_SimOpt<Real>>(dynamic_cast<ROL::Vector_SimOpt<Real>&>(*non_slack_variables));
        return non_slack_variables_simopt->get_2();
    }
    // Non-const version
    ROL::Ptr<ROL::PartitionedVector<Real>> extract_control_and_slacks( ROL::PartitionedVector<Real> &design_variables ) const
    {
        MPI_Barrier(MPI_COMM_WORLD);
        ROL::Ptr<ROL::Vector<Real>> ctl_variables = extract_control_variables( design_variables );

        std::vector<ROL::Ptr<ROL::Vector<Real>>> slack_vecs;

        slack_vecs.push_back( ctl_variables->clone() );
        slack_vecs[0]->set( *ctl_variables );

        const unsigned int n_vec = design_variables.numVectors();
        for (unsigned int i_vec = 1; i_vec < n_vec; ++i_vec) {
            slack_vecs.push_back( design_variables.get(i_vec)->clone() );
            slack_vecs[i_vec]->set( *(design_variables.get(i_vec)) );
        }
        ROL::Ptr<ROL::PartitionedVector<Real>> slack_variables_partitioned = ROL::makePtr<ROL::PartitionedVector<Real>> (slack_vecs);
        return slack_variables_partitioned;
    }

    //ROL::Ptr<const ROL::Vector<Real>> extract_control_variables( const ROL::PartitionedVector<Real> &design_variables )
    //{
    //    const unsigned int nvecs = design_variables.numVectors();
    //    if (numVectors < 2) std::abort();

    //    ROL::Ptr<const ROL::Vector<Real>> non_slack_variables = design_variables[0];
    //    ROL::Ptr<const ROL::Vector<Real>> non_slack_variables_simopt = ROL::makePtrFromRef<const ROL::Vector_SimOpt<Real>>(dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*non_slack_variables));
    //    return non_slack_variables_simopt->get_2();
    //}
    /// Constructor.
    PDAS_P24_Constrained_Preconditioner(
        const ROL::Ptr<const ROL::Vector<Real>>             design_variables,
        const ROL::Ptr<ROL::Objective<Real>>                objective,
        const ROL::Ptr<ROL::Constraint<Real>>               state_constraints,
        const ROL::Ptr<const ROL::Vector<Real>>             dual_state,
        const ROL::Ptr<ROL::Constraint_Partitioned<Real>>   equality_constraints,
        const ROL::Ptr<const ROL::PartitionedVector<Real>>  dual_equality,
        const ROL::Ptr<ROL::BoundConstraint<Real>>          bound_constraints,
        const ROL::Ptr<const ROL::Vector<Real>>             dual_inequality,
        const ROL::Ptr<const ROL::Vector<Real>>             des_plus_dual,
        const Real                                          constraint_tolerance,
        const ROL::Ptr<ROL::Secant<Real> >                  secant,
        const bool use_second_order_terms = true,
        const bool use_approximate_preconditioner = false)
        : design_variables_ (ROL::makePtrFromRef<const ROL::PartitionedVector<Real>>(dynamic_cast<const ROL::PartitionedVector<Real>&>(*design_variables)))
        , objective_(ROL::makePtrFromRef<ROL::Objective_SimOpt<Real>>(dynamic_cast<ROL::Objective_SimOpt<Real>&>(*objective)))
        , state_constraints_(ROL::makePtrFromRef<PHiLiP::FlowConstraints<PHILIP_DIM>>(dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(*state_constraints)))
        , dual_state_(dual_state)
        , equality_constraints_simopt_(equality_constraints)
        , n_equality_constraints_(equality_constraints_simopt_->get_n_constraints())
        , dual_equality_(dual_equality)
        , bound_constraints_(bound_constraints)
        , dual_inequality_(dual_inequality)
        , des_plus_dual_(des_plus_dual)
        , bounded_constraint_tolerance_(constraint_tolerance)
        , secant_(secant)
        , use_second_order_terms_(use_second_order_terms)
        , use_approximate_preconditioner_(use_approximate_preconditioner)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
        , cx(n_equality_constraints_)
        , cs(n_equality_constraints_)
        , cs_Rsinv(n_equality_constraints_)
        , cs_Rsinv_Rx(n_equality_constraints_)
        , C(n_equality_constraints_)
        , C_Lzz(n_equality_constraints_)
        , C_Lzz_Ct(n_equality_constraints_)
        , C_Lzz_Ct_inv(n_equality_constraints_)
    {
        //MPI_Barrier(MPI_COMM_WORLD);
        //if(mpi_rank==0) std::cout << __PRETTY_FUNCTION__ << std::endl;
        //std::cout << "n_equality_constraints_ " << n_equality_constraints_ << std::endl;
        //const unsigned int n_other_dual = dual_equality_->dimension();
        //std::cout << "n_other_dual " << n_other_dual << std::endl;
        //std::abort();

        simulation_and_control_variables_ = ROL::makePtrFromRef(dynamic_cast<const ROL::Vector_SimOpt<Real>&> (*(design_variables_->get(0))));
        simulation_variables_ = simulation_and_control_variables_->get_1();
        control_variables_ = simulation_and_control_variables_->get_2();

        const int error_precond1 = state_constraints_->construct_JacobianPreconditioner_1(*simulation_variables_, *control_variables_);
        const int error_precond2 = state_constraints_->construct_AdjointJacobianPreconditioner_1(*simulation_variables_, *control_variables_);
        assert(error_precond1 == 0);
        assert(error_precond2 == 0);
        (void) error_precond1;
        (void) error_precond2;


        std::vector<ROL::Ptr<ROL::Vector<Real>>> slack_vecs;
        const unsigned int n_vec = design_variables_->numVectors();
        for (unsigned int i_vec = 1; i_vec < n_vec; ++i_vec) {
            slack_vecs.push_back( design_variables_->get(i_vec)->clone() );
            slack_vecs[i_vec-1]->set( *(design_variables_->get(i_vec)) );
        }
        ROL::PartitionedVector<Real> slack_variables_partitioned (slack_vecs);
        slack_variables_ = ROL::makePtrFromRef(slack_variables_partitioned);

        initialize_constraint_sensitivities();

        // if (use_approximate_preconditioner_) {
        //     const int error_precond1 = equal_constraints_->construct_JacobianPreconditioner_1(*simulation_variables_, *control_variables_);
        //     const int error_precond2 = equal_constraints_->construct_AdjointJacobianPreconditioner_1(*simulation_variables_, *control_variables_);
        //     assert(error_precond1 == 0);
        //     assert(error_precond2 == 0);
        //     (void) error_precond1;
        //     (void) error_precond2;
        // }

        // const auto &design_simopt = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*design_variables);

        // const ROL::Ptr<const ROL::Vector<Real>> input_1 = design_simopt.get_1();
        // const ROL::Ptr<const ROL::Vector<Real>> input_2 = design_simopt.get_2();
        // const ROL::Ptr<const ROL::Vector<Real>> input_3 = dual_state;
        // const ROL::Ptr<const ROL::Vector<Real>> input_4 = other_lagrange_mult;

        // y1 = input_1->clone();
        // y2 = input_2->clone();
        // y3 = input_3->clone();
        // y4 = input_4->clone();

        // temp_1 = input_1->clone();
        // if (use_second_order_terms_) {
        //     Rsinv_y1 = input_1->clone();
        //     Lxs_Rsinv_y1 = input_2->clone();
        // }

        // const unsigned int n_other_constraints = dual_equality_->dimension();
        // cs.resize(n_other_constraints);
        // //for (unsigned int i_other_constraints = 0; i_other_constraints < n_other_constraints; ++i_other_constraints) {
        // //    cs[i_other_constraints] = input_1->clone();

        // //    const auto i_basis_vector = cs[i_other_constraints].basis(i_other_constraints);
        // //    equality_constraints_simopt_->applyAdjointJacobian_1(

        // //    equality_constraints_simopt_->applyJacobian_2
        // //}

        // RsTinv_cs = input_1->clone();
    };


    void initialize_constraint_sensitivities()
    {
        Real tol = 1e-15;
        for(unsigned int i = 0; i < n_equality_constraints_; ++ i) {

            ROL::Ptr<ROL::Vector<Real>> input_one = dual_equality_->get(i)->clone();
            if (input_one->dimension() != 1) {
                std::cout << "Aborting. The current implementation allows for additional scalar constraints, not vector constraints" << std::endl;
                std::cout << "If vector constraints are used, please break them down into scalars. " << std::endl;
                std::cout << "Othewise, the follow evaluation of the constraints' derivatives are combined into a vector instead of a matrix. " << std::endl;
                std::abort();
            }
            input_one->setScalar(one);

            const ROL::Ptr<ROL::Constraint<Real>> i_con = equality_constraints_simopt_->get(i);
            ROL::Constraint_SimOpt<Real> &i_con_simopt = dynamic_cast<ROL::Constraint_SimOpt<Real>&>(*i_con);
            ROL::Ptr<ROL::Vector<Real>> i_cs = simulation_variables_->clone();
            i_con_simopt.applyAdjointJacobian_1(*i_cs, *input_one, *simulation_variables_, *control_variables_, tol);

            cs[i] = i_cs->clone();
            cs[i]->set(*i_cs);

            state_constraints_->applyInverseAdjointJacobian_1(*i_cs, *(cs[i]), *simulation_variables_, *control_variables_, tol);
#ifdef REPLACE_INVERSE_WITH_IDENTITY
            i_cs->set(*(cs[i]));
#endif

            cs_Rsinv[i] = i_cs->clone();
            cs_Rsinv[i]->set(*i_cs);


            ROL::Ptr<ROL::Vector<Real>> i_cx = control_variables_->clone();

            i_con_simopt.applyAdjointJacobian_2(*i_cx, *input_one, *simulation_variables_, *control_variables_, tol);

            cx[i] = i_cx->clone();
            cx[i]->set(*i_cx);


            state_constraints_->applyAdjointJacobian_2(*i_cx, *(cs_Rsinv[i]), *simulation_variables_, *control_variables_, tol);
            cs_Rsinv_Rx[i] = i_cx->clone();
            cs_Rsinv_Rx[i]->set(*i_cx);

            C[i] = cx[i]->clone();
            C[i]->set(*cx[i]);
            C[i]->axpy(-one,*cs_Rsinv_Rx[i]);

            C_Lzz[i] = cx[i]->clone();
            secant_->applyB(*C_Lzz[i],*C[i]);
            C_Lzz[i]->set(*C[i]);

        }
        for(unsigned int i = 0; i < n_equality_constraints_; ++i) {
            for(unsigned int j = 0; j < n_equality_constraints_; ++j) {
                C_Lzz_Ct[i][j] = C_Lzz[i]->dot(*C[j]);
            }
        }
        C_Lzz_Ct_inv.invert(C_Lzz_Ct);
    }

    void apply( ROL::Vector<Real> &output, const ROL::Vector<Real> &input, Real &/*tol*/ ) const
    {
        (void) output;
        (void) input;
    }

    /// Applies [ cx - cs Rsinv Rx ]
    void applyC() const
    {
    }


    /// Application of KKT preconditionner on vector input outputted into output.
    //void vmult (dealiiSolverVectorWrappingROL<Real>       &output,
    //            const dealiiSolverVectorWrappingROL<Real> &input) const
    void applyInverse( ROL::Vector<Real> &output, const ROL::Vector<Real> &input, Real &tol ) const
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(mpi_rank==0) std::cout << __PRETTY_FUNCTION__ << std::endl;
        output.set(input);

        static int number_of_times = 0;
        number_of_times++;
        pcout << "Number of P4_KKT vmult = " << number_of_times << std::endl;

        // Split input vector
        //const ROL::Ptr<const ROL::Vector<Real>> input_rol = input.getVector();
        const ROL::Ptr<const ROL::Vector<Real>> input_rol = ROL::makePtrFromRef(input);
        const auto &input_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*input_rol);
        const ROL::Ptr< const ROL::Vector<Real> > input_design           = input_partitioned.get(0);
        const ROL::Ptr< const ROL::Vector<Real> > input_dual_equality    = input_partitioned.get(1);
        const ROL::Ptr< const ROL::Vector<Real> > input_dual_inequality  = input_partitioned.get(2);

        const ROL::Ptr< const ROL::Vector<Real> > input_design_sim_ctl = (dynamic_cast<const ROL::PartitionedVector<Real>&>(*input_design)).get(0);
        const ROL::Ptr< const ROL::Vector<Real> > input_design_simulation = (dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*input_design_sim_ctl)).get_1();
        const ROL::Ptr< const ROL::Vector<Real> > input_design_control    = (dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*input_design_sim_ctl)).get_2();

        //const ROL::Ptr< const ROL::PartitionedVector<Real> > input_design_slacks  = extract_slacks(dynamic_cast<const ROL::PartitionedVector<Real> &> (*input_design));

        const ROL::Ptr< const ROL::Vector<Real> > input_simdual_equality = (dynamic_cast<const ROL::PartitionedVector<Real>&>(*input_dual_equality)).get(0);

        std::vector<ROL::Ptr<ROL::Vector<Real>>> input_nonsimdual_stdvec = extract_nonsim_dual_equality( dynamic_cast<const ROL::PartitionedVector<Real>&>(*input_dual_equality) );
        const ROL::PartitionedVector<Real> &input_dual_inequality_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(*input_dual_inequality);
        const unsigned int n_vec_dual_inequality = input_dual_inequality_partitioned.numVectors();
        for (unsigned int i_vec = 0; i_vec < n_vec_dual_inequality; ++i_vec) {
            input_nonsimdual_stdvec.push_back(input_dual_inequality_partitioned.get(i_vec)->clone());
            input_nonsimdual_stdvec.back()->set(*(input_dual_inequality_partitioned.get(i_vec)));
        }
        const ROL::Ptr< const ROL::PartitionedVector<Real> > input_nonsimdual = ROL::makePtr<const ROL::PartitionedVector<Real>> (input_nonsimdual_stdvec);



        int ii = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        if(mpi_rank==0) std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;

        //ROL::Ptr<ROL::Vector<Real>> output_rol = output.getVector();
        ROL::Ptr<ROL::Vector<Real>> output_rol = ROL::makePtrFromRef(output);
        auto &output_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(*output_rol);
        const ROL::Ptr<       ROL::Vector<Real> > output_design          = output_partitioned.get(0);
        const ROL::Ptr<       ROL::Vector<Real> > output_dual_equality   = output_partitioned.get(1);
        const ROL::Ptr<       ROL::Vector<Real> > output_dual_inequality = output_partitioned.get(2);

        MPI_Barrier(MPI_COMM_WORLD);
        if(mpi_rank==0) std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;

        const ROL::Ptr< ROL::Vector<Real> > output_design_sim_ctl = (dynamic_cast<ROL::PartitionedVector<Real>&>(*output_design)).get(0);
        const ROL::Ptr< ROL::Vector<Real> > output_design_simulation = (dynamic_cast<ROL::Vector_SimOpt<Real>&>(*output_design_sim_ctl)).get_1();
        const ROL::Ptr< ROL::Vector<Real> > output_design_control    = (dynamic_cast<ROL::Vector_SimOpt<Real>&>(*output_design_sim_ctl)).get_2();

        //const ROL::Ptr< ROL::PartitionedVector<Real> > output_design_slacks  = extract_slacks(dynamic_cast<ROL::PartitionedVector<Real> &> (*output_design));

        const ROL::Ptr< ROL::Vector<Real> > output_simdual_equality = (dynamic_cast<ROL::PartitionedVector<Real>&>(*output_dual_equality)).get(0);

        std::vector<ROL::Ptr<ROL::Vector<Real>>> output_nonsimdual_stdvec = extract_nonsim_dual_equality( dynamic_cast<ROL::PartitionedVector<Real>&>(*output_dual_equality) );
        ROL::PartitionedVector<Real> &output_dual_inequality_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(*output_dual_inequality);
        for (unsigned int i_vec = 0; i_vec < n_vec_dual_inequality; ++i_vec) {
            output_nonsimdual_stdvec.push_back(output_dual_inequality_partitioned.get(i_vec)->clone());
            output_nonsimdual_stdvec.back()->set(*(output_dual_inequality_partitioned.get(i_vec)));
        }
        const ROL::Ptr< ROL::PartitionedVector<Real> > output_nonsimdual = ROL::makePtr<ROL::PartitionedVector<Real>> (output_nonsimdual_stdvec);


        const ROL::Ptr< const ROL::Vector<Real> >            z1 = input_design_simulation;
        const ROL::Ptr< const ROL::PartitionedVector<Real> > z2 = extract_control_and_slacks(dynamic_cast<const ROL::PartitionedVector<Real> &> (*input_design));
        const ROL::Ptr< const ROL::Vector<Real> >            z3 = input_simdual_equality;
        const ROL::Ptr< const ROL::PartitionedVector<Real> > z4 = input_nonsimdual;
        //std::cout << " after extract control and slacks " << input.norm() << std::endl;
        //std::cout << " input.norm() " << input.norm() << std::endl;
        //std::cout << " output.norm() " << output.norm() << std::endl;
        //std::cout << " z1->norm() " << z1->norm() << std::endl;
        //std::cout << " z2->norm() " << z2->norm() << std::endl;
        //std::cout << " z3->norm() " << z3->norm() << std::endl;
        //std::cout << " z4->norm() " << z4->norm() << std::endl;

        const ROL::Ptr< ROL::Vector<Real> >                  x1 = output_design_simulation;
        const ROL::Ptr< ROL::PartitionedVector<Real> >       x2 = extract_control_and_slacks(dynamic_cast<ROL::PartitionedVector<Real> &> (*output_design));
        const ROL::Ptr< ROL::Vector<Real> >                  x3 = output_simdual_equality;
        const ROL::Ptr< ROL::PartitionedVector<Real> >       x4 = output_nonsimdual;

        const ROL::Ptr< ROL::Vector<Real> >                  y1 = z3->clone();
        const ROL::Ptr< ROL::PartitionedVector<Real> >       y2 = ROL::makePtr<ROL::PartitionedVector<Real>>(dynamic_cast<ROL::PartitionedVector<Real>&>(*z4->clone()));
        const ROL::Ptr< ROL::PartitionedVector<Real> >       y3 = ROL::makePtr<ROL::PartitionedVector<Real>>(dynamic_cast<ROL::PartitionedVector<Real>&>(*z2->clone()));
        const ROL::Ptr< ROL::Vector<Real> >                  y4 = z1->clone();

        // y1
        y1->set(*z3);

        // y2
        const ROL::Ptr< ROL::Vector<Real> > Rsinv_y1 = y1->clone();
        if (use_approximate_preconditioner_) {
            state_constraints_->applyInverseJacobianPreconditioner_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
        } else {
            state_constraints_->applyInverseJacobian_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
        }
#ifdef REPLACE_INVERSE_WITH_IDENTITY
        Rsinv_y1->set(*y1);
#endif

        y2->set(*z4);
        const ROL::Ptr< ROL::Vector<Real> > cs_Rsinv_y1 = y2->clone();
        cs_Rsinv_y1->zero();

        for(unsigned int i = 0; i < n_equality_constraints_; ++ i) {
            const ROL::Ptr<ROL::Constraint<Real>> i_con = equality_constraints_simopt_->get(i);
            ROL::Constraint_SimOpt<Real> &i_con_simopt = dynamic_cast<ROL::Constraint_SimOpt<Real>&>(*i_con);
            (void) i_con_simopt;
            ROL::Vector<Real> &i_vec_out = *((dynamic_cast<ROL::PartitionedVector<Real>&>(*cs_Rsinv_y1)).get(i));
            i_con_simopt.applyJacobian_1(i_vec_out, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
        }

        //std::vector<ROL::Ptr<ROL::Vector<Real>>> cx(n_equality_constraints_);
        //std::vector<ROL::Ptr<ROL::Vector<Real>>> cs(n_equality_constraints_);
        //std::vector<ROL::Ptr<ROL::Vector<Real>>> cs_Rsinv(n_equality_constraints_);

        //if ( output_nonsimdual->dimension() != (int)n_equality_constraints_ ) {
        //    std::cout << "The n equality constraints contain some contraints that output more than a single constraint value" << std::endl;
        //    std::cout << "The current implementation only works if the constraints are scalars, such as lift or moment in a single direction" << std::endl;
        //    std::cout << "This is because we require the derivative of the constraints with respect to the state variables explicitly such that we applyAdjointJacobian_1 on a unit vector." << std::endl;
        //}
        //for(unsigned int i = 0; i < n_equality_constraints_; ++ i) {


        //    ROL::Ptr<ROL::Vector<Real>> input_one = output_nonsimdual->get(i)->clone();
        //    if ( input_one->dimension() != 1) {
        //        std::cout << "The n equality constraints contain some contraints that output more than a single constraint value" << std::endl;
        //        std::cout << "The current implementation only works if the constraints are scalars, such as lift or moment in a single direction" << std::endl;
        //        std::cout << "This is because we require the derivative of the constraints with respect to the state variables explicitly such that we applyAdjointJacobian_1 on a unit vector." << std::endl;
        //    }
        //    input_one->setScalar(one);

        //    const ROL::Ptr<ROL::Constraint<Real>> i_con = equality_constraints_simopt_->get(i);
        //    ROL::Constraint_SimOpt<Real> &i_con_simopt = dynamic_cast<ROL::Constraint_SimOpt<Real>&>(*i_con);
        //    ROL::Ptr<ROL::Vector<Real>> i_cs = simulation_variables_->clone();
        //    i_con_simopt.applyAdjointJacobian_1(*i_cs, *input_one, *simulation_variables_, *control_variables_, tol);

        //    cs[i] = i_cs->clone();
        //    cs[i]->set(*i_cs);

        //    state_constraints_->applyInverseAdjointJacobian_1(*i_cs, *(cs[i]), *simulation_variables_, *control_variables_, tol);

        //    cs_Rsinv[i] = i_cs->clone();
        //    cs_Rsinv[i]->set(*i_cs);


        //    ROL::Ptr<ROL::Vector<Real>> i_cx = control_variables_->clone();

        //    i_con_simopt.applyAdjointJacobian_2(*i_cx, *input_one, *simulation_variables_, *control_variables_, tol);

        //    cx[i] = i_cx->clone();
        //    cx[i]->set(*i_cx);
        //}

        //(ROL::makePtrFromRef<ROL::Constraint_Partitioned<Real>>(dynamic_cast<ROL::Constraint_Partitioned<Real>&>(*equality_constraints)))
        y2->axpy(-one, *cs_Rsinv_y1);

        // y4
        y4->set(*z1);

        // y3
        y3->set(*z2);

        const ROL::Ptr< ROL::Vector<Real> > RsTinv_y4 = y4->clone();
        if (use_approximate_preconditioner_) {
            state_constraints_->applyInverseAdjointJacobianPreconditioner_1(*RsTinv_y4, *y4, *simulation_variables_, *control_variables_, tol);
        } else {
            state_constraints_->applyInverseAdjointJacobian_1(*RsTinv_y4, *y4, *simulation_variables_, *control_variables_, tol);
        }
#ifdef REPLACE_INVERSE_WITH_IDENTITY
        RsTinv_y4->set(*y4);
#endif

        //const ROL::Ptr< ROL::Vector<Real> > RxT_RsTinv_y4 = y3->clone();
        const ROL::Ptr< ROL::Vector<Real> > RxT_RsTinv_y4 = control_variables_->clone();
        state_constraints_->applyAdjointJacobian_2(*RxT_RsTinv_y4, *RsTinv_y4, *simulation_variables_, *control_variables_, tol);

        const ROL::Ptr< ROL::Vector<Real> > RxallT_RsTinv_y4 = y3->clone();

        ROL::PartitionedVector<Real> &RxallT_RsTinv_y4_part = dynamic_cast<ROL::PartitionedVector<Real> &> (*RxallT_RsTinv_y4);

        RxallT_RsTinv_y4_part.zero();
        RxallT_RsTinv_y4_part.get(0)->set(*RxT_RsTinv_y4);


        y3->axpy(-one, *RxallT_RsTinv_y4);

        // std::cout << " C_Lzz_Ct_inv.m() " << C_Lzz_Ct_inv.m() << std::endl;
        // std::cout << " C_Lzz_Ct_inv.n() " << C_Lzz_Ct_inv.n() << std::endl;

        //std::cout << " z1->dimension() " << z1->dimension() << std::endl;
        //std::cout << " z2->dimension() " << z2->dimension() << std::endl;
        //std::cout << " z3->dimension() " << z3->dimension() << std::endl;
        //std::cout << " z4->dimension() " << z4->dimension() << std::endl;
        //std::cout << " y1->dimension() " << y1->dimension() << std::endl;
        //std::cout << " y2->dimension() " << y2->dimension() << std::endl;
        //std::cout << " y3->dimension() " << y3->dimension() << std::endl;
        //std::cout << " y4->dimension() " << y4->dimension() << std::endl;
        //std::cout << " x1->dimension() " << x1->dimension() << std::endl;
        //std::cout << " x2->dimension() " << x2->dimension() << std::endl;
        //std::cout << " x3->dimension() " << x3->dimension() << std::endl;
        //std::cout << " x4->dimension() " << x4->dimension() << std::endl;

        //std::cout << " input.norm() " << input.norm() << std::endl;
        //std::cout << " output.norm() " << output.norm() << std::endl;
        //std::cout << " z1->norm() " << z1->norm() << std::endl;
        //std::cout << " z2->norm() " << z2->norm() << std::endl;
        //std::cout << " z3->norm() " << z3->norm() << std::endl;
        //std::cout << " z4->norm() " << z4->norm() << std::endl;
        //std::cout << " y1->norm() " << y1->norm() << std::endl;
        //std::cout << " y2->norm() " << y2->norm() << std::endl;
        //std::cout << " y3->norm() " << y3->norm() << std::endl;
        //std::cout << " y4->norm() " << y4->norm() << std::endl;
        //std::cout << " x1->norm() " << x1->norm() << std::endl;
        //std::cout << " x2->norm() " << x2->norm() << std::endl;
        //std::cout << " x3->norm() " << x3->norm() << std::endl;
        //std::cout << " x4->norm() " << x4->norm() << std::endl;
        //std::abort();

        // Need to solve for x2 and x4 simultaneously
        // [ (cx - cs Rsinv Rx)          0              ]    [x2]   =   [y2]
        // [        Lzz          (cx - cs Rsinv Rx)^T   ]    [x4]   =   [y3]
        Teuchos::ParameterList gmres_setting;
        //gmres_setting.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-6);
        //gmres_setting.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-4);
        gmres_setting.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-26);
        gmres_setting.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-12);
        gmres_setting.sublist("General").sublist("Krylov").set("Iteration Limit", 300);
        //gmres_setting.sublist("General").sublist("Krylov").set("Use Initial Guess", true);
        gmres_setting.sublist("General").sublist("Krylov").set("Use Initial Guess", false);
        ROL::Ptr< ROL::Krylov<Real> > krylov = ROL::KrylovFactory<Real>(gmres_setting);
        auto &gmres = dynamic_cast<ROL::GMRES<Real>&>(*krylov);
        //if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0) gmres.enableOutput(std::cout);


        int iter_Krylov_;  ///< GMRES iteration counter
        int flag_Krylov_;  ///< GMRES termination flag

        ROL::Ptr<ROL::PartitionedVector<Real>> x2_x4 = ROL::makePtr<ROL::PartitionedVector<Real>>(
                std::vector<ROL::Ptr<ROL::Vector<Real>> >({x2, x4})
            );
        ROL::Ptr<ROL::PartitionedVector<Real>> y2_y3 = ROL::makePtr<ROL::PartitionedVector<Real>>(
                std::vector<ROL::Ptr<ROL::Vector<Real>> >({y2, y3})
            );

        const ROL::Ptr<ROL::Secant<Real> > Lzz = secant_;
        ROL::Ptr<ROL::LinearOperator<Real> > CLzzC_block_operator = ROL::makePtr<CLzzC_block>(C, Lzz, bound_constraints_, des_plus_dual_, bounded_constraint_tolerance_);
        //ROL::Ptr<ROL::LinearOperator<Real> > CLzzC_block_operator = ROL::makePtr<Identity_Preconditioner_FlipVectors>();
        ROL::Ptr<ROL::LinearOperator<Real> > precond_identity = ROL::makePtr<Identity_Preconditioner_FlipVectors>();
        gmres.run(*x2_x4,*CLzzC_block_operator,*y2_y3,*precond_identity,iter_Krylov_,flag_Krylov_);

        // Solve for x1
        ROL::Ptr<ROL::Vector<Real>> Rx_x2 = x1->clone();
        state_constraints_->applyJacobian_2(*Rx_x2, *(x2->get(0)), *simulation_variables_, *control_variables_, tol);
        ROL::Ptr<ROL::Vector<Real>> y1_minus_Rx_x2 = y1->clone();
        y1_minus_Rx_x2->set(*y1);
        y1_minus_Rx_x2->axpy(-one, *Rx_x2);
        if (use_approximate_preconditioner_) {
            state_constraints_->applyInverseJacobianPreconditioner_1(*x1, *y1_minus_Rx_x2, *simulation_variables_, *control_variables_, tol);
        } else {
            state_constraints_->applyInverseJacobian_1(*x1, *y1_minus_Rx_x2, *simulation_variables_, *control_variables_, tol);
        }
#ifdef REPLACE_INVERSE_WITH_IDENTITY
        x1->set(*y1_minus_Rx_x2);
#endif

        // Solve for x3
        x3->set(*RsTinv_y4);
        for(unsigned int i = 0; i < n_equality_constraints_; ++ i) {
            const ROL::Vector<Real> &x4_dual_equ = *(x4->get(i));
            const ROL::SingletonVector<Real> &x4_dual_equ_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(x4_dual_equ);
            const Real x4_dual_equ_val = x4_dual_equ_singleton.getValue();
            x3->axpy(x4_dual_equ_val, *cs_Rsinv[i]);
        }


 
        // // const ROL::Ptr<const ROL::Vector<Real>> src_design = output_design_constraint.get_1();
        // // const auto &src_state_split_control = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_design);
        // // const ROL::Ptr<const ROL::Vector<Real>> z1 = src_state_split_control.get_1();
        // // const ROL::Ptr<const ROL::Vector<Real>> z2 = src_state_split_control.get_2();

        // // const ROL::Ptr<const ROL::Vector<Real>> src_constraint = output_design_constraint.get_2();
        // // const auto &src_constraintpde_split_constraintother = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_constraint);
        // // const ROL::Ptr<const ROL::Vector<Real>> z3 = src_constraintpde_split_constraintother.get_1();
        // // const ROL::Ptr<const ROL::Vector<Real>> z4 = src_constraintpde_split_constraintother.get_2();

        // // // Split output vector
        // // ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();
        // // auto &dst_design_split_constraint = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_rol);

        // // ROL::Ptr<ROL::Vector<Real>> dst_design = dst_design_split_constraint.get_1();
        // // auto &dst_state_split_control = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_design);
        // // ROL::Ptr<ROL::Vector<Real>> x1 = dst_state_split_control.get_1();
        // // ROL::Ptr<ROL::Vector<Real>> x2 = dst_state_split_control.get_2();

        // // ROL::Ptr<ROL::Vector<Real>> dst_constraint = dst_design_split_constraint.get_2();
        // // auto &dst_constraintpde_split_constraintother = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_constraint);
        // // ROL::Ptr<ROL::Vector<Real>> x3 = dst_constraintpde_split_constraintother.get_1();
        // // ROL::Ptr<ROL::Vector<Real>> x4 = dst_constraintpde_split_constraintother.get_2();

        // ROL::Ptr<ROL::Vector<Real>> x1, x2, x3, x4;
        // ROL::Ptr<ROL::Vector<Real>> z1, z2, z3, z4;

        // // Evaluate y ********************

        // // Evaluate y1 = z3
        // y1->set(*z3);

        // // Evaluate y2 = z4 - (cs Rs^{-1}) y1
        // y2->set(*z4);

        // if (use_second_order_terms_) {
        //     // Evaluate Rs^{-1} y1
        //     if (use_approximate_preconditioner_) {
        //         equality_constraints_simopt_->applyInverseJacobianPreconditioner_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
        //     } else {
        //         equality_constraints_simopt_->applyInverseJacobian_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
        //     }

        //     equality_constraints_simopt_->applyAdjointHessian_11 (*temp_1, *dual_state_, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
        //     y3->axpy(-1.0, *temp_1);
        //     objective_->hessVec_11(*temp_1, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
        //     y3->axpy(-1.0, *temp_1);
        // }

        // // Evaluate y2 = z2 - Rx^{T} Rs^{-T} y3 - Lxs Rs^{-1} y1  
        // auto RsTinv_y3 = y3->clone();
        // if (use_approximate_preconditioner_) {
        //     equality_constraints_simopt_->applyInverseAdjointJacobianPreconditioner_1(*RsTinv_y3, *y3, *simulation_variables_, *control_variables_, tol);
        // } else {
        //     equality_constraints_simopt_->applyInverseAdjointJacobian_1(*RsTinv_y3, *y3, *simulation_variables_, *control_variables_, tol);
        // }
        // equality_constraints_simopt_->applyAdjointJacobian_2(*y2, *RsTinv_y3, *simulation_variables_, *control_variables_, tol);
        // y2->scale(-1.0);
        // y2->plus(*z2);

        // if (use_second_order_terms_) {
        //     equality_constraints_simopt_->applyAdjointHessian_12(*Lxs_Rsinv_y1, *dual_state_, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
        //     y2->axpy(-1.0,*Lxs_Rsinv_y1);
        //     objective_->hessVec_21(*Lxs_Rsinv_y1, *Rsinv_y1, *simulation_variables_, *control_variables_, tol);
        //     y2->axpy(-1.0,*Lxs_Rsinv_y1);
        // }

        // // Evaluate x ********************

        // // x2 = Lzz^{-1} y2
        // const bool use_secant = true;
        // if (use_secant) {
        //     secant_->applyH( *x2, *y2);
        // } else {
        //     x2->set(*y2);
        // }

        // // x1 = Rs^{-1} (y1 - Ad x2)

        // // temp1 = y1 - Ad x2
        // equality_constraints_simopt_->applyJacobian_2(*temp_1, *x2, *simulation_variables_, *control_variables_, tol);
        // temp_1->scale(-1.0);
        // temp_1->axpy(1.0, *y1);

        // auto Rsinv_y1_minus_Ad_x2 = y1->clone();
        // if (use_approximate_preconditioner_) {
        //     equality_constraints_simopt_->applyInverseJacobianPreconditioner_1(*x1, *temp_1, *simulation_variables_, *control_variables_, tol);
        // } else {
        //     equality_constraints_simopt_->applyInverseJacobian_1(*x1, *temp_1, *simulation_variables_, *control_variables_, tol);
        // }

        // // x3 = Rs^{-T} x3_rhs
        // // x3_rhs  = y3
        // auto x3_rhs = y3->clone();

        // if (use_second_order_terms_) {

        //     // x3_rhs += -(Lsx - Lss Rs^{-1} Rx x2)

        //     auto negative_Rsinv_Ad_x2 = Rsinv_y1_minus_Ad_x2;
        //     negative_Rsinv_Ad_x2->axpy(-1.0, *Rsinv_y1);

        //     equality_constraints_simopt_->applyAdjointHessian_11 (*temp_1, *dual_state_, *negative_Rsinv_Ad_x2, *simulation_variables_, *control_variables_, tol);
        //     x3_rhs->axpy(-1.0, *temp_1);
        //     objective_->hessVec_11(*temp_1, *negative_Rsinv_Ad_x2, *simulation_variables_, *control_variables_, tol);
        //     x3_rhs->axpy(-1.0, *temp_1);

        //     equality_constraints_simopt_->applyAdjointHessian_21 (*temp_1, *dual_state_, *x2, *simulation_variables_, *control_variables_, tol);
        //     x3_rhs->axpy(-1.0, *temp_1);
        //     objective_->hessVec_12(*temp_1, *x2, *simulation_variables_, *control_variables_, tol);
        //     x3_rhs->axpy(-1.0, *temp_1);
        // }

        // // x3 = Rs^{-T} x3_rhs
        // if (use_approximate_preconditioner_) {
        //     equality_constraints_simopt_->applyInverseAdjointJacobianPreconditioner_1(*x3, *x3_rhs, *simulation_variables_, *control_variables_, tol);
        // } else {
        //     equality_constraints_simopt_->applyInverseAdjointJacobian_1(*x3, *x3_rhs, *simulation_variables_, *control_variables_, tol);
        // }

        // if (mpi_rank == 0) {
        //     dealii::deallog.depth_console(99);
        // } else {
        //     dealii::deallog.depth_console(0);
        // }

    }

};

} // namespace PHiLiP
#endif
