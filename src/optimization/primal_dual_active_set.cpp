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
#include "ROL_SlacklessObjective.hpp"

#include <deal.II/lac/full_matrix.h>

#include "optimization/flow_constraints.hpp"
#include "primal_dual_active_set.hpp"
#include "optimization/dealii_solver_rol_vector.hpp"

namespace PHiLiP {

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
                for (unsigned int ieq = 0; ieq < neq; ++ieq) {
                    const ROL::SingletonVector<Real> &x4_adj_equ_var_singleton = dynamic_cast<const ROL::SingletonVector<Real>&>(*x4_adj_equ_var[ieq]);
                    y3_ctl->axpy(x4_adj_equ_var_singleton.getValue(), *C[ieq]);
                }
                y3_ctl->axpy(one, x4_active_ctl_dual_inequality);

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
            slack_vecs.push_back( design_variables_->get(i_vec)->clone() );
            slack_vecs[i_vec-1]->set( *(design_variables_->get(i_vec)) );
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
            slack_vecs.push_back( design_variables_->get(i_vec)->clone() );
            slack_vecs[i_vec]->set( *(design_variables_->get(i_vec)) );
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
            slack_vecs.push_back( design_variables_->get(i_vec)->clone() );
            slack_vecs[i_vec]->set( *(design_variables_->get(i_vec)) );
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
        const ROL::Ptr<ROL::Constraint<Real>>               equality_constraints,
        const ROL::Ptr<const ROL::Vector<Real>>             dual_equality,
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
        , equality_constraints_simopt_(ROL::makePtrFromRef<ROL::Constraint_Partitioned<Real>>(dynamic_cast<ROL::Constraint_Partitioned<Real>&>(*equality_constraints)))
        , n_equality_constraints_(equality_constraints_simopt_->get_n_constraints())
        , dual_equality_(ROL::makePtrFromRef<const ROL::PartitionedVector<Real>>(dynamic_cast<const ROL::PartitionedVector<Real>&>(*dual_equality)))
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
        //state_constraints_->applyInverseJacobian_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);
        state_constraints_->applyInverseJacobianPreconditioner_1(*Rsinv_y1, *y1, *simulation_variables_, *control_variables_, tol);

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
        //state_constraints_->applyInverseAdjointJacobian_1(*RsTinv_y4, *y4, *simulation_variables_, *control_variables_, tol);
        state_constraints_->applyInverseAdjointJacobianPreconditioner_1(*RsTinv_y4, *y4, *simulation_variables_, *control_variables_, tol);

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

        // std::cout << " x1->dimension() " << x1->dimension() << std::endl;
        // std::cout << " x2->dimension() " << x2->dimension() << std::endl;
        // std::cout << " x3->dimension() " << x3->dimension() << std::endl;
        // std::cout << " x4->dimension() " << x4->dimension() << std::endl;
        // std::cout << " y1->dimension() " << y1->dimension() << std::endl;
        // std::cout << " y2->dimension() " << y2->dimension() << std::endl;
        // std::cout << " y3->dimension() " << y3->dimension() << std::endl;
        // std::cout << " y4->dimension() " << y4->dimension() << std::endl;

        // std::cout << " input.norm() " << input.norm() << std::endl;
        // std::cout << " output.norm() " << output.norm() << std::endl;
        // std::cout << " x1->norm() " << x1->norm() << std::endl;
        // std::cout << " x2->norm() " << x2->norm() << std::endl;
        // std::cout << " x3->norm() " << x3->norm() << std::endl;
        // std::cout << " x4->norm() " << x4->norm() << std::endl;
        // std::cout << " y1->norm() " << y1->norm() << std::endl;
        // std::cout << " y2->norm() " << y2->norm() << std::endl;
        // std::cout << " y3->norm() " << y3->norm() << std::endl;
        // std::cout << " y4->norm() " << y4->norm() << std::endl;
        //std::abort();

        // Need to solve for x2 and x4 simultaneously
        // [ (cx - cs Rsinv Rx)          0              ]    [x2]   =   [y2]
        // [        Lzz          (cx - cs Rsinv Rx)^T   ]    [x4]   =   [y3]
        Teuchos::ParameterList gmres_setting;
        gmres_setting.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1e-6);
        gmres_setting.sublist("General").sublist("Krylov").set("Relative Tolerance", 1e-4);
        gmres_setting.sublist("General").sublist("Krylov").set("Iteration Limit", 300);
        gmres_setting.sublist("General").sublist("Krylov").set("Use Initial Guess", true);
        //gmres_setting.sublist("General").sublist("Krylov").set("Use Initial Guess", false);
        ROL::Ptr< ROL::Krylov<Real> > krylov = ROL::KrylovFactory<Real>(gmres_setting);
        auto &gmres = dynamic_cast<ROL::GMRES<Real>&>(*krylov);
        if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0) gmres.enableOutput(std::cout);


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
        ROL::Ptr<ROL::LinearOperator<Real> > precond_identity = ROL::makePtr<Identity_Preconditioner_FlipVectors>();
        gmres.run(*x2_x4,*CLzzC_block_operator,*y2_y3,*precond_identity,iter_Krylov_,flag_Krylov_);

        // Solve for x1
        ROL::Ptr<ROL::Vector<Real>> Rx_x2 = x1->clone();
        state_constraints_->applyJacobian_2(*Rx_x2, *(x2->get(0)), *simulation_variables_, *control_variables_, tol);
        ROL::Ptr<ROL::Vector<Real>> y1_minus_Rx_x2 = y1->clone();
        y1_minus_Rx_x2->set(*y1);
        y1_minus_Rx_x2->axpy(-one, *Rx_x2);
        //state_constraints_->applyInverseJacobian_1(*x1, *y1_minus_Rx_x2, *simulation_variables_, *control_variables_, tol);
        state_constraints_->applyInverseJacobianPreconditioner_1(*x1, *y1_minus_Rx_x2, *simulation_variables_, *control_variables_, tol);

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
PrimalDualActiveSetStep<Real>::PDAS_KKT_System::
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
{
    temp_design_          = design_variables_->clone();
    temp_dual_equality_   = dual_equality_->clone();
    temp_dual_inequality_ = design_variables_->clone();
    if ( !useSecant || secant == ROL::nullPtr ) useSecant_ = false;
}

template <typename Real>
void PrimalDualActiveSetStep<Real>::PDAS_KKT_System::
apply( ROL::Vector<Real> &Hv, const ROL::Vector<Real> &v, Real &tol ) const
{
    Real one(1);

    ROL::PartitionedVector<Real> &output_partitioned = dynamic_cast<ROL::PartitionedVector<Real>&>(Hv);
    const ROL::PartitionedVector<Real> &input_partitioned = dynamic_cast<const ROL::PartitionedVector<Real>&>(v);

    const ROL::Ptr< const ROL::Vector<Real> > input_design           = input_partitioned.get(0);
    const ROL::Ptr< const ROL::Vector<Real> > input_dual_equality    = input_partitioned.get(1);
    const ROL::Ptr< const ROL::Vector<Real> > input_dual_inequality  = input_partitioned.get(2);

    MPI_Barrier(MPI_COMM_WORLD);
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
        secant_->applyB(*getCtlOpt(*output_design),*getCtlOpt(*input_design));
    } else {
        objective_->hessVec(*output_design,*input_design,*design_variables_,tol);
        equality_constraints_->applyAdjointHessian(*temp_design_, *dual_equality_, *input_design, *design_variables_, tol);
        output_design->axpy(one,*temp_design_);

        // secant_->applyB(*getCtlOpt(*temp_design_),*getCtlOpt(*input_design));
        // output_design->axpy(one,*temp_design_);
        // output_design->scale(one/2.0);
        std::cout << "Adding identity matrix of " << add_identity_ << std::endl;
        getCtlOpt(*output_design)->axpy(add_identity_,*getCtlOpt(*input_design));
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
    if (symmetrize_matrix_) {
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
    if (symmetrize_matrix_) {
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

template <typename Real>
Real PrimalDualActiveSetStep<Real>::computeCriticalityMeasure(
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
PrimalDualActiveSetStep<Real>::PrimalDualActiveSetStep( ROL::ParameterList &parlist ) 
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
void PrimalDualActiveSetStep<Real>::initialize(
    ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &gradient_vec_to_clone, 
    ROL::Vector<Real> &dual_equality,
    const ROL::Vector<Real> &constraint_vec_to_clone, 
    ROL::Objective<Real> &objective,
    ROL::Constraint<Real> &equality_constraints, 
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    initialize(design_variables, gradient_vec_to_clone, gradient_vec_to_clone, objective, bound_constraints, algo_state);

    (void) equality_constraints;
    try {
        const ROL::Constraint_Partitioned<Real> &equality_constraints_partitioned = dynamic_cast<const ROL::Constraint_Partitioned<Real>&>(equality_constraints);
        const PHiLiP::FlowConstraints<PHILIP_DIM> &flow_constraints = dynamic_cast<const PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equality_constraints_partitioned.get(0));
        (void) flow_constraints;
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


}

template<typename Real>
void PrimalDualActiveSetStep<Real>::initialize(
    ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &search_direction_vec_to_clone,
    const ROL::Vector<Real> &gradient_vec_to_clone, 
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
void PrimalDualActiveSetStep<Real>::compute2(
    ROL::Vector<Real> &search_direction_design,
    const ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &dual_equality,
    ROL::Objective<Real> &objective,
    ROL::Constraint<Real> &equality_constraints, 
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    Real zero(0), one(1);
    search_direction_design.zero();
    new_dual_equality_->set(dual_equality);
    quadratic_residual_->set(*(step_state->gradientVec));

    new_design_variables_->set(design_variables);

    // Temporary variables
    ROL::Ptr<ROL::Vector<Real>> inactive_dual_equality_ = dual_equality.clone();
    ROL::Ptr<ROL::Vector<Real>> active_dual_equality_ = dual_equality.clone();

    ROL::Ptr<ROL::Vector<Real>> rhs_design_temp = design_variables.clone(); 

    // PDAS iterates through 3 steps.
    // 1. Estimate active set
    // 2. Use active set to determine search direction of the active constraints
    // 3. Solve KKT system for remaining inactive constraints
    for ( iter_PDAS_ = 0; iter_PDAS_ < maxit_; iter_PDAS_++ ) {

        /********************************************************************/
        // Modify iterate vector to check active set
        /********************************************************************/
        des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
        des_plus_dual_->axpy(scale_,*(dual_inequality_));    // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply

        /********************************************************************/
        // Project design_variables onto primal dual feasible set
        // Using approximation of the active set, obtain the search direction since
        // we know that step will be constrained.
        /********************************************************************/
        if (false) {
            search_direction_active_set_->zero();                                     // active_set_search_direction   = 0
        
            search_temp_->set(*bound_constraints.getUpperBound());                    // search_tmp = upper_bound
            search_temp_->axpy(-one,design_variables);                                // search_tmp = upper_bound - design_variables
            desvar_tmp_->set(*search_temp_);                                          // tmp        = upper_bound - design_variables
            bound_constraints.pruneUpperActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = (upper_bound - (upper_bound - design_variables + c*dual_variables)) < 0 ? 0 : upper_bound - design_variables
            search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(upper_bound - design_variables)

            search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(upper_bound - design_variables)
      
            search_temp_->set(*bound_constraints.getLowerBound());                    // search_tmp = lower_bound
            search_temp_->axpy(-one,design_variables);                                // search_tmp = lower_bound - design_variables
            desvar_tmp_->set(*search_temp_);                                          // tmp        = lower_bound - design_variables
            bound_constraints.pruneLowerActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = INACTIVE(lower_bound - design_variables)
            search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(lower_bound - design_variables)
            search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(lower_bound - design_variables)
        } else { 
            get_active_design_minus_bound(*search_direction_active_set_, design_variables, *des_plus_dual_, bound_constraints);
            search_direction_active_set_->scale(-1.0);
        }

        inactive_dual_equality_ = new_dual_equality_->clone();
        active_dual_equality_ = new_dual_equality_->clone();

        /********************************************************************/
        // Apply Hessian to active components of search_direction_design and remove inactive
        /********************************************************************/
        itol_ = std::sqrt(ROL::ROL_EPSILON<Real>());
        // INACTIVE(H)*active_set_search_direction = H*active_set_search_direction
        // INACTIVE(H)*active_set_search_direction = INACTIVE(H*active_set_search_direction)
        gradient_tmp1_->zero();
        if ( useSecantHessVec_ && secant_ != ROL::nullPtr ) {
            secant_->applyB(*getOpt(*gradient_tmp1_),*getOpt(*search_direction_active_set_));
        } else {
            objective.hessVec(*gradient_tmp1_,*search_direction_active_set_,design_variables,itol_);
            equality_constraints.applyAdjointHessian(*gradient_tmp2_, dual_equality, *getOpt(*search_direction_active_set_), *getOpt(design_variables), itol_);
            gradient_tmp1_->axpy(one, *gradient_tmp2_);
        }
        bound_constraints.pruneActive(*gradient_tmp1_,*des_plus_dual_,neps_);
        /********************************************************************/
        // SEPARATE ACTIVE AND INACTIVE COMPONENTS OF THE GRADIENT
        /********************************************************************/
        // Inactive components
        gradient_inactive_set_->set(*(step_state->gradientVec));
        bound_constraints.pruneActive(*gradient_inactive_set_,*des_plus_dual_,neps_);
        // Active components
        gradient_active_set_->set(*(step_state->gradientVec));
        gradient_active_set_->axpy(-one,*gradient_inactive_set_);
        /********************************************************************/
        // SOLVE REDUCED NEWTON SYSTEM 
        /********************************************************************/

        // rhs_design_temp = -(INACTIVE(gradient) + INACTIVE(H*active_set_search_direction))
        rhs_design_temp->set(*gradient_inactive_set_);
        rhs_design_temp->plus(*gradient_tmp1_);
        rhs_design_temp->scale(-one);

        search_direction_design.zero();
        if ( rhs_design_temp->norm() > zero ) {             
            // Initialize Hessian and preconditioner
            ROL::Ptr<ROL::Objective<Real> >       objective_ptr  = ROL::makePtrFromRef(objective);
            ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraint_ptr = ROL::makePtrFromRef(bound_constraints);
            ROL::Ptr<ROL::Constraint<Real> > equality_constraints_ptr = ROL::makePtrFromRef(equality_constraints);

            ROL::Ptr<ROL::Vector<Real>> search_direction_des_ptr = ROL::makePtrFromRef(search_direction_design);
            ROL::Ptr<ROL::Vector<Real>> search_direction_dual_ptr = search_direction_dual_;
            ROL::Ptr<ROL::PartitionedVector<Real>> search_partitioned = ROL::makePtr<ROL::PartitionedVector<Real>>(
                    std::vector<ROL::Ptr<ROL::Vector<Real>> >({search_direction_des_ptr, search_direction_dual_ptr})
                );

            ROL::Ptr<ROL::Vector<Real>> rhs_des_ptr = rhs_design_temp;
            ROL::Ptr<ROL::Vector<Real>> rhs_dual_ptr = (step_state->constraintVec)->clone();
            rhs_dual_ptr->scale(-one);
            ROL::Ptr<ROL::PartitionedVector<Real>> rhs_partitioned = ROL::makePtr<ROL::PartitionedVector<Real>>(std::vector<ROL::Ptr<ROL::Vector<Real>> >({rhs_des_ptr, rhs_dual_ptr}));

            ROL::Ptr<ROL::LinearOperator<Real> > hessian = ROL::makePtr<InactiveConstrainedHessian>(objective_ptr, equality_constraints_ptr, bound_constraint_ptr, algo_state.iterateVec, algo_state.lagmultVec, des_plus_dual_, neps_, secant_, useSecantHessVec_);
            ROL::Ptr<ROL::LinearOperator<Real> > precond = ROL::makePtr<InactiveConstrainedHessianPreconditioner>(objective_ptr, bound_constraint_ptr, algo_state.iterateVec, des_plus_dual_, neps_, secant_, useSecantPrecond_);

            krylov_->run(*search_partitioned, *hessian, *rhs_partitioned, *precond, iter_Krylov_, flag_Krylov_);

            bound_constraints.pruneActive(search_direction_design,*des_plus_dual_,neps_);        // search_direction_design <- inactive_search_direction
        }
        search_direction_design.plus(*search_direction_active_set_);                             // search_direction_design = inactive_search_direction + active_set_search_direction
        /********************************************************************/
        // UPDATE MULTIPLIER 
        /********************************************************************/
        rhs_design_temp->zero();
        if ( useSecantHessVec_ && secant_ != ROL::nullPtr ) {
            secant_->applyB(*getOpt(*rhs_design_temp),*getOpt(search_direction_design));
        } else {
            objective.hessVec(*rhs_design_temp,search_direction_design,design_variables,itol_);
        }
        gradient_tmp1_->set(*rhs_design_temp);
        bound_constraints.pruneActive(*gradient_tmp1_,*des_plus_dual_,neps_);

        // dual^{k+1} = - ( ACTIVE(H * search_direction_design) + gradient_active_set_ )
        dual_inequality_->set(*rhs_design_temp);
        dual_inequality_->axpy(-one,*gradient_tmp1_);
        dual_inequality_->plus(*gradient_active_set_);
        dual_inequality_->scale(-one);
        dual_inequality_->zero();
        /********************************************************************/
        // UPDATE STEP 
        /********************************************************************/
        new_design_variables_->set(design_variables);
        new_design_variables_->plus(search_direction_design);
        quadratic_residual_->set(*(step_state->gradientVec));
        quadratic_residual_->plus(*rhs_design_temp);
        // Compute criticality measure  
        desvar_tmp_->set(*new_design_variables_);
        desvar_tmp_->axpy(-one,quadratic_residual_->dual());
        bound_constraints.project(*desvar_tmp_);
        desvar_tmp_->axpy(-one,*new_design_variables_);
        pcout << "des_var_temp " << desvar_tmp_->norm() << std::endl;
        pcout << "gtol gnorm " << gtol_*algo_state.gnorm << std::endl;
        pcout << "rhs_design_temp.norm() " << rhs_design_temp->norm() << std::endl;
        if ( desvar_tmp_->norm() < gtol_*algo_state.gnorm ) {
            flag_PDAS_ = 0;
            break;
        }
        if ( search_direction_design.norm() < stol_*design_variables.norm() ) {
            flag_PDAS_ = 2;
            break;
        } 
    }
    if ( iter_PDAS_ == maxit_ ) {
        flag_PDAS_ = 1;
    } else {
        iter_PDAS_++;
    }
}


template<typename Real>
void PrimalDualActiveSetStep<Real>::compute_PDAS_rhs(
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
    MPI_Barrier(MPI_COMM_WORLD);
    static int ii = 0; (void) ii;
    std::cout << __PRETTY_FUNCTION__ << " " << ii++ << std::endl;
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
    const bool LINEAR_OBJECTIVE = false;
    if (LINEAR_OBJECTIVE) {
        objective.gradient(*rhs_design, old_design_variables, tol);
    } else {
        objective.gradient(*rhs_design, new_design_variables, tol);
    }

    ROL::Ptr<ROL::Vector<Real>> rhs_design_temp = rhs_design->clone(); 
    const bool LINEAR_CONSTRAINTS = true;//false;
    if (LINEAR_CONSTRAINTS) {
        equality_constraints.applyAdjointJacobian(*rhs_design_temp, new_dual_equality, old_design_variables, tol);
    } else {
        equality_constraints.applyAdjointJacobian(*rhs_design_temp, new_dual_equality, new_design_variables, tol);
    }
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

    if (symmetrize_matrix_) {
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
const Real get_value(unsigned int i, const ROL::Vector<Real> &vec) {

    try {
        /// Base case 1
        /// We have a VectorAdapter from deal.II which can return a value (if single processor).
        const dealii::LinearAlgebra::distributed::Vector<double> &vecdealii = PHiLiP::ROL_vector_to_dealii_vector_reference(vec);
        if (vecdealii.in_local_range(i)) {
            return vecdealii[i];
        } else {
            return -9999999999;
        }
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
void PrimalDualActiveSetStep<Real>::compute(
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

    ROL::Ptr<ROL::Vector<Real>> old_dual_equality = dual_equality.clone();
    old_dual_equality->set(dual_equality);

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

    // pcout << "Old design: " << std::endl;
    // for (int i = 0; i < design_variables.dimension(); ++i) {
    //     pcout << get_value<Real>(i, design_variables) << std::endl;
    // }
    // pcout << "Old dual equality: " << std::endl;
    // for (int i = 0; i < new_dual_equality_->dimension(); ++i) {
    //     pcout << get_value<Real>(i, *new_dual_equality_) << std::endl;
    // }
    // pcout << "Old dual inequality: " << std::endl;
    // for (int i = 0; i < dual_inequality_->dimension(); ++i) {
    //     pcout << get_value<Real>(i, *dual_inequality_) << std::endl;
    // }

    old_dual_inequality_->set(*dual_inequality_);
    new_design_variables_->set(design_variables);
    for ( iter_PDAS_ = 0; iter_PDAS_ < maxit_; iter_PDAS_++ ) {

        /********************************************************************/
        // Modify iterate vector to check active set
        /********************************************************************/
        des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
        const Real positive_scale = 0.001;
        des_plus_dual_->axpy(positive_scale,*(dual_inequality_));    // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply

        //pcout << "des_plus_dual: " << std::endl;
        //for (int i = 0; i < des_plus_dual_->dimension(); ++i) {
        //    pcout << get_value<Real>(i, *des_plus_dual_) << std::endl;
        //}
        auto des_plus_dual_clone = des_plus_dual_->clone();
        des_plus_dual_clone->set(*des_plus_dual_);
        bound_constraints.pruneActive( *des_plus_dual_clone, *des_plus_dual_, neps_);
        std::cout << "des_plus_dual_clone norm: " << des_plus_dual_clone->norm() << std::endl;
        static unsigned int index_to_project_interior = 0;

        if (des_plus_dual_clone->norm() < 1e-14) {

            des_plus_dual_clone->set(*des_plus_dual_);
            bound_constraints.projectInterior( *des_plus_dual_clone );
            //const unsigned int n = des_plus_dual_->dimension();
            const unsigned int n = getCtlOpt(design_variables)->dimension();


            const unsigned int index = index_to_project_interior % n;

            //for (int i = 0; i < des_plus_dual_->dimension(); ++i) {
            //    pcout << get_value<Real>(i, *des_plus_dual_) << std::endl;
            //}

            set_value(index, get_value(index, *des_plus_dual_clone), *des_plus_dual_);
            index_to_project_interior++;

            //for (int i = 0; i < des_plus_dual_->dimension(); ++i) {
            //    pcout << get_value<Real>(i, *des_plus_dual_) << std::endl;
            //}
            

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
        //const unsigned int rhs_size = rhs_partitioned->dimension();
        //pcout << "RHS: " << std::endl;
        //for (unsigned int i = 0; i < rhs_size; ++i) {
        //    pcout << get_value<Real>(i, *rhs_partitioned) << std::endl;
        //}
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
        const ROL::Ptr<ROL::Vector<Real> >    old_dual_equality_ptr   = ROL::makePtrFromRef(*old_dual_equality);
        //const Real add_identity_factor = 10.0;
        //const Real add_identity_factor = std::sqrt(algo_state.gnorm) * 3.0;
        //const Real add_identity_factor = 1.0;//algo_state.gnorm * 3.0;
        const Real add_identity_factor = 0.0;//algo_state.gnorm * 3.0;
        pcout << "Building PDAS KKT system..." << std::endl;
        ROL::Ptr<ROL::LinearOperator<Real> > hessian = ROL::makePtr<PDAS_KKT_System>(objective_ptr,
                                                                                     equality_constraint_ptr, bound_constraint_ptr,
                                                                                     old_design_var_ptr, old_dual_equality_ptr, des_plus_dual_,
                                                                                     add_identity_factor, neps_, secant_, useSecantHessVec_);
        ROL::Ptr<ROL::LinearOperator<Real> > precond = ROL::makePtr<Identity_Preconditioner>();

        // Get FlowConstraint
        const ROL::Ptr<ROL::Constraint_Partitioned<Real> >  equality_constraint_partitioned_ptr = ROL::makePtrFromRef(dynamic_cast<ROL::Constraint_Partitioned<Real>&>(*equality_constraint_ptr));
        const ROL::Ptr<ROL::Constraint<Real> >  state_constraints_ptr = equality_constraint_partitioned_ptr->get(0);

        pcout << "Build constraints..." << std::endl;
        // Get remaining constraints
        const unsigned int n_equality_constraints_ = equality_constraint_partitioned_ptr->get_n_constraints();
        std::vector<ROL::Ptr<ROL::Constraint<Real> >>  remaining_equality_constraints(n_equality_constraints_-1);
        std::vector<bool>  remaining_is_inequality(n_equality_constraints_-1);
        for (unsigned int i = 0; i < n_equality_constraints_-1; ++i) {
            remaining_equality_constraints[i] = equality_constraint_partitioned_ptr->get(i+1);
            remaining_is_inequality[i] = equality_constraint_partitioned_ptr->isInequality_[i+1];
        }
        ROL::Constraint_Partitioned<Real> other_equality_constraints(remaining_equality_constraints, remaining_is_inequality);
        const ROL::Ptr<ROL::Constraint_Partitioned<Real> >  other_equality_constraints_ptr = ROL::makePtrFromRef(other_equality_constraints);

        pcout << "Build dual vectors..." << std::endl;
        const ROL::Ptr<ROL::PartitionedVector<Real> >  old_dual_equality_partitioned_ptr = ROL::makePtrFromRef(dynamic_cast<ROL::PartitionedVector<Real>&>(*old_dual_equality_ptr));
        const ROL::Ptr<const ROL::Vector<Real>>   dual_state_ptr = old_dual_equality_partitioned_ptr->get(0);

        const unsigned int n_dual_vecs = old_dual_equality_partitioned_ptr->numVectors();
        std::vector<ROL::Ptr<ROL::Vector<Real>>> other_dual_equality_vectors(n_dual_vecs-1);
        for (unsigned int i_vec = 1; i_vec < n_dual_vecs; ++i_vec) {
            other_dual_equality_vectors[i_vec-1] = old_dual_equality_partitioned_ptr->get(i_vec);
        }
        const ROL::PartitionedVector<Real> other_dual_equality_partitioned(other_dual_equality_vectors);
        const ROL::Ptr<const ROL::PartitionedVector<Real> >  other_dual_equality_partitioned_ptr = ROL::makePtrFromRef(other_dual_equality_partitioned);



        if (is_full_space_) {

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
                secant_
                );
        }

        pcout << "old_design_variables norm " << design_variables.norm() << std::endl;
        pcout << "new_design_variables_ norm " << new_design_variables_->norm() << std::endl;

        //std::abort();
        bool print_KKT = false;
        //bool print_KKT = (algo_state.iter == 0);
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

                    for (int j = 0; j < basis->dimension(); ++j) {
                        pcout << get_value<Real>(j, *basis);
                    }
                    pcout << std::endl;
                    MPI_Barrier(MPI_COMM_WORLD);
                    {
                        hessian->apply(*column_of_kkt_operator,*basis, tol);
                        precond->applyInverse(*column_of_precond_kkt_operator,*column_of_kkt_operator, tol);

                        //precond->applyInverse(*column_of_kkt_operator,*basis, tol);
                        //hessian->apply(*column_of_precond_kkt_operator,*column_of_kkt_operator, tol);
                    }
                    //preconditioner.vmult(column_of_precond_kkt_operator,*basis);
                    if (do_full_matrix) {
                        for (int j = 0; j < rhs_size; ++j) {
                            fullA[j][i] = get_value(j,*column_of_precond_kkt_operator);
                            //fullA[j][i] = get_value(j,*column_of_kkt_operator);
                            //pcout<< get_value(j,*basis) << " ";
                        }
                        //pcout << std::endl;
                    }
                }
                pcout<<"Dense matrix:"<<std::endl;
                fullA.print_formatted(std::cout, 10, true, 18, "0", 1., 0.);
                std::abort();
            }
        }

        auto &gmres = dynamic_cast<ROL::GMRES<Real>&>(*krylov_);
        if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0) gmres.enableOutput(std::cout);
        gmres.run(*search_partitioned,*hessian,*rhs_partitioned,*precond,iter_Krylov_,flag_Krylov_);

        if (symmetrize_matrix_) {
            bound_constraints.pruneInactive(*search_dual_inequality, *des_plus_dual_, tol);
            //search_dual_inequality->zero();

            rhs_dual_inequality_temp->set(*dual_inequality_);
            bound_constraints.pruneActive(*rhs_dual_inequality_temp, *des_plus_dual_, tol);
            search_dual_inequality->axpy(-one, *rhs_dual_inequality_temp);

            auto search_design_temp = rhs_dual_inequality->clone();
            search_design_temp->set(*rhs_dual_inequality);
            bound_constraints.pruneInactive(*search_design_temp, *des_plus_dual_, tol);
            bound_constraints.pruneActive(*search_design, *des_plus_dual_, tol);
            search_design->axpy(one, *search_design_temp);
        }


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
        // const unsigned int sea_size = search_partitioned->dimension();
        // pcout << "Search: " << std::endl;
        // for (unsigned int i = 0; i < sea_size; ++i) {
        //     pcout << get_value<Real>(i, *search_partitioned) << std::endl;
        // }

        /********************************************************************/
        // UPDATE STEP 
        /********************************************************************/
        new_design_variables_->plus(*search_design);
        new_dual_equality_->plus(*search_dual_equality);
        dual_inequality_->plus(*search_dual_inequality);

        if ( bound_constraints.isActivated() ) {
            bound_constraints.project(*new_design_variables_);
        }

        // pcout << "new design: " << std::endl;
        // for (int i = 0; i < new_design_variables_->dimension(); ++i) {
        //     pcout << get_value<Real>(i, *new_design_variables_) << std::endl;
        // }
        // pcout << "new dual equality: " << std::endl;
        // for (int i = 0; i < new_dual_equality_->dimension(); ++i) {
        //     pcout << get_value<Real>(i, *new_dual_equality_) << std::endl;
        // }
        // pcout << "new dual inequality: " << std::endl;
        // for (int i = 0; i < dual_inequality_->dimension(); ++i) {
        //     pcout << get_value<Real>(i, *dual_inequality_) << std::endl;
        // }

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
void PrimalDualActiveSetStep<Real>::compute(
    ROL::Vector<Real> &search_direction_design,
    const ROL::Vector<Real> &design_variables,
    ROL::Objective<Real> &objective,
    ROL::BoundConstraint<Real> &bound_constraints, 
    ROL::AlgorithmState<Real> &algo_state )
{
    ROL::Ptr<ROL::StepState<Real> > step_state = ROL::Step<Real>::getState();
    Real zero(0), one(1);
    search_direction_design.zero();
    quadratic_residual_->set(*(step_state->gradientVec));

    new_design_variables_->set(design_variables);
    // PDAS iterates through 3 steps.
    // 1. Estimate active set
    // 2. Use active set to determine search direction of the active constraints
    // 3. Solve KKT system for remaining inactive constraints
    ROL::Ptr<ROL::Vector<Real>> rhs_design_temp = design_variables.clone(); 
    for ( iter_PDAS_ = 0; iter_PDAS_ < maxit_; iter_PDAS_++ ) {
        /********************************************************************/
        // Modify iterate vector to check active set
        /********************************************************************/
        des_plus_dual_->set(*new_design_variables_);    // des_plus_dual = initial_desvar
        des_plus_dual_->axpy(scale_,*(dual_inequality_));    // des_plus_dual = initial_desvar + c*dualvar, note that papers would usually divide by scale_ instead of multiply
        /********************************************************************/
        // Project design_variables onto primal dual feasible set
        // Using approximation of the active set, obtain the search direction since
        // we know that step will be constrained.
        /********************************************************************/
        search_direction_active_set_->zero();                                     // active_set_search_direction   = 0
    
        search_temp_->set(*bound_constraints.getUpperBound());                    // search_tmp = upper_bound
        search_temp_->axpy(-one,design_variables);                                // search_tmp = upper_bound - design_variables
        desvar_tmp_->set(*search_temp_);                                          // tmp        = upper_bound - design_variables
        bound_constraints.pruneUpperActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = (upper_bound - (upper_bound - design_variables + c*dual_variables)) < 0 ? 0 : upper_bound - design_variables
        search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(upper_bound - design_variables)

        search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(upper_bound - design_variables)
  
        search_temp_->set(*bound_constraints.getLowerBound());                    // search_tmp = lower_bound
        search_temp_->axpy(-one,design_variables);                                // search_tmp = lower_bound - design_variables
        desvar_tmp_->set(*search_temp_);                                          // tmp        = lower_bound - design_variables
        bound_constraints.pruneLowerActive(*desvar_tmp_,*des_plus_dual_,neps_);   // tmp        = INACTIVE(lower_bound - design_variables)
        search_temp_->axpy(-one,*desvar_tmp_);                                    // search_tmp = ACTIVE(lower_bound - design_variables)
        search_direction_active_set_->plus(*search_temp_);                        // active_set_search_direction += ACTIVE(lower_bound - design_variables)
        /********************************************************************/
        // Apply Hessian to active components of search_direction_design and remove inactive
        /********************************************************************/
        itol_ = std::sqrt(ROL::ROL_EPSILON<Real>());
        // INACTIVE(H)*active_set_search_direction = H*active_set_search_direction
        // INACTIVE(H)*active_set_search_direction = INACTIVE(H*active_set_search_direction)
        gradient_tmp1_->zero();
        if ( useSecantHessVec_ && secant_ != ROL::nullPtr ) {
            secant_->applyB(*getOpt(*gradient_tmp1_),*getOpt(*search_direction_active_set_));
        } else {
            objective.hessVec(*gradient_tmp1_,*search_direction_active_set_,design_variables,itol_);
            //gradient_tmp1_->axpy(10.0, *search_direction_active_set_);
        }
        bound_constraints.pruneActive(*gradient_tmp1_,*des_plus_dual_,neps_);
        /********************************************************************/
        // SEPARATE ACTIVE AND INACTIVE COMPONENTS OF THE GRADIENT
        /********************************************************************/
        // Inactive components
        gradient_inactive_set_->set(*(step_state->gradientVec));
        bound_constraints.pruneActive(*gradient_inactive_set_,*des_plus_dual_,neps_);
        // Active components
        gradient_active_set_->set(*(step_state->gradientVec));
        gradient_active_set_->axpy(-one,*gradient_inactive_set_);
        /********************************************************************/
        // SOLVE REDUCED NEWTON SYSTEM 
        /********************************************************************/

        // rhs_design_temp = -(INACTIVE(gradient) + INACTIVE(H*active_set_search_direction))
        rhs_design_temp->set(*gradient_inactive_set_);
        rhs_design_temp->plus(*gradient_tmp1_);
        rhs_design_temp->scale(-one);

        search_direction_design.zero();
        if ( rhs_design_temp->norm() > zero ) {             
            // Initialize Hessian and preconditioner
            ROL::Ptr<ROL::Objective<Real> >       objective_ptr  = ROL::makePtrFromRef(objective);
            ROL::Ptr<ROL::BoundConstraint<Real> > bound_constraint_ptr = ROL::makePtrFromRef(bound_constraints);
            ROL::Ptr<ROL::LinearOperator<Real> > hessian = ROL::makePtr<InactiveHessian>(objective_ptr, bound_constraint_ptr, algo_state.iterateVec, des_plus_dual_, neps_, secant_, useSecantHessVec_);
            ROL::Ptr<ROL::LinearOperator<Real> > precond = ROL::makePtr<InactiveHessianPreconditioner>(objective_ptr, bound_constraint_ptr, algo_state.iterateVec, des_plus_dual_, neps_, secant_, useSecantPrecond_);
            krylov_->run(search_direction_design,*hessian,*rhs_design_temp,*precond,iter_Krylov_,flag_Krylov_);
            bound_constraints.pruneActive(search_direction_design,*des_plus_dual_,neps_);        // search_direction_design <- inactive_search_direction
        }
        search_direction_design.plus(*search_direction_active_set_);                             // search_direction_design = inactive_search_direction + active_set_search_direction
        /********************************************************************/
        // UPDATE MULTIPLIER 
        /********************************************************************/
        rhs_design_temp->zero();
        if ( useSecantHessVec_ && secant_ != ROL::nullPtr ) {
            secant_->applyB(*getOpt(*rhs_design_temp),*getOpt(search_direction_design));
        } else {
            objective.hessVec(*rhs_design_temp,search_direction_design,design_variables,itol_);
            //rhs_design_temp->axpy(10.0, search_direction_design);
        }
        gradient_tmp1_->set(*rhs_design_temp);
        bound_constraints.pruneActive(*gradient_tmp1_,*des_plus_dual_,neps_);

        // dual^{k+1} = - ( ACTIVE(H * search_direction_design) + gradient_active_set_ )
        dual_inequality_->set(*rhs_design_temp);
        dual_inequality_->axpy(-one,*gradient_tmp1_);
        dual_inequality_->plus(*gradient_active_set_);
        dual_inequality_->scale(-one);
        /********************************************************************/
        // UPDATE STEP 
        /********************************************************************/
        new_design_variables_->set(design_variables);
        new_design_variables_->plus(search_direction_design);
        //quadratic_residual_->set(*(step_state->gradientVec));
        //quadratic_residual_->plus(*rhs_design_temp);
        quadratic_residual_->set(*rhs_design_temp);
        // Compute criticality measure  
        desvar_tmp_->set(*new_design_variables_);
        desvar_tmp_->axpy(-one,quadratic_residual_->dual());
        bound_constraints.project(*desvar_tmp_);
        desvar_tmp_->axpy(-one,*new_design_variables_);
        std::cout << "des_var_temp " << desvar_tmp_->norm() << std::endl;
        std::cout << "gtol gnorm " << gtol_*algo_state.gnorm << std::endl;
        std::cout << "rhs_design_temp.norm() " << rhs_design_temp->norm() << std::endl;

        if ( desvar_tmp_->norm() < gtol_*algo_state.gnorm ) {
            flag_PDAS_ = 0;
            break;
        }
        if ( search_direction_design.norm() < stol_*design_variables.norm() ) {
            flag_PDAS_ = 2;
            break;
        } 
    }
    if ( iter_PDAS_ == maxit_ ) {
        flag_PDAS_ = 1;
    } else {
        iter_PDAS_++;
    }
}
  
template<typename Real>
void PrimalDualActiveSetStep<Real>::update(
    ROL::Vector<Real> &design_variables,
    const ROL::Vector<Real> &search_direction_design,
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
    
    if ( secant_ != ROL::nullPtr ) {
      gradient_tmp1_->set(*(step_state->gradientVec));
    }
    algo_state.gnorm = computeCriticalityMeasure(design_variables,objective,bound_constraints,tol);
    algo_state.ngrad++;

    if ( secant_ != ROL::nullPtr ) {
      secant_->updateStorage(design_variables,*(step_state->gradientVec),*gradient_tmp1_,search_direction_design,algo_state.snorm,algo_state.iter+1);
    }
    (algo_state.iterateVec)->set(design_variables);
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
void PrimalDualActiveSetStep<Real>::update(
    ROL::Vector<Real> &design_variables,
    ROL::Vector<Real> &dual_equality,
    const ROL::Vector<Real> &search_direction_design,
    ROL::Objective<Real> &objective,
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
        Real design_snorm = getCtlOpt(*search_design)->norm();
        secant_->updateStorage(*getCtlOpt(design_variables),*getCtlOpt(*(step_state->gradientVec)),*getCtlOpt(*gradient_tmp1_),*getCtlOpt(*search_design),design_snorm,algo_state.iter+1);

        //pcout << "new gradient: " << std::endl;
        //for (int i = 0; i < step_state->gradientVec->dimension(); ++i) {
        //    pcout << get_value<Real>(i, *step_state->gradientVec) << std::endl;
        //}
        //pcout << "old gradient: " << std::endl;
        //for (int i = 0; i < step_state->gradientVec->dimension(); ++i) {
        //    pcout << get_value<Real>(i, *gradient_tmp1_) << std::endl;
        //}
    }

    algo_state.ngrad++;

    (algo_state.iterateVec)->set(design_variables);
    (algo_state.lagmultVec)->set(dual_equality);
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
std::string PrimalDualActiveSetStep<Real>::printName( void ) const
{
    std::stringstream hist;
    hist << "\nPrimal Dual Active Set Newton's Method\n";
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
  
template class PrimalDualActiveSetStep <double>;
} // namespace ROL

