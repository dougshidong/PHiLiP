#ifndef __FULLSPACE_STEP_H__
#define __FULLSPACE_STEP_H__

#include "ROL_Types.hpp"
#include "ROL_Step.hpp"
#include "ROL_LineSearch.hpp"

// Unconstrained Methods
#include "ROL_GradientStep.hpp"
#include "ROL_NonlinearCGStep.hpp"
#include "ROL_SecantStep.hpp"
#include "ROL_NewtonStep.hpp"
#include "ROL_NewtonKrylovStep.hpp"

#include <sstream>
#include <iomanip>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>

#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/dealii_solver_rol_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"

#include "optimization/kkt_operator.hpp"
#include "optimization/kkt_birosghattas_preconditioners.hpp"

namespace ROL {

template <class Real>
class FullSpace_BirosGhattas : public Step<Real> {
private:

    /// Vector used to clone a vector like the design variables' size and parallel distribution.
    ROL::Ptr<Vector<Real> > design_variable_cloner_;
    /// Vector used to clone a vector like the Lagrange variables' / constraints size and parallel distribution.
    ROL::Ptr<Vector<Real> > lagrange_variable_cloner_;

    /// Merit function used within the line search.
    ROL::Ptr<Objective<Real>> merit_function_;

    /// Lagrange multipliers search direction.
    ROL::Ptr<Vector<Real>> lagrange_mult_search_direction_;

    /// Store previous gradient for secant method.
    ROL::Ptr<Vector<Real>> previous_reduced_gradient_;

    /// Secant object (used for quasi-Newton preconditioner).
    ROL::Ptr<Secant<Real> > secant_;

    /// Line-search object for globalization.
    ROL::Ptr<LineSearch<Real> >  lineSearch_;
  
    /// Enum determines type of secant to use as reduced Hessian preconditioner.
    ESecant esec_;
    /// Enum determines type of line search.
    ELineSearch els_;
    /// Enum determines type of curvature condition.
    ECurvatureCondition econd_;
  
    int verbosity_;
    Real penalty_value_;
    bool acceptLastAlpha_;  ///< For backwards compatibility. When max function evaluations are reached take last step
  
    ROL::ParameterList parlist_;
  
    std::string lineSearchName_;  
    std::string secantName_;

    bool use_approximate_full_space_preconditioner_;
    std::string preconditioner_name_;
  
    double search_ctl_norm;
    double search_sim_norm;
    double search_adj_norm;

    int n_linesearches;
public:
  
    using Step<Real>::initialize;
    using Step<Real>::compute;
    using Step<Real>::update;
  
    /** \brief Constructor.
  
        Standard constructor to build a FullSpace_BirosGhattas object.
        Algorithmic specifications are passed in through a ROL::ParameterList.
        The line-search type, secant type, Krylov type, or nonlinear CG type can
        be set using user-defined objects or will automatically be initialized through
        the parameter list.
  
        @param[in]     parlist    is a parameter list containing algorithmic specifications
        @param[in]     lineSearch is a user-defined line search object
        @param[in]     secant     is a user-defined secant object
    */
    FullSpace_BirosGhattas(
        ROL::ParameterList &parlist,
        const ROL::Ptr<LineSearch<Real> > &lineSearch = ROL::nullPtr,
        const ROL::Ptr<Secant<Real> > &secant = ROL::nullPtr);

    void computeLagrangianGradient(
        Vector<Real> &lagrangian_gradient,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        const Vector<Real> &objective_gradient,
        Constraint<Real> &equal_constraints) const;

    void computeInitialLagrangeMultiplier(
        Vector<Real> &lagrange_mult,
        const Vector<Real> &design_variables,
        const Vector<Real> &objective_gradient,
        Constraint<Real> &equal_constraints) const;

  
    virtual void initialize(
        Vector<Real> &design_variables,
        const Vector<Real> &gradient,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &equal_constraints_values,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        AlgorithmState<Real> &algo_state ) override;

    void initialize(
        Vector<Real> &design_variables,
        const Vector<Real> &gradient,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &equal_constraints_values,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        BoundConstraint<Real> &bound_constraints,
        AlgorithmState<Real> &algo_state ) override;

    Real computeAugmentedLagrangianPenalty(
        const Vector<Real> &search_direction,
        const Vector<Real> &lagrange_mult_search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &objective_gradient,
        const Vector<Real> &equal_constraints_values,
        const Vector<Real> &adjoint_jacobian_lagrange,
        Constraint<Real> &equal_constraints,
        const Real offset);

    template<typename MatrixType, typename VectorType, typename PreconditionerType>
    std::vector<double>
    solve_linear (
        MatrixType &matrix_A,
        VectorType &right_hand_side,
        VectorType &solution,
        PreconditionerType &preconditioner);
        //const PHiLiP::Parameters::LinearSolverParam & param = );

    std::vector<Real> solve_KKT_system(
        Vector<Real> &search_direction,
        Vector<Real> &lag_search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints);

    void compute(
        Vector<Real> &search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        AlgorithmState<Real> &algo_state ) override;

    void compute(
        Vector<Real> &search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        BoundConstraint<Real> &bound_constraints,
        AlgorithmState<Real> &algo_state ) override;
  
    /** \brief Update step, if successful.
  
        Given a trial step, \f$s_k\f$, this function updates \f$x_{k+1}=x_k+s_k\f$. 
        This function also updates the secant approximation.
  
        @param[in,out]   design_variables        is the updated iterate
        @param[in]       search_direction        is the computed trial step
        @param[in]       objective               is the objective function
        @param[in]       equal_constraints       are the bound equal_constraints
        @param[in]       algo_state              contains the current state of the algorithm
    */
    void update(
        Vector<Real> &design_variables,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &search_direction,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        AlgorithmState<Real> &algo_state ) override;

    void update (
        Vector<Real> &design_variables,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &search_direction,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        BoundConstraint< Real > &bound_constraints,
        AlgorithmState<Real> &algo_state ) override;
  
    /** \brief Print iterate header.
  
        This function produces a string containing header information.
    */
    std::string printHeader( void ) const override;
    
    /** \brief Print step name.
  
        This function produces a string containing the algorithmic step information.
    */
    std::string printName( void ) const override;
  
    /** \brief Print iterate status.
  
        This function prints the iteration status.
  
        @param[in]     algo_state    is the current state of the algorithm
        @param[in]     printHeader   if set to true will print the header at each iteration
    */
    std::string print( AlgorithmState<Real> & algo_state, bool print_header = false ) const override;

private:
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // class FullSpace_BirosGhattas

} // namespace ROL
#endif
