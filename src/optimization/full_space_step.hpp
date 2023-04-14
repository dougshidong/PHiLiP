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

/// Full-space optimization step where the full KKT linear system is solved with a 
/// preconditioner based on the reduced-space.
/** See Biros and Ghattas' 2005 paper.
 */
template <class Real>
class FullSpace_BirosGhattas : public Step<Real> {
private:

    /// Vector used to clone a vector like the design variables' size and parallel distribution.
    ROL::Ptr<Vector<Real> > design_variable_cloner_;
    /// Vector used to clone a vector like the Lagrange variables' / constraints size and parallel distribution.
    ROL::Ptr<Vector<Real> > lagrange_variable_cloner_;

    /// Merit function used within the line search.
    /** Currently use augmented Lagrangian. */
    ROL::Ptr<Objective<Real>> merit_function_;
    /// Penalty value of the augmented Lagrangian.
    Real penalty_value_;

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
  
    /// Print verbosity.
    /** Does nothing for now. Can add more stuff later */
    int verbosity_;
    /// Whether the last line search's step length is accepted when the maximum iterations is reached.
    /** Currently not used. */
    bool acceptLastAlpha_;
  
    /// Parameter list.
    ROL::ParameterList parlist_;
  
    /// Line search name.
    std::string lineSearchName_;  
    /// Name of secant used as a reduced-Hessian preconditioner.
    std::string secantName_;

    /// Use the Tilde{P} version of Biros and Ghattas' preconditioner.
    bool use_approximate_full_space_preconditioner_;
    /// Preconditioner name.
    /** Either P2, P4, P2A, P4A, Identity */
    std::string preconditioner_name_;
  
    /// Norm of the control search direction.
    double search_ctl_norm;
    /// Norm of the simulation search direction.
    double search_sim_norm;
    /// Norm of the adjoint search direction.
    double search_adj_norm;

    /// Number of line searches used in the last design cycle.
    int n_linesearches;

    /// Number of the output file. Mesh and solution are output in update(), after updating the variables.   
    unsigned int output_count = 2000;
public:
  
    using Step<Real>::initialize; ///< See base class.
    using Step<Real>::compute; ///< See base class.
    using Step<Real>::update; ///< See base class.
  
    /** \brief Constructor.
  
        Standard constructor to build a FullSpace_BirosGhattas object.
        Algorithmic specifications are passed in through a ROL::ParameterList.
  
        @param[in]     parlist    is a parameter list containing algorithmic specifications
        @param[in]     lineSearch is a user-defined line search object
        @param[in]     secant     is a user-defined secant object
    */
    FullSpace_BirosGhattas(
        ROL::ParameterList &parlist,
        const ROL::Ptr<LineSearch<Real> > &lineSearch = ROL::nullPtr,
        const ROL::Ptr<Secant<Real> > &secant = ROL::nullPtr);

    /** Evaluates the gradient of the Lagrangian
     */
    void computeLagrangianGradient(
        Vector<Real> &lagrangian_gradient,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        const Vector<Real> &objective_gradient,
        Constraint<Real> &equal_constraints) const;

    /** Evaluates the initial Lagrange multipler by solving the augmented system.
     *  Note that we could simplify this by simply solving for the adjoint the typical
     *  manner. Using this simply because it was used by a previous ROL algorithm.
     *  Has a higher initial cost than solving the adjoint directly.
     */
    void computeInitialLagrangeMultiplier(
        Vector<Real> &lagrange_mult,
        const Vector<Real> &design_variables,
        const Vector<Real> &objective_gradient,
        Constraint<Real> &equal_constraints) const;

  
    /// Initialize with objective and equality constraints.
    /** Simply calls the more general initialize with null bounded constraints.
     */
    virtual void initialize(
        Vector<Real> &design_variables,
        const Vector<Real> &gradient,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &equal_constraints_values,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        AlgorithmState<Real> &algo_state ) override;

    /// Initialize with objective, equality constraints, and bounded constraints.
    /** Note that the current setup does not work for bounded constraints. It will
     *  simply ignore the bounded constraints.
     */
    void initialize(
        Vector<Real> &design_variables,
        const Vector<Real> &gradient,
        Vector<Real> &lagrange_mult,
        const Vector<Real> &equal_constraints_values,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        BoundConstraint<Real> &bound_constraints,
        AlgorithmState<Real> &algo_state ) override;

    /// Evaluates the penalty of the augmented Lagrangian function using Biros and Ghattas' lower bound.
    Real computeAugmentedLagrangianPenalty(
        const Vector<Real> &search_direction,
        const Vector<Real> &lagrange_mult_search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &objective_gradient,
        const Vector<Real> &equal_constraints_values,
        const Vector<Real> &adjoint_jacobian_lagrange,
        Constraint<Real> &equal_constraints,
        const Real offset);

    /// Solve a linear system using deal.II's F/GMRES solver.
    template<typename MatrixType, typename VectorType, typename PreconditionerType>
    std::vector<double>
    solve_linear (
        MatrixType &matrix_A,
        VectorType &right_hand_side,
        VectorType &solution,
        PreconditionerType &preconditioner);
        //const PHiLiP::Parameters::LinearSolverParam & param = );

    /// Setup and solve the large KKT system.
    std::vector<Real> solve_KKT_system(
        Vector<Real> &search_direction,
        Vector<Real> &lag_search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints);

    /// Computes the search directions.
    /** Uses the more general function with bounded constraints.
     */
    void compute(
        Vector<Real> &search_direction,
        const Vector<Real> &design_variables,
        const Vector<Real> &lagrange_mult,
        Objective<Real> &objective,
        Constraint<Real> &equal_constraints,
        AlgorithmState<Real> &algo_state ) override;

    /// Computes the search directions.
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
  
        @param[in,out]   design_variables        are the updated design variables (control and simulation)
        @param[in,out]   lagrange_mult           are the updated dual variables
        @param[in]       search_direction        is the computed design step
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

    /** \brief Update step, if successful.
  
        Given a trial step, \f$s_k\f$, this function updates \f$x_{k+1}=x_k+s_k\f$. 
        This function also updates the secant approximation.
  
        @param[in,out]   design_variables        are the updated design variables (control and simulation)
        @param[in,out]   lagrange_mult           are the updated dual variables
        @param[in]       search_direction        is the computed design step
        @param[in]       objective               is the objective function
        @param[in]       equal_constraints       are the equality constraints
        @param[in]       bound_constraints       are the bounded constraints
        @param[in]       algo_state              contains the current state of the algorithm
    */
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
        @param[in]     print_header  if set to true will print the header at each iteration
    */
    std::string print( AlgorithmState<Real> & algo_state, bool print_header = false ) const override;

private:
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    unsigned int linear_iteration_limit; ///< Linear iteration limit
    Real regularization_parameter; ///< Factor multiplied by identity to be added to the control hessian.
    Real regularization_scaling; ///< Scaling of regularization parameter depending on control variable's search direction.
    Real regularization_tol_low; ///< Tolerance below which regularization parameter is decreased.
    Real regularization_tol_high; ///< Control search direction tolerance above which the regularization parameter is increased.

}; // class FullSpace_BirosGhattas

} // namespace ROL
#endif
