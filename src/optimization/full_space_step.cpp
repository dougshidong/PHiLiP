#include "ROL_AugmentedLagrangian.hpp"
#include "optimization/full_space_step.hpp"
#include "optimization/kkt_operator.hpp"
#include "optimization/kkt_birosghattas_preconditioners.hpp"

#include "global_counter.hpp"

namespace ROL {

template <class Real>
FullSpace_BirosGhattas<Real>::
FullSpace_BirosGhattas(
    ROL::ParameterList &parlist,
    const ROL::Ptr<LineSearch<Real> > &lineSearch,
    const ROL::Ptr<Secant<Real> > &secant)
    : Step<Real>()
    , secant_(secant)
    , lineSearch_(lineSearch)
    , els_(LINESEARCH_USERDEFINED)
    , econd_(CURVATURECONDITION_WOLFE)
    , verbosity_(0)
    , parlist_(parlist)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    // Parse parameter list
    ROL::ParameterList& Llist = parlist.sublist("Step").sublist("Line Search");
    ROL::ParameterList& Glist = parlist.sublist("General");
    econd_ = StringToECurvatureCondition(Llist.sublist("Curvature Condition").get("Type","Strong Wolfe Conditions") );
    acceptLastAlpha_ = Llist.get("Accept Last Alpha", false);
    verbosity_ = Glist.get("Print Verbosity",0);

    preconditioner_name_ = parlist.sublist("Full Space").get("Preconditioner","P4");
    use_approximate_full_space_preconditioner_ = (preconditioner_name_ == "P2A" || preconditioner_name_ == "P4A");

    // Initialize Line Search
    if (lineSearch_ == ROL::nullPtr) {
        lineSearchName_ = Llist.sublist("Line-Search Method").get("Type","Backtracking");
        els_ = StringToELineSearch(lineSearchName_);
        lineSearch_ = LineSearchFactory<Real>(parlist);
    }
    else { // User-defined linesearch provided
        lineSearchName_ = Llist.sublist("Line-Search Method").get("User Defined Line-Search Name",
                                                                  "Unspecified User Defined Line-Search");
    }

    secantName_ = Glist.sublist("Secant").get("Type","Limited-Memory BFGS");
    esec_ = StringToESecant(secantName_);
    secant_ = SecantFactory<Real>(parlist);

}

template <class Real>
void FullSpace_BirosGhattas<Real>::computeLagrangianGradient(
    Vector<Real> &lagrangian_gradient,
    const Vector<Real> &design_variables,
    const Vector<Real> &lagrange_mult,
    const Vector<Real> &objective_gradient,
    Constraint<Real> &equal_constraints) const
{
    /* Apply adjoint of constraint Jacobian to current multiplier. */
    Real tol = std::sqrt(ROL_EPSILON<Real>());

    auto &flow_constraint = (dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(equal_constraints));
    const double old_CFL = flow_constraint.flow_CFL_;
    // Lagrangian of the gradient should not be influenced by the constraint regularization (CFL)
    // Otherwise, you when end magnifying the dual variable values since
    // dL/dW = dI/dW + \lambda^T (M/dt + dRdW)
    flow_constraint.flow_CFL_ = 0.0;
    equal_constraints.applyAdjointJacobian(lagrangian_gradient, lagrange_mult, design_variables, tol);
    flow_constraint.flow_CFL_ = old_CFL;
    lagrangian_gradient.plus(objective_gradient);
}

template <class Real>
void FullSpace_BirosGhattas<Real>::computeInitialLagrangeMultiplier(
    Vector<Real> &lagrange_mult,
    const Vector<Real> &design_variables,
    const Vector<Real> &objective_gradient,
    Constraint<Real> &equal_constraints) const
{
    Real one(1);

    /* Form right-hand side of the augmented system. */
    ROL::Ptr<Vector<Real> > rhs1 = design_variable_cloner_->clone();
    ROL::Ptr<Vector<Real> > rhs2 = lagrange_variable_cloner_->clone();

    // rhs1 is the negative gradient of the Lagrangian
    computeLagrangianGradient(*rhs1, design_variables, lagrange_mult, objective_gradient, equal_constraints);
    rhs1->scale(-one);
    // rhs2 is zero
    rhs2->zero();

    /* Declare left-hand side of augmented system. */
    ROL::Ptr<Vector<Real> > lhs1 = design_variable_cloner_->clone();
    ROL::Ptr<Vector<Real> > lhs2 = lagrange_variable_cloner_->clone();

    /* Compute linear solver tolerance. */
    //Real b1norm  = rhs1->norm();
    Real tol = std::sqrt(ROL_EPSILON<Real>());

    /* Solve augmented system. */
    const std::vector<Real> augiters = equal_constraints.solveAugmentedSystem(*lhs1, *lhs2, *rhs1, *rhs2, design_variables, tol);

    /* Return updated Lagrange multiplier. */
    // lhs2 is the multiplier update
    lagrange_mult.plus(*lhs2);

    // // Evaluate the full gradient wrt u
    // obj_->gradient_1(*dualstate_,*state_,z,tol);
    // // Solve adjoint equation
    // con_->applyInverseAdjointJacobian_1(*adjoint_,*dualstate_,*state_,z,tol);
    // adjoint_->scale(static_cast<Real>(-one));

}


template <class Real>
void FullSpace_BirosGhattas<Real>::initialize(
    Vector<Real> &design_variables,
    const Vector<Real> &gradient,
    Vector<Real> &lagrange_mult,
    const Vector<Real> &equal_constraints_values,
    Objective<Real> &objective,
    Constraint<Real> &equal_constraints,
    AlgorithmState<Real> &algo_state )
{
    BoundConstraint<Real> bound_constraints;
    bound_constraints.deactivate();
    initialize(
        design_variables,
        gradient,
        lagrange_mult,
        equal_constraints_values,
        objective,
        equal_constraints,
        bound_constraints, // new argument
        algo_state);
}

template <class Real>
void FullSpace_BirosGhattas<Real>::initialize(
    Vector<Real> &design_variables,
    const Vector<Real> &gradient,
    Vector<Real> &lagrange_mult,
    const Vector<Real> &equal_constraints_values,
    Objective<Real> &objective,
    Constraint<Real> &equal_constraints,
    BoundConstraint<Real> &bound_constraints,
    AlgorithmState<Real> &algo_state )
{
    pcout << __PRETTY_FUNCTION__ << std::endl;
    Real tol = ROL_EPSILON<Real>();
    Real zero(0);

    // Initialize the algorithm state
    algo_state.iter  = 0;
    algo_state.nfval = 0;
    algo_state.ncval = 0;
    algo_state.ngrad = 0;

    ROL::Ptr<StepState<Real> > step_state = Step<Real>::getState();
    design_variable_cloner_ = design_variables.clone();
    design_variable_cloner_ = gradient.clone();
    lagrange_variable_cloner_ = lagrange_mult.clone();
    lagrange_variable_cloner_ = equal_constraints_values.clone();

    lagrange_mult_search_direction_ = lagrange_mult.clone();

    // Initialize state descent direction and gradient storage
    step_state->descentVec  = design_variables.clone();
    step_state->gradientVec = gradient.clone();
    step_state->constraintVec = equal_constraints_values.clone();
    step_state->searchSize  = zero;

    // Project design_variables onto bound_constraints set
    if ( bound_constraints.isActivated() ) {
      bound_constraints.project(design_variables);
    }
    // Update objective function, get value, and get gradient
    const bool changed_design_variables = true;
    objective.update(design_variables, changed_design_variables, algo_state.iter);
    algo_state.value = objective.value(design_variables, tol);
    algo_state.nfval++;
    objective.gradient(*(step_state->gradientVec), design_variables, tol);
    algo_state.ngrad++;

    // Update equal_constraints.
    equal_constraints.update(design_variables,true,algo_state.iter);
    equal_constraints.value(*(step_state->constraintVec), design_variables, zero);
    algo_state.cnorm = lagrange_variable_cloner_->norm();
    algo_state.ncval++;

    //computeInitialLagrangeMultiplier(lagrange_mult, design_variables, *(step_state->gradientVec), equal_constraints);
    auto &equal_constraints_sim_opt = dynamic_cast<ROL::Constraint_SimOpt<Real>&>(equal_constraints);
    const auto &objective_ctl_gradient = *(dynamic_cast<const Vector_SimOpt<Real>&>(*(step_state->gradientVec)).get_1());
    const auto &design_variables_sim_opt = dynamic_cast<ROL::Vector_SimOpt<Real>&>(design_variables);
    const auto &simulation_variables = *(design_variables_sim_opt.get_1());
    const auto &control_variables    = *(design_variables_sim_opt.get_2());
    equal_constraints_sim_opt.applyInverseAdjointJacobian_1(lagrange_mult, objective_ctl_gradient, simulation_variables, control_variables, tol);
    lagrange_mult.scale(-1.0);

    // Compute gradient of Lagrangian at new multiplier guess.
    ROL::Ptr<Vector<Real> > lagrangian_gradient = step_state->gradientVec->clone();
    computeLagrangianGradient(*lagrangian_gradient, design_variables, lagrange_mult, *(step_state->gradientVec), equal_constraints);
    const auto &lagrangian_gradient_simopt = dynamic_cast<const Vector_SimOpt<Real>&>(*lagrangian_gradient);
    previous_reduced_gradient_ = lagrangian_gradient_simopt.get_2()->clone();
    algo_state.ngrad++;

    auto &flow_constraint = (dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(equal_constraints));
    //flow_constraint.flow_CFL_ = 1.0/std::pow(algo_state.cnorm, 0.5);
    //flow_constraint.flow_CFL_ = 1.0/std::pow(lagrangian_gradient->norm(), 1.00);
    //flow_constraint.flow_CFL_ = -std::max(1.0/std::pow(algo_state.cnorm, 2.0), 100.0);
    //flow_constraint.flow_CFL_ = -1e-0;
    flow_constraint.flow_CFL_ = -100;

    // // Not sure why this is done in ROL_Step.hpp
    // if ( bound_constraints.isActivated() ) {
    //     ROL::Ptr<Vector<Real> > xnew = design_variables.clone();
    //     xnew->set(design_variables);
    //     xnew->axpy(-one,(step_state->gradientVec)->dual());
    //     bound_constraints.project(*xnew);
    //     xnew->axpy(-one,design_variables);
    //     algo_state.gnorm = xnew->norm();
    // }
    // else {
    //     algo_state.gnorm = (step_state->gradientVec)->norm();
    // }

    // I don't have to initialize with merit function since it does nothing
    // with it. But might as well be consistent.
    penalty_value_ = 1.0;
    merit_function_ = ROL::makePtr<ROL::AugmentedLagrangian<Real>> (
            makePtrFromRef<Objective<Real>>(objective),
            makePtrFromRef<Constraint<Real>>(equal_constraints),
            lagrange_mult,
            penalty_value_,
            design_variables,
            equal_constraints_values,
            parlist_);

    // Dummy search direction vector used to initialize the linesearch.
    ROL::Ptr<Vector<Real> > search_direction_dummy = design_variables.clone();
    lineSearch_->initialize(design_variables, *search_direction_dummy, gradient, *merit_function_, bound_constraints);
}

template <class Real>
Real FullSpace_BirosGhattas<Real>::computeAugmentedLagrangianPenalty(
    const Vector<Real> &search_direction,
    const Vector<Real> &lagrange_mult_search_direction,
    const Vector<Real> &design_variables,
    const Vector<Real> &objective_gradient,
    const Vector<Real> &equal_constraints_values,
    const Vector<Real> &adjoint_jacobian_lagrange,
    Constraint<Real> &equal_constraints,
    const Real offset)
{
    pcout << __PRETTY_FUNCTION__ << std::endl;
    // Biros and Ghattas 2005, Part II
    // Equation (2.10)
    Real penalty = objective_gradient.dot(search_direction);
    penalty += adjoint_jacobian_lagrange.dot(search_direction);
    penalty += equal_constraints_values.dot(lagrange_mult_search_direction);

    const ROL::Ptr<Vector<Real>> jacobian_search_direction = equal_constraints_values.clone();
    Real tol = std::sqrt(ROL_EPSILON<Real>());
    equal_constraints.applyJacobian(*jacobian_search_direction, search_direction, design_variables, tol);

    Real denom = jacobian_search_direction->dot(equal_constraints_values);

    penalty /= denom;

    // Note that the offset is not on the fraction as in the paper.
    // The penalty term should always be positive and towards infinity.
    // It is a mistake from the paper since the numerator and denominator can be
    // small and negative. Therefore, the positive offset on a small negative
    // numerator with a small negative denominator might result in a large negative
    // penalty value.
    // if (penalty > 0.0) {
    //     penalty += offset;
    // } else {
    //     penalty = 1.0;
    // }
    penalty += offset;

    return penalty;
}

template <class Real>
template<typename MatrixType, typename VectorType, typename PreconditionerType>
std::vector<double> FullSpace_BirosGhattas<Real>::solve_linear (
    MatrixType &matrix_A,
    VectorType &right_hand_side,
    VectorType &solution,
    PreconditionerType &preconditioner)
    //const PHiLiP::Parameters::LinearSolverParam & param = )
{
    const bool print_kkt_operator = false;
    const bool print_precond_kkt_operator = false;
    // This will only work with 1 process.
    if (print_kkt_operator) {
        matrix_A.print(right_hand_side);
    }

    if (print_precond_kkt_operator) {
       const int do_full_matrix = (1 == dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
       pcout << "do_full_matrix: " << do_full_matrix << std::endl;
       if (do_full_matrix) {
           dealiiSolverVectorWrappingROL<Real> column_of_kkt_operator, column_of_precond_kkt_operator;
           column_of_kkt_operator.reinit(right_hand_side);
           column_of_precond_kkt_operator.reinit(right_hand_side);
           dealii::FullMatrix<double> fullA(right_hand_side.size());
           for (int i = 0; i < right_hand_side.size(); ++i) {
               pcout << "COLUMN NUMBER: " << i+1 << " OUT OF " << right_hand_side.size() << std::endl;
               auto basis = right_hand_side.basis(i);
               MPI_Barrier(MPI_COMM_WORLD);
               {
                   matrix_A.vmult(column_of_kkt_operator,*basis);
                   preconditioner.vmult(column_of_precond_kkt_operator,column_of_kkt_operator);
               }
               //preconditioner.vmult(column_of_precond_kkt_operator,*basis);
               if (do_full_matrix) {
                   for (int j = 0; j < right_hand_side.size(); ++j) {
                       fullA[j][i] = column_of_precond_kkt_operator[j];
                       //fullA[j][i] = column_of_kkt_operator[j];
                   }
               }
           }
           pcout<<"Dense matrix:"<<std::endl;
           fullA.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
           std::abort();
       }
    }

    enum Solver_types { gmres, fgmres };




    const double rhs_norm = right_hand_side.l2_norm();
    (void) rhs_norm;
    // const double tolerance = rhs_norm*rhs_norm;
    //const double tolerance = std::max(1e-8 * rhs_norm, 1e-14);

    //const double tolerance = std::max(rhs_norm * rhs_norm, 1e-12);
    //const double tolerance = 1e-11;
    //const double tolerance = std::max(1e-3 * rhs_norm, 1e-11);
    // Used for almost all the results:
    const double tolerance = std::min(1e-4, std::max(1e-6 * rhs_norm, 1e-11));

    dealii::SolverControl solver_control(2000, tolerance, true, true);
    solver_control.enable_history_data();

    (void) preconditioner;
    const unsigned int     max_n_tmp_vectors = 1000;
    //Solver_types solver_type = gmres;
    // Used for most results
    Solver_types solver_type = fgmres;
    switch(solver_type) {

        case gmres: {
            const bool     right_preconditioning = true; // default: false
            const bool     use_default_residual = true;//false; // default: true
            const bool     force_re_orthogonalization = false; // default: false
            typedef typename dealii::SolverGMRES<VectorType>::AdditionalData AddiData_GMRES;
            AddiData_GMRES add_data_gmres( max_n_tmp_vectors, right_preconditioning, use_default_residual, force_re_orthogonalization);
            dealii::SolverGMRES<VectorType> solver_gmres(solver_control, add_data_gmres);
            solution = right_hand_side;
            try {
                solver_gmres.solve(matrix_A, solution, right_hand_side
                //, dealii::PreconditionIdentity());
                , preconditioner);
            } catch(...) {
            }
            break;
        }
        case fgmres: {
            typedef typename dealii::SolverFGMRES<VectorType>::AdditionalData AddiData_FGMRES;
            AddiData_FGMRES add_data_fgmres( max_n_tmp_vectors );
            dealii::SolverFGMRES<VectorType> solver_fgmres(solver_control, add_data_fgmres);
            try {
                solver_fgmres.solve(matrix_A, solution, right_hand_side
                //, dealii::PreconditionIdentity());
                , preconditioner);
            } catch(...) {
                solution = right_hand_side;
            }
            break;
        }
        default: break;
    }
    return solver_control.get_history_data();

}

template <class Real>
std::vector<Real> FullSpace_BirosGhattas<Real>::solve_KKT_system(
    Vector<Real> &search_direction,
    Vector<Real> &lag_search_direction,
    const Vector<Real> &design_variables,
    const Vector<Real> &lagrange_mult,
    Objective<Real> &objective,
    Constraint<Real> &equal_constraints)
{
    Real tol = std::sqrt(ROL_EPSILON<Real>());
    const Real one = 1.0;

    /* Form gradient of the Lagrangian. */
    ROL::Ptr<Vector<Real> > objective_gradient = design_variable_cloner_->clone();
    objective.gradient(*objective_gradient, design_variables, tol);
    // Apply adjoint of equal_constraints Jacobian to current Lagrange multiplier.
    ROL::Ptr<Vector<Real> > adjoint_jacobian_lagrange = design_variable_cloner_->clone();
    equal_constraints.applyAdjointJacobian(*adjoint_jacobian_lagrange, lagrange_mult, design_variables, tol);

    /* Form right-hand side of the augmented system. */
    ROL::Ptr<Vector<Real> > rhs1 = design_variable_cloner_->clone();
    ROL::Ptr<Vector<Real> > rhs2 = lagrange_variable_cloner_->clone();
    // rhs1 is the negative gradient of the Lagrangian
    computeLagrangianGradient(*rhs1, design_variables, lagrange_mult, *objective_gradient, equal_constraints);
    rhs1->scale(-one);
    // rhs2 is the contraint value
    equal_constraints.value(*rhs2, design_variables, tol);
    rhs2->scale(-one);

    //pcout << " norm(rhs1) : " << rhs1->norm() << std::endl;
    //pcout << " norm(rhs2) : " << rhs2->norm() << std::endl;

    /* Declare left-hand side of augmented system. */
    ROL::Ptr<Vector<Real> > lhs1 = rhs1->clone();
    ROL::Ptr<Vector<Real> > lhs2 = rhs2->clone();

    ROL::Vector_SimOpt lhs_rol(lhs1, lhs2);
    ROL::Vector_SimOpt rhs_rol(rhs1, rhs2);

    KKT_Operator kkt_operator(
        makePtrFromRef<Objective<Real>>(objective),
        makePtrFromRef<Constraint<Real>>(equal_constraints),
        makePtrFromRef<const Vector<Real>>(design_variables),
        makePtrFromRef<const Vector<Real>>(lagrange_mult));


    std::shared_ptr<BirosGhattasPreconditioner<Real>> kkt_precond =
        BirosGhattasPreconditionerFactory<Real>::create_KKT_preconditioner( parlist_,
                                   objective,
                                   equal_constraints,
                                   design_variables,
                                   lagrange_mult,
                                   secant_);

    dealiiSolverVectorWrappingROL<double> lhs(makePtrFromRef(lhs_rol));
    dealiiSolverVectorWrappingROL<double> rhs(makePtrFromRef(rhs_rol));

    std::vector<double> linear_residuals = solve_linear (kkt_operator, rhs, lhs, *kkt_precond);
    pcout << "Solving the KKT system took "
        << linear_residuals.size() << " iterations "
        << " to achieve a residual of " << linear_residuals.back() << std::endl;

    search_direction.set(*(lhs_rol.get_1()));
    lag_search_direction.set(*(lhs_rol.get_2()));

    //pcout << " norm(lhs1) : " << search_direction.norm() << std::endl;
    //pcout << " norm(lhs2) : " << lag_search_direction.norm() << std::endl;

    // {
    //     kkt_operator.vmult(residual,lhs);
    //     residual.add(-1.0, rhs);
    //     pcout << "linear residual after solve_KKT_system: " << residual.l2_norm() << std::endl;

    //     pcout<<"solution:"<<std::endl;
    //     lhs.print();
    //     pcout<<"right hand side:"<<std::endl;
    //     rhs.print();

    //     // MPI_Barrier(MPI_COMM_WORLD);
    //     // solution.print();
    //     // std::abort();
    // }


    return linear_residuals;

}

template <class Real>
void FullSpace_BirosGhattas<Real>::compute(
    Vector<Real> &search_direction,
    const Vector<Real> &design_variables,
    const Vector<Real> &lagrange_mult,
    Objective<Real> &objective,
    Constraint<Real> &equal_constraints,
    AlgorithmState<Real> &algo_state )
{
    BoundConstraint<Real> bound_constraints;
    bound_constraints.deactivate();
    compute( search_direction, design_variables, lagrange_mult, objective, equal_constraints, bound_constraints, algo_state );
}

template <class Real>
void FullSpace_BirosGhattas<Real>::compute(
    Vector<Real> &search_direction,
    const Vector<Real> &design_variables,
    const Vector<Real> &lagrange_mult,
    Objective<Real> &objective,
    Constraint<Real> &equal_constraints,
    BoundConstraint<Real> &bound_constraints,
    AlgorithmState<Real> &algo_state )
{
    pcout << __PRETTY_FUNCTION__ << std::endl;
    ROL::Ptr<StepState<Real> > step_state = Step<Real>::getState();

    Real tol = std::sqrt(ROL_EPSILON<Real>());
    const Real one = 1.0;

    /* Form gradient of the Lagrangian. */
    ROL::Ptr<Vector<Real> > objective_gradient = design_variable_cloner_->clone();
    objective.gradient(*objective_gradient, design_variables, tol);
    // Apply adjoint of equal_constraints Jacobian to current Lagrange multiplier.
    ROL::Ptr<Vector<Real> > adjoint_jacobian_lagrange = design_variable_cloner_->clone();
    equal_constraints.applyAdjointJacobian(*adjoint_jacobian_lagrange, lagrange_mult, design_variables, tol);

    /* Form right-hand side of the augmented system. */
    ROL::Ptr<Vector<Real> > rhs1 = design_variable_cloner_->clone();
    ROL::Ptr<Vector<Real> > rhs2 = lagrange_variable_cloner_->clone();
    // rhs1 is the negative gradient of the Lagrangian
    ROL::Ptr<Vector<Real> > lagrangian_gradient = step_state->gradientVec->clone();
    computeLagrangianGradient(*lagrangian_gradient, design_variables, lagrange_mult, *objective_gradient, equal_constraints);
    rhs1->set(*lagrangian_gradient);
    rhs1->scale(-one);
    // rhs2 is the contraint value
    equal_constraints.value(*rhs2, design_variables, tol);
    rhs2->scale(-one);

    /* Declare left-hand side of augmented system. */
    ROL::Ptr<Vector<Real> > lhs1 = design_variable_cloner_->clone();
    ROL::Ptr<Vector<Real> > lhs2 = lagrange_variable_cloner_->clone();

    // /* Compute linear solver tolerance. */
    // Real b1norm  = rhs1->norm();
    // Real tol = setTolOSS(lmhtol_*b1norm);

    /* Solve augmented system. */
    //const std::vector<Real> augiters = equal_constraints.solveAugmentedSystem(*lhs1, *lhs2, *rhs1, *rhs2, design_variables, tol);
    pcout
        << "Startingto solve augmented system..."
        << std::endl;
    //const std::vector<Real> kkt_iters = equal_constraints.solveAugmentedSystem(*lhs1, *lhs2, *rhs1, *rhs2, design_variables, tol);
    // {
    //     // Top left block times top vector
    //     objective.hessVec(*lhs1, search_direction, design_variables, tol);
    //     rhs1->axpy(-1.0,*lhs1);
    //     equal_constraints.applyAdjointHessian(*lhs1, lagrange_mult, search_direction, design_variables, tol);
    //     rhs1->axpy(-1.0,*lhs1);

    //     // Top right block times bottom vector
    //     equal_constraints.applyAdjointJacobian(*lhs1, *lagrange_mult_search_direction_, design_variables, tol);
    //     rhs1->axpy(-1.0,*lhs1);

    //     // Bottom left left block times top vector
    //     equal_constraints.applyJacobian(*lhs2, search_direction, design_variables, tol);
    //     rhs2->axpy(-1.0,*lhs2);
    //     pcout << "rhs1->norm() before solve " << rhs1->norm() << std::endl;
    //     pcout << "rhs2->norm() before solve " << rhs2->norm() << std::endl;
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     // Reset rhs
    //     // rhs1 is the negative gradient of the Lagrangian
    //     computeLagrangianGradient(*rhs1, design_variables, lagrange_mult, *objective_gradient, equal_constraints);
    //     rhs1->scale(-one);
    //     // rhs2 is the contraint value
    //     equal_constraints.value(*rhs2, design_variables, tol);
    //     rhs2->scale(-one);
    // }
    const std::vector<Real> kkt_iters = solve_KKT_system(*lhs1, *lhs2, design_variables, lagrange_mult, objective, equal_constraints);

    step_state->SPiter = kkt_iters.size();
    pcout << "Finished solving augmented system..." << std::endl;

    {
        search_direction.set(*lhs1);
        lagrange_mult_search_direction_->set(*lhs2);
    }
    // std::cout
    //     << "search_direction.norm(): "
    //     << search_direction.norm()
    //     << std::endl;
    // std::cout
    //     << "lagrange_mult_search_direction_.norm(): "
    //     << lagrange_mult_search_direction_->norm()
    //     << std::endl;
    // {
    //     // Top left block times top vector
    //     objective.hessVec(*lhs1, search_direction, design_variables, tol);
    //     rhs1->axpy(-1.0,*lhs1);
    //     equal_constraints.applyAdjointHessian(*lhs1, lagrange_mult, search_direction, design_variables, tol);
    //     rhs1->axpy(-1.0,*lhs1);

    //     // Top right block times bottom vector
    //     equal_constraints.applyAdjointJacobian(*lhs1, *lagrange_mult_search_direction_, design_variables, tol);
    //     rhs1->axpy(-1.0,*lhs1);

    //     // Bottom left left block times top vector
    //     equal_constraints.applyJacobian(*lhs2, search_direction, design_variables, tol);
    //     rhs2->axpy(-1.0,*lhs2);

    //     std::cout << "rhs1->norm() after solve " << rhs1->norm() << std::endl;
    //     std::cout << "rhs2->norm() after solve " << rhs2->norm() << std::endl;
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     //std::abort();
    // }

    //#pen_ = parlist.sublist("Step").sublist("Augmented Lagrangian").get("Initial Penalty Parameter",ten);
    /* Create merit function based on augmented Lagrangian */
    const Real penalty_offset = 10;//1e-4;
    penalty_value_ = computeAugmentedLagrangianPenalty(
        search_direction,
        *lagrange_mult_search_direction_,
        design_variables,
        *objective_gradient,
        *(step_state->constraintVec),
        *adjoint_jacobian_lagrange,
        equal_constraints,
        penalty_offset);
    const auto reduced_gradient = (dynamic_cast<Vector_SimOpt<Real>&>(*lagrangian_gradient)).get_2();
    penalty_value_ = std::max(1e-0/reduced_gradient->norm(), 1.0);
    //penalty_value_ = std::max(1e-2/lagrangian_gradient->norm(), 1.0);
    pcout
        << "Finished computeAugmentedLagrangianPenalty..."
        << std::endl;
    AugmentedLagrangian<Real> &augLag = dynamic_cast<AugmentedLagrangian<Real>&>(*merit_function_);

    step_state->nfval = 0;
    step_state->ngrad = 0;
    Real merit_function_value = 0.0;

    bool linesearch_success = false;
    Real fold = 0.0;
    int n_searches = 0;
    while (!linesearch_success) {

        augLag.reset(lagrange_mult, penalty_value_);

        const bool changed_design_variables = true;
        merit_function_->update(design_variables, changed_design_variables, algo_state.iter);
        fold = merit_function_value;
        ROL::Ptr<Vector<Real> > merit_function_gradient = design_variable_cloner_->clone();
        merit_function_->gradient( *merit_function_gradient, design_variables, tol );
        Real directional_derivative_step = merit_function_gradient->dot(search_direction);
        directional_derivative_step += step_state->constraintVec->dot(*lagrange_mult_search_direction_);
        pcout
            << "Penalty value: " << penalty_value_
            << "Directional_derivative_step (Should be negative for descent direction)"
            << directional_derivative_step
            << std::endl;
        //if (directional_derivative_step > 0.0) {
        //    pcout << "Increasing penalty value to obtain descent direction..." << std::endl;
        //    penalty_value_ *= 2.0;
        //    continue;
        //}

        /* Perform line-search */
        merit_function_value = merit_function_->value(design_variables, tol );
        pcout
            << "Performing line search..."
            << " Initial merit function value = " << merit_function_value
            << std::endl;
        lineSearch_->setData(algo_state.gnorm,*merit_function_gradient);

        n_linesearches = 0;
        lineSearch_->run(step_state->searchSize,
                         merit_function_value,
                         n_linesearches,
                         step_state->ngrad,
                         directional_derivative_step,
                         search_direction,
                         design_variables,
                         *merit_function_,
                         bound_constraints);
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

            if (n_searches > 1) {
                pcout << " Linesearch failed, searching other direction " << std::endl;
                search_direction.scale(-1.0);
                penalty_value_ = std::max(1e-0/reduced_gradient->norm(), 1.0);
            }
            if (n_searches > 2) {
                pcout << " Linesearch failed in other direction... ending " << std::endl;
                std::abort();
            }
        }
        lineSearch_->setMaxitUpdate(step_state->searchSize, merit_function_value, fold);
    }

    pcout
        << "End of line search... searchSize is..."
        << step_state->searchSize
        << " and number of function evaluations: "
        << step_state->nfval
        << " Final merit function value = " << merit_function_value
        << std::endl;

    // // Make correction if maximum function evaluations reached
    // if(!acceptLastAlpha_) {
    //     lineSearch_->setMaxitUpdate(step_state->searchSize,fval_,algo_state.value);
    // }
    // Compute scaled descent direction
    lagrange_mult_search_direction_->scale(step_state->searchSize);
    search_direction.scale(step_state->searchSize);
    if ( bound_constraints.isActivated() ) {
        search_direction.plus(design_variables);
        bound_constraints.project(search_direction);
        search_direction.axpy(static_cast<Real>(-1),design_variables);
    }
    pcout
        << "End of compute..."
        << std::endl;

}

template <class Real>
void FullSpace_BirosGhattas<Real>::update(
    Vector<Real> &design_variables,
    Vector<Real> &lagrange_mult,
    const Vector<Real> &search_direction,
    Objective<Real> &objective,
    Constraint<Real> &equal_constraints,
    AlgorithmState<Real> &algo_state )
{
    pcout << __PRETTY_FUNCTION__ << std::endl;
    BoundConstraint<Real> bound_constraints;
    bound_constraints.deactivate();
    update( design_variables, lagrange_mult, search_direction, objective, equal_constraints, bound_constraints, algo_state );
}

template <class Real>
void FullSpace_BirosGhattas<Real>::update(
    Vector<Real> &design_variables,
    Vector<Real> &lagrange_mult,
    const Vector<Real> &search_direction,
    Objective<Real> &objective,
    Constraint<Real> &equal_constraints,
    BoundConstraint< Real > &bound_constraints,
    AlgorithmState<Real> &algo_state )
{
    pcout << __PRETTY_FUNCTION__ << std::endl;
    Real tol = std::sqrt(ROL_EPSILON<Real>());
    (void) bound_constraints;
    design_variables.plus(search_direction);
    lagrange_mult.plus(*lagrange_mult_search_direction_);

    // Update StepState
    ROL::Ptr<StepState<Real> > step_state = Step<Real>::getState();
    step_state->descentVec  = design_variables.clone();
    objective.gradient(*(step_state->gradientVec), design_variables, tol);
    equal_constraints.value(*(step_state->constraintVec), design_variables, tol);


    ROL::Ptr<Vector<Real> > lagrangian_gradient = step_state->gradientVec->clone();
    computeLagrangianGradient(*lagrangian_gradient, design_variables, lagrange_mult, *(step_state->gradientVec), equal_constraints);

    algo_state.nfval += step_state->nfval;
    algo_state.ngrad += step_state->ngrad;


    algo_state.value = objective.value(design_variables, tol);
    algo_state.gnorm = lagrangian_gradient->norm();
    algo_state.cnorm = step_state->constraintVec->norm();
    search_sim_norm = ((dynamic_cast<const Vector_SimOpt<Real>&>(search_direction)).get_1())->norm();
    search_ctl_norm = ((dynamic_cast<const Vector_SimOpt<Real>&>(search_direction)).get_2())->norm();
    search_adj_norm = lagrange_mult_search_direction_->norm();

    algo_state.snorm = std::pow(search_sim_norm,2) +
                       std::pow(search_ctl_norm,2) +
                       std::pow(search_adj_norm,2);
    algo_state.snorm = std::sqrt(algo_state.snorm);

    auto &flow_constraint = (dynamic_cast<PHiLiP::FlowConstraints<PHILIP_DIM>&>(equal_constraints));
    //flow_constraint.update(
    //    *((dynamic_cast<const Vector_SimOpt<Real>&>(design_variables)).get_1()),
    //    *((dynamic_cast<const Vector_SimOpt<Real>&>(design_variables)).get_2()),
    //    true,
    //    algo_state.iter); // Prints out the solution.
    //flow_constraint.flow_CFL_ = 1.0/std::pow(algo_state.cnorm, 0.5);
    //flow_constraint.flow_CFL_ = 10 + 1.0/std::pow(algo_state.cnorm, 2.0);
    //flow_constraint.flow_CFL_ = 1.0/std::pow(algo_state.cnorm, 2.0);
    flow_constraint.flow_CFL_ = -10000*std::max(1.0, 1.0/std::pow(algo_state.cnorm, 2.00));
    //flow_constraint.flow_CFL_ = 0.0;
    //flow_constraint.flow_CFL_ = 1e-6;

    algo_state.iterateVec->set(design_variables);
    algo_state.lagmultVec->set(lagrange_mult);
    algo_state.iter++;

    const auto current_reduced_gradient = (dynamic_cast<Vector_SimOpt<Real>&>(*lagrangian_gradient)).get_2();
    const auto control_search_direction = (dynamic_cast<const Vector_SimOpt<Real>&>(search_direction)).get_2();
    const double search_norm = control_search_direction->norm();
    // Updates if (search.dot(gradDiff) > ROL_EPSILON<Real>()*snorm*snorm) {
    const double modified_search_norm = std::sqrt(std::pow(search_norm,2) * 1e-8 / ROL_EPSILON<Real>());
    if (n_linesearches <= 3) {
        secant_->updateStorage(design_variables, *current_reduced_gradient, *previous_reduced_gradient_, *control_search_direction, modified_search_norm, algo_state.iter+1);
    }
    //secant_->updateStorage(design_variables, *current_reduced_gradient, *previous_reduced_gradient_, *control_search_direction, search_norm, algo_state.iter);
    previous_reduced_gradient_ = current_reduced_gradient;

    MPI_Barrier(MPI_COMM_WORLD);
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        pcout
        << " algo_state.iter: "  <<   algo_state.iter << std::endl
        << " penalty_value_: "<< penalty_value_ << std::endl
        << " step_state->searchSize: " << step_state->searchSize << std::endl
        << " algo_state.value: "  <<   algo_state.value << std::endl
        << " algo_state.gnorm: "  <<   algo_state.gnorm << std::endl
        << " algo_state.cnorm: "  <<   algo_state.cnorm << std::endl
        << " algo_state.snorm: "  <<   algo_state.snorm << std::endl
        << " n_vmult_total: "  <<   n_vmult << std::endl
        << "  dRdW_form " << dRdW_form << std::endl
        << "  dRdW_mult " << dRdW_mult << std::endl
        << "  dRdX_mult " << dRdX_mult << std::endl
        << "  d2R_mult  " << d2R_mult  << std::endl
        ;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

template <class Real>
std::string FullSpace_BirosGhattas<Real>::printHeader( void ) const
{
  //head.erase(std::remove(head.end()-3,head.end(),'\n'), head.end());
  std::stringstream hist;
  // hist.write(head.c_str(),head.length());
  // hist << std::setw(18) << std::left << "ls_#fval";
  // hist << std::setw(18) << std::left << "ls_#grad";
  hist << std::setw(18) << std::left << "Iteration";
  hist << std::setw(18) << std::left << "Func. val.";
  hist << std::setw(18) << std::left << "||Lagr. grad.||";
  hist << std::setw(18) << std::left << "||Constraint||";
  hist << std::setw(18) << std::left << "||Search dir||";
  hist << std::setw(18) << std::left << "search_ctl_norm";
  hist << std::setw(18) << std::left << "search_sim_norm";
  hist << std::setw(18) << std::left << "search_adj_norm";
  hist << std::setw(18) << std::left << "n_kkt_iter";
  hist << std::setw(18) << std::left << "n_linesearches";
  hist << std::setw(18) << std::left << "n_grad";
  hist << std::setw(18) << std::left << "n_vmult";
  hist << std::setw(18) << std::left << "dRdW_form";
  hist << std::setw(18) << std::left << "dRdW_mult";
  hist << std::setw(18) << std::left << "dRdX_mult";
  hist << std::setw(18) << std::left << "d2R_mult";
  hist << "\n";
  return hist.str();
}

template <class Real>
std::string FullSpace_BirosGhattas<Real>::printName( void ) const
{
  std::stringstream hist;
  //hist << name;
  hist << "********************************************************" << std::endl;
  hist << "Biros and Ghattas Full-space method...";
  hist << "with " + preconditioner_name_ + " preconditioner" << std::endl;
  const auto &design_variable_simopt = dynamic_cast<const Vector_SimOpt<Real>&>(*design_variable_cloner_);
  hist << "Using "
      << design_variable_cloner_->dimension() << " design variables: "
      << design_variable_simopt.get_1()->dimension() << " simulation variables and "
      << design_variable_simopt.get_2()->dimension() << " control variables."
      << std::endl;

  hist << "Line Search: " << lineSearchName_;
  hist << " satisfying " << ECurvatureConditionToString(econd_) << "\n";
  hist << "********************************************************" << std::endl;
  return hist.str();
}

template <class Real>
std::string FullSpace_BirosGhattas<Real>::print( AlgorithmState<Real> & algo_state, bool print_header) const
{
  const ROL::Ptr<const StepState<Real> > step_state = Step<Real>::getStepState();
  // desc.erase(std::remove(desc.end()-3,desc.end(),'\n'), desc.end());
  // size_t pos = desc.find(name);
  // if ( pos != std::string::npos ) {
  //   desc.erase(pos, name.length());
  // }

  std::stringstream hist;
  if ( algo_state.iter == 0 ) {
    hist << printName();
  }
  (void) print_header;
  if ( algo_state.iter == 0 ) {
  //if ( print_header ) {
    hist << printHeader();
  }
  //hist << desc;
  if ( algo_state.iter == 0 ) {
    //hist << "\n";
  }
  else {
    hist << std::setw(18) << std::left << algo_state.iter;
    hist << std::setw(18) << std::left << algo_state.value;
    hist << std::setw(18) << std::left << algo_state.gnorm;
    hist << std::setw(18) << std::left << algo_state.cnorm;
    hist << std::setw(18) << std::left << algo_state.snorm;
    hist << std::setw(18) << std::left << search_ctl_norm;
    hist << std::setw(18) << std::left << search_sim_norm;
    hist << std::setw(18) << std::left << search_adj_norm;
    hist << std::setw(18) << std::left << step_state->SPiter;
    hist << std::setw(18) << std::left << step_state->nfval;
    hist << std::setw(18) << std::left << step_state->ngrad;
    hist << std::setw(18) << std::left << n_vmult;
    hist << std::setw(18) << std::left << dRdW_form;
    hist << std::setw(18) << std::left << dRdW_mult;
    hist << std::setw(18) << std::left << dRdX_mult;
    hist << std::setw(18) << std::left << d2R_mult;
    hist << std::endl;
  }
  return hist.str();
}

template class FullSpace_BirosGhattas<double>;

} // ROL namespace

