#ifndef __JFNK_SOLVER__
#define __JFNK_SOLVER__

#include "dg/dg.h"
#include <deal.II/lac/solver_gmres.h>
#include "jacobian_vector_product.h"
#include <deal.II/base/conditional_ostream.h>

namespace PHiLiP {
namespace ODE{

/// Implicit solver for an implicit Euler step using the Jacobian-free Newton-Krylov method
template <int dim, typename real, typename MeshType>
class JFNKSolver{
public:

    /// Constructor
    JFNKSolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);
    
    /// Destructor
    ~JFNKSolver() {};

    /// Solve an implicit Euler step according to Jacobian-free Newton-Krylov method
    /** See for example Knoll & Keyes 2004 "Jacobian-free Newton-Krylov methods; a survey of approaches and applications
     * Solves J(wk) * dwk = -R*(wk), where R*= dw/dt - R is unsteady residual and J is its Jacobian
     * Consists of outer loop (Newton iteration)
     * Calls solver_GMRES.solve(...) for inner loop (GMRES iterations)
     */
    void solve(real dt,
               dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution);

    /// current estimate for the solution
    dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate;
    
protected:

    /// output on processor 0
    dealii::ConditionalOStream pcout;

    /// pointer to input parameters
    const Parameters::AllParameters *const all_parameters;
    
    /// Input linear solver parameters
    const Parameters::LinearSolverParam linear_param;

    /// small pertubation for finite difference
    const double perturbation_magnitude;

    /// tolerance for Newton iterations 
    const double epsilon_Newton;

    /// tolerance for GMRES iterations
    const double epsilon_GMRES;

    /// maximum number of temporary vectors for GMRES - GMRES is restarted after this many iterations
    const int max_num_temp_vectors;

    /// maximum number of GMRES iterations
    const int max_GMRES_iter;

    /// maximum number of Newton iterations
    const int max_Newton_iter;

    /// linear solve output (true indicates verbose output)
    const bool do_output;

    /// Jacobian-vector product utilities
    JacobianVectorProduct<dim,real,MeshType> jacobian_vector_product;

    /// Solver control object
    dealii::SolverControl solver_control;
    
    /// GMRES solver
    dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver_GMRES;
    
    /// Update to solution during Newton iterations
    dealii::LinearAlgebra::distributed::Vector<double> solution_update_newton;
};

}
}
#endif
