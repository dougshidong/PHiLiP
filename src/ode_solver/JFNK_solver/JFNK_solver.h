#ifndef __JFNK_SOLVER__
#define __JFNK_SOLVER__

#include "dg/dg.h"
#include <deal.II/lac/solver_gmres.h>
#include "jacobian_vector_product.h"
#include <deal.II/base/conditional_ostream.h>

namespace PHiLiP {
namespace ODE{

template <int dim, typename real, typename MeshType>
class JFNKSolver{
public:

    /// Constructor
    JFNKSolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);
    
    /// Destructor
    ~JFNKSolver() {};

    /// Solve an implicit Euler step according to Jacobian-free Newton-Krylov method
    /** Reinitializes jacobian_vector_product for next step
     * Consists of outer loop (Newton iteration)
     * Calls solver.solve(...) for inner loop (GMRES iterations)
     * TO DO: how to return ? use pointer maybe ?
     * for now, access public current_solution_estimate
     */
    void solve(real dt,
               dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution);

    /// current estimate for the solution
    // COULD USE A POINTER HERE
    dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate;
    
protected:

    /// pointer to dg
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    /// output on processor 0
    dealii::ConditionalOStream pcout;

    /// pointer to input parameters
    const Parameters::AllParameters *const all_parameters;
    
    /// Input linear solver parameters
    const Parameters::LinearSolverParam linear_param;

    /// small number for finite difference
    const double epsilon_jacobian;

    /// tolerance for Newton iterations 
    const double epsilon_Newton;

    /// tolerance for GMRES 
    const double epsilon_GMRES;

    /// maximum number of temporary vectors for GMRES
    const int max_num_temp_vectors;

    /// maximum number of GMRES iterations
    const int max_GMRES_iter;

    /// maximum number of Newton iterations
    const int max_Newton_iter;

    /// ODE output (true indicates verbose output)
    const bool do_output;

    /// Jacobian-vector product utilities
    JacobianVectorProduct<dim,real,MeshType> jacobian_vector_product;

    /// Solver control object
    dealii::SolverControl solver_control;
    
    /// Solver
    dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver_GMRES;
    
    /// update to solution during Newton iterations
    dealii::LinearAlgebra::distributed::Vector<double> solution_update_newton;
};

}
}
#endif
