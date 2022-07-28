#ifndef __JFNK_SOLVER__
#define __JFNK_SOLVER__

#include "dg/dg.h"
#include <deal.II/lac/solver_gmres.h>
#include "jacobian_vector_product.h"

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
    /* Reinitializes Jv for next step
     * Consists of outer loop (Newton iteration)
     * Calls solver.solve(...) for inner loop (GMRES iterations)
     * TO DO: how to return ? use pointer maybe ?
     * for now, access public current_solution_estimate
     */
    void solve(real dt,
               dealii::LinearAlgebra::distributed::Vector<double> previous_step_solution_input);

    /// current estimate for the solution
    // COULD USE A POINTER HERE
    dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate;
    
protected:

    /// pointer to dg
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;
    
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

    /// Jacobian-vector product utilities
    JacobianVectorProduct<dim,real,MeshType> Jv;

    /// Solver control object
    dealii::SolverControl solver_control;
    
    /// Solver
    dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver_GMRES;
    
    /// solution at previous timestep
    // COULD USE A POINTER HERE
    dealii::LinearAlgebra::distributed::Vector<double> previous_step_solution;
    
    /// update to solution during Newton iterations
    dealii::LinearAlgebra::distributed::Vector<double> solution_update_newton;
};

}
}
#endif
