#ifndef __JACOBIAN_VECTOR_PRODUCT__
#define __JACOBIAN_VECTOR_PRODUCT__

#include "dg/dg.h"

namespace PHiLiP {
namespace ODE{

/// Class to store information for the JFNK solver, and interact with dg
template <int dim, typename real, typename MeshType>
class JacobianVectorProduct{
public:
    /// Constructor
    JacobianVectorProduct(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    ///Destructor
    ~JacobianVectorProduct() {};

    /// Reinitializes the stored data for a new timestep.
    void reinit_for_next_timestep(const double dt_input,
                const double fd_perturbation_input,
                const dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution_input);

    /// Reinitializes the stored data for the next Newton iteration.
    void reinit_for_next_Newton_iter(const dealii::LinearAlgebra::distributed::Vector<double> &current_solution_estimate_input);

    /// Returns the product of the Jacobian with vector w, computed with a matrix-free finite difference approximation
    /** Write the results into destination. */
    void vmult (dealii::LinearAlgebra::distributed::Vector<double> &destination,
                const dealii::LinearAlgebra::distributed::Vector<double> &w) const;
    
    /// Unsteady residual = dw/dt - R
    dealii::LinearAlgebra::distributed::Vector<double> compute_unsteady_residual(const dealii::LinearAlgebra::distributed::Vector<double> &solution,
            const bool do_negate = false) const;
protected:

    /// pointer to dg
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    /// timestep size for implicit Euler step
    double dt;
    
    /// small number for finite difference
    double fd_perturbation;
    
    /// solution at previous timestep
    dealii::LinearAlgebra::distributed::Vector<double> previous_step_solution;
    
    /// pointer to current estimate for the solution
    std::unique_ptr<dealii::LinearAlgebra::distributed::Vector<double>> current_solution_estimate;
    
    /// residual of current estimate for the solution
    dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate_residual;
    
    /// Compute residual from dg,  R(w) = IMM * RHS where RHS is evaluated using solution=w, and store in destination
    void compute_dg_residual(dealii::LinearAlgebra::distributed::Vector<double> &destination,
            const dealii::LinearAlgebra::distributed::Vector<double> &w) const;
};

}
}
#endif
