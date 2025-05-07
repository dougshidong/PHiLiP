#ifndef __POD_GALERKIN_RUNGE_KUTTA_ODESOLVER__
#define __POD_GALERKIN_RUNGE_KUTTA_ODESOLVER__

#include "JFNK_solver/JFNK_solver.h"
#include "dg/dg_base.hpp"
#include "runge_kutta_base.h"
#include "runge_kutta_methods/rk_tableau_base.h"
#include "relaxation_runge_kutta/empty_RRK_base.h"


namespace PHiLiP {
namespace ODE {

/*  Reference for Galerkin Runge-Kutta see equations 3.9 through 3.11 in
 *  Carlberg, K., Barone, M., & Antil, H. (2017). Galerkin v. least-squares Petrov–Galerkin projection in nonlinear model
 *  reduction. Journal of Computational Physics, 330, 693–734. https://doi.org/10.1016/j.jcp.2016.10.033
 *
 *  This class preforms the Runge Kutta Method for a Galerkin POD Reduced order model.
 *  This assumes that the test basis, W, is equal to the trail basis, i.e. W=V.
 *  This then preforms the following:  V^T*M*V*du/dt = V^T*R
 *
 *  In this class, we are solving for the solution in the reduced order space and then project it back into the
 *  Full order space
 */
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PODGalerkinRungeKuttaODESolver: public RungeKuttaBase <dim, real, n_rk_stages, MeshType>
{
public:
    PODGalerkinRungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input,
            std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod); ///< Constructor.

    /// Destructor
    virtual ~PODGalerkinRungeKuttaODESolver() override {};

    void allocate_runge_kutta_system () override;

    void calculate_stage_solution (int istage, real dt, const bool pseudotime) override;

    void calculate_stage_derivative (int istage, real dt) override;

    void sum_stages (real dt, const bool pseudotime) override;

    void apply_limiter () override;

    real adjust_time_step (real dt) override;

    /// Generate test basis
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(const Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &pod_basis);

    /// Generate reduced LHS
    std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(const Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &test_basis);

protected:
    /// Stores Butcher tableau a and b, which specify the RK method
    std::shared_ptr<RKTableauBase<dim,real,MeshType>> butcher_tableau;
    
    /// Reduced Space sized Runge Kutta Stages
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> reduced_rk_stage;

    /// POD Basis
    Epetra_CrsMatrix epetra_pod_basis;

    /// System Matrix (Unsure if needed, do some testing)
    Epetra_CrsMatrix epetra_system_matrix; 

    /// Pointer to Epetra Matrix for Test Basis
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis;

    /// Pointer to Epetra Matrix for LHS
    std::shared_ptr<Epetra_CrsMatrix> epetra_reduced_lhs;

    /// dealII indexset for FO solution
    dealii::IndexSet solution_index;

    /// dealII indexset for RO solution
    dealii::IndexSet reduced_index;

private:
    /// Function to multiply a dealii vector by an Epetra Matrix
    int multiply(Epetra_CrsMatrix &epetra_matrix,
                 dealii::LinearAlgebra::distributed::Vector<double> &input_dealii_vector,
                 dealii::LinearAlgebra::distributed::Vector<double> &output_dealii_vector,
                 const dealii::IndexSet &index_set,
                 const bool transpose);

    /// Function to convert a epetra_vector to dealii
    void epetra_to_dealii(Epetra_Vector &epetra_vector,
                          dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                          const dealii::IndexSet &index_set);
};

} // ODE namespace
} // PHiLiP namespace

#endif

