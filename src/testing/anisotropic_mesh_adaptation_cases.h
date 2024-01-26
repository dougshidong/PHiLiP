#ifndef __ANISOTROPICMESHADAPTATIONCASES_H__ 
#define __ANISOTROPICMESHADAPTATIONCASES_H__ 

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Test to check anisotropic mesh adaptation.
template <int dim, int nstate>
class AnisotropicMeshAdaptationCases : public TestsBase
{
public:
    /// Constructor
    AnisotropicMeshAdaptationCases(const Parameters::AllParameters *const parameters_input,
                                       const dealii::ParameterHandler &parameter_handler_input);
    
    /// Parameter handler.
    const dealii::ParameterHandler &parameter_handler;

    /// Runs the test related to anisotropic mesh adaptation.
    int run_test() const;

    /// Checks PHiLiP::FEValuesShapeHessian for MappingFEField with dealii's shape hessian for MappingQGeneric.
    void verify_fe_values_shape_hessian(const DGBase<dim, double> &dg) const;
    
    /// Evaluates \f[ J_exact - J(u_h) \f].
    double evaluate_functional_error(std::shared_ptr<DGBase<dim,double>> dg) const;
    
    /// Evaluates \f[ J_exact - J(u_h) \f].
    double evaluate_abs_dwr_error(std::shared_ptr<DGBase<dim,double>> dg) const;
    
    /// Outputs vtk files with primal and adjoint solutions.
    double output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg, const int count_val) const;
    
    /// Evaluates l2 norm of solution error.
    double evaluate_enthalpy_error(std::shared_ptr<DGBase<dim,double>> dg) const;
    
    /// Project surface nodes to cylinder.
    void project_surface_nodes_on_cylinder(std::shared_ptr<DGBase<dim,double>> dg) const;

    void evaluate_regularization_matrix(
        dealii::TrilinosWrappers::SparseMatrix &regularization_matrix,
        std::shared_ptr<DGBase<dim,double>> dg) const;

    void increase_grid_degree_and_interpolate_solution(std::shared_ptr<DGBase<dim,double>> dg) const;
    
    void refine_mesh_and_interpolate_solution(std::shared_ptr<DGBase<dim,double>> dg) const;

    void test_numerical_flux(std::shared_ptr<DGBase<dim,double>> dg) const;

}; 

} // Tests namespace
} // PHiLiP namespace

#endif

