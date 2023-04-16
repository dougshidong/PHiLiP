#ifndef __CELL_DISTORTION_FUNCTIONAL_H__ 
#define __CELL_DISTORTION_FUNCTIONAL_H__ 

#include "functional.h"

namespace PHiLiP {

/// Class to compute the weight of mesh to prevent distortion of mesh during optimization.
/**
 *  Comutes \f[\mathcal{J}(\mathbf{u},\mathbf{x}) = \mu \sum_k \frac{1}{\Omega_k^2} \f]
 */
template <int dim, int nstate, typename real>
class CellDistortion : public Functional<dim, nstate, real> // using default MeshType
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

public: 
    
    /// Constructor
    CellDistortion( 
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = false,
        const bool uses_solution_gradient = false);

    /// Destructor
    ~CellDistortion(){}

    /// Templated function to evaluate a cell's volume weight.
    template <typename real2>
    real2 evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real2> &/*physics*/,
        const std::vector< real2 > &/*soln_coeff*/,
        const dealii::FESystem<dim> &/*fe_solution*/,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const
{
    const unsigned int n_vol_quad_pts = volume_quadrature.size();
    const unsigned int n_metric_dofs_cell = coords_coeff.size();

    real2 cell_distortion_measure = 0.0;
    real2 cell_volume = 0.0;
    for (unsigned int iquad=0; iquad<n_vol_quad_pts; ++iquad) {

        const dealii::Point<dim,double> &ref_point = volume_quadrature.point(iquad);
        const double quad_weight = volume_quadrature.weight(iquad);

        std::array< dealii::Tensor<1,dim,real2>, dim > coord_grad; // Tensor initialize with zeros
        dealii::Tensor<2,dim,real2> metric_jacobian;

        for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) {
            const unsigned int axis = fe_metric.system_to_component_index(idof).first;
            coord_grad[axis] += coords_coeff[idof] * fe_metric.shape_grad (idof, ref_point);
        }
        real2 jacobian_frobenius_norm_squared = 0.0;
        for (int row=0;row<dim;++row) {
            for (int col=0;col<dim;++col) {
                metric_jacobian[row][col] = coord_grad[row][col];
                jacobian_frobenius_norm_squared += pow(coord_grad[row][col], 2);
            }
        }
        const real2 jacobian_determinant = dealii::determinant(metric_jacobian);

        real2 integrand_distortion = jacobian_frobenius_norm_squared/pow(jacobian_determinant, 2/dim);
        integrand_distortion = pow(integrand_distortion, mesh_volume_power);
        cell_distortion_measure += integrand_distortion * jacobian_determinant * quad_weight;
        cell_volume += 1.0 * jacobian_determinant * quad_weight;
    } // quad loop ends

    real2 cell_volume_obj_func = mesh_weight_factor * scaling_w_elements * cell_distortion_measure/cell_volume;

    if(dim == 1)
    {
        cell_volume_obj_func = mesh_weight_factor*scaling_w_elements*(scaling_w_elements/cell_volume - 1.0);
    }

    return cell_volume_obj_func;
}
    
    /// Corresponding real function to evaluate a cell's volume functional. Overrides function in Functional.
    real evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real> &physics,
        const std::vector< real > &soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const override
    {
        return evaluate_volume_cell_functional<real>(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
    }
    
    /// Corresponding FadFadType function to evaluate a cell's volume functional. Overrides function in Functional.
    FadFadType evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,FadFadType> &physics_fad_fad,
        const std::vector< FadFadType > &soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< FadFadType > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const override
        {
            return evaluate_volume_cell_functional<FadFadType>(physics_fad_fad, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
        }

private:
    /// Stores the weight of mesh to be used to evaluate this function. 
    /** It is parameter \f[\mu \f] in \f[\mathcal{J}(\mathbf{u},\mathbf{x}) = \mu \sum_k \frac{1}{\Omega_k^2} \f] 
     */
    const real mesh_weight_factor;

    /// Stores power of mesh cell volume
    /** It is parameter \f[\gamma \f] in \f[\mathcal{J}(\mathbf{u},\mathbf{x}) = \mu \sum_k \Omega_k^\gamma \f] 
     */
    const int mesh_volume_power;

    /// Scales mesh weight with number of elements.
    const real scaling_w_elements;
}; // class ends

} // namespace PHiLiP
#endif
