#ifndef __MESH_JACOBIAN_DEVIATION_H__
#define __MESH_JACOBIAN_DEVIATION_H__

#include "functional.h"

namespace PHiLiP {

template<int dim, int nstate, typename real>
class MeshJacobianDeviation : public Functional<dim,nstate,real> 
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>; ///< Alias for dealii's parallel distributed vector.

public:
    MeshJacobianDeviation(
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = false,
        const bool uses_solution_gradient = false);

    ~MeshJacobianDeviation() {}

    real evaluate_functional(
        const bool compute_dIdw = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

private:
    void store_initial_rmsh();

    template<typename real2>
    real2 evaluate_cell_volume_term(
        const std::vector<real2> &coords_coeff,
        const real initial_cell_rmsh,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const;

    template<typename real2>
    real2 compute_cell_rmsh(
        const std::vector<real2> &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const;

    dealii::Vector<real> initial_rmsh;

    const real mesh_weight;

};
} // PHiLiP namespace
#endif
