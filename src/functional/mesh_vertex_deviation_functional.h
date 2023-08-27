#ifndef __MESH_VERTEX_DEVIATION_H__
#define __MESH_VERTEX_DEVIATION_H__

#include "functional.h"

namespace PHiLiP {

template<int dim, int nstate, typename real>
class MeshVertexDeviation : public Functional<dim,nstate,real> 
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>; ///< Alias for dealii's parallel distributed vector.

public:
    MeshVertexDeviation(
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = false,
        const bool uses_solution_gradient = false);

    ~MeshVertexDeviation() {}

    real evaluate_functional(
        const bool compute_dIdw = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

private:
    template<typename real2>
    real2 evaluate_cell_volume_term(
        const std::vector<real2> &coords_coeff,
        const std::vector<real> &target_values,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const;

    VectorType initial_vol_nodes;
    const double mesh_weight;

};
} // PHiLiP namespace
#endif
