#ifndef __SKEWSYMMETRIC_CURVED_PERIODIC_GRID_H__
#define __SKEWSYMMETRIC_CURVED_PERIODIC_GRID_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a Gaussian bump grid with an associated nonlinear manifold.
/** Numbers used are the ones from the High-Order Prediction Workshop (HOPW)
 */
template<int dim, typename TriangulationType>
void skewsymmetric_curved_grid(
    TriangulationType &grid,
    const unsigned int n_subdivisions);

/// Gaussian bump manifold.
template<int dim,int spacedim,int chartdim>
class SkewsymmetricCurvedGridManifold: public dealii::ChartManifold<dim,spacedim,chartdim> {
protected:
    static constexpr double pi = atan(1) * 4.0; ///< PI.
public:
    /// Constructor.
    SkewsymmetricCurvedGridManifold()
    : dealii::ChartManifold<dim,spacedim,chartdim>()
    {};
    ///The mapping for the 2D grid follows from Hennemann, Sebastian, et al. "A provably entropy stable subcell shock capturing approach for high order split form DG for the compressible Euler equations." Journal of Computational Physics 426 (2021): 109935. Eq.(59).
    template<typename real>
    dealii::Point<spacedim,real> mapping(const dealii::Point<chartdim,real> &chart_point) const; ///< Templated mapping from square to the nonsymmetric warping.

    virtual dealii::Point<chartdim> pull_back(const dealii::Point<spacedim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<spacedim> push_forward(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,chartdim,spacedim> push_forward_gradient(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<dim,spacedim> > clone() const override; ///< See dealii::Manifold.

};

} // namespace Grids
} // namespace PHiLiP
#endif

