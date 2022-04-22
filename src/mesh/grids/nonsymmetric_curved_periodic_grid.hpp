#ifndef __NONSYMMETRIC_CURVED_PERIODIC_GRID_H__
#define __NONSYMMETRIC_CURVED_PERIODIC_GRID_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a nonsymmetric grid with an associated nonlinear manifold.
/*The mapping for the 2D and 3D grids follows from Cicchino, Alexander, et al. "Provably Stable Flux Reconstruction High-Order Methods on Curvilinear Elements.". The 2D is Eq. (66), and the 3D is Eq. (64).
*/
template<int dim, typename TriangulationType>
void nonsymmetric_curved_grid(
    TriangulationType &grid,
    const unsigned int n_subdivisions);

/// Nonsymmetric manifold.
template<int dim,int spacedim,int chartdim>
class NonsymmetricCurvedGridManifold: public dealii::ChartManifold<dim,spacedim,chartdim> {
protected:
    static constexpr double pi = atan(1) * 4.0; ///< PI.
public:
    /// Constructor.
    NonsymmetricCurvedGridManifold()
    : dealii::ChartManifold<dim,spacedim,chartdim>()
    {};
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

