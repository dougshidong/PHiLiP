#ifndef __WAVY_PERIODIC_GRID_H__
#define __WAVY_PERIODIC_GRID_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a Gaussian bump grid with an associated nonlinear manifold.
/** Numbers used are the ones from the High-Order Prediction Workshop (HOPW)
 */
template<int dim, typename TriangulationType>
void wavy_grid_Abe_2015(
    TriangulationType &grid,
    const std::vector<unsigned int> n_subdivisions);

/// Gaussian bump manifold.
template<int dim,int spacedim,int chartdim>
class WavyManifold: public dealii::ChartManifold<dim,spacedim,chartdim> {
protected:
    static constexpr double pi = atan(1) * 4.0; ///< PI.

    const double L0 = 10; ///< Hypercube's width.
    const double amplitude = 0.4; ///< Amplitude of the waves.
    const double n = 4; ///< Number of sine bumps in 1 dimension.
    const std::vector<unsigned int> n_subdivisions; ///< Number of cells in each directions.
    std::array<double,dim> dx; ///< Cell spacing.
public:
    /// Constructor.
    WavyManifold( const std::vector<unsigned int> n_cells )
    : dealii::ChartManifold<dim,spacedim,chartdim>()
    , n_subdivisions(n_cells)
    {
        for (int d=0; d<dim; ++d) {
            dx[d] = L0/n_subdivisions[d];
        }
    };

    template<typename real>
    dealii::Point<spacedim,real> mapping(const dealii::Point<chartdim,real> &chart_point) const; ///< Templated mapping from square to the bump.

    virtual dealii::Point<chartdim> pull_back(const dealii::Point<spacedim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<spacedim> push_forward(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,chartdim,spacedim> push_forward_gradient(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<dim,spacedim> > clone() const override; ///< See dealii::Manifold.

};

} // namespace Grids
} // namespace PHiLiP
#endif


