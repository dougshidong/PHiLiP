#ifndef __GAUSSIAN_BUMP_H__
#define __GAUSSIAN_BUMP_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a Gaussian bump grid with an associated nonlinear manifold.
/** Numbers used are the ones from the High-Order Prediction Workshop (HOPW)
 */
void gaussian_bump(
    dealii::parallel::distributed::Triangulation<2> &grid,
    const std::vector<unsigned int> n_subdivisions,
    const double channel_length,
    const double channel_heigh,
    const double bump_height = 0.0625);

/// Gaussian bump manifold.
class BumpManifold: public dealii::ChartManifold<2,2,2> {
protected:
    double channel_height; ///< Channel height.
    double bump_height; ///< Bump height.
    static constexpr double coeff_expx = -25; ///< Bump exponent (variance).
    static constexpr double coeff_expy = -30; ///< Bump propagation in the domain.
public:
    BumpManifold(const double channel_height, const double bump_height)
    : dealii::ChartManifold<2,2,2>()
    , channel_height(channel_height)
    , bump_height(bump_height)
    {};
    template<typename real>
    dealii::Point<2,real> mapping(const dealii::Point<2,real> &chart_point) const; ///< Templated mapping from square to the bump.

    virtual dealii::Point<2> pull_back(const dealii::Point<2> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<2> push_forward(const dealii::Point<2> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,2,2> push_forward_gradient(const dealii::Point<2> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<2,2> > clone() const override; ///< See dealii::Manifold.

    /// Used to deform a hypercube Triangulation into the Gaussian bump.
    static dealii::Point<2> warp (const dealii::Point<2> &p);

};

} // namespace Grids
} // namespace PHiLiP
#endif
