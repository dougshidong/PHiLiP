#ifndef __GAUSSIAN_BUMP_H__
#define __GAUSSIAN_BUMP_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a Gaussian bump grid with an associated nonlinear manifold.
void gaussian_bump(
    dealii::parallel::distributed::Triangulation<2> &grid,
    const std::vector<unsigned int> n_subdivisions,
    const double channel_length,
    const double y_height);

/// Gaussian bump manifold.
class BumpManifold: public dealii::ChartManifold<2,2,2> {
protected:
    static constexpr double y_height = 0.8;
    static constexpr double bump_height = 0.0625; // High-Order Prediction Workshop
    static constexpr double coeff_expx = -25; // High-Order Prediction Workshop
    static constexpr double coeff_expy = -30;
public:
    template<typename real>
    dealii::Point<2,real> mapping(const dealii::Point<2,real> &chart_point) const;

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
