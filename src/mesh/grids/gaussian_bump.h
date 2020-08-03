#ifndef __GAUSSIAN_BUMP_H__
#define __GAUSSIAN_BUMP_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a Gaussian bump grid with an associated nonlinear manifold.
/** Numbers used are the ones from the High-Order Prediction Workshop (HOPW)
 */
template<int dim>
void gaussian_bump(
    dealii::parallel::distributed::Triangulation<dim> &grid,
    const std::vector<unsigned int> n_subdivisions,
    const double channel_length,
    const double channel_height,
    const double bump_height = 0.0625,
    const double channel_width = 1.0);

/// Gaussian bump manifold.
template<int dim>
class BumpManifold: public dealii::ChartManifold<dim,dim,dim> {
protected:
    double channel_height; ///< Channel height.
    double channel_width; ///< Channel width in 3D
    double bump_height; ///< Bump height.
    static constexpr double coeff_expx = -25; ///< Bump exponent (variance).
    static constexpr double coeff_expy = -30; ///< Bump propagation in the domain.
public:
    /// Constructor.
    BumpManifold(const double channel_height, const double bump_height, const double channel_width = 1.0)
    : dealii::ChartManifold<dim,dim,dim>()
    , channel_height(channel_height)
    , channel_width(channel_width)
    , bump_height(bump_height)
    {};
    template<typename real>
    dealii::Point<dim,real> mapping(const dealii::Point<dim,real> &chart_point) const; ///< Templated mapping from square to the bump.

    virtual dealii::Point<dim> pull_back(const dealii::Point<dim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<dim> push_forward(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,dim,dim> push_forward_gradient(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<dim,dim> > clone() const override; ///< See dealii::Manifold.

    /// Used to deform a hypercube Triangulation into the Gaussian bump.
    static dealii::Point<dim> warp (const dealii::Point<dim> &p);

};

} // namespace Grids
} // namespace PHiLiP
#endif
