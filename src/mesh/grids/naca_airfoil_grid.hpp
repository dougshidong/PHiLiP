#ifndef __NACA_H__
#define __NACA_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a Gaussian bump grid with an associated nonlinear manifold.
/** Numbers used are the ones from the High-Order Prediction Workshop (HOPW)
 */
void naca_airfoil(
    dealii::parallel::distributed::Triangulation<2> &grid,
    dealii::GridGenerator::Airfoil::AdditionalData airfoil_data);

/// NACA airfoil manifold.
template<int dim = 2, int chartdim = 1>
class NACAManifold: public dealii::ChartManifold<dim,dim,chartdim> {
protected:
    const std::string serial_number; ///< NACA serial number. String should be 4 char long.
    const bool is_upper; ///< Flag for upper surface (suction side) versus lower surface (pressure side).

    const std::array<unsigned int,4> serial_digits; ///< Conversion NACA string serial number (char * ) to int

    const double thickness; ///< Maximum thickness in percentage of the cord

    /// Templated push-forward mapping.
    template<typename real>
    dealii::Point<dim,real> push_forward_mapping(const dealii::Point<chartdim,real> &chart_point) const;

public:
    /// Constructor.
    NACAManifold(const std::string serial_number, const bool is_upper);

    virtual dealii::Point<chartdim> pull_back(const dealii::Point<dim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<dim> push_forward(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,chartdim,dim> push_forward_gradient(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.

    virtual std::unique_ptr<dealii::Manifold<dim,dim> > clone() const override; ///< See dealii::Manifold.
};

} // namespace Grids
} // namespace PHiLiP
#endif

