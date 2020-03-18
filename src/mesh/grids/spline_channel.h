#ifndef __SPLINE_CHANNEL__
#define __SPLINE_CHANNEL__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

#include "../high_order_grid.h"

namespace PHiLiP {
namespace Grids {

/// Create a Gaussian bump grid with an associated nonlinear manifold.
/** Numbers used are the ones from the High-Order Prediction Workshop (HOPW)
 */
// void spline_channel(
//     dealii::parallel::distributed::Triangulation<2> &grid,
//     const std::vector<unsigned int> n_subdivisions,
//     const double channel_length,
//     const double y_height);

/// Clampled B-spline manifold.
/** Currently uses uniform knots.
 *  The spline degree P is specified by letting the knot multiplicity be 1
 *  within the domain and the clamped ends have a knot multiplicity of P+1.
 */
template<int dim,int chartdim = dim-1>
class BSplineManifold: public dealii::ChartManifold<dim,dim,chartdim> {
public:
    /// Constructor.
    BSplineManifold( const unsigned int _spline_degree, const unsigned int _n_control_pts );
    // /// Chartdim will always be dim-1 since we will be using BSplineManifold to represent
    // /// the domain boundary.
    //static constexpr int chartdim = dim-1;
protected:
    // template<typename real>
    // real getbij(real u, int i, int degree, std::vector<double> 1d_knot) {
    //     if (degree==0) {
    //         if (1d_knot[i] <= u && u < 1d_knot[i+1]) return 1;
    //         else return 0;
    //     }

    //     double h = getbij(u, i,   degree-1, 1d_knot);
    //     double k = getbij(u, i+1, degree-1, 1d_knot);

    //     double bij = 0;

    //     if (h!=0) bij += (u        - 1d_knot[i]) / (1d_knot[i+degree]   - 1d_knot[i]  ) * h;
    //     if (k!=0) bij += (1d_knot[i+degree+1] - u   ) / (1d_knot[i+degree+1] - 1d_knot[i+1]) * k;

    //     return bij;
    // }
    unsigned int spline_degree; ///< B-spline degree.
    unsigned int n_1d_control_pts; ///< Number of control points in one of the chartdim direction.
    unsigned int n_control_pts; ///< Total number of control points.
    unsigned int n_1d_knots; ///< Number of knots in one of the chartdim direction.
    std::array<std::vector<double>,chartdim> knot_vector; ///< Knot vector.

    std::vector<dealii::Point<dim>> control_points; ///< Control points.

    /// DeBoor algorithm for a 1D chartdim BSpline.
    template<typename real>
    dealii::Point<dim,real> DeBoor_1D(
        const real chart_point
        , const unsigned int degree
        , const unsigned int knot_index
        , const std::vector<double> &knot_vector_1d
        , const std::vector<dealii::Point<dim,real>> &control_points
        ) const;

    /// DeBoor algorithm for a N-D chartdim BSpline.
    template<typename real>
    dealii::Point<dim,real> DeBoor(
        const dealii::Point<chartdim,real> &chart_point
        , const unsigned int degree
        , std::array<std::vector<double>,chartdim> knot_vector
        , const std::vector<dealii::Point<dim,real>> control_points
        ) const;

    /// Generate the clamped uniform knot vector of degree spline_degree with n_control_pts.
    std::array<std::vector<double>,chartdim> generate_clamped_uniform_knot_vector()
    {
        // Knot vector starting at epsilon since we will have
        // to perform std::pow(knot_value,0) at some point and 0 to the
        // power of 0 is undefined. The literature for B-splines use the 
        // definition that std::pow(0,0) is equal to 1.
        const double zero = std::numeric_limits<double>::epsilon();
        const double knot_start = zero-zero;
        const double knot_end = 1.0;

        std::vector<double> knots(n_1d_knots);
        // Repeated knots at clamped ends
        int n_outer = spline_degree + 1;
        for (int iknot = 0; iknot < n_outer; iknot++) {
            knots[iknot] = knot_start;
            knots[n_1d_knots-1 - iknot] = knot_end;
        }
        // Uniform knot Vector
        int n_inner = n_1d_knots - 2*n_outer;
        double knot_dx = (knot_end - knot_start) / (n_inner + 1);
        for (int iknot = 0; iknot < n_inner; iknot++) {
            knots[iknot + n_outer] = iknot * knot_dx;
        }

        std::array<std::vector<double>,chartdim> all_knots;
        for (int d=0;d<chartdim;++d) {
            all_knots[d] = knots;
        }
        return all_knots;
    };

    /// Perform a least-squares to fit B-spline to Triangulation surface.
    double fit_spline(
        const HighOrderGrid<dim,double> &high_order_grid,
        const unsigned int boundary_user_index,
        const std::vector<dealii::Point<dim>> clamped_points
    );

public:
    template<typename real>
    dealii::Point<dim,real> mapping(const dealii::Point<chartdim,real> &chart_point) const; ///< Templated mapping from square to the bump.

    virtual dealii::Point<chartdim> pull_back(const dealii::Point<dim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<dim> push_forward(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,chartdim,dim> push_forward_gradient(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<dim,dim> > clone() const override; ///< See dealii::Manifold.

};

} // namespace Grids
} // namespace PHiLiP
#endif
