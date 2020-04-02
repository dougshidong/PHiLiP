#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "gaussian_bump.h"

namespace PHiLiP {
namespace Grids {

void gaussian_bump(
    dealii::parallel::distributed::Triangulation<2> &grid,
    const std::vector<unsigned int> n_subdivisions,
    const double channel_length,
    const double channel_height,
    const double bump_height)
{
    const double x_start = channel_length * 0.5;
    const dealii::Point<2> p1(-x_start,0.0), p2(x_start,channel_height);
    const bool colorize = true;
    dealii::GridGenerator::subdivided_hyper_rectangle (grid, n_subdivisions, p1, p2, colorize);

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 2 || current_id == 3) cell->face(face)->set_boundary_id (1001); // Bottom and top wall
                if (current_id == 1) cell->face(face)->set_boundary_id (1002); // Outflow with supersonic or back_pressure
                if (current_id == 0) cell->face(face)->set_boundary_id (1003); // Inflow

                if (current_id == 2) {
                    cell->face(face)->set_user_index(1); // Bottom wall
                } else {
                    cell->face(face)->set_user_index(-1); // All other boundaries.
                }
            }
        }
    }

    const BumpManifold bump_manifold(channel_height, bump_height);

    // Warp grid to be a gaussian bump
    //dealii::GridTools::transform (&(BumpManifold::warp), grid);
    dealii::GridTools::transform (
        [&bump_manifold](const dealii::Point<2> &chart_point) {
          return bump_manifold.push_forward(chart_point);}, grid);
    
    // Assign a manifold to have curved geometry
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid.reset_all_manifolds();
    grid.set_all_manifold_ids(manifold_id);
    grid.set_manifold ( manifold_id, bump_manifold );
   
    // // Set Flat manifold on the domain, but not on the boundary.
    grid.set_manifold(0, dealii::FlatManifold<2>());
    grid.set_all_manifold_ids_on_boundary(1001,1);
    grid.set_manifold(1, bump_manifold);
}

template<typename real>
dealii::Point<2,real> BumpManifold::mapping(const dealii::Point<2,real> &chart_point) const 
{
    const real x_ref = chart_point[0];
    const real y_ref = chart_point[1];
    const real x_phys = x_ref;//-1.5+x_ref*3.0;

    //const real y_phys = channel_height*y_ref + exp(coeff_expy*y_ref*y_ref)*bump_height*exp(coeff_expx*x_phys*x_phys) * (1.0+0.7*x_phys);

    const double coeff2 = 2; // Increase for more aggressive INITIAL exponential spacing.
    real y_scaled = channel_height;
    y_scaled *= (exp(std::pow(y_ref,coeff2))-1.0);
    y_scaled /= (exp(std::pow(channel_height,coeff2))-1.0); // [0,channel_height]
    const real y_lower = bump_height*exp(coeff_expx*x_ref*x_ref);
    const real perturbation = y_lower * exp(coeff_expy*y_scaled*y_scaled);
    const real y_phys = y_scaled + perturbation;

    //std::cout << x_ref << " " << y_ref << " " << x_phys << " " << y_phys << std::endl;
    return dealii::Point<2,real> ( x_phys, y_phys); // Trigonometric
}

dealii::Point<2> BumpManifold::pull_back(const dealii::Point<2> &space_point) const {
    double x_phys = space_point[0];
    double y_phys = space_point[1];
    double x_ref = x_phys;

    double y_ref = y_phys;

    using ADtype = Sacado::Fad::DFad<double>;
    ADtype x_ref_ad = x_ref;
    ADtype y_ref_ad = y_ref;
    y_ref_ad.diff(0,1);
    for (int i=0; i<200; i++) {
        dealii::Point<2,ADtype> chart_point_ad(x_ref_ad,y_ref_ad);
        dealii::Point<2,ADtype> new_point = mapping<ADtype>(chart_point_ad);

        const double fun = new_point[1].val() - y_phys;
        const double derivative = new_point[1].dx(0);
        y_ref_ad = y_ref_ad - fun/derivative;
        if(std::abs(fun) < 1e-15) break;
    }

    dealii::Point<2,ADtype> chart_point_ad(x_ref_ad,y_ref_ad);
    dealii::Point<2,ADtype> new_point = mapping<ADtype>(chart_point_ad);
    const double fun = new_point[1].val();
    const double error = std::abs(fun - y_phys);
    x_ref = x_ref_ad.val();
    y_ref = y_ref_ad.val();
    if (error > 1e-13) {
        std::cout << "Large error " << error << std::endl;
        std::cout << "xref " << x_ref << " yref " << y_ref << " y_phys " << y_phys << " " << fun << " " << error << std::endl;
    }

    dealii::Point<2> p(x_ref, y_ref);
    return p;
}

dealii::Point<2> BumpManifold::push_forward(const dealii::Point<2> &chart_point) const 
{
    return mapping<double>(chart_point);
}

dealii::DerivativeForm<1,2,2> BumpManifold::push_forward_gradient(const dealii::Point<2> &chart_point) const
{
    using ADtype = Sacado::Fad::DFad<double>;
    ADtype x_ref = chart_point[0];
    ADtype y_ref = chart_point[1];
    x_ref.diff(0,2);
    y_ref.diff(1,2);
    dealii::Point<2,ADtype> chart_point_ad(x_ref,y_ref);
    dealii::Point<2,ADtype> new_point = mapping<ADtype>(chart_point_ad);

    dealii::DerivativeForm<1, 2, 2> dphys_dref;
    dphys_dref[0][0] = new_point[0].dx(0);
    dphys_dref[0][1] = new_point[0].dx(1);
    dphys_dref[1][0] = new_point[1].dx(0);
    dphys_dref[1][1] = new_point[1].dx(1);

    return dphys_dref;
}

std::unique_ptr<dealii::Manifold<2,2> > BumpManifold::clone() const
{
    return std::make_unique<BumpManifold>(channel_height,bump_height);
}

} // namespace Grids
} // namespace PHiLiP
