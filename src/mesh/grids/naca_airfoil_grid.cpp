#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "naca_airfoil_grid.hpp"

namespace PHiLiP {
namespace Grids {

void naca_airfoil(
    dealii::parallel::distributed::Triangulation<2> &grid,
    dealii::GridGenerator::Airfoil::AdditionalData airfoil_data)
{
    // dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    // airfoil_data.airfoil_type = "NACA";
    // airfoil_data.naca_id      = naca_id;
    // airfoil_data.airfoil_length = 1.0;
    // airfoil_data.height         = farfield_length;
    // airfoil_data.length_b2      = farfield_length;
    // airfoil_data.incline_factor = 0.0;
    // airfoil_data.bias_factor    = 4.5; // default good enough?
    // airfoil_data.refinements    = 0;


    // const int n_cells_airfoil = n_subdivisions[0] * 2 / 3;
    // const int n_cells_downstream = n_subdivisions[0] - n_cells_airfoil;
    // airfoil_data.n_subdivision_x_0 = n_cells_airfoil / 2;
    // airfoil_data.n_subdivision_x_1 = n_cells_airfoil - airfoil_data.n_subdivision_x_0;
    // airfoil_data.n_subdivision_x_2 = n_cells_downstream;
    // airfoil_data.n_subdivision_y = n_subdivisions[1];
    // airfoil_data.airfoil_sampling_factor = 3; // default 2
    dealii::GridGenerator::Airfoil::create_triangulation(grid, airfoil_data);

    // Assign a manifold to have curved geometry
    unsigned int manifold_id = 0;
    grid.reset_all_manifolds();
    grid.set_all_manifold_ids(manifold_id);
    // // Set Flat manifold on the domain, but not on the boundary.
    grid.set_manifold(manifold_id, dealii::FlatManifold<2>());

    manifold_id = 1;
    bool is_upper = true;
    const NACAManifold<2,1> upper_naca(airfoil_data.naca_id, is_upper);
    grid.set_all_manifold_ids_on_boundary(2,manifold_id); // upper airfoil side
    grid.set_manifold(manifold_id, upper_naca);

    is_upper = false;
    const NACAManifold<2,1> lower_naca(airfoil_data.naca_id, is_upper);
    manifold_id = 2;
    grid.set_all_manifold_ids_on_boundary(3,manifold_id); // lower airfoil side
    grid.set_manifold(manifold_id, lower_naca);


    // manifold_id = 0;
    // dealii::TransfiniteInterpolationManifold<2,2> inner_manifold;
    // inner_manifold.initialize(grid);
    // grid.set_manifold(manifold_id, inner_manifold);
    // grid.refine_global(1);
    // inner_manifold.initialize(grid);


    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 1 || current_id == 4 || current_id == 5) {
                    cell->face(face)->set_boundary_id (1004); // farfield
                } else {
                    cell->face(face)->set_boundary_id (1001); // wall bc
                }
            }
        }
    }
}

template<int dim, int chartdim>
NACAManifold<dim,chartdim>::NACAManifold(const std::string serial_number, const bool is_upper)
    : dealii::ChartManifold<dim,dim,chartdim>()
    , serial_number(serial_number)
    , is_upper(is_upper)
    , serial_digits ({{ (unsigned int)(serial_number[0] - '0'),
                       (unsigned int)(serial_number[1] - '0'),
                       (unsigned int)(serial_number[2] - '0'),
                       (unsigned int)(serial_number[3] - '0') }})
    , thickness(static_cast<double>( (10 * serial_digits[2] + serial_digits[3]) / 100.0))
{ }


template<int dim, int chartdim>
template<typename real>
dealii::Point<dim,real> NACAManifold<dim,chartdim>::push_forward_mapping(const dealii::Point<chartdim,real> &chart_point) const 
{
    dealii::Point<dim,real> physical_point;
  
    const real x = chart_point[0];
    if (x > 1.0) {
        physical_point[0] = x;
        physical_point[1] = 0.0;
        return physical_point;
    }

    const real thickness_ad = thickness;
    real y_t = 5 * thickness_ad *
                 (0.2969 * std::pow(x, 0.5) - 0.126 * x -
                  0.3516 * std::pow(x, 2) + 0.2843 * std::pow(x, 3) -
                  0.1036 * std::pow(x, 4)); // half thickness_ad at a position x

    if (!is_upper) {
        y_t *= -1.0;
    }
    if (serial_digits[0] == 0 && serial_digits[1] == 0) { // is symmetric
        physical_point[0] = x;
        physical_point[1] = y_t;
    } else { // is asymmetric
        const real m = 1.0 * serial_digits[0] / 100; // max. chamber
        const real p = 1.0 * serial_digits[1] / 10; // location of max. chamber
  
        real y_c, dy_c;
        if (x <= p) {
            y_c = m / std::pow(p, 2) * (2 * p * x - std::pow(x, 2));
            dy_c = 2 * m / std::pow(p, 2) * (p - x);
        } else {
            y_c = m / std::pow(1 - p, 2) * ((1 - 2 * p) + 2 * p * x - std::pow(x, 2));
            dy_c = 2 * m / std::pow(1 - p, 2) * (p - x);
        }
  
        const real theta = std::atan(dy_c);
  
        physical_point[0] = x - y_t * std::sin(theta);
        physical_point[1] = y_c + y_t * std::cos(theta);
    }

    return physical_point;
}


template<int dim, int chartdim>
dealii::Point<dim> NACAManifold<dim,chartdim>::push_forward(const dealii::Point<chartdim> &chart_point) const 
{
    return push_forward_mapping<double>(chart_point);
}

template<int dim, int chartdim>
dealii::Point<chartdim> NACAManifold<dim,chartdim>::pull_back(const dealii::Point<dim> &physical_point) const {
    double x_phys = physical_point[0];
    double y_phys = physical_point[1];

    dealii::Point<chartdim> chart_point;
    if (chartdim==2) chart_point[1] = y_phys;

    if (x_phys > 1.0) {
        chart_point[0] = x_phys;
        return chart_point;
    }


    if (serial_digits[0] == 0 && serial_digits[1] == 0) { // is symmetric
        chart_point[0] = x_phys;
        if(chartdim==2) chart_point[1] = y_phys;
    } else {
        /// Initial guess
        double x_chart = x_phys;

        using FadType = Sacado::Fad::DFad<double>;
        FadType x_chart_ad = x_chart;
        FadType y_chart_ad = y_phys;
        for (int i=0; i<200; i++) {
            dealii::Point<chartdim,FadType> chart_point_ad;
            chart_point_ad[0] = x_chart_ad;
            if(chartdim==2) chart_point_ad[1] = y_chart_ad;
            dealii::Point<dim,FadType> new_phys_point = push_forward_mapping<FadType>(chart_point_ad);

            const double fun = new_phys_point[1].val() - y_phys;
            const double derivative = new_phys_point[1].dx(0);
            x_chart_ad = x_chart_ad - fun/derivative;
            if(std::abs(fun) < 1e-15) break;
        }

        chart_point[0] = x_chart_ad.val();
        if (chartdim==2) chart_point[1] = y_phys;
        dealii::Point<dim,double> new_phys_point = push_forward_mapping<double>(chart_point);
        const double error = (new_phys_point - physical_point).norm();
        if (error > 1e-13) {
            std::cout << "Large error " << error << std::endl;
            std::cout << "x_chart " << x_chart
                << " physical_point " << physical_point
                << " new_phys_point " << new_phys_point;
        }

        chart_point[0] = x_chart;
        if (chartdim==2) chart_point[1] = y_phys;
    }
    return chart_point;
}

template<int dim, int chartdim>
dealii::DerivativeForm<1,chartdim,dim> NACAManifold<dim,chartdim>::push_forward_gradient(const dealii::Point<chartdim> &chart_point) const
{
    using FadType = Sacado::Fad::DFad<double>;

    dealii::Point<chartdim,FadType> chart_point_ad;
    for (int dc=0; dc<chartdim; ++dc) {
        chart_point_ad[dc] = chart_point[dc];
        chart_point_ad[dc].diff(dc,chartdim);
    }

    dealii::Point<dim,FadType> new_point = push_forward_mapping<FadType>(chart_point_ad);

    dealii::DerivativeForm<1, chartdim, dim> dphys_dref;
    for (int d=0; d<dim; ++d) {
        for (int dc=0; dc<chartdim; ++dc) {
            dphys_dref[d][dc] = new_point[d].dx(dc);
        }
    }

    return dphys_dref;
}

template<int dim, int chartdim>
std::unique_ptr<dealii::Manifold<dim,dim> > NACAManifold<dim,chartdim>::clone() const
{
    return std::make_unique<NACAManifold<dim,chartdim>>(serial_number, is_upper);
}

} // namespace Grids
} // namespace PHiLiP

