#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "spline_channel.h"

namespace PHiLiP {
namespace Grids {

// void spline_channel(
//     dealii::parallel::distributed::Triangulation<2> &grid,
//     const std::vector<unsigned int> n_subdivisions,
//     const double channel_length,
//     const double channel_height)
// {
//     const double x_start = channel_length * 0.5;
//     const dealii::Point<2> p1(-x_start,0.0), p2(x_start,channel_height);
//     const bool colorize = true;
//     dealii::GridGenerator::subdivided_hyper_rectangle (grid, n_subdivisions, p1, p2, colorize);
// 
//     // Set boundary type and design type
//     for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
//         for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
//             if (cell->face(face)->at_boundary()) {
//                 unsigned int current_id = cell->face(face)->boundary_id();
//                 if (current_id == 2 || current_id == 3) cell->face(face)->set_boundary_id (1001); // Bottom and top wall
//                 if (current_id == 1) cell->face(face)->set_boundary_id (1002); // Outflow with supersonic or back_pressure
//                 if (current_id == 0) cell->face(face)->set_boundary_id (1003); // Inflow
// 
//                 if (current_id == 2) {
//                     cell->face(face)->set_user_index(1); // Bottom wall
//                 } else {
//                     cell->face(face)->set_user_index(-1); // All other boundaries.
//                 }
//             }
//         }
//     }
// 
//     const BSplineManifold spline_manifold;
// 
//     // Warp grid to be a gaussian spline
//     //dealii::GridTools::transform (&(BSplineManifold::warp), grid);
//     dealii::GridTools::transform (
//         [&spline_manifold](const dealii::Point<2> &chart_point) {
//           return spline_manifold.push_forward(chart_point);}, grid);
//     
//     // Assign a manifold to have curved geometry
//     unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
//     grid.reset_all_manifolds();
//     grid.set_all_manifold_ids(manifold_id);
//     grid.set_manifold ( manifold_id, spline_manifold );
//    
//     // // Set Flat manifold on the domain, but not on the boundary.
//     grid.set_manifold(0, dealii::FlatManifold<2>());
//     grid.set_all_manifold_ids_on_boundary(1001,1);
//     grid.set_manifold(1, spline_manifold);
// }

template<int dim,int chartdim>
BSplineManifold<dim,chartdim>::BSplineManifold(
    const unsigned int _spline_degree,
    const unsigned int _n_control_pts
    )
    : spline_degree(_spline_degree)
    , n_1d_control_pts(_n_control_pts)
    , n_control_pts(std::pow(n_1d_control_pts,chartdim))
    , n_1d_knots(n_1d_control_pts + spline_degree + 1)
    , knot_vector(generate_clamped_uniform_knot_vector())
{
    control_points.resize(n_control_pts);
}

template <int chartdim>
void global_to_grid ( const int index, const int n_1d_pts, std::array<int,chartdim> &grid_index )
{
    if constexpr (chartdim == 1) {
        grid_index[0] = index;
    } else {
        int current_direction_index = index / n_1d_pts;
        grid_index[chartdim-1] = current_direction_index;
        const int remaining_index = index % n_1d_pts;

        global_to_grid<chartdim-1>( remaining_index, n_1d_pts, grid_index );
    }
}

template <int chartdim>
int grid_to_global ( const int n_1d_pts, const std::array<int,chartdim> &grid_index )
{
    int global_index = 0;
    for (int d=0;d<chartdim;d++) {
        global_index += grid_index[d] * std::pow(n_1d_pts,d);
    }
    return global_index;
}


template<typename real>
int get_knot_interval(const real val, const std::vector<double> knot_vector)
{
    // Binary search to find interval i such that
    // knot_vector[i] <= val < knot_vector[i+1]
    const int n_knots = knot_vector.size();
    for (int i = 0; i < n_knots; ++i) {
    }
    int low = 0, high = n_knots - 1 - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if(knot_vector[mid] <= val && val < knot_vector[mid+1]) {
            return mid;
        } else if(val >= knot_vector[mid+1]) {
            low=mid-1;
        } else if(val < knot_vector[mid]) {
            high=mid+1;
        }
        else{
           return mid;
        }
   }
   assert();
   return -1;
}


template<int dim, int chartdim>
template<typename real>
dealii::Point<dim,real> BSplineManifold<dim,chartdim>
::DeBoor_1D(  const real chart_point
            , const unsigned int degree
            , const unsigned int knot_index
            , const std::vector<double> &knot_vector_1d
            , const std::vector<dealii::Point<dim,real>> &control_points
    ) const 
{
    // https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
    std::vector<dealii::Point<dim,real>> d_points(degree);
    for (unsigned int j = 0; j < degree+1; ++j) {
        d_points[j] = control_points[j + knot_index - degree];
    }
    for (unsigned int r = 1; r < degree+1; ++r) {
        for (unsigned int j = degree; j > r-1; --j) {
            real alpha = (chart_point - knot_vector_1d[j+knot_index-degree]) / (knot_vector_1d[j+1+knot_index-r] - knot_vector_1d[j+knot_index-degree]);
            for (int d=0; d<dim; ++d) {
                d_points[j][d] = (1.0 - alpha) * d_points[j-1][d] + alpha * d_points[j][d];
            }
        }
    }
    return d_points[degree];
}

template<int dim, int chartdim>
template<typename real>
dealii::Point<dim,real> BSplineManifold<dim,chartdim>::DeBoor(
    const dealii::Point<chartdim,real> &chart_point
    , const unsigned int degree
    , std::array<std::vector<double>,chartdim> knot_vector
    , const std::vector<dealii::Point<dim,real>> control_points
    ) const
{
    // https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-de-boor.html
    dealii::Point<dim,real> new_point;
    if constexpr (chartdim == 1) {
        const unsigned int cdim = 0;
        const unsigned int knot_index = get_knot_interval(chart_point[cdim], knot_vector[cdim]);
        new_point = DeBoor_1D( chart_point[cdim], degree , knot_index , knot_vector[cdim], control_points );
    }
    if constexpr (chartdim == 2) {

        std::array<unsigned int, chartdim> knot_indices;
        for (int cdim = 0; cdim < chartdim; ++cdim) {
            knot_indices[cdim] = get_knot_interval(chart_point[cdim], knot_vector[cdim]);
        }

        /// new_point = sum_j (N_j(v) q_j)
        /// where q_j = sum_i (N_i(u) p_ij)
        const unsigned int n_new_points_j = n_control_pts / n_1d_control_pts;
        std::vector<dealii::Point<dim,real>> new_points_j(n_new_points_j);

        std::vector<dealii::Point<dim,real>> used_control_points(n_1d_control_pts);
        for (int j = 0; j < n_1d_control_pts; ++j) {

            std::array<int,chartdim> grid_index;
            grid_index[1] = j;

            for (int i = 0; i < n_1d_control_pts; ++i) {

                grid_index[0] = i;
                const int global_used_ctl_index = grid_to_global<chartdim> ( n_1d_control_pts, grid_index );

                for (int d=0;d<dim;++d) {
                    used_control_points[i][d] = control_points[global_used_ctl_index][d];
                }
            }
            new_points_j[j] = DeBoor_1D( chart_point[0], degree , knot_indices[0], knot_vector[0], used_control_points );
        }

        new_point = DeBoor_1D( chart_point[1], degree , knot_indices[1], knot_vector[1], new_points_j );
    }
    // unsigned int n_new_points = control_points.size();
    // for (unsigned int cdim = 0; cdim < chartdim; ++cdim) {

    //     n_new_points /= n_1d_control_pts;


    //     real chart_val = chart_point[cdim];
    //     const unsigned int knot_index = get_knot_interval(chart_val, knot_vector[cdim]);

    //     const std::vector<dealii::Point<dim,real>> new_points(n_new_points);

    //     for (int i_new_pt = 0; i_new_pt < n_new_points; ++i_new_pt) {

    //         std::array<int,chartdim-cdim-1> new_grid_index;
    //         global_to_grid ( i_new_pt, n_1d_control_pts, new_grid_index );

    //         for (int i_used_pt = 0; i_used_pt < n_1d_control_pts; i_used_pt) {
    //             
    //             std::array<int,chartdim-cdim> used_grid_index;
    //             if (cdim == 0) {
    //                 used_grid_index[0] = i_used_pt;
    //                 used_grid_index[0] = new_grid_index
    //                 used_grid_index[0] = i_used_pt;
    //             }
    //             const int global_ctl_index = grid_to_global ( n_1d_control_pts, const std::array<int,chartdim> &grid_index )
    //         }
    //     }
    //     
    //     const std::vector<dealii::Point<dim,real>> used_control_points(n_1d_control_pts);
    //     for (int i_new_pt = 0; i_new_pt < n_control_pts; ++i_new_pt) {
    //         std::array<int,chartdim> grid_index;
    //         global_to_grid ( i_new_pt, n_1d_control_pts, grid_index );

    //         used_control_points

    //     }
    //             for (int ictl = 0; ictl < n_1d_control_pts; ++i_ctl) {
    //         }
    //         new_points[i_new_pt] = 
    //     }
    //     const std::array<int,chartdim> &grid_index
    //     const int global_ctl_index = grid_to_global ( n_1d_control_pts, const std::array<int,chartdim> &grid_index )
    // }
    // return global_index;

    return new_point;
}


// dealii::Point<2> BSplineManifold<dim,chartdim>::pull_back(const dealii::Point<2> &space_point) const {
//     double x_phys = space_point[0];
//     double y_phys = space_point[1];
//     double x_ref = x_phys;
// 
//     double y_ref = y_phys;
// 
//     using ADtype = Sacado::Fad::DFad<double>;
//     ADtype x_ref_ad = x_ref;
//     ADtype y_ref_ad = y_ref;
//     y_ref_ad.diff(0,1);
//     for (int i=0; i<200; i++) {
//         dealii::Point<2,ADtype> chart_point_ad(x_ref_ad,y_ref_ad);
//         dealii::Point<2,ADtype> new_point = DeBoor<ADtype>(chart_point_ad);
// 
//         const double fun = new_point[1].val() - y_phys;
//         const double derivative = new_point[1].dx(0);
//         y_ref_ad = y_ref_ad - fun/derivative;
//         if(std::abs(fun) < 1e-15) break;
//     }
// 
//     dealii::Point<2,ADtype> chart_point_ad(x_ref_ad,y_ref_ad);
//     dealii::Point<2,ADtype> new_point = DeBoor<ADtype>(chart_point_ad);
//     const double fun = new_point[1].val();
//     const double error = std::abs(fun - y_phys);
//     x_ref = x_ref_ad.val();
//     y_ref = y_ref_ad.val();
//     if (error > 1e-13) {
//         std::cout << "Large error " << error << std::endl;
//         std::cout << "xref " << x_ref << " yref " << y_ref << " y_phys " << y_phys << " " << fun << " " << error << std::endl;
//     }
// 
//     dealii::Point<2> p(x_ref, y_ref);
//     return p;
// }
// 
template<int dim, int chartdim>
dealii::Point<dim> BSplineManifold<dim,chartdim>::push_forward(const dealii::Point<chartdim> &chart_point) const 
{
    return DeBoor(chart_point, spline_degree, knot_vector, control_points);
}

template<int dim, int chartdim>
double BSplineManifold<dim,chartdim>::fit_spline(
        const HighOrderGrid<dim,double> &high_order_grid,
        const unsigned int boundary_user_index,
        const std::vector<dealii::Point<dim>> clamped_points
    )
{
    double l2error = 0;

    (void) high_order_grid;
    (void) boundary_user_index;
    (void) clamped_points;
    // if constexpr (chartdim == 1) assert(clamped_points == 2);
    // // else figure out how to do this for a surface bspline in 3D.
    // n_surface_indices = high_order_grid.locally_relevant_surface_nodes_indices.size();

    // std::vector<dealii::Point<dim> local_surface_points_on_user_index;
    // for (int i = 0; i < n_surface_indices; ++i)
    //     if (high_order_grid.locally_relevant_surface_nodes_user_index[i] == boundary_user_index) {
    //         const unsigned int global_index = high_order_grid.locally_relevant_surface_nodes_indices[i];

    //         high_order_grid.locally_relevant_surface_nodes_user_index[i] == boundary_user_index) {
    //     }
    // }
   
    // for (const auto &cell : high_order_grid.dof_handler_grid.active_cell_iterators()) {
    //     if (cell->is_locally_owned()) {
    //         fe_v.reinit(cell);
    //         cell->get_dof_indices(dofs);
    //         const std::vector<dealii::Point<dim>> &points = fe_v.get_quadrature_points();
    //         for (unsigned int q = 0; q < points.size(); ++q) {
    //             const unsigned int comp = fe.system_to_component_index(q).first;
    //             if (fe_mask[comp]) ::dealii::internal::ElementAccess<VectorType>::set(points[q][fe_to_real[comp]], dofs[q], position_vector);
    //         }
    //     }
    // }

    return l2error;
}

template<int dim, int chartdim>
dealii::DerivativeForm<1,chartdim,dim> BSplineManifold<dim,chartdim>::push_forward_gradient(const dealii::Point<chartdim> &chart_point) const
{
    using ADtype = Sacado::Fad::DFad<double>;

    dealii::Point<chartdim,ADtype> chart_point_ad;

    for (int cdim=0; cdim<chartdim; ++cdim) {
        chart_point_ad[cdim] = chart_point[cdim];
        chart_point_ad[cdim].diff(cdim,chartdim);
    }

    std::vector<dealii::Point<dim,ADtype>> control_points_ad(control_points.size());
    for (unsigned int ictl = 0; ictl < control_points.size(); ++ictl) {
        control_points_ad[ictl] = control_points[ictl];
    }

    dealii::Point<dim,ADtype> new_point = DeBoor(chart_point_ad, spline_degree, knot_vector, control_points_ad);

    dealii::DerivativeForm<1, chartdim, dim> dphys_dref;

    for (int d=0; d<dim; ++d) {
        for (int cdim=0; cdim<chartdim; ++cdim) {
            dphys_dref[d][cdim] = new_point[d].dx(cdim);
        }
    }

    return dphys_dref;
}


template<int dim, int chartdim>
std::unique_ptr<dealii::Manifold<dim,dim>> BSplineManifold<dim,chartdim>::clone() const
{
    return std::make_unique<BSplineManifold<dim,chartdim>>(spline_degree, n_1d_control_pts);
}

#if PHILIP_DIM!=1
    template class BSplineManifold <PHILIP_DIM>;
#endif

} // namespace Grids
} // namespace PHiLiP
