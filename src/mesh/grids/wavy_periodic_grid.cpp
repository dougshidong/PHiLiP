#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "wavy_periodic_grid.hpp"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void wavy_grid_Abe_2015(
    TriangulationType &grid,
    const std::vector<unsigned int> n_subdivisions)
{
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    for (int d=0; d<dim; ++d) {
        p1[d] = -5.0;
        p2[d] = 5.0;
    }
    const bool colorize = true;
    dealii::GridGenerator::subdivided_hyper_rectangle (grid, n_subdivisions, p1, p2, colorize);

    std::vector<dealii::GridTools::PeriodicFacePair<typename TriangulationType::cell_iterator> > matched_pairs;
    if (dim>=1) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(grid,0,1,0,matched_pairs);
        grid.add_periodicity(matched_pairs);
    }
    if (dim>=2) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(grid,2,3,1,matched_pairs);
        grid.add_periodicity(matched_pairs);
    }
    if (dim>=3) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(grid,4,5,2,matched_pairs);
        grid.add_periodicity(matched_pairs);
    }

    const WavyManifold<dim,dim,dim> wave_manifold(n_subdivisions);

    dealii::GridTools::transform (
        [&wave_manifold](const dealii::Point<dim> &chart_point) {
          return wave_manifold.push_forward(chart_point);}, grid);
    
    // Assign a manifold to have curved geometry
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid.reset_all_manifolds();
    grid.set_all_manifold_ids(manifold_id);
    grid.set_manifold ( manifold_id, wave_manifold );

    //grid.reset_all_manifolds();
    //for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
    //    // Set a dummy boundary ID
    //    cell->set_material_id(9002);
    //    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
    //        if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
    //    }
    //}
}

template<int dim,int spacedim,int chartdim>
template<typename real>
dealii::Point<spacedim,real> WavyManifold<dim,spacedim,chartdim>::mapping(const dealii::Point<chartdim,real> &chart_point) const 
{
    dealii::Point<spacedim,real> phys_point;

    phys_point[0] = chart_point[0];
    if constexpr (dim >= 2) {

        real x_perturbation = amplitude*dx[0];
        x_perturbation *= sin(n*pi*chart_point[1] / L0);

        phys_point[0] = chart_point[0];
        phys_point[0] += x_perturbation;

        real y_perturbation = amplitude*dx[1];
        y_perturbation *= sin(n*pi*chart_point[0] / L0);

        phys_point[1] = chart_point[1];
        phys_point[1] += y_perturbation;
    }
    if constexpr (dim >= 3) {
        phys_point[0] = chart_point[0];
        phys_point[0] += amplitude*sin(pi*chart_point[1]);
        phys_point[1] = chart_point[1];
        phys_point[1] += amplitude*sin(pi*chart_point[2]);
        phys_point[2] = chart_point[2];
        phys_point[2] += amplitude*sin(pi*chart_point[0]);

        real x_perturbation = amplitude*dx[0];
        x_perturbation *= sin(n*pi*chart_point[1] / L0) * sin (n*pi*chart_point[2] / L0);

        phys_point[0] = chart_point[0];
        phys_point[0] += x_perturbation;

        real y_perturbation = amplitude*dx[1];
        y_perturbation *= sin(n*pi*chart_point[0] / L0) * sin (n*pi*chart_point[2] / L0);

        phys_point[1] = chart_point[1];
        phys_point[1] += y_perturbation;

        real z_perturbation = amplitude*dx[2];
        z_perturbation *= sin(n*pi*chart_point[0] / L0) * sin (n*pi*chart_point[1] / L0);

        phys_point[2] = chart_point[2];
        phys_point[2] += z_perturbation;
    }
    //phys_point[sd] += chart_point[cd]+sin(pi*chart_point[cd]);
    //for (int sd=0; sd<spacedim; ++sd) {
    //    phys_point[sd] = 0.0;
    //    for (int cd=0; cd<chartdim; ++cd) {
    //        if (cd != sd) {
    //            phys_point[sd] += sin(pi*chart_point[cd]);
    //        }
    //    }
    //    phys_point[sd] *= 0.1;
    //}
    return phys_point;
}

template<int dim,int spacedim,int chartdim>
dealii::Point<chartdim> WavyManifold<dim,spacedim,chartdim>::pull_back(const dealii::Point<spacedim> &space_point) const {

    using FadType = Sacado::Fad::DFad<double>;
    dealii::Point<chartdim,FadType> chart_point_ad;
    for (int d=0; d<chartdim; ++d) {
        chart_point_ad[d] = space_point[d];
    }
    for (int i=0; i<200; i++) {
        for (int d=0; d<chartdim; ++d) {
            chart_point_ad[d].diff(d,chartdim);
        }
        dealii::Point<spacedim,FadType> new_point = mapping<FadType>(chart_point_ad);

        dealii::Tensor<1,dim,double> fun;
        for (int d=0; d<chartdim; ++d) {
            fun[d] = new_point[d].val() - space_point[d];
        }
        double l2_norm = fun.norm();
        if(l2_norm < 1e-15) break;

        dealii::Tensor<2,dim,double> derivative;
        for (int sd=0; sd<spacedim; ++sd) {
            for (int cd=0; cd<chartdim; ++cd) {
                derivative[sd][cd] = new_point[sd].dx(cd);
            }
        }
        dealii::Tensor<2,dim,double> inv_jac = dealii::invert(derivative);
        dealii::Tensor<1,dim,double> dx = - inv_jac * fun;

        for (int d=0; d<chartdim; ++d) {
            chart_point_ad[d] = chart_point_ad[d].val() + dx[d];
        }
    }


    dealii::Point<dim,double> chart_point;
    for (int d=0; d<chartdim; ++d) {
        chart_point[d] = chart_point_ad[d].val();
    }
    dealii::Point<spacedim,double> new_point = mapping<double>(chart_point);
    dealii::Tensor<1,dim,double> fun = new_point - space_point;

    const double error = fun.norm();
    if (error > 1e-13) {
        std::cout << "Large error " << error << std::endl;
        std::cout << "Input space_point: " << space_point
                  << " Output space_point " << new_point << std::endl;
    }

    return chart_point;
}

template<int dim,int spacedim,int chartdim>
dealii::Point<spacedim> WavyManifold<dim,spacedim,chartdim>::push_forward(const dealii::Point<chartdim> &chart_point) const 
{
    return mapping<double>(chart_point);
}

template<int dim,int spacedim,int chartdim>
dealii::DerivativeForm<1,chartdim,spacedim> WavyManifold<dim,spacedim,chartdim>::push_forward_gradient(const dealii::Point<chartdim> &chart_point) const
{
    using FadType = Sacado::Fad::DFad<double>;
    dealii::Point<chartdim,FadType> chart_point_ad;
    for (int d=0; d<chartdim; ++d) {
        chart_point_ad[d] = chart_point[d];
    }
    for (int d=0; d<chartdim; ++d) {
        chart_point_ad[d].diff(d,chartdim);
    }
    dealii::Point<spacedim,FadType> new_point = mapping<FadType>(chart_point_ad);

    dealii::DerivativeForm<1, chartdim, spacedim> dphys_dref;
    for (int sd=0; sd<spacedim; ++sd) {
        for (int cd=0; cd<chartdim; ++cd) {
            dphys_dref[sd][cd] = new_point[sd].dx(cd);
        }
    }
    return dphys_dref;
}

template<int dim,int spacedim,int chartdim>
std::unique_ptr<dealii::Manifold<dim,spacedim> > WavyManifold<dim,spacedim,chartdim>::clone() const
{
    return std::make_unique<WavyManifold<dim,spacedim,chartdim>>(n_subdivisions);
}

template void wavy_grid_Abe_2015<1, dealii::Triangulation<1> >                       (dealii::Triangulation<1> &grid, const std::vector<unsigned int> n_subdivisions);
template void wavy_grid_Abe_2015<2, dealii::parallel::distributed::Triangulation<2>> (dealii::parallel::distributed::Triangulation<2> &grid, const std::vector<unsigned int> n_subdivisions);
template void wavy_grid_Abe_2015<3, dealii::parallel::distributed::Triangulation<3>> (dealii::parallel::distributed::Triangulation<3> &grid, const std::vector<unsigned int> n_subdivisions);

} // namespace Grids
} // namespace PHiLiP


