#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "nonsymmetric_curved_periodic_grid.hpp"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void nonsymmetric_curved_grid(
    TriangulationType &grid,
    const unsigned int n_subdivisions)
{
    const double left = -1.0;
    const double right = 1.0;
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube (grid, left, right, colorize);

    std::vector<dealii::GridTools::PeriodicFacePair<typename TriangulationType::cell_iterator> > matched_pairs;
    if constexpr(dim>=1) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(grid,0,1,0,matched_pairs);
        grid.add_periodicity(matched_pairs);
    }
    if constexpr(dim>=2) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(grid,2,3,1,matched_pairs);
        grid.add_periodicity(matched_pairs);
    }
    if constexpr(dim>=3) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(grid,4,5,2,matched_pairs);
        grid.add_periodicity(matched_pairs);
    }

    grid.refine_global(n_subdivisions);

    const NonsymmetricCurvedGridManifold<dim,dim,dim> periodic_nonsym_curved_manifold;

    dealii::GridTools::transform (
        [&periodic_nonsym_curved_manifold](const dealii::Point<dim> &chart_point) {
          return periodic_nonsym_curved_manifold.push_forward(chart_point);}, grid);
    
    // Assign a manifold to have curved geometry
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid.reset_all_manifolds();
    grid.set_all_manifold_ids(manifold_id);
    grid.set_manifold ( manifold_id, periodic_nonsym_curved_manifold );

    // grid.reset_all_manifolds();
    // for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
    //    // Set a dummy boundary ID
    //    cell->set_material_id(9002);
    //    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
    //        if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
    //    }
    // }
}

template<int dim,int spacedim,int chartdim>
template<typename real>
dealii::Point<spacedim,real> NonsymmetricCurvedGridManifold<dim,spacedim,chartdim>
::mapping(const dealii::Point<chartdim,real> &p) const 
{
    dealii::Point<dim> q = p;

    const double beta = 1.0/10.0;
    const double alpha = 1.0/10.0;
    if constexpr(dim == 1) {
        q[dim-1] = p[dim-1] + cos(2.0 * pi* p[dim-1]);
    }
    if constexpr(dim == 2) {
        //non sym transform
        q[dim-2] =p[dim-2] +  beta*std::cos(pi/2.0 * p[dim-2]) * std::cos(3.0 * pi/2.0 * p[dim-1]);
        q[dim-1] =p[dim-1] +  beta*std::sin(2.0 * pi * (p[dim-2])) * std::cos(pi /2.0 * p[dim-1]);
    }
    if constexpr(dim==3) {
       //periodic non sym 
        double temp1 = alpha*(std::sin(pi * p[0]) * std::sin(pi * p[1]));
        double temp2 = alpha*exp(1.0-p[1])*(std::sin(pi * p[0]) * std::sin(pi* p[1]));
        q[0] =p[0] + temp1; 
        q[1] =p[1] + temp2;
        q[2] =p[2] +  1.0/20.0*( std::sin(2.0 * pi * q[0]) + std::sin(2.0 * pi * q[1]));
    }

    return q;
}

template<int dim,int spacedim,int chartdim>
dealii::Point<chartdim> NonsymmetricCurvedGridManifold<dim,spacedim,chartdim>
::pull_back(const dealii::Point<spacedim> &space_point) const 
{
    dealii::Point<dim> x_ref;
    dealii::Point<dim> x_phys;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = space_point[idim];
        x_phys[idim] = space_point[idim];
    }
    dealii::Vector<double> function(dim);
    dealii::FullMatrix<double> derivative(dim);
    const double beta = 1.0/10.0;
    const double alpha = 1.0/10.0;
    int flag =0;
    while(flag != dim){
        //set function value
        if constexpr(dim==1) {
            function[0] = x_ref[0] - x_phys[0] +cos(2.0*pi*x_ref[0]);
        }
        if constexpr(dim==2) {
            function[0] = x_ref[0] - x_phys[0] +beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
            function[1] = x_ref[1] - x_phys[1] +beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
        }
        else if constexpr(dim==3) {
            double temp1 = alpha*(std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]));
            double temp2 = alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]));
            function[0] = x_ref[0] - x_phys[0] + temp1;
            function[1] = x_ref[1] - x_phys[1] + temp2;
            function[2] = x_ref[2] - x_phys[2] +1.0/20.0*( std::sin(2.0 * pi * (temp1+x_ref[0])) + std::sin(2.0 * pi * (temp2+x_ref[1])));
        }
        //set derivative value
        if constexpr(dim==1) {
            derivative[0][0] = 1.0 - 2.0*pi*sin(2.0*pi*x_ref[0]);
        }
        if constexpr(dim==2) {
            derivative[0][0] = 1.0 - beta* pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
            derivative[0][1] =  - beta*3.0 *pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1]);
        
            derivative[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
            derivative[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1]);  
        }
        else if constexpr(dim==3) {
            derivative[0][0] = 1.0 + alpha*pi*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
            derivative[0][1] =  alpha*pi*std::sin(pi*x_ref[0])*std::cos(pi*x_ref[1]);
            derivative[0][2] = 0.0; 
        
            derivative[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
            derivative[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::sin(pi*x_ref[1])) +alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::cos(pi*x_ref[1]));  
            derivative[1][2] =  0.0;
            double xtemp = x_ref[0] + alpha*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]); 
            double ytemp = x_ref[1] + alpha*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]); 
            derivative[2][0] = 1.0/10.0*pi*(std::cos(2.0*pi*xtemp)*derivative[0][0] + std::cos(2.0*pi*ytemp)*derivative[1][0]);
            derivative[2][1] = 1.0/10.0*pi*(std::cos(2.0*pi*xtemp)*derivative[0][1] + std::cos(2.0*pi*ytemp)*derivative[1][1]);
            derivative[2][2] = 1.0;
        }

        dealii::FullMatrix<double> Jacobian_inv(dim);
        Jacobian_inv.invert(derivative);
        dealii::Vector<double> Newton_Step(dim);
        Jacobian_inv.vmult(Newton_Step, function);
        for(int idim=0; idim<dim; idim++) {
            x_ref[idim] -= Newton_Step[idim];
        }
        flag = 0;
        for(int idim=0; idim<dim; idim++) {
            if(std::abs(function[idim]) < 1e-15) flag++;
        }
        if(flag == dim) break;
    }
    std::vector<double> function_check(dim);
    if constexpr(dim==1) {
        function_check[0] = x_ref[0] + cos(2.0*pi*x_ref[0]);
    }
    if constexpr(dim==2) {
        function_check[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        function_check[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
    }
    else if constexpr(dim==3) {
        double temp1 = alpha*(std::sin(pi * x_ref[0]) * std::sin(pi * x_ref[1]));
        double temp2 = alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) * std::sin(pi* x_ref[1]));
        function_check[0] = x_ref[0] + temp1;
        function_check[1] = x_ref[1] + temp2;
        function_check[2] = x_ref[2] +1.0/20.0*( std::sin(2.0 * pi * (temp1+x_ref[0])) + std::sin(2.0 * pi * (temp2+x_ref[1])));
    }
    std::vector<double> error(dim);
    for(int idim=0; idim<dim; idim++) 
        error[idim] = std::abs(function_check[idim] - x_phys[idim]);
    if (error[0] > 1e-13) {
        std::cout << "Large error " << error[0] << std::endl;
        for(int idim=0;idim<dim; idim++)
        std::cout << "dim " << idim << " xref " << x_ref[idim] <<  " x_phys " << x_phys[idim] << " function Check  " << function_check[idim] << " Error " << error[idim] << " Flag " << flag << std::endl;
    }

    return x_ref;
}

template<int dim,int spacedim,int chartdim>
dealii::Point<spacedim> NonsymmetricCurvedGridManifold<dim,spacedim,chartdim>
::push_forward(const dealii::Point<chartdim> &chart_point) const 
{
    return mapping<double>(chart_point);
}

template<int dim,int spacedim,int chartdim>
dealii::DerivativeForm<1,chartdim,spacedim> NonsymmetricCurvedGridManifold<dim,spacedim,chartdim>
::push_forward_gradient(const dealii::Point<chartdim> &chart_point) const
{
    dealii::DerivativeForm<1, dim, dim> dphys_dref;
    const double beta = 1.0/10.0;
    const double alpha = 1.0/10.0;
    dealii::Point<dim> x_ref;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = chart_point[idim];
    }

    if constexpr(dim==1) {
        dphys_dref[0][0] = 1.0 - 2.0*pi*sin(2.0*pi*x_ref[0]);
    }
    if constexpr(dim==2) {
        dphys_dref[0][0] = 1.0 - beta*pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        dphys_dref[0][1] =  - beta*3.0*pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1]);

        dphys_dref[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
        dphys_dref[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1]);  
    }
    else if constexpr(dim==3) {
        dphys_dref[0][0] = 1.0 + alpha*pi*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
        dphys_dref[0][1] =  alpha*pi*std::sin(pi*x_ref[0])*std::cos(pi*x_ref[1]);
        dphys_dref[0][2] = 0.0; 

        dphys_dref[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0])*std::sin(pi*x_ref[1]);
        dphys_dref[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::sin(pi*x_ref[1])) +alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::cos(pi*x_ref[1]));  
        dphys_dref[1][2] =  0.0;
        double xtemp = x_ref[0] + alpha*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]); 
        double ytemp = x_ref[1] + alpha*exp(1.0-x_ref[1])*std::sin(pi*x_ref[0])*std::sin(pi*x_ref[1]); 
        dphys_dref[2][0] = 1.0/10.0*pi*(std::cos(2.0*pi*xtemp)*dphys_dref[0][0] + std::cos(2.0*pi*ytemp)*dphys_dref[1][0]);
        dphys_dref[2][1] = 1.0/10.0*pi*(std::cos(2.0*pi*xtemp)*dphys_dref[0][1] + std::cos(2.0*pi*ytemp)*dphys_dref[1][1]);
        dphys_dref[2][2] = 1.0;
    }

    return dphys_dref;
}

template<int dim,int spacedim,int chartdim>
std::unique_ptr<dealii::Manifold<dim,spacedim> > NonsymmetricCurvedGridManifold<dim,spacedim,chartdim>
::clone() const
{
    return std::make_unique<NonsymmetricCurvedGridManifold<dim,spacedim,chartdim>>();
}

template void nonsymmetric_curved_grid<1, dealii::Triangulation<1> >                       (dealii::Triangulation<1> &grid, const unsigned int n_subdivisions);
template void nonsymmetric_curved_grid<2, dealii::parallel::distributed::Triangulation<2>> (dealii::parallel::distributed::Triangulation<2> &grid, const unsigned int n_subdivisions);
template void nonsymmetric_curved_grid<3, dealii::parallel::distributed::Triangulation<3>> (dealii::parallel::distributed::Triangulation<3> &grid, const unsigned int n_subdivisions);

} // namespace Grids
} // namespace PHiLiP

