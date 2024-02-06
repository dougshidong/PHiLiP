#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include "dg/dg_base.hpp"
#include "dg/dg_factory.hpp"
#include "operators/operators.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
//#include <GCL_test.h>

const double TOLERANCE = 1E-6;
using namespace std;
//namespace PHiLiP {


template <int dim>
class CurvManifold: public dealii::ChartManifold<dim,dim,dim> {
    virtual dealii::Point<dim> pull_back(const dealii::Point<dim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<dim> push_forward(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,dim,dim> push_forward_gradient(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<dim,dim> > clone() const override; ///< See dealii::Manifold.
};
template<int dim>
dealii::Point<dim> CurvManifold<dim>::pull_back(const dealii::Point<dim> &space_point) const 
{
    using namespace PHiLiP;
    const double pi = atan(1)*4.0;
    dealii::Point<dim> x_ref;
    dealii::Point<dim> x_phys;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = space_point[idim];
        x_phys[idim] = space_point[idim];
    }
    dealii::Vector<double> function(dim);
    dealii::FullMatrix<double> derivative(dim);
    double beta =1.0/10.0;
    double alpha =1.0/10.0;
    int flag =0;
    while(flag != dim){
    if(dim==2){
        function[0] = x_ref[0] - x_phys[0] +beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        function[1] = x_ref[1] - x_phys[1] +beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
    }
    else{
        function[0] = x_ref[0] - x_phys[0] +alpha*(std::cos(pi * x_ref[2]) + std::cos(pi * x_ref[1]));
        function[1] = x_ref[1] - x_phys[1] +alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) + std::sin(pi* x_ref[2]));
        function[2] = x_ref[2] - x_phys[2] +1.0/20.0*( std::sin(2.0 * pi * x_ref[0]) + std::sin(2.0 * pi * x_ref[1]));
    }


    if(dim==2){
        derivative[0][0] = 1.0 - beta* pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        derivative[0][1] =  - beta*3.0 *pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1]);

        derivative[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
        derivative[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1]);  
    }
    else{
        derivative[0][0] = 1.0;
        derivative[0][1] =      - alpha*pi*std::sin(pi*x_ref[1]);
        derivative[0][2] =   - alpha*pi*std::sin(pi*x_ref[2]);

        derivative[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0]);
        derivative[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::sin(pi*x_ref[2]));  
        derivative[1][2] =  alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[2]);
        derivative[2][0] = 1.0/10.0*pi*std::cos(2.0*pi*x_ref[0]);
        derivative[2][1] = 1.0/10.0*pi*std::cos(2.0*pi*x_ref[1]);
        derivative[2][2] = 1.0;
    }

        dealii::FullMatrix<double> Jacobian_inv(dim);
        Jacobian_inv.invert(derivative);
        dealii::Vector<double> Newton_Step(dim);
        Jacobian_inv.vmult(Newton_Step, function);
        for(int idim=0; idim<dim; idim++){
            x_ref[idim] -= Newton_Step[idim];
        }
        flag=0;
        for(int idim=0; idim<dim; idim++){
            if(std::abs(function[idim]) < 1e-15)
                flag++;
        }
        if(flag == dim)
            break;
    }
    std::vector<double> function_check(dim);
    if(dim==2){
        function_check[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        function_check[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
    }
    else{
        function_check[0] = x_ref[0] +alpha*(std::cos(pi * x_ref[2]) + std::cos(pi * x_ref[1]));
        function_check[1] = x_ref[1] +alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) + std::sin(pi* x_ref[2]));
        function_check[2] = x_ref[2] +1.0/20.0*( std::sin(2.0 * pi * x_ref[0]) + std::sin(2.0 * pi * x_ref[1]));
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

template<int dim>
dealii::Point<dim> CurvManifold<dim>::push_forward(const dealii::Point<dim> &chart_point) const 
{
    const double pi = atan(1)*4.0;

    dealii::Point<dim> x_ref;
    dealii::Point<dim> x_phys;
    for(int idim=0; idim<dim; idim++)
        x_ref[idim] = chart_point[idim];
    double beta = 1.0/10.0;
    double alpha = 1.0/10.0;
    if(dim==2){
        x_phys[0] = x_ref[0] + beta*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        x_phys[1] = x_ref[1] + beta*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
    }
    else{
        x_phys[0] =x_ref[0] +  alpha*(std::cos(pi * x_ref[2]) + std::cos(pi * x_ref[1]));
        x_phys[1] =x_ref[1] +  alpha*exp(1.0-x_ref[1])*(std::sin(pi * x_ref[0]) + std::sin(pi* x_ref[2]));
        x_phys[2] =x_ref[2] +  1.0/20.0*( std::sin(2.0 * pi * x_ref[0]) + std::sin(2.0 * pi * x_ref[1]));
    }
    return dealii::Point<dim> (x_phys); // Trigonometric
}

template<int dim>
dealii::DerivativeForm<1,dim,dim> CurvManifold<dim>::push_forward_gradient(const dealii::Point<dim> &chart_point) const 
{
    const double pi = atan(1)*4.0;
    dealii::DerivativeForm<1, dim, dim> dphys_dref;
    double beta = 1.0/10.0;
    double alpha = 1.0/10.0;
    dealii::Point<dim> x_ref;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = chart_point[idim];
    }

    if(dim==2){
        dphys_dref[0][0] = 1.0 - beta*pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1]);
        dphys_dref[0][1] =  - beta*3.0*pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1]);

        dphys_dref[1][0] =  beta*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1]);
        dphys_dref[1][1] =  1.0 -beta*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1]);  
    }
    else{
        dphys_dref[0][0] = 1.0;
        dphys_dref[0][1] =      - alpha*pi*std::sin(pi*x_ref[1]);
        dphys_dref[0][2] =   - alpha*pi*std::sin(pi*x_ref[2]);

        dphys_dref[1][0] =       alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[0]);
        dphys_dref[1][1] =  1.0 -alpha*exp(1.0-x_ref[1])*(std::sin(pi*x_ref[0])+std::sin(pi*x_ref[2]));  
        dphys_dref[1][2] =  alpha*pi*exp(1.0-x_ref[1])*std::cos(pi*x_ref[2]);
        dphys_dref[2][0] = 1.0/10.0*pi*std::cos(2.0*pi*x_ref[0]);
        dphys_dref[2][1] = 1.0/10.0*pi*std::cos(2.0*pi*x_ref[1]);
        dphys_dref[2][2] = 1.0;
    }

    return dphys_dref;
}

template<int dim>
std::unique_ptr<dealii::Manifold<dim,dim> > CurvManifold<dim>::clone() const 
{
    return std::make_unique<CurvManifold<dim>>();
}

template <int dim>
static dealii::Point<dim> warp (const dealii::Point<dim> &p)
{
    const double pi = atan(1)*4.0;
    dealii::Point<dim> q = p;

    double beta =1.0/10.0;
    double alpha =1.0/10.0;
    if (dim == 2){
        q[dim-2] =p[dim-2] +  beta*std::cos(pi/2.0 * p[dim-2]) * std::cos(3.0 * pi/2.0 * p[dim-1]);
        q[dim-1] =p[dim-1] +  beta*std::sin(2.0 * pi * (p[dim-2])) * std::cos(pi /2.0 * p[dim-1]);
    }
    if(dim==3){
        q[0] =p[0] +  alpha*(std::cos(pi * p[2]) + std::cos(pi * p[1]));
        q[1] =p[1] +  alpha*exp(1.0-p[1])*(std::sin(pi * p[0]) + std::sin(pi* p[2]));
        q[2] =p[2] +  1.0/20.0*( std::sin(2.0 * pi * p[0]) + std::sin(2.0 * pi * p[1]));
    }

    return q;
}

/****************************
 * End of Curvilinear Grid
 * ***************************/

int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);

    double left = 0.0;
    double right = 1.0;
    const bool colorize = true;
    //Generate a standard grid

    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
        dealii::GridGenerator::hyper_cube (*grid, left, right, colorize);
        grid->refine_global(0);

//Warp the grid
//IF WANT NON-WARPED GRID COMMENT UNTIL SAYS "NOT COMMENT"
    dealii::GridTools::transform (&warp<dim>, *grid);

// Assign a manifold to have curved geometry
    const CurvManifold<dim> curv_manifold;
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid->reset_all_manifolds();
    grid->set_all_manifold_ids(manifold_id);
    grid->set_manifold ( manifold_id, curv_manifold );
//"END COMMENT" TO NOT WARP GRID
    double max_GCL = 0.0;
    for(unsigned int poly_degree = 2; poly_degree<5; poly_degree++){
        unsigned int grid_degree = poly_degree;

        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        dg->allocate_system ();

        dealii::QGaussLobatto<1> grid_quad(grid_degree +1);
        const dealii::FE_DGQ<1> fe_grid(grid_degree);
        const dealii::FESystem<1,1> fe_sys_grid(fe_grid, nstate);
        dealii::QGauss<1> flux_quad(poly_degree +1);
        dealii::QGauss<0> flux_quad_face(poly_degree +1);

        PHiLiP::OPERATOR::mapping_shape_functions<dim,2*dim,real> mapping_basis(nstate,poly_degree,grid_degree);
        mapping_basis.build_1D_shape_functions_at_grid_nodes(fe_sys_grid, grid_quad);
        mapping_basis.build_1D_shape_functions_at_flux_nodes(fe_sys_grid, flux_quad, flux_quad_face);

        const unsigned int n_quad_pts = pow(poly_degree+1,dim);

        const dealii::FESystem<dim> &fe_metric = (dg->high_order_grid->fe_system);
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell; 
        auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
        for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
            if (!current_cell->is_locally_owned()) continue;
        
            pcout<<" degree "<<grid_degree<<" metric dofs "<<n_metric_dofs<<std::endl;
            std::vector<dealii::types::global_dof_index> current_metric_dofs_indices(n_metric_dofs);
            metric_cell->get_dof_indices (current_metric_dofs_indices);
            std::array<std::vector<real>,dim> mapping_support_points;
            for(int idim=0; idim<dim; idim++){
                mapping_support_points[idim].resize(n_metric_dofs/dim);
            }
            dealii::QGaussLobatto<dim> vol_GLL(grid_degree +1);
            for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
                for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
                    const real val = (dg->high_order_grid->volume_nodes[current_metric_dofs_indices[idof]]);
                    const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
                    mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate); 
                }
            }

            PHiLiP::OPERATOR::metric_operators<real,dim,2*dim> metric_oper(nstate,poly_degree,grid_degree);
            metric_oper.build_volume_metric_operators(
                n_quad_pts, n_metric_dofs/dim,
                mapping_support_points,
                mapping_basis,
                true);

            std::array<std::vector<real>,dim> GCL;
            for(int idim=0; idim<dim; idim++){
                GCL[idim].resize(n_quad_pts);
            }

            const dealii::FE_DGQArbitraryNodes<1> fe_poly(flux_quad);
            const dealii::FESystem<1,1> fe_sys_poly(fe_poly, nstate);
            PHiLiP::OPERATOR::basis_functions_state<dim,nstate,2*dim,real> flux_basis_quad(poly_degree, 1);
            flux_basis_quad.build_1D_gradient_state_operator(fe_sys_poly, flux_quad);
            flux_basis_quad.build_1D_volume_state_operator(fe_sys_poly, flux_quad);
            for(int idim=0; idim<dim; idim++){
                flux_basis_quad.divergence_matrix_vector_mult(metric_oper.metric_cofactor_vol[idim], GCL[idim],
                                                              flux_basis_quad.oneD_vol_state_operator[0],
                                                              flux_basis_quad.oneD_vol_state_operator[0],
                                                              flux_basis_quad.oneD_vol_state_operator[0],
                                                              flux_basis_quad.oneD_grad_state_operator[0],
                                                              flux_basis_quad.oneD_grad_state_operator[0],
                                                              flux_basis_quad.oneD_grad_state_operator[0]);
            }

            for(int idim=0; idim<dim; idim++){
               // printf("\n GCL for derivative x_%d \n", idim);
                for(unsigned int idof=0; idof<n_quad_pts; idof++){
                //    printf(" %.16g \n", GCL[idim][idof]);
                    if( std::abs(GCL[idim][idof]) > max_GCL){
                        max_GCL = std::abs(GCL[idim][idof]);
                    }
                }
            }

        }

    }//end poly degree loop
    const double max_GCL_mpi= (dealii::Utilities::MPI::max(max_GCL, MPI_COMM_WORLD));

    if( max_GCL_mpi > 1e-10){
        pcout<<" Metrics Do NOT Satisfy GCL Condition\n"<<std::endl;
        return 1;
    }
    else{
        pcout<<" Metrics Satisfy GCL Condition\n"<<std::endl;
        return 0;
    }
}

//}//end PHiLiP namespace

