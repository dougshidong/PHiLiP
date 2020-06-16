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
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
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
    //for (int i=0; i<200; i++) {
    int flag =0;
    while(flag != dim){
//#if 0
        for(int idim=0;idim<dim;idim++){
            //function[idim] = 1.0 / 8.0; 
            //function[idim] = 1.0 / 8.0 /2.0; 
            function[idim] = 1.0/20.0; 
            for(int idim2=0;idim2<dim;idim2++){
                //function[idim] *= std::cos(3.0 * pi/2.0 * x_ref[idim2]);
                function[idim] *= std::cos(2.0 * pi* x_ref[idim2]);
            }
            function[idim] += x_ref[idim] - x_phys[idim];
        }
//#endif
#if 0
        function[0] = x_ref[0] - x_phys[0] +1.0/40.0*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        function[1] = x_ref[1] - x_phys[1] +1.0/40.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        function[2] = x_ref[2] - x_phys[2] +1.0/40.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
#endif
   // #if 0
        for(int idim=0; idim<dim; idim++){
            for(int idim2=0; idim2<dim;idim2++){
                //derivative[idim][idim2] = - 3.0 * pi / 16.0;
                //derivative[idim][idim2] = - 3.0 * pi / 16.0 /2.0;
                derivative[idim][idim2] = - 1.0/20.0*2.0 * pi;
                for(int idim3 =0;idim3<dim; idim3++){
                    if(idim2 == idim3)
                        //derivative[idim][idim2] *=std::sin(3.0 * pi/2.0 * x_ref[idim3]);
                        derivative[idim][idim2] *=std::sin(2.0 * pi * x_ref[idim3]);
                    else
                        //derivative[idim][idim2] *=std::cos(3.0 * pi/2.0 * x_ref[idim3]);
                        derivative[idim][idim2] *=std::cos(2.0 * pi* x_ref[idim3]);
                }
                if(idim == idim2)
                    derivative[idim][idim2] += 1.0;
            }
        }
//#endif
#if 0
        derivative[0][0] = 1.0 - 1.0/40.0* pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        derivative[0][1] =  - 1.0/40.0*3.0 *pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        derivative[0][2] =  1.0/40.0*2.0*pi * std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(2.0*pi*(x_ref[2]));

        derivative[1][0] =  1.0/40.0*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        derivative[1][1] =  1.0 -1.0/40.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));  
        derivative[1][2] =  1.0/40.0*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::cos(3.0*pi/2.0*(x_ref[2]));

        derivative[2][0] = 1.0/40.0*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        derivative[2][1] = - 1.0/40.0*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        derivative[2][2] = 1.0 + 1.0/40.0*5.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(5.0*pi/2.0*(x_ref[2]));
#endif

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
//#if 0
    for(int idim=0;idim<dim; idim++){
        //function_check[idim] = 1.0/8.0;
        //function_check[idim] = 1.0/8.0 /2.0;
        function_check[idim] = 1.0/20.0;
        for(int idim2=0; idim2<dim; idim2++){
            //function_check[idim] *= std::cos(3.0 * pi/2.0 * x_ref[idim2]);
            function_check[idim] *= std::cos(2.0 * pi * x_ref[idim2]);
        }
        function_check[idim] += x_ref[idim];
    }
//#endif
#if 0
        function_check[0] = x_ref[0] + 1.0/40.0*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        function_check[1] = x_ref[1] + 1.0/40.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        function_check[2] = x_ref[2] + 1.0/40.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
#endif
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
//#if 0
    for(int idim=0; idim<dim; idim++){
        //x_phys[idim] = 1.0/8.0;
       // x_phys[idim] = 1.0/8.0 /2.0;
        x_phys[idim] = 1.0/20.0;
        for(int idim2=0;idim2<dim; idim2++){
           //x_phys[idim] *= std::cos( 3.0 * pi/2.0 * x_ref[idim2]);
           x_phys[idim] *= std::cos( 2.0 * pi * x_ref[idim2]);
        }
        x_phys[idim] += x_ref[idim];
    }
//#endif
#if 0
        x_phys[0] = x_ref[0] + 1.0/40.0*std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        x_phys[1] = x_ref[1] + 1.0/40.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        x_phys[2] = x_ref[2] + 1.0/40.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
#endif
    return dealii::Point<dim> (x_phys); // Trigonometric
}

template<int dim>
dealii::DerivativeForm<1,dim,dim> CurvManifold<dim>::push_forward_gradient(const dealii::Point<dim> &chart_point) const 
{
    const double pi = atan(1)*4.0;
    dealii::DerivativeForm<1, dim, dim> dphys_dref;
//#if 0
    dealii::Point<dim> x;
    for(int idim=0; idim<dim; idim++)
        x[idim] = chart_point[idim];
    for(int idim=0; idim<dim; idim++){
        for(int idim2=0; idim2<dim;idim2++){
            //dphys_dref[idim][idim2] = - 3.0 * pi / 16.0;
            //dphys_dref[idim][idim2] = - 3.0 * pi / 16.0 /2.0;
            dphys_dref[idim][idim2] = - 1.0/20.0*2.0 * pi;
            for(int idim3 =0;idim3<dim; idim3++){
                if(idim2 == idim3)
                    //dphys_dref[idim][idim2] *=std::sin(3.0 * pi/2.0 * x[idim3]);
                    dphys_dref[idim][idim2] *=std::sin(2.0 * pi * x[idim3]);
                else
                   //  dphys_dref[idim][idim2] *=std::cos(3.0 * pi/2.0 * x[idim3]);
                     dphys_dref[idim][idim2] *=std::cos(2.0 * pi* x[idim3]);
            }     
            if(idim == idim2)
                dphys_dref[idim][idim2] += 1.0;
        }
    }
//#endif
#if 0
    dealii::Point<dim> x_ref;
    for(int idim=0; idim<dim; idim++){
        x_ref[idim] = chart_point[idim];
    }

        dphys_dref[0][0] = 1.0 - 1.0/40.0*pi/2.0 * std::sin(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        dphys_dref[0][1] =  - 1.0/40.0*3.0*pi/2.0 * std::cos(pi/2.0*x_ref[0])*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(2.0*pi*(x_ref[2]));
        dphys_dref[0][2] =  1.0/40.0*2.0*pi * std::cos(pi/2.0*x_ref[0])*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(2.0*pi*(x_ref[2]));

        dphys_dref[1][0] =  1.0/40.0*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));
        dphys_dref[1][1] =  1.0 -1.0/40.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(pi/2.0*x_ref[1])*std::sin(3.0*pi/2.0*(x_ref[2]));  
        dphys_dref[1][2] =  1.0/40.0*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(pi/2.0*x_ref[1])*std::cos(3.0*pi/2.0*(x_ref[2]));

        dphys_dref[2][0] = 1.0/40.0*2.0*pi*std::cos(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        dphys_dref[2][1] = -1.0/40.0*3.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::sin(3.0*pi/2.0*x_ref[1])*std::sin(5.0*pi/2.0*(x_ref[2]));
        dphys_dref[2][2] = 1.0 + 1.0/40.0*5.0*pi/2.0*std::sin(2.0*pi*(x_ref[0]))*std::cos(3.0*pi/2.0*x_ref[1])*std::cos(5.0*pi/2.0*(x_ref[2]));
#endif

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

    if (dim == 2){
        q[dim-1] = p[dim-1] + 1.0/8.0 * std::cos(3.0 * pi/2.0 * p[dim-1]) * std::cos(3.0 * pi/2.0 * p[dim-2]);
        q[dim-2] = p[dim-2] + 1.0/8.0 * std::cos(3.0 * pi/2.0 * p[dim-1]) * std::cos(3.0 * pi/2.0 * p[dim-2]);
        //q[dim-1] = p[dim-1] +  std::cos(2.0 * pi * p[dim-1]) * std::cos(2.0 * pi* p[dim-2]);
        //q[dim-2] = p[dim-2] +  std::cos(2.0 * pi * p[dim-1]) * std::cos(2.0 * pi* p[dim-2]);
    }
    if(dim==3){
       //q[dim-1] = p[dim-1] + 1.0/16.0 * cos(3.0 * pi/2.0 * p[dim-1]) * cos(3.0 * pi/2.0 * p[dim-2]) * cos(3.0 * pi/2.0 * p[dim-3]);
       //q[dim-2] = p[dim-2] + 1.0/16.0 * cos(3.0 * pi/2.0 * p[dim-1]) * cos(3.0 * pi/2.0 * p[dim-2]) * cos(3.0 * pi/2.0 * p[dim-3]);
       //q[dim-3] = p[dim-3] + 1.0/16.0 * cos(3.0 * pi/2.0 * p[dim-1]) * cos(3.0 * pi/2.0 * p[dim-2]) * cos(3.0 * pi/2.0 * p[dim-3]);
       //transform David
   //    #if 0
        q[dim-1] =p[dim-1] + 1.0/20.0*  std::cos(2.0 * pi * p[dim-1]) * std::cos(2.0 * pi * p[dim-2]) * std::cos(2.0 * pi * p[dim-3]);
        q[dim-2] =p[dim-2] +  1.0/20.0* std::cos(2.0 * pi * p[dim-1]) * std::cos(2.0 * pi * p[dim-2]) * std::cos(2.0 * pi * p[dim-3]);
        q[dim-3] =p[dim-3] +  1.0/20.0* std::cos(2.0 * pi * p[dim-1]) * std::cos(2.0 * pi * p[dim-2]) * std::cos(2.0 * pi * p[dim-3]);
    //    #endif
        //non sym transform
        #if 0
        q[dim-1] =p[dim-1] +  1.0/40.0*std::cos(pi/2.0 * p[dim-1]) * std::cos(3.0 * pi/2.0 * p[dim-2]) * std::sin(2.0 * pi * (p[dim-3]));
        q[dim-2] =p[dim-2] +  1.0/40.0*std::sin(2.0 * pi * (p[dim-1])) * std::cos(pi /2.0 * p[dim-2]) * std::sin(3.0 * pi /2.0 * p[dim-3]);
        q[dim-3] =p[dim-3] +  1.0/40.0*std::sin(2.0 * pi * (p[dim-1])) * std::cos(3.0 * pi/2.0 * p[dim-2]) * std::cos(5.0 * pi/2.0 * p[dim-3]);
        #endif
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
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);

    unsigned int poly_degree = 3;
    double left = 0.0;
    double right = 1.0;
    const bool colorize = true;
    //Generate a standard grid

        dealii::parallel::distributed::Triangulation<dim> grid(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
    dealii::GridGenerator::hyper_cube (grid, left, right, colorize);
    grid.refine_global(0);

//Warp the grid
//IF WANT NON-WARPED GRID COMMENT UNTIL SAYS "NOT COMMENT"
//#if 0
    dealii::GridTools::transform (&warp<dim>, grid);

// Assign a manifold to have curved geometry
    const CurvManifold<dim> curv_manifold;
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid.reset_all_manifolds();
    grid.set_all_manifold_ids(manifold_id);
    grid.set_manifold ( manifold_id, curv_manifold );
//#endif
//"END COMMENT" TO NOT WARP GRID

//setup DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, &grid);
    dg->allocate_system ();
    
    double max_GCL = 0.0;

    const unsigned int n_quad_pts      = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell     =dg->fe_collection[poly_degree].dofs_per_cell;
    dealii::QGauss<dim> quad_val(poly_degree+1);
    //dealii::QGaussLobatto<dim> quad_val(poly_degree+1);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid.mapping_fe_field));
    // dealii::FEValues<dim,dim> fe_values_test(*(dg->high_order_grid.mapping_fe_field), dg->fe_collection[poly_degree], quad_val, 
    dealii::FEValues<dim,dim> fe_values_vol(mapping, dg->fe_collection[poly_degree], quad_val, 
                                dealii::update_values | dealii::update_JxW_values | 
                                dealii::update_quadrature_points | dealii::update_inverse_jacobians);
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::FullMatrix<real>> local_derivative_operator(dim);
    for(int idim=0; idim<dim; idim++){
       local_derivative_operator[idim].reinit(n_quad_pts, n_dofs_cell);
    }
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            dealii::Tensor<1,dim,real> derivative;
            const dealii::Point<dim> qpoint  = dg->volume_quadrature_collection[poly_degree].point(iquad);
            derivative = dg->fe_collection[poly_degree].shape_grad_component(idof, qpoint, istate);
            for (int idim=0; idim<dim; idim++){
                local_derivative_operator[idim][iquad][idof] = derivative[idim];//store dChi/dXi
            }
        }
    }
    }
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
	
                fe_values_vol.reinit (current_cell);

                std::vector<dealii::FullMatrix<real>> Jacobian_inv(n_quad_pts);
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    Jacobian_inv[iquad].reinit(dim, dim);
                }
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    dealii::DerivativeForm<1, dim, dim> temp;
                    temp=fe_values_vol.inverse_jacobian(iquad);
                    for(int idim=0; idim<dim; idim++){
                        for(int idim2=0; idim2<dim; idim2++){
                            Jacobian_inv[iquad][idim][idim2] = temp[idim][idim2];
                        }
                    }
                }

                std::vector< std::vector<dealii::FullMatrix<real>>>  Gij(dim);
                for(int idim=0; idim<dim; idim++){
                    Gij[idim].resize(dim);
                    for(int idim2=0; idim2<dim; idim2++){
                        Gij[idim][idim2].reinit(n_dofs_cell, n_dofs_cell);
                    }
                }

                const std::vector<real> &quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights ();
                for(int idim=0; idim<dim;idim++){
                    for(int idim2=0; idim2<dim; idim2++){
                        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                                if(idof==iquad){
                                    Gij[idim][idim2][idof][iquad] = fe_values_vol.JxW(iquad)/quad_weights[iquad]*Jacobian_inv[iquad][idim][idim2];
                                }
                            }
                        }
                    }
                }

                std::vector<dealii::Vector<real>> GCL(dim);
                for(int idim=0; idim<dim; idim++){
                    GCL[idim].reinit(n_dofs_cell);
                }
                dealii::Vector<real> ones(n_dofs_cell);
                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    ones[idof] = 1.0;
                }

                for(int idim=0; idim< dim; idim++){
                    for(int idim2=0; idim2<dim;idim2++){
                        dealii::FullMatrix<real> temp(n_dofs_cell);
                        local_derivative_operator[idim2].mmult(temp, Gij[idim2][idim]);
                        dealii::Vector<real> temp2(n_dofs_cell);
                        temp.vmult(temp2, ones);
                        GCL[idim].add(1, temp2);
                    }
                }

                for(int idim=0; idim<dim; idim++){
                    printf("\n GCL for derivative x_%d \n", idim);
                    for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                        printf(" %.16g \n", GCL[idim][idof]);
                        if( std::abs(GCL[idim][idof]) > max_GCL){
                            max_GCL = GCL[idim][idof];
                        }
                    }
                }

            }


    if( max_GCL > 1e-10){
        printf(" Metrics Do NOT Satisfy GCL Condition\n");
        return 1;
    }
    else{
        printf(" Metrics Satisfy GCL Condition\n");
        return 0;
    }
}

//}//end PHiLiP namespace
