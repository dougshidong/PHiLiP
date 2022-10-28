#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "operators/operators.h"

const double TOLERANCE = 1E-6;
using namespace std;
//namespace PHiLiP {

template <int dim, int nstate>
void assemble_weak_auxiliary_volume(
    std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg,
    const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
    const unsigned int poly_degree,
    PHiLiP::OPERATOR::basis_functions<dim,2*dim> &soln_basis,
    PHiLiP::OPERATOR::metric_operators<double,dim,2*dim> &metric_oper,
    std::vector<dealii::Tensor<1,dim,double>> &local_auxiliary_RHS)
{
    const unsigned int n_quad_pts  = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    const std::vector<double> &quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();

    //Fetch the modal soln coefficients and the modal auxiliary soln coefficients
    //We immediately separate them by state as to be able to use sum-factorization
    //in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
    //mult would sum the states at the quadrature point.
    //That is why the basis functions are of derived class state rather than base.
    std::array<std::vector<double>,nstate> soln_coeff;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
        if(ishape == 0)
            soln_coeff[istate].resize(n_shape_fns);

        soln_coeff[istate][ishape] = dg->solution(current_dofs_indices[idof]);
    }

    //Interpolate coefficients to quad points and evaluate rhs
    for(int istate=0; istate<nstate; istate++){
        std::vector<double> soln_at_q(n_shape_fns);
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q,
                                         soln_basis.oneD_vol_operator);

        for(int idim=0; idim<dim; idim++){
            std::vector<double> rhs(n_shape_fns);
            for(int jdim=0; jdim<dim; jdim++){
                std::vector<double> metric_cofactor_times_quad_weights(n_quad_pts);
                //For Reference transofrmation
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    metric_cofactor_times_quad_weights[iquad] = metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                              * quad_weights[iquad];
                }
                //solve inner product
                soln_basis.inner_product(soln_at_q, metric_cofactor_times_quad_weights,
                                         rhs,
                                         (jdim==0) ? soln_basis.oneD_grad_operator : soln_basis.oneD_vol_operator,
                                         (jdim==1) ? soln_basis.oneD_grad_operator : soln_basis.oneD_vol_operator,
                                         (jdim==2) ? soln_basis.oneD_grad_operator : soln_basis.oneD_vol_operator,
                                         true, -1.0);
            }
            //write the the auxiliary rhs for the test function.
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                local_auxiliary_RHS[istate*n_shape_fns + ishape][idim] += rhs[ishape];
            }
        }
    }
}
template <int dim, int nstate>
void assemble_face_term_auxiliary_weak(
    std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg,
    const unsigned int iface, const unsigned int neighbor_iface,
    const dealii::types::global_dof_index /*current_cell_index*/,
    const dealii::types::global_dof_index /*neighbor_cell_index*/,
    const unsigned int poly_degree_int, 
    const unsigned int /*poly_degree_ext*/,
    const unsigned int n_dofs_int,
    const unsigned int n_dofs_ext,
    const unsigned int n_face_quad_pts,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    PHiLiP::OPERATOR::basis_functions<dim,2*dim> &soln_basis_int,
    PHiLiP::OPERATOR::basis_functions<dim,2*dim> &soln_basis_ext,
    PHiLiP::OPERATOR::metric_operators<double,dim,2*dim> &metric_oper_int,
    std::vector<dealii::Tensor<1,dim,double>> &local_auxiliary_RHS_int,
    std::vector<dealii::Tensor<1,dim,double>> &local_auxiliary_RHS_ext)
{

    const unsigned int n_shape_fns_int = n_dofs_int / nstate;
    const unsigned int n_shape_fns_ext = n_dofs_ext / nstate;
    //Extract interior modal coefficients of solution
    std::array<std::vector<double>,nstate> soln_coeff_int;
    std::array<std::vector<double>,nstate> soln_coeff_ext;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        const unsigned int istate = dg->fe_collection[poly_degree_int].system_to_component_index(idof).first;
        const unsigned int ishape = dg->fe_collection[poly_degree_int].system_to_component_index(idof).second;
        //allocate
        if(ishape == 0)
            soln_coeff_int[istate].resize(n_shape_fns_int);

        //solve
        soln_coeff_int[istate][ishape] = dg->solution(dof_indices_int[idof]);
        //allocate
        if(ishape == 0)
            soln_coeff_ext[istate].resize(n_shape_fns_ext);

        //solve
        soln_coeff_ext[istate][ishape] = dg->solution(dof_indices_ext[idof]);
    }
    //Interpolate soln modal coefficients to the facet
    std::array<std::vector<double>,nstate> soln_at_surf_q_int;
    std::array<std::vector<double>,nstate> soln_at_surf_q_ext;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     soln_coeff_int[istate], soln_at_surf_q_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     soln_coeff_ext[istate], soln_at_surf_q_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);
    }

    //evaluate physical facet fluxes dot product with physical unit normal scaled by determinant of metric facet Jacobian
    //the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> surf_num_flux_int_dot_normal;
    std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> surf_num_flux_ext_dot_normal;
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        //Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        //The way it is stored in metric_operators is to use sum-factorization in each direction,
        //but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        //Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        //This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,double> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad];
            }
        }
        //numerical fluxes
        dealii::Tensor<1,dim,double> unit_phys_normal_int;
        metric_oper_int.transform_reference_to_physical(unit_ref_normal_int,
                                                        metric_cofactor_surf,
                                                        unit_phys_normal_int);
        const double face_Jac_norm_scaled = unit_phys_normal_int.norm();
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it. 

        std::array<double,nstate> diss_soln_num_flux;
        std::array<double,nstate> soln_state_int;
        std::array<double,nstate> soln_state_ext;
        for(int istate=0; istate<nstate; istate++){
            soln_state_int[istate] = soln_at_surf_q_int[istate][iquad];
            soln_state_ext[istate] = soln_at_surf_q_ext[istate][iquad];
        }

        for(int istate=0; istate<nstate; istate++){
            diss_soln_num_flux[istate] = 0.5 * ( soln_state_int[istate] + soln_state_ext[istate]);
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    surf_num_flux_int_dot_normal[istate][idim].resize(n_face_quad_pts);
                    surf_num_flux_ext_dot_normal[istate][idim].resize(n_face_quad_pts);
                }
                //solve
                surf_num_flux_int_dot_normal[istate][idim][iquad]
                    = (diss_soln_num_flux[istate]) * unit_phys_normal_int[idim] * face_Jac_norm_scaled;

                //for the external I compare it to the strong form's external
                surf_num_flux_ext_dot_normal[istate][idim][iquad]
                    = (diss_soln_num_flux[istate] - soln_at_surf_q_ext[istate][iquad]) * (- unit_phys_normal_int[idim]) * face_Jac_norm_scaled;
            }
        }
    }

    //solve residual and set
    const std::vector<double> &surf_quad_weights = dg->face_quadrature_collection[poly_degree_int].get_weights();
    for(int istate=0; istate<nstate; istate++){
        for(int idim=0; idim<dim; idim++){
            std::vector<double> rhs_int(n_shape_fns_int);

            soln_basis_int.inner_product_surface_1D(iface, 
                                                surf_num_flux_int_dot_normal[istate][idim],
                                                surf_quad_weights, rhs_int,
                                                soln_basis_int.oneD_surf_operator,
                                                soln_basis_int.oneD_vol_operator,
                                                true, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln

            for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
                local_auxiliary_RHS_int[istate*n_shape_fns_int + ishape][idim] += rhs_int[ishape]; 
            }
            std::vector<double> rhs_ext(n_shape_fns_ext);

            soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                surf_num_flux_ext_dot_normal[istate][idim],
                                                surf_quad_weights, rhs_ext,
                                                soln_basis_ext.oneD_surf_operator,
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln

            for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
                local_auxiliary_RHS_ext[istate*n_shape_fns_ext + ishape][idim] += rhs_ext[ishape]; 
            }
        }
    }

}

int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);

    all_parameters_new.use_curvilinear_split_form=true;

    double left = 0.0;
    double right = 1.0;
    const unsigned int igrid_start = 2;
    const unsigned int n_grids = 3;
    const unsigned int final_poly_degree = 4;
    for(unsigned int poly_degree = 3; poly_degree<final_poly_degree; poly_degree++){
        const unsigned int grid_degree = 1;
        for(unsigned int igrid=igrid_start; igrid<n_grids; ++igrid){
            pcout<<" Grid Index"<<igrid<<std::endl;
            //Generate a standard grid

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
            using Triangulation = dealii::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
            using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                MPI_COMM_WORLD,
                typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
//straight
            dealii::GridGenerator::hyper_cube(*grid, left, right, true);
#if PHILIP_DIM==1
            std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
            dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
            grid->add_periodicity(matched_pairs);
#else
	    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
		dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
                if(dim >= 2)
		dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
                if(dim>=3)
		dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
		grid->add_periodicity(matched_pairs);
#endif
            grid->refine_global(igrid);
            pcout<<" made grid for Index"<<igrid<<std::endl;
             
            //setup DG
            using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
            using ODE_enum = Parameters::ODESolverParam::ODESolverEnum;
            //choose NS equations
            all_parameters_new.pde_type = PDE_enum::navier_stokes;
            all_parameters_new.use_weak_form = false;
            all_parameters_new.use_periodic_bc = true;
            all_parameters_new.ode_solver_param.ode_solver_type = ODE_enum::explicit_solver;//auxiliary only works explicit for now
            all_parameters_new.use_inverse_mass_on_the_fly = true;
            std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
            dg->allocate_system (false,false,false);
            if(!all_parameters_new.use_inverse_mass_on_the_fly){
                dg->evaluate_mass_matrices(true);
            }
    
            //set solution as some random number between [1e-8,30] at each dof
            //loop over cells as to write only to local solution indices
            const unsigned int n_dofs = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
            std::vector<dealii::types::global_dof_index> current_dofs_indices(n_dofs);
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
                for(unsigned int i=0; i<n_dofs; i++){
                    dg->solution[current_dofs_indices[i]] = 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8)));
                }
            }

            const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
            const unsigned int n_quad_pts  = dg->volume_quadrature_collection[poly_degree].size();
             
            const dealii::FESystem<dim> &fe_metric = (dg->high_order_grid->fe_system);
            const unsigned int n_metric_dofs = fe_metric.dofs_per_cell; 
            const unsigned int n_grid_nodes = n_metric_dofs / dim;
            auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
             
            //build 1D reference operators
            PHiLiP::OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(dg->nstate, poly_degree, 1);
            mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
            mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, dg->oneD_quadrature_collection[poly_degree], dg->oneD_face_quadrature);
             
            PHiLiP::OPERATOR::basis_functions<dim,2*dim> basis(dg->nstate, dg->max_degree, dg->max_grid_degree);
            basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
            basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
            basis.build_1D_surface_operator(dg->oneD_fe_collection_1state[dg->max_degree], dg->oneD_face_quadrature);

            PHiLiP::OPERATOR::basis_functions<dim,2*dim> flux_basis(dg->nstate, dg->max_degree, dg->max_grid_degree);
            flux_basis.build_1D_volume_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
            flux_basis.build_1D_gradient_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
            flux_basis.build_1D_surface_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_face_quadrature);
             
            //loop over cells and compare rhs strong versus rhs weak
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
                if (!current_cell->is_locally_owned()) continue;
            
                //get mapping support points
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
             
                //build volume metric operators
                PHiLiP::OPERATOR::metric_operators<double,dim,2*dim> metric_oper(dg->nstate,poly_degree,1,true);
                metric_oper.build_volume_metric_operators(n_quad_pts, n_grid_nodes,
                                                          mapping_support_points,
                                                          mapping_basis);
             
                std::vector<dealii::types::global_dof_index> current_dofs_indices;
                std::vector<dealii::types::global_dof_index> neighbor_dofs_indices;
                current_dofs_indices.resize(n_dofs_cell);
                current_cell->get_dof_indices (current_dofs_indices);
                const dealii::types::global_dof_index current_cell_index = current_cell->active_cell_index();

                std::vector<dealii::Tensor<1,dim,real>> rhs_strong(n_dofs_cell);
                std::vector<dealii::Tensor<1,dim,real>> rhs_weak(n_dofs_cell);

                //assemble DG strong rhs auxiliary
                dg->assemble_volume_term_auxiliary_equation (
                    current_dofs_indices,
                    poly_degree,
                    basis,
                    flux_basis,
                    metric_oper,
                    rhs_strong);
                //assemble weak DG auxiliary eq
                assemble_weak_auxiliary_volume<PHILIP_DIM,PHILIP_DIM+2>(
                    dg,
                    current_dofs_indices,
                    poly_degree,
                    basis,
                    metric_oper,
                    rhs_weak);

                //loop over faces
                for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
                    const auto neighbor_cell = current_cell->neighbor_or_periodic_neighbor(iface);
                    std::vector<dealii::Tensor<1,dim,real>> rhs_ext_strong(n_dofs_cell);
                    std::vector<dealii::Tensor<1,dim,real>> rhs_ext_weak(n_dofs_cell);

                    //get facet metric operators
                    metric_oper.build_facet_metric_operators(
                        iface,
                        dg->face_quadrature_collection[poly_degree].size(),
                        n_grid_nodes,
                        mapping_support_points,
                        mapping_basis,
                        false);

                    const unsigned int neighbor_iface = current_cell->periodic_neighbor_of_periodic_neighbor(iface);
                    neighbor_dofs_indices.resize(n_dofs_cell);
                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);
                    const dealii::types::global_dof_index neighbor_cell_index = neighbor_cell->active_cell_index();
                     
                    //evaluate facet auxiliary RHS
                    dg->assemble_face_term_auxiliary (
                        iface, neighbor_iface, 
                        current_cell_index, neighbor_cell_index,
                        poly_degree, poly_degree,
                        current_dofs_indices, neighbor_dofs_indices,
                        basis, basis,
                        metric_oper,
                        rhs_strong, rhs_ext_strong);
                     
                    const unsigned int n_face_quad_pts = dg->face_quadrature_collection[poly_degree].size();//assume interior cell does the work
                    //assemble facet auxiliary WEAK DG RHS
                    //note that for the ext rhs, this function will return the DG strong 
                    //facet rhs in rhs_ext_weak to directly compare to the above's neighbour
                    assemble_face_term_auxiliary_weak<PHILIP_DIM,PHILIP_DIM+2> (
                        dg,
                        iface, neighbor_iface, 
                        current_cell_index, neighbor_cell_index,
                        poly_degree, poly_degree,
                        n_dofs_cell, n_dofs_cell,
                        n_face_quad_pts,
                        current_dofs_indices, neighbor_dofs_indices,
                        basis, basis,
                        metric_oper,
                        rhs_weak, rhs_ext_weak);

                    for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                        for(int idim=0; idim<dim; idim++){
                            if(std::abs(rhs_ext_strong[idof][idim]-rhs_ext_weak[idof][idim])>1e-13){
                                pcout<<"The strong external cell face RHS is not correct."<<std::endl;
                                return 1;
                            }
                        }
                    }

    
                    
                }//end of face loop

                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    for(int idim=0; idim<dim; idim++){
                        if(std::abs(rhs_strong[idof][idim]-rhs_weak[idof][idim])>1e-13){
                            pcout<<"The strong and weak RHS are not equivalent interior cell."<<std::endl;
                            return 1;
                        }
                    }
                }
            }//end of cell loop

        }//end of grid loop

    }//end poly degree loop
    pcout<<"The weak and strong auxiliary RHS are equivalent."<<std::endl;
    return 0;
}

//}//end PHiLiP namespace

