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
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include "dg/dg_factory.hpp"
#include "operators/operators.h"
//#include <GCL_test.h>

#include <deal.II/grid/grid_out.h>
#include "mesh/gmsh_reader.hpp"
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/reference_cell.h>

const double TOLERANCE = 1E-6;
using namespace std;
//namespace PHiLiP {



int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    //dealii::ReferenceCell reference_cell = dealii::ReferenceCells::Hexahedron;// = dealii::internal::make_reference_cell_from_int(7);
    //dealii::ReferenceCell reference_cell =  dealii::ReferenceCell::get_hypercube(dim);
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    parameter_handler.enter_subsection("flow_solver");
     parameter_handler.enter_subsection("grid");
      parameter_handler.set("input_mesh_filename", "/home/nyaki/Codes_2/PHiLiP/build_release/tests/unit_tests/operator_tests/3D_Cube_18_Nodes");
      //std::string input_mesh_filename = parameter_handler.get("input_mesh_filename");
     parameter_handler.leave_subsection();
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("flow_solver");
     parameter_handler.enter_subsection("grid");
      {
          std::string input_mesh_filename = parameter_handler.get("input_mesh_filename");
      }
     parameter_handler.leave_subsection();
    parameter_handler.leave_subsection();
    //std::string input_mesh_filename = parameter_handler.get("input_mesh_filename");
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    std::string input_mesh_filename = all_parameters_new.flow_solver_param.input_mesh_filename+std::string(".msh");
    all_parameters_new.euler_param.parse_parameters (parameter_handler);

    std::shared_ptr< HighOrderGrid<dim, double> > high_order_grid = read_gmsh <dim, dim> (input_mesh_filename,false);

    bool det_Jac_neg = false;
    bool det_match = true;
    double largest_error = 1e-13;
    double relative_error = 1e-13;
    for(unsigned int poly_degree = 1; poly_degree<3; poly_degree++){
        //unsigned int poly_degree = 1;
        unsigned int grid_degree = 1;
        std::cout<<"\nPoly degree: "<<poly_degree<<"\n";
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, high_order_grid->triangulation);
        dg->allocate_system ();

        dealii::QGaussLobatto<1> grid_quad(grid_degree +1);
        const dealii::FE_DGQ<1> fe_grid(grid_degree);
        const dealii::FESystem<1,1> fe_sys_grid(fe_grid, nstate);
        dealii::QGauss<1> flux_quad(poly_degree +1);
        dealii::QGauss<0> flux_quad_face(poly_degree +1);

        PHiLiP::OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(nstate,poly_degree,grid_degree);
        mapping_basis.build_1D_shape_functions_at_grid_nodes(fe_sys_grid, grid_quad);
        mapping_basis.build_1D_shape_functions_at_flux_nodes(fe_sys_grid, flux_quad, flux_quad_face);

        const unsigned int n_quad_pts = pow(poly_degree+1,dim);
        const unsigned int n_face_quad_pts = pow(poly_degree+1,dim-1);

        const dealii::FESystem<dim> &fe_metric = (dg->high_order_grid->fe_system);
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell; 
        auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
        for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
            if (!current_cell->is_locally_owned()) continue;
        
            std::vector<dealii::types::global_dof_index> current_metric_dofs_indices(n_metric_dofs);
            metric_cell->get_dof_indices (current_metric_dofs_indices);
            std::array<std::vector<real>,dim> mapping_support_points;
            for(int idim=0; idim<dim; idim++){
                mapping_support_points[idim].resize(n_metric_dofs/dim);
            }
            // }
            dealii::QGaussLobatto<dim> vol_GLL(grid_degree +1);
            //dealii::QGauss<dim> vol_GL(grid_degree +1);
            for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
                //std::cout<<"igrid_node: "<<igrid_node<<"\n";
                for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
                    const real val = (dg->high_order_grid->volume_nodes[current_metric_dofs_indices[idof]]);
                    const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
                    mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate);
                    //std::cout<<"shape value component["<<idof<<"]["<<igrid_node<<"]: "<<fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate)<<"\n";
                    //std::cout<<"mapping_support_points["<<istate<<"]["<<igrid_node<<"]: "<<mapping_support_points[istate][igrid_node]<<"\n";
                    //std::cout<<"faces for vertex["<<igrid_node<<"]"<<dealii::ReferenceCell::faces_for_given_vertex(igrid_node)<<"\n";
                }
                //std::cout<<"\n";
            }
            // for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
            //     for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            //         const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            //         std::cout<<"mapping_support_points["<<istate<<"]["<<igrid_node<<"]: "<<mapping_support_points[istate][igrid_node]<<"\n";
            //     }
            //     std::cout<<"\n";
            // }
            // std::cout<<"\n";

            PHiLiP::OPERATOR::metric_operators<real,dim,2*dim> metric_oper(nstate,poly_degree,grid_degree);
            metric_oper.build_volume_metric_operators(
                n_quad_pts, n_metric_dofs/dim,
                mapping_support_points,
                mapping_basis,
                false);

            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                if(metric_oper.det_Jac_vol[iquad]<0)
                    det_Jac_neg = true;
            }
            dealii::FEValues<dim,dim> fe_values(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], dg->volume_quadrature_collection[poly_degree], dealii::update_JxW_values);
            dealii::FEFaceValues<dim,dim> fe_values_surf(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], dg->face_quadrature_collection[poly_degree], /*dealii::update_JxW_values,*/ dealii::update_jacobians);
            fe_values.reinit(current_cell);
            //fe_values_surf.reinit(current_cell);
            const std::vector<double> &quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();
            const std::vector<double> &quad_weights_surf = dg->face_quadrature_collection[poly_degree].get_weights();
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                if(std::abs(fe_values.JxW(iquad)/quad_weights[iquad] - metric_oper.det_Jac_vol[iquad])>1e-13)
                    det_match = false;
            }
            const dealii::types::global_dof_index current_cell_index = current_cell->active_cell_index();
            //std::cout<<"\ncurrent_cell_index: " << current_cell_index<<"\n";
            for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
                fe_values_surf.reinit(current_cell, iface);
                std::array<std::vector<real>,dim> mapping_support_points_corrected = mapping_support_points;
                // if(iface == 2){
                //     for(int idim=0; idim<dim; idim++){
                //         mapping_support_points_corrected[idim][1] = mapping_support_points[idim][4];
                //         mapping_support_points_corrected[idim][4] = mapping_support_points[idim][1];
                //         //mapping_support_points_corrected[idim][3] = mapping_support_points[idim][6];
                //         //mapping_support_points_corrected[idim][6] = mapping_support_points[idim][3];
                //     }
                // }
                // if(iface == 3){
                //     for(int idim=0; idim<dim; idim++){
                //         //mapping_support_points_corrected[idim][1] = mapping_support_points[idim][4];
                //         //mapping_support_points_corrected[idim][4] = mapping_support_points[idim][1];
                //         mapping_support_points_corrected[idim][3] = mapping_support_points[idim][6];
                //         mapping_support_points_corrected[idim][6] = mapping_support_points[idim][3];
                //     }
                // }
                metric_oper.build_facet_metric_operators(
                iface, n_face_quad_pts, n_metric_dofs/dim,
                mapping_support_points_corrected,
                mapping_basis,
                false);
                const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
                for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
                    double JXW = fe_values_surf.JxW(iquad);
                    dealii::Tensor<2,dim,real> metric_cofactor_surf;
                    for(int idim=0; idim<dim; idim++){
                        for(int jdim=0; jdim<dim; jdim++){
                            metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad];
                        }
                    }
                    dealii::Tensor<1,dim,real> unit_phys_normal_int;
                    metric_oper.transform_reference_to_physical(unit_ref_normal_int,
                                                                metric_cofactor_surf,
                                                                unit_phys_normal_int);
                    const double face_Jac_norm_scaled = unit_phys_normal_int.norm();
                    if(std::abs(/*fe_values_surf.JxW(iquad)*/JXW/quad_weights_surf[iquad] - face_Jac_norm_scaled/*metric_oper.det_Jac_surf[iquad]*/)>1e-13){
                        det_match = false;
                        std::cout<<"\ncurrent_cell_index: " << current_cell_index<<"\n";
                        std::cout<<"iface: " << iface<<"\n";
                        std::cout<<"iquad: " << iquad<<"\n";
                        std::cout<<"fe_values_surf.JxW/quad_weights_surf: "<<fe_values_surf.JxW(iquad)/quad_weights_surf[iquad]<<"\n";
                        std::cout<<"metric_oper.det_Jac_surf: "<< face_Jac_norm_scaled<<"\n\n";
                        // std::cout<<"fe_values_surf.jacobian: ";
                        // for(int idim=0; idim<dim; idim++){
                        //     std::cout<<fe_values_surf.jacobian(iquad)[idim]<<" ";
                        // }
                        // std::cout<<"\nmetric_oper.metric_Jacobian_vol_cubature: ";
                        // for(int idim=0; idim<dim; idim++){
                        //     for(int jdim=0; jdim<dim; jdim++){
                        //             std::cout<<metric_oper.metric_Jacobian_vol_cubature[idim][jdim][iquad]<<" ";
                        //     }
                        // }
                    }
                    std::cout<<"\ncurrent_cell_index: " << current_cell_index<<"\n";
                    std::cout<<"iface: " << iface<<"\n";
                    for(int idim=0; idim<dim; idim++){
                        for(int jdim=0; jdim<dim; jdim++){
                                if(std::abs(fe_values_surf.jacobian(iquad)[idim][jdim] - metric_oper.metric_Jacobian_vol_cubature[idim][jdim][iquad])>1e-13){
                                    std::cout<<"iquad: "<<iquad<<"\n";
                                    std::cout<<"idim: "<<idim<<"\n";
                                    std::cout<<"jdim: "<<jdim<<"\n";
                                    std::cout<<"fe_values_surf.jacobian: "<<fe_values_surf.jacobian(iquad)[idim][jdim]<<"\n";
                                    std::cout<<"metric_oper.metric_Jacobian_vol_cubature: "<<metric_oper.metric_Jacobian_vol_cubature[idim][jdim][iquad]<<"\n";
                                }
                        }
                    }
                    if(std::abs(fe_values_surf.JxW(iquad)/quad_weights_surf[iquad] - face_Jac_norm_scaled)>largest_error){
                        largest_error = std::abs(fe_values_surf.JxW(iquad)/quad_weights_surf[iquad] - face_Jac_norm_scaled);
                        relative_error = std::abs(fe_values_surf.JxW(iquad)/quad_weights_surf[iquad] - face_Jac_norm_scaled)/std::max(fe_values_surf.JxW(iquad)/quad_weights_surf[iquad],face_Jac_norm_scaled);
                    }
                    // auto current_face = current_cell->face(iface);
                    // if ((current_face->at_boundary()))
                    // {
                    //     const unsigned int boundary_id = current_face->boundary_id();
                    //     std::cout<<"\ncurrent_cell_index: " << current_cell_index<<"\n";
                    //     std::cout<<"iface: " << iface<<"\n";
                    //     std::cout<<"boundary_id: " << boundary_id<<"\n";
                    //     // for(unsigned int i_face_vertex = 0; i_face_vertex<4; i_face_vertex++){
                    //     //     unsigned int ivertex = dealii::ReferenceCell::internal::Info::Hex::standard_to_real_face_vertex(i_face_vertex, iface, true) 	
                    //     //     //dealii::Point<dim,real> ivertex = current_cell->reference_cell().face_vertex_location(iface, i_face_vertex);
                    //     //     //dealii::Point<dim,real> ivertex = face_vertex_location;
                    //     //     //dealii::Point<dim,real> ivertex = dealii::ReferenceCell<dim> face_vertex_location(iface, i_face_vertex);//reference_cell.face_vertex_location<dim>(iface, i_face_vertex);
                    //     //     std::cout<<"vertex["<<i_face_vertex<<"] location: " << ivertex <<"\n";
                    //     // }
                    // }
                }
            }
        }

    }//end poly degree loop

    if( det_Jac_neg){
        pcout<<" Metrics give negative determinant of Jacobian\n"<<std::endl;
        return 1;
    }
    if(!det_match){
        pcout<<"Determiannt of metric Jacobian not match dealii value"<<std::endl;
        pcout<<"Largest error: "<<largest_error<<std::endl;
        pcout<<"Relative error: "<<relative_error<<std::endl;
        return 1;
    }
    else{
        pcout<<" Metrics Satisfy Determinant Jacobian Condition\n"<<std::endl;
        return 0;
    }
}

//}//end PHiLiP namespace

