#include <fstream>

#include <deal.II/grid/grid_out.h>
#include "mesh/gmsh_reader.hpp"
#include <deal.II/grid/tria_accessor.h>
#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <ctime>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/qprojector.h>

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

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_fe_field.h> 

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "operators/operators.h"

const double TOLERANCE = 1E-6;
using namespace std;

// // generates fe_collection for dof_handler initialization
// template <int dim>
// dealii::hp::FECollection<dim> get_fe_collection(
//     const unsigned int max_degree,
//     const int nstate,
//     const bool use_collocated_nodes)
// {
//     dealii::hp::FECollection<dim> fe_coll;
    
//     // collocated nodes repeat degree = 1
//     unsigned int degree = use_collocated_nodes?1:0;
//     const dealii::FE_DGQ<dim> fe_dg(degree);
//     const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
//     fe_coll.push_back(fe_system);

//     // looping over remaining degrees
//     for(unsigned int degree = 1; degree <= max_degree; ++degree){
//         const dealii::FE_DGQ<dim> fe_dg(degree);
//         const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
//         fe_coll.push_back(fe_system);
//     }

//     // returning
//     return fe_coll;
// }

int main (/*int argc, char * argv[]*/)
{

    // dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    // const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    // using namespace PHiLiP;
    // using real = double;

    // std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    // const int dim = PHILIP_DIM;
    // const int nstate = 1;
    // dealii::ParameterHandler parameter_handler;
    // PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    // PHiLiP::Parameters::AllParameters all_parameters_new;
    // all_parameters_new.parse_parameters (parameter_handler);
    // all_parameters_new.nstate = nstate;
    // dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    // using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    // all_parameters_new.flux_reconstruction_type = FR_enum::cHU;

    // bool equiv = true;
    // bool sum_fact = true;
    // int fail_bool = false;

    // // std::string filename;
    // // for (int i = 1; i < argc; i++) {
    // //    std::string s(argv[i]);
    // //    if (s.rfind("--input=", 0) == 0) {
    // //        filename = s.substr(std::string("--input=").length());
    // //        std::ifstream f(filename);
    // //        std::cout << "iproc = " << mpi_rank << " : File " << filename;
    // //        if (f.good()) std::cout << " exists" << std::endl;
    // //        else std::cout << " not found" << std::endl;
    // //        }
    // //    else {
    // //        std::cout << "iproc = " << mpi_rank << " : Unknown: " << s << std::endl;
    // //    }
    // // }

    // // const bool do_renumber_dofs =  false;//true;
    // // std::shared_ptr< HighOrderGrid<dim, double> > high_order_grid = read_gmsh <dim, dim> (filename,do_renumber_dofs);


    // // high_order_grid->output_results_vtk(0);
    // unsigned int poly_degree=1;
    // const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);
    // dealii::QGauss<dim> vol_quad_dim (poly_degree+1);
    // const dealii::FE_DGQ<dim> fe_dim(poly_degree);
    // const dealii::FESystem<dim,dim> fe_system_dim(fe_dim, nstate);
    
    // dealii::QGauss<1> quad_1D (poly_degree+1);
    // const dealii::FE_DGQ<1> fe(poly_degree);
    // const dealii::FESystem<1,1> fe_system(fe, nstate);
    // PHiLiP::OPERATOR::basis_functions<dim,2*dim> basis_1D(nstate, poly_degree, 1);
    // basis_1D.build_1D_volume_operator(fe, quad_1D);
    // basis_1D.build_1D_gradient_operator(fe, quad_1D);
    // dealii::FullMatrix<double> basis_dim(n_dofs);
    // basis_dim = basis_1D.tensor_product(basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator);

    // // std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (this->mpi_communicator); // Mesh smoothing is set to none by default.
    // dealii::parallel::distributed::Triangulation<dim> grid(
    // MPI_COMM_WORLD,
    // typename dealii::Triangulation<dim>::MeshSmoothing(
    //     dealii::Triangulation<dim>::smoothing_on_refinement |
    //     dealii::Triangulation<dim>::smoothing_on_coarsening));
    // //const unsigned int number_of_refinements = 0;//this->all_param.flow_solver_param.number_of_mesh_refinements;
    // const double domain_left = 0;//this->all_param.flow_solver_param.grid_left_bound;
    // const double domain_right = 1;//this->all_param.flow_solver_param.grid_right_bound;
    // const bool colorize = true;
    
    // dealii::GridGenerator::hyper_cube(grid, domain_left, domain_right, colorize);
    // //dealii::hp::FECollection<dim> fe_collection = get_fe_collection<dim>(1, nstate, true);
    // dealii::DoFHandler<dim> dof_handler(grid);
    // dof_handler.initialize(grid, fe_system_dim);
    // dof_handler.distribute_dofs(fe_system_dim);

    // for(unsigned int idof=0; idof<n_dofs; idof++){
    //     for(unsigned int iquad=0; iquad<n_dofs; iquad++){
    //         dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
    //         if(fe_system_dim.shape_grad_component(idof,qpoint,0)[0] != basis_dim[iquad][idof])
    //             equiv = false;
    //     } 
    // } 
    // if(dim >= 2){
    //     basis_dim = basis_1D.tensor_product(basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator,basis_1D.oneD_vol_operator);
    //     for(unsigned int idof=0; idof<n_dofs; idof++){
    //         for(unsigned int iquad=0; iquad<n_dofs; iquad++){
    //             dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
    //             if(fe_system_dim.shape_grad_component(idof,qpoint,0)[1] != basis_dim[iquad][idof])
    //                 equiv = false;
    //         } 
    //     } 
    // }
    // if(dim >= 3){
    //     basis_dim = basis_1D.tensor_product(basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator);
    //     for(unsigned int idof=0; idof<n_dofs; idof++){
    //         for(unsigned int iquad=0; iquad<n_dofs; iquad++){
    //             dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
    //             if(fe_system_dim.shape_grad_component(idof,qpoint,0)[2] != basis_dim[iquad][idof])
    //                 equiv = false;
    //         } 
    //     } 
    // }
    // //const unsigned int n_face_quad_pts = face_quadrature_collection[poly_degree].size();

    // // std::vector<double> sol_hat(n_dofs);
    // // for(unsigned int idof=0; idof<n_dofs; idof++){
    // //     // sol_hat[idof] = 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8)));
    // //     sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8))) );
    // // }
    // // std::vector<double> sol_dim(n_dofs);
    // // for(unsigned int idof=0; idof<n_dofs; idof++){
    // //     sol_dim[idof] = 0.0;
    // //     for(unsigned int iquad=0; iquad<n_dofs; iquad++){
    // //         sol_dim[idof] += basis_dim[idof][iquad] * sol_hat[iquad];
    // //     }
    // // }
    // // std::vector<double> sol_sum_fact(n_dofs);
    // // if(dim==1)
    // //     basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator, basis_1D.oneD_vol_operator);
    // // if(dim==2)
    // //     basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator);
    // // if(dim==3)
    // //     basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_vol_operator, basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator);
    

    // // for(unsigned int idof=0; idof<n_dofs; idof++){
    // //     if(std::abs(sol_dim[idof] - sol_sum_fact[idof])>1e-12){
    // //         sum_fact = false;
    // //         pcout<<"sum fact wrong "<<sol_dim[idof]<<" "<<sol_sum_fact[idof]<<std::endl;
    // //     }
    // // }

    // // dealii::hp::FECollection<dim> fe_coll;
    // // fe_coll.push_back(fe_system_dim);
    // // dealii::DoFHandler<dim> dof_handler(*(high_order_grid->triangulation));
    // // dof_handler.initialize(*(high_order_grid->triangulation), fe_coll);
    // dealii::hp::QCollection<dim-1>   face_quadrature_collection;
    // const unsigned int n_face_quad_pts = 4;
    // for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    //     const int i_fele = cell->active_fe_index();
    //     const int i_quad = i_fele;
    //     const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad];
    //     //this->fe_collection[i_fele];
    //     for(unsigned int iface = 0; iface<6;iface++){
    //         const auto face_data_set = dealii::QProjector<dim>::DataSetDescriptor::face( 
    //                                                                                 dealii::ReferenceCell::get_hypercube(dim),
    //                                                                                 iface,
    //                                                                                 /*cell->face_orientation(iface)*/true,
    //                                                                                 cell->face_flip(iface),
    //                                                                                 cell->face_rotation(iface),
    //                                                                                 used_face_quadrature.size());
    //         dealii::Quadrature<dim> face_quadrature;
    //         if constexpr (dim < 3) {
    //         face_quadrature = dealii::QProjector<dim>::project_to_face(dealii::ReferenceCell::get_hypercube(dim),
    //                                                                     used_face_quadrature,
    //                                                                     iface);
    //         } else{
    //             const dealii::Quadrature<dim> all_faces_quad = dealii::QProjector<dim>::project_to_all_faces (dealii::ReferenceCell::get_hypercube(dim), used_face_quadrature);
    //                         std::vector< dealii::Point< dim >> points(n_face_quad_pts);
    //             std::vector< double > weights(n_face_quad_pts);
    //             for (unsigned int iquad = 0; iquad < n_face_quad_pts; ++iquad) {
    //                 points[iquad] = all_faces_quad.point(iquad+face_data_set);
    //                 weights[iquad] = all_faces_quad.weight(iquad+face_data_set);
    //             }
    //             face_quadrature = dealii::Quadrature<dim>(points, weights);           
    //         }
    //         const std::vector<dealii::Point<dim,double>> &unit_quad_pts = face_quadrature.get_points();
    //         // for (unsigned int idof = 0; idof < n_dofs; ++idof) {
    //         //     const unsigned int istate = this->fe_collection[i_fele].system_to_component_index(idof).first;
    //         //     const unsigned int ishape = this->fe_collection[i_fele].system_to_component_index(idof).second;
    //         //     for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
    //         //         std::cout<<"shape value int["<<idof<<"]: "<<this->fe_collection[i_fele].shape_value(idof,this->volume_quadrature_collection[poly_degree_int].point(iquad))<<"\n";
    //         //     }
    //         //     std::cout<<"\n";
    //         // }
    //     }
    // }


    // if (fail_bool) {
    //     pcout << "Test failed. The estimated error should be the same for a given p, even after refinement and translation." << std::endl;
    // } else {
    //     pcout << "Test successful." << std::endl;
    // }
    // if( equiv == false){
    //     pcout<<" Tensor product not recover original !"<<std::endl;
    //     return 1;
    // }
    // if(sum_fact == false){
    //     pcout<<" sum fcatorization not recover A*u"<<std::endl;
    //     return 1;
    // }
    // return fail_bool;
}
