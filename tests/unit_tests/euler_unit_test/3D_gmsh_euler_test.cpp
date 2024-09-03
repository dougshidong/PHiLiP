#include <fenv.h> // catch nan
#include <random>
#include <set>
#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/numerics/vector_tools.h> // interpolate initial conditions

#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/grids/curved_periodic_grid.hpp"
#include "mesh/grids/wavy_periodic_grid.hpp"

#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/euler.h"
#include "dg/dg_factory.hpp"
#include "dg/dg.h"
#include "ode_solver/ode_solver_factory.h"

#include <deal.II/grid/grid_out.h>
#include "mesh/gmsh_reader.hpp"
#include <deal.II/grid/tria_accessor.h>

using namespace PHiLiP;

const bool NONCONFORMING = false;//true;
enum GridType { eccentric_hyper_shell, abe2015_wavy, naca0012 };
const GridType GRID_TYPE = eccentric_hyper_shell;
const double PERT_SIZE = 1.0e-5;
const int POLY_DEGREE_START = 0;
const int POLY_DEGREE_END = 4;
const int GRID_DEGREE_START = 1;
const int GRID_DEGREE_END = 5;
const int OVERINTEGRATION = 0;
// const unsigned int NX_CELL = 2;
// const unsigned int NY_CELL = 3;
// const unsigned int NZ_CELL = 4;

// double random_pert(double lower, double upper)
// {
//     double f = (double)rand() / RAND_MAX;
//     return lower + f * (upper - lower);
// }

// template<int dim>
// void perturb_high_order_grid ( std::shared_ptr < DGBase<dim, double> > dg, const double perturbation_size )
// {
//     const dealii::DoFHandler<dim> &DH_grid = dg->high_order_grid->dof_handler_grid;
//     const dealii::FESystem<dim,dim> &fe_grid = DH_grid.get_fe();
//     dealii::IndexSet locally_owned_dofs_grid = DH_grid.locally_owned_dofs();
//     const unsigned int dofs_per_cell = fe_grid.dofs_per_cell;
//     const unsigned int dofs_per_face = fe_grid.dofs_per_face;

//     std::vector<dealii::types::global_dof_index> dof_indices(fe_grid.dofs_per_cell);

//     for (auto cell = DH_grid.begin_active(); cell != DH_grid.end(); ++cell) {

//         if (!cell->is_locally_owned()) continue;

//         cell->get_dof_indices(dof_indices);

//         // Store boundary face dofs.
//         std::set<dealii::types::global_dof_index> boundary_face_dofs;
//         for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
//             if (cell->face(iface)->at_boundary()) {
//                 for (unsigned int idof_face=0; idof_face<dofs_per_face; ++idof_face) {
//                     unsigned int idof_cell = fe_grid.face_to_cell_index(idof_face, iface);
//                     boundary_face_dofs.insert(idof_cell);
//                 }
//             }
//         }

//         for (unsigned int idof=0; idof<dofs_per_cell; ++idof) {
//             const bool is_not_boundary_dof = (boundary_face_dofs.find(idof) == boundary_face_dofs.end());
//             if (is_not_boundary_dof) {
//                 const dealii::types::global_dof_index global_idof_index = dof_indices[idof];
//                 double pert = random_pert(-perturbation_size, perturbation_size);
//                 if (dim == 3) pert /= 3;
//                 dg->high_order_grid->volume_nodes[global_idof_index] += pert;
//             }
//         }

//     }
//     dg->high_order_grid->ensure_conforming_mesh();
// }

// template<int dim>
// void create_curved_grid (std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> grid, const GridType grid_type) {

//     using Triangulation = dealii::parallel::distributed::Triangulation<dim>;

//     if (grid_type == GridType::eccentric_hyper_shell) {
//         dealii::Point< dim > inner_center, outer_center;
//         for (int d=0; d<dim; ++d) {
//             inner_center[d] = 0.5;
//             outer_center[d] = 0.0;
//         }
//         const double 	inner_radius = 1.0;
//         const double 	outer_radius = 3.0;
//         const unsigned int n_cells = 0;
//         dealii::GridGenerator::eccentric_hyper_shell<dim>( *grid, inner_center, outer_center, inner_radius, outer_radius, n_cells);
//         for (typename Triangulation::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
//             for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
//                 if (cell->face(face)->at_boundary()) {
//                     cell->face(face)->set_boundary_id (1004); // riemann
//                     //cell->face(face)->set_boundary_id (1005); // farfield
//                 }
//             }
//         }
//         grid->refine_global();
//     }


//     if (grid_type == GridType::naca0012) {
//         if constexpr (dim == 2) {
//             grid->clear();
//             dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
//             airfoil_data.airfoil_type = "NACA";
//             airfoil_data.naca_id      = "0012";
//             airfoil_data.airfoil_length = 1.0;
//             airfoil_data.height         = 150.0; // Farfield radius.
//             airfoil_data.length_b2      = 150.0;
//             airfoil_data.incline_factor = 0.0;
//             airfoil_data.bias_factor    = 4.5;
//             airfoil_data.refinements    = 0;

//             airfoil_data.n_subdivision_x_0 = 15;
//             airfoil_data.n_subdivision_x_1 = 15;
//             airfoil_data.n_subdivision_x_2 = 15;
//             airfoil_data.n_subdivision_y = 15;

//             airfoil_data.airfoil_sampling_factor = 10000;

//             std::vector<unsigned int> n_subdivisions(dim);
//             n_subdivisions[0] = airfoil_data.n_subdivision_x_0 + airfoil_data.n_subdivision_x_1 + airfoil_data.n_subdivision_x_2;
//             n_subdivisions[1] = airfoil_data.n_subdivision_y;
//             Grids::naca_airfoil(*grid, airfoil_data);
//             for (typename Triangulation::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
//                 for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
//                     if (cell->face(face)->at_boundary()) {
//                         cell->face(face)->set_boundary_id (1004); // riemann
//                         //cell->face(face)->set_boundary_id (1005); // farfield
//                     }
//                 }
//             }
//         } else {
//             create_curved_grid (grid, GridType::eccentric_hyper_shell);
//         }
//     }
//     if (grid_type == GridType::abe2015_wavy) {
//         grid->clear();
//         std::vector<unsigned int> n_subdivisions(dim);
//         for (int d=0; d<dim; ++d) {
//             n_subdivisions[d] = 7;
//         }
//         Grids::wavy_grid_Abe_2015<dim>(*grid, n_subdivisions);
//         // Has periodic BC. Should not replace BC conditions.
//     }

//     if (NONCONFORMING) {
//         const int n_refine = 2;
//         for (int i=0; i<n_refine;i++) {
//             grid->prepare_coarsening_and_refinement();
//             unsigned int icell = 0;
//             for (auto cell = grid->begin_active(); cell!=grid->end(); ++cell) {
//                 if (!cell->is_locally_owned()) continue;
//                 icell++;
//                 if (icell < grid->n_active_cells()/7) {
//                     cell->set_refine_flag();
//                 }
//                 if (icell == 1 && dim == 3) {
//                     cell->set_refine_flag();
//                 }
//             }
//             grid->execute_coarsening_and_refinement();
//         }
//     }
// }

template <int dim, typename real, typename MeshType>
template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
bool DGBase<dim,real,MeshType>::current_cell_should_do_the_work (
    const DoFCellAccessorType1 &current_cell, 
    const DoFCellAccessorType2 &neighbor_cell) const
{
    if (neighbor_cell->has_children()) {
    // Only happens in 1D where neither faces have children, but neighbor has some children
    // Can't do the computation now since we need to query the children's DoF
        AssertDimension(dim,1);
        return false;
    } else if (neighbor_cell->is_ghost()) {
    // In the case the neighbor is a ghost cell, we let the processor with the lower rank do the work on that face
    // We cannot use the cell->index() because the index is relative to the distributed triangulation
    // Therefore, the cell index of a ghost cell might be different to the physical cell index even if they refer to the same cell
        return (current_cell->subdomain_id() < neighbor_cell->subdomain_id());
        //return true;
    } else {
    // Locally owned neighbor cell
        Assert(neighbor_cell->is_locally_owned(), dealii::ExcMessage("If not ghost, neighbor should be locally owned."));

        if (current_cell->index() < neighbor_cell->index()) {
        // Cell with lower index does work
            return true;
        } else if (neighbor_cell->index() == current_cell->index()) {
        // If both cells have same index
        // See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
        // then cell at the lower level does the work
            return (current_cell->level() < neighbor_cell->level());
        }
        return false;
    }
    Assert(0==1, dealii::ExcMessage("Should not have reached here. Somehow another possible case has not been considered when two cells have the same coarseness."));
    return false;
}

template<int dim>
int test()
{
    srand (1.0);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    //int test_error = 0;

    // using namespace PHiLiP;
    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    parameter_handler.set("pde_type", "euler");
    parameter_handler.set("use_weak_form", "false");
    parameter_handler.set("use_invariant_curl_form", "false");
    parameter_handler.set("conv_num_flux", "roe");
    parameter_handler.set("dimension", (long int)dim);
    parameter_handler.set("flux_nodes_type", "GLL");
    parameter_handler.set("use_split_form", "false");
    parameter_handler.enter_subsection("euler"); 
    parameter_handler.set("mach_infinity", 0.5);
    parameter_handler.set("angle_of_attack", 5.0);
    parameter_handler.set("side_slip_angle", 0.0);
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("navier_stokes");
    parameter_handler.set("reynolds_number_inf", 5000.0);
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("ODE solver");
    parameter_handler.set("ode_solver_type", "runge_kutta");
    parameter_handler.set("nonlinear_max_iterations", (long int) 1);
    double time_step = 1e-8;
    parameter_handler.set("initial_time_step", time_step);
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("flow_solver");
     parameter_handler.enter_subsection("grid");
      parameter_handler.set("input_mesh_filename", "NACA0012_Coarse_Slip_Wall");
     parameter_handler.leave_subsection();
    parameter_handler.leave_subsection();
    parameter_handler.enter_subsection("grid");
    {
        std::string input_mesh_filename = parameter_handler.get("input_mesh_filename");
    }
    parameter_handler.leave_subsection();
    Parameters::AllParameters param;
    param.parse_parameters (parameter_handler);
    std::string input_mesh_filename = param.flow_solver_param.input_mesh_filename+std::string(".msh");
    param.euler_param.parse_parameters (parameter_handler);


    std::shared_ptr< HighOrderGrid<dim, double> > high_order_grid = read_gmsh <dim, dim> (input_mesh_filename,false);

    // using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    // std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
    //     MPI_COMM_WORLD,
    //     typename dealii::Triangulation<dim>::MeshSmoothing(
    //         dealii::Triangulation<dim>::smoothing_on_refinement |
    //         dealii::Triangulation<dim>::smoothing_on_coarsening));

    // // create_curved_grid (grid, GRID_TYPE);
    // std::shared_ptr<dealii::parallel::distributed::Triangulation<2> > grid = std::make_shared<dealii::parallel::distributed::Triangulation<2> > (
    // MPI_COMM_WORLD);
    // dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    // airfoil_data.airfoil_type = "NACA";
    // airfoil_data.naca_id      = "0012";
    // airfoil_data.airfoil_length = 1.0;
    // airfoil_data.height         = 5.0;
    // airfoil_data.length_b2      = 10.0;
    // airfoil_data.incline_factor = 0.1;
    // airfoil_data.bias_factor    = 5.0; 
    // airfoil_data.refinements    = 0;
    // airfoil_data.n_subdivision_x_0 = 5;
    // airfoil_data.n_subdivision_x_1 = 4;
    // airfoil_data.n_subdivision_x_2 = 5;
    // airfoil_data.n_subdivision_y = 6;
    // airfoil_data.airfoil_sampling_factor = 10; 

    // dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);

    // // Set boundary type and design type
    // for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
    //     for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
    //         if (cell->face(face)->at_boundary()) {
    //             unsigned int current_id = cell->face(face)->boundary_id();
    //             if (current_id == 0 || current_id == 1 || current_id == 4 || current_id == 5) {
    //                 cell->face(face)->set_boundary_id (1004); // farfield
    //             } else {
    //                 cell->face(face)->set_boundary_id (1001); // wall
    //             }
    //         }
    //     }
    // }
    // std::shared_ptr<dealii::parallel::distributed::Triangulation<2> > grid_2D = grid;
    // std::shared_ptr<dealii::parallel::distributed::Triangulation<3>> grid_3D = std::make_shared<dealii::parallel::distributed::Triangulation<3>>(
    // MPI_COMM_WORLD
    // );
    // const unsigned int n_slices = 17;
    // const double height = 0.2;
    // const bool copy_manifold_ids = false;
    // //const std::vector<types::manifold_id> manifold_priorities = {};
    // dealii::GridGenerator::extrude_triangulation(*grid_2D, n_slices, height, *grid_3D, copy_manifold_ids);


    // //Loop through cells to define boundary id's (periodic) on the z plane. 
    // for (typename dealii::parallel::distributed::Triangulation<3>::active_cell_iterator cell = grid_3D->begin_active(); cell != grid_3D->end(); ++cell) {
    //     for (unsigned int face=0; face<dealii::GeometryInfo<3>::faces_per_cell; ++face) {
    //         if (cell->face(face)->at_boundary()) {
    //             unsigned int current_id = cell->face(face)->boundary_id();
    //             if (current_id == 1005) { //Dealii automatically assigns the next available number to the new boundaries when creating the 3D mesh. Thus, since the largest number used is 1005, it assigns 1006 and 1007 to the new boundaries. 
    //                 cell->face(face)->set_boundary_id (1006); //(2005); // z = 0 boundaries
    //             } else if(current_id == 1006){
    //                 cell->face(face)->set_boundary_id (1006); //(2006); // z = height boundaries
    //             }
    //         }
    //     }
    // }

    // // Periodic boundary parameters
    // // const bool periodic_x = false;
    // // const bool periodic_y = false;
    // // const bool periodic_z = true;
    // // const int x_periodic_1 = 0; 
    // // const int x_periodic_2 = 0;
    // // const int y_periodic_1 = 0; 
    // // const int y_periodic_2 = 0;
    // // const int z_periodic_1 = 2005; 
    // // const int z_periodic_2 = 2006;

    // // //Check for periodic boundary conditions and apply
    // // std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
    
    // // if (periodic_x) {
    // //     dealii::GridTools::collect_periodic_faces(*grid_3D, x_periodic_1, x_periodic_2, 0, matched_pairs);
    // // }

    // // if (periodic_y) {
    // //     dealii::GridTools::collect_periodic_faces(*grid_3D, y_periodic_1, y_periodic_2, 1, matched_pairs);
    // // }

    // // if (periodic_z) {
    // //     dealii::GridTools::collect_periodic_faces(*grid_3D, z_periodic_1, z_periodic_2, 2, matched_pairs);
    // // }

    // // if (periodic_x || periodic_y || periodic_z) {
    // //     grid_3D->add_periodicity(matched_pairs);
    // // }

    // //std::vector<dealii::types::global_dof_index> dof_indices(high_order_grid->fe_system.dofs_per_cell);
    // //pcout << "Number of cells: " << high_order_grid->n_active_cells() << std::endl;

    std::vector< std::vector<double> > residuals(10, std::vector<double>(10,1));

    // for (int POLY_DEGREE = POLY_DEGREE_START; POLY_DEGREE <= POLY_DEGREE_END; PtrueOLY_DEGREE++) {
    //     for (int GRID_DEGREE = GRID_DEGREE_START; GRID_DEGREE <= GRID_DEGREE_END; GRID_DEGREE++) {
    int POLY_DEGREE = 1;
    int GRID_DEGREE = 1;
    std::cout << " POLY_DEGREE : " << POLY_DEGREE
                << " OVERINTEGRATION : " << OVERINTEGRATION
                << " GRID_DEGREE : " << GRID_DEGREE << std::endl;

    std::cout << " Integration strength : " << 2*POLY_DEGREE+1 + OVERINTEGRATION
                << " INTEGRAND : " << POLY_DEGREE-1+2*(GRID_DEGREE-1) << std::endl;

    parameter_handler.set("overintegration", (long int) OVERINTEGRATION);
    //parameter_handler.set("overintegration", (long int) std::max(0,(POLY_DEGREE+2*GRID_DEGREE) - (2*POLY_DEGREE-1))+10);

    // Update param with new overintegration parameter.
    param.parse_parameters (parameter_handler);

    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, POLY_DEGREE, POLY_DEGREE, GRID_DEGREE, /*grid_3D*/high_order_grid->triangulation);
    dg->allocate_system ();
    
    //perturb_high_order_grid (dg, PERT_SIZE);
    //dg->high_order_grid->output_results_vtk(9999);

    //Initialize coarse grid solution with free-stream
    Physics::Euler<dim,dim+2,double> euler_physics_double = Physics::Euler<dim, dim+2, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    FreeStreamInitialConditions<dim,dim+2,double> initial_conditions(euler_physics_double);


    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

    dg->assemble_residual();

    //auto metric_cell = high_order_grid->dof_handler_grid.begin_active();
    const unsigned int n_soln_dofs = dg->fe_collection[POLY_DEGREE].dofs_per_cell;;//fe_soln.dofs_per_cell;
    for (auto soln_cell = dg->dof_handler.begin_active(); soln_cell != dg->dof_handler.end(); ++soln_cell/*,++metric_cell*/) {
        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        std::vector<dealii::types::global_dof_index> current_metric_dofs_indices;
        current_dofs_indices.resize(n_soln_dofs);
        //current_metric_dofs_indices.resize(n_soln_dofs);
        //metric_cell->get_dof_indices (current_metric_dofs_indices);
        soln_cell->get_dof_indices (current_dofs_indices);
        const dealii::types::global_dof_index current_cell_index = soln_cell->active_cell_index();
        std::cout<<"\ncurrent_cell_index: " << current_cell_index<<"\n";
        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
            if(soln_cell->face_orientation(iface)){
                std::cout<<"current_cell iface: "<<iface<<" has standard orientation!\n";
            } else {
                std::cout<<"current_cell iface: "<<iface<<" DOES NOT have standard orientation!\n";
            }
        }
        for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
            std::cout<<"right_hand_side["<<idof<<"]: "<<dg->right_hand_side(current_dofs_indices[idof])<<"\n";
            std::cout<<"dg solution["<<idof<<"]: "<<dg->solution(current_dofs_indices[idof])<<"\n";
            //std::cout<<"metric_right_hand_side["<<idof<<"]: "<<dg->right_hand_side(current_metric_dofs_indices[idof])<<"\n";
        }
        //std::cout<<dg->right_hand_side(current_dofs_indices[idof]);

        
        // std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // std::vector<dealii::types::global_dof_index> neighbor_dofs_indices;
        // current_dofs_indices.resize(n_soln_dofs);
        // soln_cell->get_dof_indices (current_dofs_indices);
        // for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
        //     auto current_face = soln_cell->face(iface);
        //     const dealii::types::global_dof_index current_cell_index = soln_cell->active_cell_index();
        //     std::cout<<"\ncurrent_cell_index: " << current_cell_index<<"\n";
        //     // std::cout<<"iface: " << iface<<"\n";
        //     if(soln_cell->face_orientation(iface)){
        //         std::cout<<"current_cell iface: "<<iface<<" has standard orientation!\n";
        //     } else {
        //         std::cout<<"current_cell iface: "<<iface<<" DOES NOT have standard orientation!\n";
        //     }
        //     if ((current_face->at_boundary() && !soln_cell->has_periodic_neighbor(iface)))
        //     {
        //         const unsigned int boundary_id = current_face->boundary_id();
        //         std::cout<<"\nboundary_id: " << boundary_id<<"\n";
        //         for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        //             std::cout<<"soln_coeff["<<idof<<"]: "<<dg->solution(current_dofs_indices[idof])<<"\n";
        //         }
        //     }
        //     else
        //     {
        //         const auto neighbor_cell = soln_cell->neighbor(iface);
        //         const dealii::types::global_dof_index neighbor_cell_index = neighbor_cell->active_cell_index();
        //         neighbor_dofs_indices.resize(n_soln_dofs);
        //         neighbor_cell->get_dof_indices (neighbor_dofs_indices);
        //         const unsigned int neighbor_iface = soln_cell->neighbor_face_no(iface);
        //         std::cout<<"\nneighbor_cell_index: " << neighbor_cell_index<<"\n";
        //         // std::cout<<"neighbor_iface " << neighbor_iface<<"\n";
        //         if(neighbor_cell->face_orientation(neighbor_iface)){
        //             std::cout<<"neighbor_cell iface: "<<neighbor_iface<<" has standard orientation!\n";
        //         } else {
        //             std::cout<<"neighbor_cell iface: "<<neighbor_iface<<" DOES NOT have standard orientation!\n";
        //         }
        //         for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        //             std::cout<<"soln_coeff_int["<<idof<<"]: "<<dg->solution(current_dofs_indices[idof])<<"\n";
        //             std::cout<<"soln_coeff_ext["<<idof<<"]: "<<dg->solution(neighbor_dofs_indices[idof])<<"\n";
        //         }   

        //     }
        // }
    }
    double residual_norm = dg->get_residual_l2norm();
    residuals[POLY_DEGREE][GRID_DEGREE] = residual_norm;
    pcout << "Residual norm: " << residual_norm << std::endl;
    if (residual_norm > 1e-13) {
        std::cout << "Freestream flow is not preserved" << std::endl;
        //dg->output_results_vtk(9999);
        //return 1;
    }

    dg->output_results_vtk(9000+GRID_DEGREE);

    //     }
    // }

    // std::cout << std::setprecision(6) << std::scientific;
    // std::cout << "S/G ";
    // for (int GRID_DEGREE = GRID_DEGREE_START; GRID_DEGREE <= GRID_DEGREE_END; GRID_DEGREE++) {
    //     std::cout << GRID_DEGREE << "            ";
    // }
    // std::cout << std::endl;
    // for (int POLY_DEGREE = POLY_DEGREE_START; POLY_DEGREE <= POLY_DEGREE_END; POLY_DEGREE++) {
    //     std::cout << POLY_DEGREE << "  ";
    //     for (int GRID_DEGREE = GRID_DEGREE_START; GRID_DEGREE <= GRID_DEGREE_END; GRID_DEGREE++) {
    //         std::cout << " " << residuals[POLY_DEGREE][GRID_DEGREE];
    //         if (GRID_DEGREE <= POLY_DEGREE+2 && residuals[POLY_DEGREE][GRID_DEGREE] > 1e-14) {
    //             test_error += 1;
    //         }
    //     }
    //     std::cout << std::endl;
    // }

    // if (test_error) {
    //     std::cout << "Freestream preservation is not satisfied for grids of order p_g <= p+2" << std::endl;
    // }

    return false;
}


int main (int argc, char * argv[])
{
#if !defined(__APPLE__)
    feenableexcept(FE_INVALID | FE_OVERFLOW); // catch nan
#endif
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    // std::string filename;
    // for (int i = 1; i < argc; i++) {
    //    std::string s(argv[i]);
    //    if (s.rfind("--input=", 0) == 0) {
    //        filename = s.substr(std::string("--input=").length());
    //        std::ifstream f(filename);
    //        std::cout << "iproc = " << mpi_rank << " : File " << filename;
    //        if (f.good()) std::cout << " exists" << std::endl;
    //        else std::cout << " not found" << std::endl;
    //        }
    //    else {
    //        std::cout << "iproc = " << mpi_rank << " : Unknown: " << s << std::endl;
    //    }
    // }
    int test_error = false;
    try {
         test_error += test<PHILIP_DIM>();
    }
    catch (std::exception &exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }
    catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        throw;
    }

    return test_error;
}


