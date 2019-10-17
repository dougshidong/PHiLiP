#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/solution_transfer.h>

//#include "assert_compare_array.h"
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "testing/tests.h"
#include "testing/grid_study.h"
#include "physics/manufactured_solution.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

//#include <deal.II/lac/solver_control.h>
//#include <deal.II/lac/trilinos_precondition.h>
//#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

//#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/fe/mapping_q.h> // Might need mapping_q
#include <deal.II/fe/mapping_manifold.h> 
#include <deal.II/fe/mapping_fe_field.h> 

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>


const double TOLERANCE = 1E-6;
using namespace std;

void get_projection_operator(
                const dealii::FEValues<PHILIP_DIM,PHILIP_DIM> &fe_values_volume, unsigned int n_quad_pts,
                 unsigned int n_dofs_cell, dealii::FullMatrix<double> &projection_matrix)
{

    using real = double;

        dealii::FullMatrix<real> local_mass_matrix(n_dofs_cell);
        dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
        dealii::FullMatrix<real> local_vandermonde(n_quad_pts, n_dofs_cell);
        dealii::FullMatrix<real> local_weights(n_quad_pts);
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_volume.get_fe().system_to_component_index(itest).first;
            for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {
                const unsigned int istate_trial = fe_values_volume.get_fe().system_to_component_index(itrial).first;
                real value = 0.0;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    value +=
                        fe_values_volume.shape_value_component(itest,iquad,istate_test)
                        * fe_values_volume.shape_value_component(itrial,iquad,istate_trial)
                        * fe_values_volume.JxW(iquad);
                }
                local_mass_matrix[itrial][itest] = 0.0;
                local_mass_matrix[itest][itrial] = 0.0;
                if(istate_test==istate_trial) { 
                    local_mass_matrix[itrial][itest] = value;
                    local_mass_matrix[itest][itrial] = value;
                }
            }
        }
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_volume.get_fe().system_to_component_index(itest).first;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                        local_vandermonde[iquad][itest] = fe_values_volume.shape_value_component(itest,iquad,istate_test);
                }
        }
        for( unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for( unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                local_weights[iquad][iquad2] = 0.0;
                if(iquad==iquad2)
                    local_weights[iquad][iquad2] = fe_values_volume.JxW(iquad);
            }
        }

        local_inverse_mass_matrix.invert(local_mass_matrix);
        dealii::FullMatrix<real> V_trans_W(n_dofs_cell, n_quad_pts);

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            for( unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                V_trans_W[itest][iquad] = local_vandermonde[iquad][itest] * fe_values_volume.JxW(iquad);
            }
        }


    for (unsigned int idof=0; idof<n_dofs_cell; idof++){
        for (unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            projection_matrix[idof][iquad] = 0.0;
            for (unsigned int idof2=0; idof2<n_dofs_cell; idof2++){
                projection_matrix[idof][iquad] += local_inverse_mass_matrix[idof][idof2] * V_trans_W[idof2][iquad];
            }
        }
    } 


}
int main (int /*argc*/, char * /*argv*/[])
{

    using real = double;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = dim;
    const unsigned int n_grids = 6;
    std::array<double,n_grids> grid_size;
    std::array<double,n_grids> grid_size_flux;
    std::array<double,n_grids> l2_error_poly;
    std::array<double,n_grids> l2_error_projected;
    std::array<double,n_grids> slope;
    std::array<double,n_grids> slope_proj;

    std::array<double,n_grids> l2_error_div;
    std::array<double,n_grids> l2_error_proj_div;
    std::array<double,n_grids> slope_div;
    std::array<double,n_grids> slope_proj_div;

    unsigned int poly_degree = 2;
    using ADtype = double;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    MPI_Comm mpi_communicator;

    std::vector<int> n_1d_cells(n_grids);
    n_1d_cells[0] =1;
    for (unsigned int igrid=1;igrid<n_grids;++igrid) {
        n_1d_cells[igrid] = static_cast<int>(n_1d_cells[igrid-1]*1.5) + 2;
    }

    for (unsigned int igrid=0; igrid<n_grids; ++igrid) {
        // Note that Triangulation must be declared before DG
        // DG will be destructed before Triangulation
        // thus removing any dependence of Triangulation and allowing Triangulation to be destructed
        // Otherwise, a Subscriptor error will occur
//#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
        dealii::Triangulation<dim> grid(
            typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
//#else
#if 0
        dealii::parallel::distributed::Triangulation<dim> grid(
            mpi_communicator,
            typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

        // Generate hypercube

            dealii::GridGenerator::subdivided_hyper_cube(grid, n_1d_cells[igrid]);
            for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
                // Set a dummy boundary ID
                cell->set_material_id(9002);
                for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                    if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
                }
            }


        //build the fe_collections
            dealii::hp::FECollection<dim> fe_collection;
            dealii::hp::FECollection<dim> fe_collection_flux;

            dealii::hp::QCollection<dim> volume_quadrature_collection;
            dealii::hp::QCollection<dim> volume_quadrature_collection_flux;

            const dealii::FE_DGQ<dim> fe_dg(poly_degree);
            const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
            fe_collection.push_back (fe_system);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (poly_degree+1);
            volume_quadrature_collection.push_back(vol_quad_Gauss_Legendre);

            const dealii::FE_DGQ<dim> fe_dg_flux(poly_degree+1);
            const dealii::FESystem<dim,dim> fe_system_flux(fe_dg_flux, nstate);
            fe_collection_flux.push_back (fe_system_flux);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre_flux (poly_degree+2);
            volume_quadrature_collection_flux.push_back(vol_quad_Gauss_Legendre_flux);

        //allocate a dof_handler, solution, and projected solution
            dealii::hp::DoFHandler<dim> dof_handler;
            dealii::IndexSet locally_owned_dofs;
            dealii::IndexSet ghost_dofs;
            dealii::IndexSet locally_relevant_dofs;
            dealii::LinearAlgebra::distributed::Vector<double> solution_projected;
            dealii::LinearAlgebra::distributed::Vector<double> solution;

            dof_handler.initialize(grid, fe_collection_flux);

            dof_handler.distribute_dofs(fe_collection_flux);
            locally_owned_dofs = dof_handler.locally_owned_dofs();
            dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, ghost_dofs);
            locally_relevant_dofs = ghost_dofs;
            ghost_dofs.subtract_set(locally_owned_dofs);
            solution_projected.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);

            const unsigned int max_dofs_per_cell_flux = dof_handler.get_fe_collection().max_dofs_per_cell();
            std::vector<dealii::types::global_dof_index> current_dofs_indices_flux(max_dofs_per_cell_flux);

            dof_handler.initialize(grid, fe_collection);

            dof_handler.distribute_dofs(fe_collection);
            locally_owned_dofs = dof_handler.locally_owned_dofs();
            dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, ghost_dofs);
            locally_relevant_dofs = ghost_dofs;
            ghost_dofs.subtract_set(locally_owned_dofs);
            solution.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);

            const dealii::FE_DGQ<dim> fe(poly_degree);
            const dealii::FESystem<dim,dim> fe_sys(fe_dg, nstate);

            dealii::DoFHandler<dim> dof_handler_grid;
            dof_handler_grid.initialize(grid, fe_sys);
            dof_handler_grid.distribute_dofs(fe_sys);
            #if PHILIP_DIM==1
            using VectorType = dealii::Vector<double>;
            VectorType nodes;
            nodes.reinit(dof_handler.n_dofs());
            using DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>;
            #else
            using  VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
            VectorType nodes;
            nodes.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
            using  DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>;
            #endif
            const dealii::ComponentMask mask(dim, true);
            dealii::VectorTools::get_position_vector(dof_handler_grid, nodes,mask);
            nodes.update_ghost_values();
            const auto mapping =  dealii::MappingFEField<dim, dim, VectorType,DoFHandlerType> (dof_handler_grid, nodes, mask);


            const dealii::hp::MappingCollection<dim> mapping_collection (mapping);

            dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points); ///< FEValues of volume.

//solution basis functions evaluated at flux volume nodes
            dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_soln_flux (mapping_collection, fe_collection, volume_quadrature_collection_flux, dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points); ///< FEValues of volume.

            dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_flux (mapping_collection, fe_collection_flux, volume_quadrature_collection_flux, dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points); ///< FEValues of volume.


            const unsigned int max_dofs_per_cell = dof_handler.get_fe_collection().max_dofs_per_cell();
            std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
            for (auto current_cell = dof_handler.begin_active(); current_cell!=dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
                const unsigned int mapping_index = 0;
                const unsigned int fe_index_curr_cell = current_cell->active_fe_index();
                const unsigned int quad_index = fe_index_curr_cell;
                const dealii::FESystem<dim,dim> &current_fe_ref = fe_collection[fe_index_curr_cell];
                const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

                // Obtain the mapping from local dof indices to global dof indices
                current_dofs_indices.resize(n_dofs_curr_cell);
                current_cell->get_dof_indices (current_dofs_indices);

                //FOR FLUX POINTS
                const dealii::FESystem<dim,dim> &current_fe_ref_flux = fe_collection_flux[fe_index_curr_cell];

                fe_values_collection_volume.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
                const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

                dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (current_cell);


                fe_values_collection_volume_flux.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);
                const dealii::FEValues<dim,dim> &fe_values_volume_flux = fe_values_collection_volume_flux.get_present_fe_values();

                fe_values_collection_volume_soln_flux.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);
                const dealii::FEValues<dim,dim> &fe_values_volume_soln_flux = fe_values_collection_volume_soln_flux.get_present_fe_values();

                const unsigned int n_quad_pts      = fe_values_volume.n_quadrature_points;
                const unsigned int n_dofs_cell     = fe_values_volume.dofs_per_cell;
                //FOR FLUX POINTS
                const unsigned int n_quad_pts_flux      = fe_values_volume_flux.n_quadrature_points;
                const unsigned int n_dofs_cell_flux     = fe_values_volume_flux.dofs_per_cell;

                current_dofs_indices_flux.resize(n_dofs_cell_flux);
                for (unsigned int idof_flux = 0; idof_flux<n_dofs_cell_flux; idof_flux++){
                    current_dofs_indices_flux[idof_flux] = (current_dofs_indices[0]/n_dofs_cell) * n_dofs_cell_flux + idof_flux;
                }


                dealii::FullMatrix<real> local_inverse_vandermonde(n_dofs_cell, n_quad_pts);
                get_projection_operator(fe_values_volume, n_quad_pts, n_dofs_cell, local_inverse_vandermonde);
                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    solution[current_dofs_indices[idof]]=0.0;
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        const dealii::Point<dim> qpoint = (fe_values_volume.quadrature_point(iquad));
                       for (int idim=0; idim<dim; idim++){
                        solution[current_dofs_indices[idof]] +=local_inverse_vandermonde[idof][iquad] *sin(2*qpoint[idim]); 
                        }
                    }   
                }

                
                dealii::FullMatrix<real> projection_matrix(n_dofs_cell_flux, n_quad_pts_flux);
                get_projection_operator(fe_values_volume_flux, n_quad_pts_flux, n_dofs_cell_flux, projection_matrix);
        

                std::vector< ADArray > soln_at_q_flux(n_quad_pts_flux);
                for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
                    for (int istate=0; istate<nstate; istate++) { 
                    // Interpolate solution to the flux volume quadrature points
                        soln_at_q_flux[iquad][istate]      = 0;
                    }
                }
                for (unsigned int iquad=0; iquad<n_quad_pts_flux; ++iquad) {
                    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
                        const unsigned int istate = fe_values_volume_soln_flux.get_fe().system_to_component_index(idof).first;
                        soln_at_q_flux[iquad][istate]      += solution[current_dofs_indices[idof]] * fe_values_volume_soln_flux.shape_value_component(idof, iquad, istate);
                    }
                }
                for (unsigned int idof=0; idof<n_dofs_cell_flux; idof++){
                     solution_projected[current_dofs_indices_flux[idof]] = 0.0;
                        for (unsigned int iquad=0; iquad<n_quad_pts_flux; iquad++){
                           const dealii::Point<dim> qpoint = (fe_values_volume_flux.quadrature_point(iquad));
                           for(int idim=0;idim<dim;idim++){
                                solution_projected[current_dofs_indices_flux[idof]] += projection_matrix[idof][iquad] * sin(2*qpoint[idim]);
                           }
                        }
                }
            }

            const unsigned int n_global_active_cells = grid.n_global_active_cells();
            const unsigned int n_dofs = dof_handler.n_dofs();
           std::cout << "Dimension: " << dim
                 << "\t Polynomial degree p: " << poly_degree
                 << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::hp::QCollection<dim> quadrature_collection_extra;
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
            quadrature_collection_extra.push_back (quad_extra);

            dealii::hp::FEValues<dim,dim> fe_values_extra_coll(mapping_collection, fe_collection, quadrature_collection_extra, dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points | dealii::update_gradients);
            dealii::hp::FEValues<dim,dim> fe_values_extra_proj_coll(mapping_collection, fe_collection_flux, quadrature_collection_extra, dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points | dealii::update_gradients);

            std::array<double,nstate> soln_at_q;
            std::array<double,nstate> soln_at_q_proj;

            ADArrayTensor1 soln_div;
            ADArrayTensor1 soln_proj_div;

            double l2error = 0.0;
            double l2error_proj = 0.0;

            double l2error_div = 0.0;
            double l2error_proj_div = 0.0;

            // Integrate solution error and output error

            std::vector<dealii::types::global_dof_index> dofs_indices (max_dofs_per_cell);
            for (auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;

                const unsigned int mapping_index = 0;
                const unsigned int fe_index_curr_cell = cell->active_fe_index();
                const unsigned int quad_index = fe_index_curr_cell;

                fe_values_extra_coll.reinit (cell, quad_index, mapping_index, fe_index_curr_cell);

                dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (cell);
                fe_values_extra_proj_coll.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);

                const dealii::FEValues<dim,dim> &fe_values_extra = fe_values_extra_coll.get_present_fe_values();
                const dealii::FEValues<dim,dim> &fe_values_extra_proj = fe_values_extra_proj_coll.get_present_fe_values();
                cell->get_dof_indices (dofs_indices);
                const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;

                const unsigned int n_dofs_cell          = fe_values_extra.dofs_per_cell;
                const unsigned int n_dofs_cell_flux     = fe_values_extra_proj.dofs_per_cell;
                current_dofs_indices_flux.resize(n_dofs_cell_flux);
                for (unsigned int idof_flux = 0; idof_flux<n_dofs_cell_flux; idof_flux++){
                    current_dofs_indices_flux[idof_flux] = (dofs_indices[0]/n_dofs_cell) * n_dofs_cell_flux + idof_flux;
                }

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                    std::fill(soln_at_q_proj.begin(), soln_at_q_proj.end(), 0.0);

                    std::fill(soln_div.begin(), soln_div.end(), 0.0);
                    std::fill(soln_proj_div.begin(), soln_proj_div.end(), 0.0);

                    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }
                    for (unsigned int idof=0; idof<n_dofs_cell_flux; ++idof) {
                        const unsigned int istate = fe_values_extra_proj.get_fe().system_to_component_index(idof).first;
                        soln_at_q_proj[istate] += solution_projected[current_dofs_indices_flux[idof]] * fe_values_extra_proj.shape_value_component(idof, iquad, istate);
                    }

                    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_div[istate] += solution[dofs_indices[idof]] * fe_values_extra.shape_grad_component(idof, iquad, istate);
                    }
                    for (unsigned int idof=0; idof<n_dofs_cell_flux; ++idof) {
                        const unsigned int istate = fe_values_extra_proj.get_fe().system_to_component_index(idof).first;
                        soln_proj_div[istate] += solution_projected[current_dofs_indices_flux[idof]] * fe_values_extra_proj.shape_grad_component(idof, iquad, istate);
                    }



                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                    for (int istate=0; istate<nstate; ++istate) {
                        double uexact = 0.0;
                        std::array<double,dim> uexact_grad;
                        for(int idim=0; idim<dim; idim++)
                            uexact_grad[idim] = 0.0;
                       for (int idim=0; idim<dim; idim++){
                        uexact += sin(2 * qpoint[idim]);
                        uexact_grad[idim] += 2 * cos(2 * qpoint[idim]);
                        }
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                        l2error_proj += pow(soln_at_q_proj[istate] - uexact, 2) * fe_values_extra_proj.JxW(iquad);
                        for(int idim=0; idim<dim;idim++){
                            l2error_div += pow(soln_div[istate][idim] - uexact_grad[idim], 2) * fe_values_extra.JxW(iquad);
                            l2error_proj_div += pow(soln_proj_div[istate][idim] - uexact_grad[idim], 2) * fe_values_extra_proj.JxW(iquad);
                        }
                    }
                }

            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));
            const double l2error_mpi_sum_proj = std::sqrt(dealii::Utilities::MPI::sum(l2error_proj, mpi_communicator));

            const double l2error_mpi_sum_div = std::sqrt(dealii::Utilities::MPI::sum(l2error_div, mpi_communicator));
            const double l2error_mpi_sum_proj_div = std::sqrt(dealii::Utilities::MPI::sum(l2error_proj_div, mpi_communicator));

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            const double dx_flux = 1.0/pow(max_dofs_per_cell_flux * n_global_active_cells,(1.0/dim));
           // const double dx_flux =dx; 
            grid_size[igrid] = dx;
            grid_size_flux[igrid] = dx_flux;
            l2_error_poly[igrid] = l2error_mpi_sum;
            l2_error_projected[igrid] = l2error_mpi_sum_proj;

            l2_error_div[igrid] = l2error_mpi_sum_div;
            l2_error_proj_div[igrid] = l2error_mpi_sum_proj_div;
           // output_error[igrid] = std::abs(solution_integral - exact_solution_integral);

//            convergence_table.add_value("p", poly_degree);
 //           convergence_table.add_value("cells", n_global_active_cells);
  //          convergence_table.add_value("DoFs", n_dofs);
   //         convergence_table.add_value("dx", dx);
    //        convergence_table.add_value("soln_L2_error", l2error_mpi_sum);


           std::cout << " Grid size h: " << dx 
                 << " L2-soln_error: " << l2error_mpi_sum
                 << " L2-soln_error_projected: " << l2error_mpi_sum_proj
                 << std::endl;

            if (igrid > 0) {
                const double slope_soln_err = log(l2_error_poly[igrid]/l2_error_poly[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                slope[igrid] = slope_soln_err;
                const double slope_soln_err_div = log(l2_error_div[igrid]/l2_error_div[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                slope_div[igrid] = slope_soln_err_div;
                std::cout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  solution_error1 " << l2_error_poly[igrid-1]
                     << "  solution_error2 " << l2_error_poly[igrid]
                     << "  slope " << slope_soln_err
                     << std::endl;
            }
            if (igrid > 0) {
                const double slope_soln_err_proj = log(l2_error_projected[igrid]/l2_error_projected[igrid-1])
                                      / log(grid_size_flux[igrid]/grid_size_flux[igrid-1]);
                slope_proj[igrid] = slope_soln_err_proj;
                const double slope_soln_err_proj_div = log(l2_error_proj_div[igrid]/l2_error_proj_div[igrid-1])
                                      / log(grid_size_flux[igrid]/grid_size_flux[igrid-1]);
                slope_proj_div[igrid] = slope_soln_err_proj_div;
                std::cout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree+1
                     << std::endl
                     << "  Projected solution_error1 " << l2_error_projected[igrid-1]
                     << " Projected solution_error2 " << l2_error_projected[igrid]
                     << " Projected slope " << slope_soln_err_proj
                     << std::endl;
            }

        }
        slope[0]=0;
        slope_proj[0]=0;
        printf("\nConvergence Summary\n");
        fflush(stdout);
        printf("dx | dx projected | l2 soln error | l2 projected error | slope solution | slope projected\n");
        fflush(stdout);
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {
            printf(" %g    %g     %g        %g       %g        %g\n",grid_size[igrid], grid_size_flux[igrid],l2_error_poly[igrid],l2_error_projected[igrid],slope[igrid],slope_proj[igrid]);
            fflush(stdout);
        }
        slope_div[0]=0;
        slope_proj_div[0]=0;
        printf("\nDivergence\n");
        fflush(stdout);
        printf("dx | dx projected | l2 soln error | l2 projected error | slope solution | slope projected\n");
        fflush(stdout);
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {
            printf(" %g    %g     %g        %g       %g        %g\n",grid_size[igrid], grid_size_flux[igrid],l2_error_div[igrid],l2_error_proj_div[igrid],slope_div[igrid],slope_proj_div[igrid]);
            fflush(stdout);
        }
        


    return 0;
}

