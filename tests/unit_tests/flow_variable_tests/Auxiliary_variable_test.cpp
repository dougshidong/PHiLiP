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

#include <deal.II/base/convergence_table.h>

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
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);

   // all_parameters_new.use_collocated_nodes=true;
    all_parameters_new.use_curvilinear_split_form=true;
    const int dim_check = 0;

    //unsigned int poly_degree = 3;
    double left = 0.0;
    double right = 1.0;
//    const bool colorize = true;
    dealii::ConvergenceTable convergence_table;
    const unsigned int igrid_start = 2;
    const unsigned int n_grids = 7;
    std::array<double,n_grids> grid_size;
    std::array<double,n_grids> soln_error;
    std::array<double,n_grids> soln_error_inf;
    for(unsigned int poly_degree = 2; poly_degree<6; poly_degree++){
        unsigned int grid_degree = 1;
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
#else
	std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
		dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
                if(dim == 2)
		dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
                if(dim==3)
		dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
		grid->add_periodicity(matched_pairs);
#endif
        grid->refine_global(igrid);
pcout<<" made grid for Index"<<igrid<<std::endl;

    //setup operator
    OPERATOR::OperatorBase<dim,real> operators(&all_parameters_new, nstate, poly_degree, poly_degree, grid_degree); 
//setup DG
   // std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, grid);
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    using ODE_enum = Parameters::ODESolverParam::ODESolverEnum;
    all_parameters_new.pde_type = PDE_enum::diffusion;
    all_parameters_new.use_weak_form = false;
    all_parameters_new.use_periodic_bc = true;
    all_parameters_new.ode_solver_param.ode_solver_type = ODE_enum::explicit_solver;//auxiliary only works explicit for now
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();
    dg->evaluate_mass_matrices(true);
    

 //           dealii::IndexSet locally_owned_dofs;
 //           dealii::IndexSet ghost_dofs;
 //           dealii::IndexSet locally_relevant_dofs;
 //           locally_owned_dofs = dg->dof_handler.locally_owned_dofs();
 //           dealii::DoFTools::extract_locally_relevant_dofs(dg->dof_handler, ghost_dofs);
 //           locally_relevant_dofs = ghost_dofs;
 //           ghost_dofs.subtract_set(locally_owned_dofs);
 //           dealii::LinearAlgebra::distributed::Vector<double> solution;
 //           solution.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
 //           dealii::LinearAlgebra::distributed::Vector<double> solution_deriv;
//            solution_deriv.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
//Interpolate IC

            const double pi = atan(1)*4.0;
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts      = dg->volume_quadrature_collection[poly_degree].size();

            const dealii::FESystem<dim> &fe_metric = (dg->high_order_grid->fe_system);
            const unsigned int n_metric_dofs = fe_metric.dofs_per_cell; 
            auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
                if (!current_cell->is_locally_owned()) continue;
	
//pcout<<"grid degree "<<grid_degree<<" metric dofs "<<n_metric_dofs<<std::endl;
                std::vector<dealii::types::global_dof_index> current_metric_dofs_indices(n_metric_dofs);
                metric_cell->get_dof_indices (current_metric_dofs_indices);
                std::vector<std::vector<real>> mapping_support_points(dim);
                std::vector<std::vector<real>> phys_quad_pts(dim);
                for(int idim=0; idim<dim; idim++){
                    mapping_support_points[idim].resize(n_metric_dofs/dim);
                    phys_quad_pts[idim].resize(n_quad_pts);
                }
                dealii::QGaussLobatto<dim> vol_GLL(grid_degree +1);
                for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
                    for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
                        const real val = (dg->high_order_grid->volume_nodes[current_metric_dofs_indices[idof]]);
                        const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
                        mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate); 
                    }
                }
            //interpolate solution
                current_dofs_indices.resize(n_dofs_cell);
                current_cell->get_dof_indices (current_dofs_indices);
                for(int idim=0; idim<dim; idim++){
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        phys_quad_pts[idim][iquad] = 0.0;
                        for(unsigned int jquad=0; jquad<n_metric_dofs/dim; jquad++){
                            phys_quad_pts[idim][iquad] += operators.mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][iquad][jquad]
                                                        * mapping_support_points[idim][jquad];
                        }
                    }
                }
                std::vector<real> soln(n_quad_pts);
                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    dg->solution[current_dofs_indices[idof]]=0.0;
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        double exact = 1.0;
                       for (int idim=0; idim<dim; idim++){
                               // exact *= exp(-(phys_quad_pts[idim][iquad])*(phys_quad_pts[idim][iquad]));
                                exact *= sin(2.0*pi*phys_quad_pts[idim][iquad]) * cos(4.0*pi*phys_quad_pts[idim][iquad]);
                        }
                        dg->solution[current_dofs_indices[idof]] += operators.vol_projection_operator[poly_degree][idof][iquad] *exact; 
                    }   
                }
                //end interpolated solution
            }
    dg->solution.update_ghost_values();

    //Get auxiliary solution
    dg->assemble_auxiliary_residual();


    //TEST ERROR OOA

pcout<<"OOA here"<<std::endl;
            double l2error = 0.0;
            double linf_error = 0.0;
            int overintegrate = 4;
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
       //     const dealii::MappingQGeneric<dim, dim> mapping_collection2 (poly_degree+1);
           // dealii::FEValues<dim,dim> fe_values_extra(mapping_collection2, dg->fe_collection[0], quad_extra, 
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                                dealii::update_values | dealii::update_JxW_values | 
                                dealii::update_jacobians |  
                                dealii::update_quadrature_points | dealii::update_inverse_jacobians);
            const unsigned int n_quad_pts_extra = fe_values_extra.n_quadrature_points;
            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            dealii::Vector<real> soln_at_q(n_quad_pts_extra);
            for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
                if (!current_cell->is_locally_owned()) continue;
                fe_values_extra.reinit(current_cell);
                dofs_indices.resize(fe_values_extra.dofs_per_cell);
                current_cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts_extra; ++iquad) {
                    soln_at_q[iquad] = 0.0;
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                       // soln_at_q[iquad] += solution_deriv[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, 0);
                        soln_at_q[iquad] += dg->auxiliary_solution[dim_check][dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, 0);
                        //soln_at_q[iquad] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, 0);
                    }
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    double uexact_x=1.0;
                    for(int idim=0; idim<dim; idim++){
                       // uexact_x *= exp(-((qpoint[idim]) * (qpoint[idim])));
                       if(idim!=dim_check)
                            uexact_x *= sin(2.0*pi*qpoint[idim]) * cos(4.0*pi*qpoint[idim]);
                    }
                   // uexact_x *= - 2.0 * qpoint[dim_check];
                    uexact_x *=  (2.0 * pi*cos(2.0*pi*qpoint[dim_check])*cos(4.0*pi*qpoint[dim_check])
                                + sin(2.0*pi*qpoint[dim_check])* (-4.0)* pi *sin(4.0*pi*qpoint[dim_check]));
//pcout<<" soln "<<soln_at_q[iquad]<<" exact "<<uexact_x<<std::endl;
                    l2error += pow(soln_at_q[iquad] - uexact_x, 2) * fe_values_extra.JxW(iquad);
                    double inf_temp = std::abs(soln_at_q[iquad]-uexact_x);
                    if(inf_temp > linf_error){
                        linf_error = inf_temp;
                    }
                }

            }
pcout<<"got OOA here"<<std::endl;

    const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, MPI_COMM_WORLD));
            const double linferror_mpi= (dealii::Utilities::MPI::max(linf_error, MPI_COMM_WORLD));
            // Convergence table
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;
            soln_error_inf[igrid] = linferror_mpi;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);
            convergence_table.add_value("soln_Linf_error", linferror_mpi);


            pcout << " Grid size h: " << dx 
                 << " L2-soln_error: " << l2error_mpi_sum
                 << " Linf-soln_error: " << linferror_mpi
                 << std::endl;


            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                const double slope_soln_err_inf = log(soln_error_inf[igrid]/soln_error_inf[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  solution_error1 " << soln_error[igrid-1]
                     << "  solution_error2 " << soln_error[igrid]
                     << "  slope " << slope_soln_err
                     << "  solution_error1_inf " << soln_error_inf[igrid-1]
                     << "  solution_error2_inf " << soln_error_inf[igrid]
                     << "  slope " << slope_soln_err_inf
                     << std::endl;
            }


    //end test error OOA



    }//end grid refinement loop

    const int igrid = n_grids-1;
   // const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
   //                       / log(grid_size[igrid]/grid_size[igrid-1]);
    const double slope_soln_err = log(soln_error_inf[igrid]/soln_error_inf[igrid-1])
                          / log(grid_size[igrid]/grid_size[igrid-1]);
    if(std::abs(slope_soln_err-poly_degree)>0.08 && poly_degree % 2 == 1){
        return 1;
    }
    if(std::abs(slope_soln_err-(poly_degree+1))>0.05 && poly_degree % 2 == 0){
    //if(std::abs(slope_soln_err-(poly_degree))>0.05 && poly_degree % 2 == 0){
        return 1;
    }
    
        pcout << " ********************************************"
             << std::endl
             << " Convergence rates for p = " << poly_degree
             << std::endl
             << " ********************************************"
             << std::endl;
        convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_Linf_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("soln_L2_error", true);
        convergence_table.set_scientific("soln_Linf_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

    }//end poly degree loop
}

//}//end PHiLiP namespace
