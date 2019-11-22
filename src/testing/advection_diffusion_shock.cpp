// includes
#include <stdlib.h>
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include "advection_diffusion_shock.h"

#include "parameters/all_parameters.h"

#include "physics/physics_factory.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include "post_processor/physics_post_processor.h"

namespace PHiLiP {
namespace Tests {

// include parameters S[0],S[1],S[2]
// n_shocks as an array

template <int dim, typename real>
real ManufacturedSolutionShocked<dim,real>::value(const dealii::Point<dim,real>  &pos, const unsigned int /*istate*/) const
{
    real val = 1;

    for(unsigned int i = 0; i < dim; ++i){
        real x = pos[i];
        real val_dim = 0;
        for(unsigned int j = 0; j < n_shocks[i]; ++j){
            // taking the product of function in each direction
            val_dim += std::atan(S_j[i][j]*(x-x_j[i][j]));
        }
        val *= val_dim;
    }

    return val;
}

template <int dim, typename real>
dealii::Tensor<1,dim,real> ManufacturedSolutionShocked<dim,real>::gradient(const dealii::Point<dim,real> &pos, const unsigned int /*istate*/) const
{
    dealii::Tensor<1,dim,real> grad;

    for(unsigned int k = 0; k < dim; ++k){
        // taking the k^th derivative
        real grad_dim = 1;
        for(unsigned int i = 0; i < dim; ++i){
            real x = pos[i];
            real val_dim = 0;
            for(unsigned int j = 0; j < n_shocks[i]; ++j){
                if(i==k){
                    // taking the derivative dimension
                    real coeff = S_j[i][j]*(x-x_j[i][j]);
                    val_dim += S_j[i][j]/(std::pow(coeff,2)+1);
                }else{
                    // value product unaffected
                    val_dim += std::atan(S_j[i][j]*(x-x_j[i][j]));
                }
            }
            grad_dim *= val_dim;
        }
        grad[k] = grad_dim;
    }

    return grad;
}

template <int dim, typename real>
dealii::SymmetricTensor<2,dim,real> ManufacturedSolutionShocked<dim,real>::hessian(const dealii::Point<dim,real> &pos, const unsigned int /*istate*/) const
{
    dealii::SymmetricTensor<2,dim,real> hes;

    for(unsigned int k1 = 0; k1 < dim; ++k1){
        // taking the k1^th derivative
        for(unsigned int k2 = 0; k2 < dim; ++k2){
            // taking the k2^th derivative
            real hes_dim = 1;
            for(unsigned int i = 0; i < dim; ++i){
                real x = pos[i];
                real val_dim = 0;
                for(unsigned int j = 0; j < n_shocks[i]; ++j){
                    if(i == k1 && i == k2){
                        // taking the second derivative in this dim
                        real coeff = S_j[i][j]*(x-x_j[i][j]);
                        val_dim += -2.0*std::pow(S_j[i][j],2)*coeff/std::pow(std::pow(coeff,2)+1,2);
                    }else if(i == k1 || i == k2){
                        // taking the first derivative in this dim
                        real coeff = S_j[i][j]*(x-x_j[i][j]);
                        val_dim += S_j[i][j]/(std::pow(coeff,2)+1);
                    }else{
                        // taking the value in this dim
                        val_dim += std::atan(S_j[i][j]*(x-x_j[i][j]));
                    }
                }
                hes_dim *= val_dim;
            }
            hes[k1][k2] = hes_dim;
        }
    }

    return hes;
}

template <int dim, int nstate>
AdvectionDiffusionShock<dim, nstate>::AdvectionDiffusionShock(const Parameters::AllParameters *const parameters_input): 
    TestsBase::TestsBase(parameters_input){}

template <int dim, int nstate>
int AdvectionDiffusionShock<dim,nstate>::run_test() const
{
    pcout << "Running the advection-diffusion shocked test case." << std::endl;

    // getting the parameter set
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    using PdeEnum = Parameters::AllParameters::PartialDifferentialEquation;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    // forcing parameter to be used but importing own
    param.manufactured_convergence_study_param.use_manufactured_source_term = true;

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start = manu_grid_conv_param.degree_start;
    const unsigned int p_end   = manu_grid_conv_param.degree_end;
    const unsigned int n_grids = manu_grid_conv_param.number_of_grids;

    // generating the physics of the problem
    using ADtype = Sacado::Fad::DFad<double>;
    std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,double>> physics_double 
        = PHiLiP::Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&param);
    std::shared_ptr <PHiLiP::Physics::PhysicsBase<dim,nstate,ADtype>> physics_adtype 
        = PHiLiP::Physics::PhysicsFactory<dim, nstate, ADtype>::create_Physics(&param);

    // replacing the manufactured solution with the one defined above
    // *** change the manufactured solution pointer to an instance of the one defined here
    physics_double->manufactured_solution_function 
        = std::shared_ptr< ManufacturedSolutionShocked<dim,double> >(new ManufacturedSolutionShocked<dim,double>(nstate));
    physics_adtype->manufactured_solution_function 
        = std::shared_ptr< ManufacturedSolutionShocked<dim,ADtype> >(new ManufacturedSolutionShocked<dim,ADtype>(nstate));

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for(unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree){

        std::vector<double> soln_error(n_grids);
        std::vector<double> output_error(n_grids);
        std::vector<double> grid_size(n_grids);

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

        // building the triangulation (must be declared before DG)
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
        dealii::Triangulation<dim> grid(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
        dealii::parallel::distributed::Triangulation<dim> grid(
            this->mpi_communicator,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::MeshSmoothing::smoothing_on_refinement));
                //dealii::Triangulation<dim>::smoothing_on_refinement |
                //dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

        // dimensions of the mesh
        const double left  = -1.0;
        const double right =  1.0;

        dealii::Vector<float> estimated_error_per_cell;
        for(unsigned int igrid=0; igrid<n_grids; ++igrid) {
            // generating the grid
            grid.clear();
            dealii::GridGenerator::subdivided_hyper_cube(grid, n_1d_cells[igrid], left, right);

            // setting the boundary conditions
            for(auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
                cell->set_material_id(9002);
                for(unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                    if(cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id(1000);
                }
            }

            // generating the DG solver
            std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);

            // casting it to a weak DG
            std::shared_ptr< DGWeak<dim,nstate,double> > dg_weak = std::dynamic_pointer_cast< DGWeak<dim,nstate,double> >(dg);

            // overriding the physics with the one with the custom manufactured solution
            dg_weak->set_physics(physics_double);
            dg_weak->set_physics(physics_adtype);

            dg->allocate_system();

            // create an ODE solver
            std::shared_ptr <ODE::ODESolver<dim, double>> ode_solver
                = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

            // solving the steady state
            ode_solver->steady_state();

            // performing the error integration
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(poly_degree+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid.mapping_fe_field), dg->fe_collection[poly_degree], quad_extra,
                dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;

            std::array<double,nstate> soln_at_q;
            double l2error = 0;

            std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
            for(auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                fe_values_extra.reinit (cell);
                cell->get_dof_indices(dofs_indices);

                double cell_l2error = 0;
                for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad){
                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);

                    for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                    for (int istate = 0; istate < nstate; ++istate){
                        const double soln_exact = physics_double->manufactured_solution_function->value(qpoint, istate);
                        
                        // comparing the converged solution to the manufactured solution
                        cell_l2error += std::pow(soln_at_q[istate] - soln_exact, 2) * fe_values_extra.JxW(iquad);
                    }
                }

                // Adding contributions to the global error
                l2error += cell_l2error;
            }
            const double l2error_mpi = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));
    
            pcout << "Error is: " << l2error_mpi << std::endl;

            // adding values to the convergence 
            const int n_dofs = dg->dof_handler.n_dofs();
            const double dx = 1.0/pow(n_dofs,(1.0/dim));

            // adding terms to the table
            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", grid.n_global_active_cells());
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("l2error", l2error_mpi);

            // ouputing it to a VTK file
            dg->output_results_vtk(igrid);
        }
        // obtaining the convergence rates
        convergence_table.evaluate_convergence_rates("l2error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx",true);
        convergence_table.set_scientific("l2error",true);

        // adding it to the final list
        convergence_table_vector.push_back(convergence_table);

    }

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }

    return 0;
}

template class AdvectionDiffusionShock <PHILIP_DIM,1>;

} // namespace Tests
} // namespace PHiLiP
