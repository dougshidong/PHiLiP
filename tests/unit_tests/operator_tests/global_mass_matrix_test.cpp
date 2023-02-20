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
#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"

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
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    all_parameters_new.flow_solver_param.flow_case_type = FlowCaseEnum::taylor_green_vortex;
    all_parameters_new.use_inverse_mass_on_the_fly = true;
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    all_parameters_new.flux_reconstruction_type = FR_enum::cPlus;
    all_parameters_new.use_weak_form = false;
    using PDE_enum   = Parameters::AllParameters::PartialDifferentialEquation;
    all_parameters_new.pde_type = PDE_enum::navier_stokes;

    bool different = false;
    for(unsigned int grid_type=0; grid_type<2; grid_type++){
        //Generate a standard grid
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
         
        const unsigned int n_refinements = 2;
        const unsigned int poly_degree = 3;
         
        // set the warped grid
        const unsigned int grid_degree = (grid_type == 1) ? poly_degree : 1;
        if(grid_type == 1){
            PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);
        }
         
        if(grid_type == 0){
            double left = 0.0;
            double right = 2 * dealii::numbers::PI;
            const bool colorize = true;
            dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
            std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
            dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
            dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
            if constexpr(PHILIP_DIM == 3)
                dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
            grid->add_periodicity(matched_pairs);
            grid->refine_global(n_refinements);
        }
         
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        dg->allocate_system ();
        //initialize IC
        //set solution as some random number between [1e-8,30] at each dof
        //loop over cells as to write only to local solution indices
        const unsigned int n_dofs = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
        std::vector<dealii::types::global_dof_index> current_dofs_indices(n_dofs);
        for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
            if (!current_cell->is_locally_owned()) continue;
            current_cell->get_dof_indices (current_dofs_indices);
            for(unsigned int i=0; i<n_dofs; i++){
                dg->solution[current_dofs_indices[i]] = 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8)));
            }
        }
        dg->solution.update_ghost_values();
         
        dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
        dealii::LinearAlgebra::distributed::Vector<double> mass_inv_mass_matrix_times_solution(dg->right_hand_side);
        dg->apply_inverse_global_mass_matrix(mass_matrix_times_solution, mass_inv_mass_matrix_times_solution); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
        mass_inv_mass_matrix_times_solution.update_ghost_values();
        for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
            if (!current_cell->is_locally_owned()) continue;
            current_cell->get_dof_indices (current_dofs_indices);
            for(unsigned int i=0; i<n_dofs; i++){
                if(abs(dg->solution[current_dofs_indices[i]] - mass_inv_mass_matrix_times_solution[current_dofs_indices[i]])>1e-12){
                    different = true;
                    std::cout<<dg->solution[current_dofs_indices[i]]<<" i "<<current_dofs_indices[i]<<" "<<mass_inv_mass_matrix_times_solution[current_dofs_indices[i]]<<std::endl;
                }
            }
        }
    }//end of grid type loop

    if(different){
        std::cout<<"The application of mass matrix and inverse on the fly does NOT recover same vector."<<std::endl;
        return 1;
    }
    else{
        pcout<<"The application of mass matrix and inverse on the fly recovers the same vector.\n"<<std::endl;
        return 0;
    }
}

//}//end PHiLiP namespace

