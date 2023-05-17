#include <fstream>
#include "dg/dg_factory.hpp"
#include "TGV_scaling.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"
#include "mesh/grids/straight_periodic_cube.hpp"
#include <time.h>
#include <deal.II/base/timer.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerTaylorGreenScaling<dim, nstate>::EulerTaylorGreenScaling(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int EulerTaylorGreenScaling<dim, nstate>::run_test() const
{
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    using real = double;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    double left = 0.0;
    double right = 2 * dealii::numbers::PI;

    const unsigned int n_refinements = all_parameters->flow_solver_param.number_of_mesh_refinements;
    if(all_parameters->use_curvilinear_grid){
        //if curvilinear
        PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);
    }
    else{
        //if straight
        PHiLiP::Grids::straight_periodic_cube<dim,Triangulation>(grid, left, right, pow(2.0,n_refinements));
    }

    std::ofstream myfile (all_parameters_new.energy_file + ".gpl"  , std::ios::trunc);
    const unsigned int poly_degree_start= all_parameters->flow_solver_param.poly_degree;

    const unsigned int poly_degree_end = 16;
    std::array<double,poly_degree_end> time_to_run;
    std::array<double,poly_degree_end> time_to_run_mpi;

    //poly degree loop
    for(unsigned int poly_degree = poly_degree_start; poly_degree<poly_degree_end; poly_degree++){

        // set the warped grid
        const unsigned int grid_degree = (all_parameters->use_curvilinear_grid) ? poly_degree : 1;
         
        if(all_parameters->overintegration == 100){
            if(all_parameters->use_curvilinear_grid){
                all_parameters_new.overintegration = 2*(poly_degree+1);
            }
            else{
                all_parameters_new.overintegration = poly_degree+1;
            }
        }
         
        // Create DG
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        dg->allocate_system (false,false,false);
         
        //Apply initial condition for TGV
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                    InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);
        //Create ODE system. 
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
         
        ode_solver->current_iteration = 0;
        ode_solver->allocate_ode_system();
        MPI_Barrier(MPI_COMM_WORLD);
        pcout << "ODE solver successfully created. This verifies no memory jump from ODE Solver." << std::endl;
        dealii::LinearAlgebra::distributed::Vector<double> solution_update;
        solution_update.reinit(dg->locally_owned_dofs, dg->ghost_dofs, this->mpi_communicator);

        //Perform 10 steps solving the reidual and applying the inverse of the mass matrix.
        for(unsigned int i_step=0; i_step<10; i_step++){
            dg->assemble_residual();
            if(all_parameters->use_inverse_mass_on_the_fly){
                dg->apply_inverse_global_mass_matrix(dg->right_hand_side, solution_update);
            } else{
                dg->global_inverse_mass_matrix.vmult(solution_update, dg->right_hand_side);
            }
        }

        //store local cpu time
        time_to_run[poly_degree] = dg->assemble_residual_time;
        //store mpi summed cpu time
        time_to_run_mpi[poly_degree] = dealii::Utilities::MPI::sum(time_to_run[poly_degree], this->mpi_communicator);
        
        std::cout<<"Poly Degree "<<poly_degree<<" time to run local cpu "<<std::fixed << std::setprecision(16) << (double)time_to_run[poly_degree]<<std::endl;
        pcout<<"Poly Degree "<<poly_degree<<" time to run Mpi "<<std::fixed << std::setprecision(16) << (double)time_to_run_mpi[poly_degree]<<std::endl;
        myfile << poly_degree << " " << std::fixed << std::setprecision(16) << time_to_run_mpi[poly_degree]<< std::endl;
    }//end of poly loop


    myfile.close();
    double avg_slope = 0.0;
    //Print a chart for time and slopes.
    pcout<<"Times for one timestep"<<std::endl;
    pcout<<"local time  | Slope |  "<<"MPI sum time | Slope "<<std::endl;
    for(unsigned int i=poly_degree_start+1; i<poly_degree_end; i++){
        const double slope_local = std::log(((double)time_to_run[i]) / ((double)time_to_run[i-1]))
                        / std::log((double)((i+1.0)/(i)));
        const double slope_mpi = std::log(((double)time_to_run_mpi[i]) /( (double)time_to_run_mpi[i-1]))
                        / std::log((double)((i+1.0)/(i)));
        pcout<<(double)time_to_run[i]<<" "<< slope_local <<" "<<
        (double)time_to_run_mpi[i]<<" "<< slope_mpi <<
        std::endl;
        if(i>poly_degree_end-4){
            avg_slope += slope_mpi;
        }
    }
    avg_slope /= (3.0);

    if(avg_slope - (dim + 1) > 0.5){
        pcout<<"Did not scale at dim + 1. Instead scaled at "<<avg_slope<<std::endl;
        return 1;
    }

    //check that it can run up to p=50 for Cartesian or p=20 for curvilinear without running out of memory.
    const unsigned int poly_degree = (all_parameters->use_curvilinear_grid) ? 20 : 50;
    const unsigned int grid_degree = (all_parameters->use_curvilinear_grid) ? poly_degree : 1;
     
    if(all_parameters->overintegration == 100){
        if(all_parameters->use_curvilinear_grid){
            all_parameters_new.overintegration = 2*(poly_degree+1);
        }
        else{
            all_parameters_new.overintegration = poly_degree+1;
        }
    }
     
    pcout<<"Checking that it does not run out of memory for poly degree "<<poly_degree<<std::endl;
    // Create DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system (false,false,false);
     
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);
     
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
     
    ode_solver->current_iteration = 0;
    ode_solver->allocate_ode_system();
    MPI_Barrier(MPI_COMM_WORLD);
    pcout << "ODE solver successfully created. This verifies no memory jump from ODE Solver." << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> solution_update;
    solution_update.reinit(dg->locally_owned_dofs, dg->ghost_dofs, this->mpi_communicator);

    for(unsigned int i_step=0; i_step<10; i_step++){
        dg->assemble_residual();
        if(all_parameters->use_inverse_mass_on_the_fly){
            dg->apply_inverse_global_mass_matrix(dg->right_hand_side, solution_update);
        } else{
            dg->global_inverse_mass_matrix.vmult(solution_update, dg->right_hand_side);
        }
    }
    //if it reaches here, then there is no memory issue.


    return 0;
}

#if PHILIP_DIM==3
    template class EulerTaylorGreenScaling <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

