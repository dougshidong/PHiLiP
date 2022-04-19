#include <fstream>

#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "burgers_stability.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersEnergyStability<dim, nstate>::BurgersEnergyStability(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
double BurgersEnergyStability<dim, nstate>::compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const
{
 double energy = 0.0;
 
 for (unsigned int i = 0; i < dg->solution.size(); ++i)
 {
  energy += 1./(dg->global_inverse_mass_matrix(i,i)) * dg->solution(i) * dg->solution(i);
  //energy += dg->solution(i) * dg->solution(i);
 }
 
 //energy = (dg->solution) * (dg->solution);
 return energy;
}

template <int dim, int nstate>
int BurgersEnergyStability<dim, nstate>::run_test() const
{
    pcout << " Running Burgers energy stability. " << std::endl;
    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>();
   
    double left = 0.0;
    double right = 2.0;
    const bool colorize = true;
    int n_refinements = 5;
    unsigned int poly_degree = 3;
    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
   
    //std::vector<dealii::GridTools::PeriodicFacePair<typename Triangulation::cell_iterator> > matched_pairs;
    //dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
    //grid->add_periodicity(matched_pairs);
    //Imposing periodicity through use_periodic_bc param
   
    grid->refine_global(n_refinements);
    pcout << "Grid generated and refined" << std::endl;
   
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
    pcout << "dg created" <<std::endl;
    dg->allocate_system ();
   
    pcout << "Implement initial conditions" << std::endl;
    dealii::FunctionParser<1> initial_condition;
    std::string variables = "x";
    std::map<std::string,double> constants;
    //constants["pi"] = dealii::numbers::PI;
    //std::string expression = "sin(pi*(x)) + 0.01";
    std::string expression = "exp(-30*(x-1)*(x-1))";
    initial_condition.initialize(variables,
                                 expression,
                                 constants);
    dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);
    // Create ODE solver using the factory and providing the DG object
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
   
    double finalTime = 0.01;
   
    //double dt = all_parameters->ode_solver_param.initial_time_step;
    //(void) dt;
   
    //need to call ode_solver before calculating energy because mass matrix isn't allocated yet.

    ode_solver->advance_solution_time(1E-10);
    double initial_energy = compute_energy(dg);
    pcout << "Initial energy is " << initial_energy << std::endl;

    //currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
    //this causes some issues with outputs (only one file is output, which is overwritten at each time step)
    //also the ode solver output doesn't make sense (says "iteration 1 out of 1")
    //but it works. I'll keep it for now and need to modify the output functions later to account for this.
    std::ofstream myfile ("energy_plot.gpl" , std::ios::trunc);


/*
   
    for (int i = 0; i < std::ceil(finalTime/dt); ++ i) {
        ode_solver->advance_solution_time(dt);
        double current_energy = compute_energy(dg);
        current_energy -= initial_energy;//subtract initial energya
        current_energy = abs(current_energy);
        std::cout << std::setprecision(16) << std::fixed;
        pcout << "Change in energy (current - initial) at time " << i * dt << " is " << current_energy << std::endl;
        myfile << i * dt << " " << current_energy << std::endl;
        if (current_energy - initial_energy >= 0.1) {
            //return 1;
            //break;
            pcout << "WARNING: Large change in energy!" << std::endl;
        }
    }
*/


    std::cout << std::setprecision(16) << std::fixed;
    ode_solver->advance_solution_time(finalTime);
    double end_energy = compute_energy(dg);
    end_energy = abs(end_energy - initial_energy);
    //std::cout << std::setprecision(16) << std::fixed;
    pcout << "Initial energy is " << initial_energy << std::endl;
    pcout << "Change in energy at t = " << ode_solver->current_time << " is " << end_energy << std::endl;

    myfile.close();

   
    //ode_solver->advance_solution_time(finalTime);
   
    return 0; //need to change
}
#if PHILIP_DIM==1
    template class BurgersEnergyStability<PHILIP_DIM,PHILIP_DIM>;
#endif
} // Tests namespace
} // PHiLiP namespace
