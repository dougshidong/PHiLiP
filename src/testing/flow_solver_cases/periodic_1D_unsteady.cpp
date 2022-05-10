#include "periodic_1D_unsteady.h"
#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "mesh/grids/straight_periodic_cube.hpp"
#include <deal.II/base/table_handler.h>

namespace PHiLiP {

namespace Tests {

//=========================================================
// PERIODIC 1D DOMAIN FOR UNSTEADY CALCULATIONS
//=========================================================

template <int dim, int nstate>
Periodic1DUnsteady<dim, nstate>::Periodic1DUnsteady(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicCubeFlow<dim, nstate>(parameters_input)
{
}

template <int dim, int nstate>
void Periodic1DUnsteady<dim, nstate>::compute_unsteady_data_and_write_to_table(
       const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>>/* dg */,
        const std::shared_ptr <dealii::TableHandler> /* unsteady_data_table */)
{
    double dt = this->all_param.ode_solver_param.initial_time_step;
    int output_solution_every_n_iterations = (int) (this->all_param.ode_solver_param.output_solution_every_dt_time_intervals/dt);
 
    if ((current_iteration % output_solution_every_n_iterations) == 0){
        this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << current_time
                    << std::endl;
    }
}

#if PHILIP_DIM==1
template class Periodic1DUnsteady <PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace

