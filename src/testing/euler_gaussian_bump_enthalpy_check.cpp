#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include "euler_gaussian_bump_enthalpy_check.h"
#include "mesh/grids/gaussian_bump.h"

#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerGaussianBumpEnthalpyCheck<dim,nstate>::EulerGaussianBumpEnthalpyCheck(const Parameters::AllParameters *const parameters_input)
 : TestsBase::TestsBase(parameters_input)
 {}

template<int dim, int nstate>
int EulerGaussianBumpEnthalpyCheck<dim,nstate>::run_test () const
{
    const Parameters::AllParameters param_transonic = *(TestsBase::all_parameters);
    Parameters::AllParameters param_subsonic = *(TestsBase::all_parameters);
    param_subsonic.artificial_dissipation_param.add_artificial_dissipation = false;
    param_subsonic.euler_param.mach_inf = 0.5;

    EulerGaussianBump<dim,nstate> gaussian_bump_transonic(&param_transonic);
    EulerGaussianBump<dim,nstate> gaussian_bump_subsonic(&param_subsonic);

    const double error_transonic = gaussian_bump_transonic.run_euler_gaussian_bump();
    const double error_subsonic = gaussian_bump_subsonic.run_euler_gaussian_bump();

    pcout << "Error transonic = "<< error_transonic << std::endl;
    pcout << "Error subsonic = "<< error_subsonic << std::endl;

    if (abs(error_transonic - error_subsonic) > 3.1e-3) 
    {
        pcout<< "Enthalpy is not conserved. Test failed" << std::endl;
        return 1;
    }
    pcout<< " Test passed" << std::endl;
    return 0;

}




#if PHILIP_DIM==2
 template class EulerGaussianBumpEnthalpyCheck <PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // Tests namespace
} // PHiLiP namespace


