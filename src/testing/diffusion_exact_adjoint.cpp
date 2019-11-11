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

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include "diffusion_exact_adjoint.h"

#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

namespace PHiLiP {
namespace Tests {
// need to build my own physics classes to override the source term in dg
// would be nice if there was a way to pass this directly to the dg class
// (otherwise this would need to be added to the physics enum)

/* Defining the physics objects to be used  */
template <int dim, int nstate, typename real>
std::array<real,nstate> diffusion_u<dim,nstate,real>::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;

    double x = pos[0];

    for (int istate=0; istate<nstate; istate++) {
        source[istate] = std::pow(x, 3) * std::pow(1-x, 3);
    }
    return source;
}

template <int dim, int nstate, typename real>
real diffusion_u<dim,nstate,real>::objective_function (
    const dealii::Point<dim,double> &pos) const
{
    double x = pos[0];

    const double pi = std::acos(-1);

    return std::sin(pi * x);
}

template <int dim, int nstate, typename real>
std::array<real,nstate> diffusion_v<dim,nstate,real>::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;

    double x = pos[0];

    const double pi = std::acos(-1);

    for (int istate=0; istate<nstate; istate++) {
        source[istate] = std::sin(pi * x);
    }
    return source;
}

template <int dim, int nstate, typename real>
real diffusion_v<dim,nstate,real>::objective_function (
    const dealii::Point<dim,double> &pos) const
{
    double x = pos[0];

    return std::pow(x, 3) * std::pow(1-x, 3);
}

/* Defining the funcitonal that performs the inner product over the entire domain */
template <int dim, int nstate, typename real>
class diffusion_functional : public Functional<dim, nstate, real>
{
    public:
        template <typename real2>
        real2 evaluate_cell_volume(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
			const dealii::FEValues<dim,dim> &fe_values_volume,
			std::vector<real2> local_solution)
        {
            unsigned int n_quad_pts = fe_values_volume.n_quadrature_points;

			std::array<real2,nstate> soln_at_q;

			real2 val = 0;

            // casting our physics object into a diffusion_objective object 
            const diffusion_objective<dim,nstate,real2>& diff_physics = dynamic_cast<const diffusion_objective<dim,nstate,real2>&>(physics);
            
            // looping over the quadrature points
            for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad){
                std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                for (unsigned int idof=0; idof<fe_values_volume.dofs_per_cell; ++idof) {
					const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
					soln_at_q[istate] += local_solution[idof] * fe_values_volume.shape_value_component(idof, iquad, istate);
				}

                const dealii::Point<dim> qpoint = (fe_values_volume.quadrature_point(iquad));

                // evaluating the associated objective function weighting at the quadrature point
                real2 objective_value = diff_physics.objetive_function(qpoint);

                // integrating over the domain (adding istate loop but should always be 1)
                for (int istate=0; istate<nstate; ++istate) {
                    val += soln_at_q[istate] * objective_value * fe_values_volume.JxW(iquad);
                }
            }

            return val;
        }

    	// non-template functions to override the template classes
		real evaluate_cell_volume(
			const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
			const dealii::FEValues<dim,dim> &fe_values_volume,
			std::vector<real> local_solution) override
		{
			return evaluate_cell_volume<>(physics, fe_values_volume, local_solution);
		}
		Sacado::Fad::DFad<real> evaluate_cell_volume(
			const PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics,
			const dealii::FEValues<dim,dim> &fe_values_volume,
			std::vector<Sacado::Fad::DFad<real>> local_solution) override
		{
			return evaluate_cell_volume<>(physics, fe_values_volume, local_solution);
		}
};

template <int dim, int nstate>
DiffusionExactAdjoint<dim, nstate>::DiffusionExactAdjoint(const Parameters::AllParameters *const parameters_input): 
    TestsBase::TestsBase(parameters_input){}

template <int dim, int nstate>
int DiffusionExactAdjoint<dim,nstate>::run_test() const
{
    std::cout << "Test setup correctly check." << std::endl;

    return 0;
}

#if PHILIP_DIM==1
    template class DiffusionExactAdjoint <PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace