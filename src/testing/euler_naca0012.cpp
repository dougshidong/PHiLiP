#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <deal.II/grid/grid_refinement.h>
#include "physics/manufactured_solution.h"
#include "euler_naca0012.hpp"
#include "flow_solver.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerNACA0012<dim,nstate>::EulerNACA0012(const Parameters::AllParameters *const parameters_input,
                                         const dealii::ParameterHandler &parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
int EulerNACA0012<dim,nstate>
::run_test () const
{
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    const unsigned int p_start             = param.manufactured_convergence_study_param.degree_start;
    const unsigned int p_end               = param.manufactured_convergence_study_param.degree_end;
    const unsigned int n_grids_input       = param.manufactured_convergence_study_param.number_of_grids;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {
        for (unsigned int igrid=0; igrid<n_grids_input; ++igrid) {
            param.grid_refinement_study_param.poly_degree = poly_degree;
            param.grid_refinement_study_param.num_refinements = igrid;
            std::unique_ptr<FlowSolver<dim,nstate>> flow_solver = FlowSolverFactory<dim,nstate>::create_FlowSolver(&param, parameter_handler);
            flow_solver->run_test();
        }
    }
    return 0;
}


#if PHILIP_DIM==2
    template class EulerNACA0012 <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


