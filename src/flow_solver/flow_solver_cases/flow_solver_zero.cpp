#include "flow_solver_zero.h"

#include <stdlib.h>

#include <iostream>

#include "dg/dg_base.hpp"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nspecies, int nstate>
FlowSolverCaseZero<dim, nspecies, nstate>::FlowSolverCaseZero(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nspecies, nstate>(parameters_input)
{
}

template <int dim, int nspecies, int nstate>
std::shared_ptr<Triangulation> FlowSolverCaseZero<dim, nspecies, nstate>::generate_grid() const
{
    return nullptr;
}

template <int dim, int nspecies, int nstate>
void FlowSolverCaseZero<dim, nspecies, nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout <<  "No flow case has been created for " << dim << "D and " << nspecies << " species. Implement a test for the templated configuration." << std::endl;
    std::abort();
}

template class FlowSolverCaseZero<PHILIP_DIM,PHILIP_SPECIES,PHILIP_DIM+2+(PHILIP_SPECIES-1)>;
} // FlowSolver namespace
} // PHiLiP namespace
