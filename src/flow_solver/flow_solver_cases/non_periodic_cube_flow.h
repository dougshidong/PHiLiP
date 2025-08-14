#ifndef __NON_PERIODIC_CUBE_FLOW__
#define __NON_PERIODIC_CUBE_FLOW__

#include "flow_solver_case_base.h"
#include "cube_flow_uniform_grid.h"

namespace PHiLiP {
namespace FlowSolver {
#if PHILIP_DIM==1
     using Triangulation = dealii::Triangulation<PHILIP_DIM>;
 #else
     using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
 #endif
 
template <int dim, int nspecies, int nstate>
class NonPeriodicCubeFlow : public CubeFlow_UniformGrid<dim, nspecies, nstate>
{
 public:
     explicit NonPeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);
     
     std::shared_ptr<Triangulation> generate_grid() const override;

     void display_additional_flow_case_specific_parameters() const override;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
