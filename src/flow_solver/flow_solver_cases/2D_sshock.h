#ifndef __SSHOCK_MESH_ADAPTATION__
#define __SSHOCK_MESH_ADAPTATION__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class SShock : public FlowSolverCaseBase<dim, nstate>
{
#if PHILIP_DIM==1
     using Triangulation = dealii::Triangulation<PHILIP_DIM>;
 #else
     using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
 #endif

 public:
     SShock(const Parameters::AllParameters *const parameters_input);
     
     ~SShock() {};
 
     std::shared_ptr<Triangulation> generate_grid() const override;

     void display_additional_flow_case_specific_parameters() const override;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif

