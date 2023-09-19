#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1_eulerian.h>


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/fe_values.h>

#include "bound_preserving_limiter.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
// Constructor
template <int dim, typename real>
BoundPreservingLimiter<dim, real>::BoundPreservingLimiter(
    const int nstate_input,
    const Parameters::AllParameters* const parameters_input)
    : nstate(nstate_input)
    , all_parameters(parameters_input) {}

template class BoundPreservingLimiter <PHILIP_DIM, double>;
} // PHiLiP namespace