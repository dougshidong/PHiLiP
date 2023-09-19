#ifndef __BOUND_PRESERVING_LIMITER__
#define __BOUND_PRESERVING_LIMITER__

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

#include "dg/dg.h"
#include "physics/physics.h"

namespace PHiLiP {
template<int dim, typename real>
class BoundPreservingLimiter
{
public:
    /// Constructor
    BoundPreservingLimiter(
        const int nstate_input,//number of states input
        const Parameters::AllParameters* const parameters_input);//pointer to parameters

    /// Destructor
    ~BoundPreservingLimiter() {};

    /// Number of states
    const int nstate;

    /// Pointer to parameters object
    const Parameters::AllParameters* const all_parameters;

    /// Function to limit the solution
    virtual void limit(
        dealii::LinearAlgebra::distributed::Vector<double>& solution,
        const dealii::DoFHandler<dim>& dof_handler,
        const dealii::hp::FECollection<dim>& fe_collection,
        dealii::hp::QCollection<dim>                            volume_quadrature_collection,
        unsigned int                                            tensor_degree,
        unsigned int                                            max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        dealii::hp::QCollection<1>                              oneD_quadrature_collection) = 0;

}; // End of BoundPreservingLimiter Class
} // PHiLiP namespace

#endif

