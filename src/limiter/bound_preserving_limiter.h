#ifndef __BOUND_PRESERVING_LIMITER__
#define __BOUND_PRESERVING_LIMITER__

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/parameter_handler.h>

#include "parameters/all_parameters.h"
#include "operators/operators.h"
#include "physics/euler.h"

namespace PHiLiP {

/// Base Class for implementation of bound preserving limiters
/**
* Bound preserving limiters were developed by Zhang and Shu (2010)
* to maintain a physical bound for numerical approximations
* Currently, there are three forms of bound preserving limiters implemented:
* (1) Maximum-Principle-Satisfying limiter
* (2) Positivity-Preserving Limiter
* (3) TVB/TVD Limiter
*/
template<int dim, typename real>
class BoundPreservingLimiter
{
public:
    /// Constructor
    explicit BoundPreservingLimiter(
        const int nstate_input,//number of states input
        const Parameters::AllParameters* const parameters_input);//pointer to parameters

    /// Destructor
    ~BoundPreservingLimiter() = default;

    /// Number of states
    const int nstate;

    /// Pointer to parameters object
    const Parameters::AllParameters* const all_parameters;

    /// Function to limit the solution
    virtual void limit(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const dealii::DoFHandler<dim>&                          dof_handler,
        const dealii::hp::FECollection<dim>&                    fe_collection,
        const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
        const unsigned int                                      grid_degree,
        const unsigned int                                      max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        const dealii::hp::QCollection<1>                        oneD_quadrature_collection,
        double                                                  dt) = 0;
}; // End of BoundPreservingLimiter Class

/// Base Class for bound preserving limiters templated on state
template<int dim, int nstate, typename real>
class BoundPreservingLimiterState : public BoundPreservingLimiter <dim, real>
{
public:
    /// Pointer to parameters object
    using BoundPreservingLimiter<dim, real>::all_parameters;

    /// Constructor
    explicit BoundPreservingLimiterState(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~BoundPreservingLimiterState() = default;

    /// Function to limit the solution
    virtual void limit(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const dealii::DoFHandler<dim>&                          dof_handler,
        const dealii::hp::FECollection<dim>&                    fe_collection,
        const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
        const unsigned int                                      grid_degree,
        const unsigned int                                      max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        const dealii::hp::QCollection<1>                        oneD_quadrature_collection,
        double                                                  dt) = 0;

    /// Function to obtain the solution cell average
    std::array<real, nstate> get_soln_cell_avg(
        const std::array<std::vector<real>, nstate>&            soln_at_q,
        const unsigned int                                      n_quad_pts,
        const std::vector<real>&                                quad_weights);

}; // End of BoundPreservingLimiterState Class
} // PHiLiP namespace

#endif

