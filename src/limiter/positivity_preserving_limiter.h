#ifndef __POSITIVITY_PRESERVING_LIMITER__
#define __POSITIVITY_PRESERVING_LIMITER__

#include "bound_preserving_limiter.h"

namespace PHiLiP {
/**********************************
* Zhang, Xiangxiong, and Chi-Wang Shu. 
* "On positivity-preserving high order discontinuous Galerkin schemes 
* for compressible Euler equations on rectangular meshes." 
* Journal of Computational Physics 229.23 (2010): 8918-8934.
**********************************/
template<int dim, int nstate, typename real>
class PositivityPreservingLimiter_Zhang2010 : public BoundPreservingLimiterState <dim, nstate, real>
{
public:
    /// Constructor
    explicit PositivityPreservingLimiter_Zhang2010(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~PositivityPreservingLimiter_Zhang2010() = default;

    /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
    std::shared_ptr<BoundPreservingLimiterState<dim, nstate, real>> tvbLimiter;
    // Euler physics pointer. Used to compute pressure.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

    /// Function to obtain the solution cell average
    using BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg;

    /// Applies positivity-preserving limiter to the solution.
    /** Using Zhang,Shu November 2010 Eq 3.14-3.19 we apply a limiter on the global solution
    */
    void limit(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const dealii::DoFHandler<dim>&                          dof_handler,
        const dealii::hp::FECollection<dim>&                    fe_collection,
        const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
        const unsigned int                                      grid_degree,
        const unsigned int                                      max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        const dealii::hp::QCollection<1>                        oneD_quadrature_collection);

}; // End of PositivityPreservingLimiter_Zhang2010 Class

/**********************************
* Wang, Cheng, et al. 
* "Robust high order discontinuous Galerkin schemes for two-dimensional gaseous detonations." 
* Journal of Computational Physics 231.2 (2012): 653-665.
**********************************/
template<int dim, int nstate, typename real>
class PositivityPreservingLimiter_Wang2012 : public BoundPreservingLimiterState <dim, nstate, real>
{
public:
    /// Constructor
    explicit PositivityPreservingLimiter_Wang2012(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~PositivityPreservingLimiter_Wang2012() = default;

    /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
    std::shared_ptr<BoundPreservingLimiterState<dim, nstate, real>> tvbLimiter;
    // Euler physics pointer. Used to compute pressure.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

    /// Function to obtain the solution cell average
    using BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg;

    /// Applies positivity-preserving limiter to the solution.
    /** Using Wang,Shu January 2012 Eq 3.2-3.7 we apply a limiter on the global solution
    */
    void limit(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const dealii::DoFHandler<dim>&                          dof_handler,
        const dealii::hp::FECollection<dim>&                    fe_collection,
        const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
        const unsigned int                                      grid_degree,
        const unsigned int                                      max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        const dealii::hp::QCollection<1>                        oneD_quadrature_collection);

}; // End of PositivityPreservingLimiter_Wang2012 Class
} // PHiLiP namespace

#endif

