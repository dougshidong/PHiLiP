#ifndef __MAXIMUM_PRINCIPLE_LIMITER__
#define __MAXIMUM_PRINCIPLE_LIMITER__

#include "bound_preserving_limiter.h"

namespace PHiLiP {
/// Class for implementation of Maximum-Principle-Satisfying limiter derived from BoundPreservingLimiterState class
/**********************************
* Zhang, Xiangxiong, and Chi-Wang Shu. 
* "On maximum-principle-satisfying high order schemes for scalar conservation laws." 
* Journal of Computational Physics 229.9 (2010): 3091-3120.
**********************************/
template<int dim, int nstate, typename real>
class MaximumPrincipleLimiter : public BoundPreservingLimiterState <dim, nstate, real>
{
public:
    /// Constructor
    explicit MaximumPrincipleLimiter(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~MaximumPrincipleLimiter() = default;

    /// Maximum of initial solution for each state in domain.
    std::vector<real> global_max;
    /// Minimum of initial solution for each state in domain.
    std::vector<real> global_min;

    /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
    std::shared_ptr<BoundPreservingLimiter<dim, real>> tvbLimiter;

private:
    /// Function to obtain the maximum and minimum of the initial solution for each state
    void get_global_max_and_min_of_solution(
        const dealii::LinearAlgebra::distributed::Vector<double>&       solution,
        const dealii::DoFHandler<dim>&                                  dof_handler,
        const dealii::hp::FECollection<dim>&                            fe_collection);

    /// Function to obtain the solution cell average
    using BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg;
    
    /// Function to verify the limited solution satisfies the strict maximum principle
    /// and write back limited solution
    void write_limited_solution(
        dealii::LinearAlgebra::distributed::Vector<double>&      solution,
        const std::array<std::vector<real>, nstate>&             soln_coeff,
        const unsigned int                                       n_shape_fns,
        const std::vector<dealii::types::global_dof_index>&      current_dofs_indices);

public:
    /// Applies maximum-principle-satisfying limiter to the solution.
    /// Using Zhang,Shu May 2010 Eq 3.8 and 3.9 we apply a limiter on the global solution
    void limit(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const dealii::DoFHandler<dim>&                          dof_handler,
        const dealii::hp::FECollection<dim>&                    fe_collection,
        const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
        const unsigned int                                      grid_degree,
        const unsigned int                                      max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        const dealii::hp::QCollection<1>                        oneD_quadrature_collection,
        double                                                  dt) override;

}; // End of MaximumPrincipleLimiter Class
} // PHiLiP namespace

#endif

