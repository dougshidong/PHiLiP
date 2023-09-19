#ifndef __MAXIMUM_PRINCIPLE_LIMITER__
#define __MAXIMUM_PRINCIPLE_LIMITER__

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
#include "bound_preserving_limiter.h"

namespace PHiLiP {
template<int dim, int nstate, typename real>
class MaximumPrincipleLimiter : public BoundPreservingLimiter <dim, real>
{
public:
    /// Constructor
    MaximumPrincipleLimiter(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~MaximumPrincipleLimiter() {};

    /// Maximum of initial solution in domain.
    std::vector<real> global_max;
    /// Minimum of initial solution in domain.
    std::vector<real> global_min;

    /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
    std::shared_ptr<BoundPreservingLimiter<dim, real>> tvbLimiter;

private:
    /// Function to obtain the maximum and minimum of the initial solution
    void get_global_max_and_min_of_solution(
        dealii::LinearAlgebra::distributed::Vector<double>      solution,
        const dealii::DoFHandler<dim>&                          dof_handler,
        const dealii::hp::FECollection<dim>&                    fe_collection);

    /// Function to obtain the solution cell average
    std::array<real, nstate> get_soln_cell_avg(
        std::array<std::vector<real>, nstate> soln_at_q,
        const unsigned int n_quad_pts,
        const std::vector<real>& quad_weights);
    
    /// Function to verify the limited solution satisfies the strict maximum principle
    void write_limited_solution(
        dealii::LinearAlgebra::distributed::Vector<double>      solution,
        std::array<std::vector<real>, nstate>                   soln_dofs,
        const unsigned int                                      n_shape_fns,
        std::vector<dealii::types::global_dof_index>            current_dofs_indices);

public:
    /// Applies maximum-principle-satisfying limiter to the solution.
    /** Using Zhang,Shu May 2010 Eq 3.8 and 3.9 we apply a limiter on the global solution
    */
    void limit(
        dealii::LinearAlgebra::distributed::Vector<double>& solution,
        const dealii::DoFHandler<dim>& dof_handler,
        const dealii::hp::FECollection<dim>& fe_collection,
        dealii::hp::QCollection<dim>                            volume_quadrature_collection,
        unsigned int                                            tensor_degree,
        unsigned int                                            max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        dealii::hp::QCollection<1>                              oneD_quadrature_collection);

}; // End of MaximumPrincipleLimiter Class
} // PHiLiP namespace

#endif

