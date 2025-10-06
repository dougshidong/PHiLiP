#ifndef __TVB_LIMITER__
#define __TVB_LIMITER__

#include "bound_preserving_limiter.h"

namespace PHiLiP {
/// Class for implementation of a TVD/TVB limiter derived from BoundPreservingLimiterState class
/**********************************
* Chen, Tianheng, and Chi-Wang Shu. 
* "Entropy stable high order discontinuous Galerkin methods with  
* suitable quadrature rules for hyperbolic conservation laws." 
* Journal of Computational Physics 345 (2017): 427-461.
**********************************/
template<int dim, int nstate, typename real>
class TVBLimiter : public BoundPreservingLimiterState <dim, nstate, real>
{
public:
    /// Constructor
    explicit TVBLimiter(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~TVBLimiter() = default;

private:
    /// Function to limit cell - apply minmod function, obtain theta (linear scaling value) and apply limiter
    std::array<std::vector<real>, nstate> limit_cell(
        std::array<std::vector<real>, nstate>                   soln_at_q,
        const unsigned int                                      n_quad_pts,
        const std::array<real, nstate>                          prev_cell_avg,
        const std::array<real, nstate>                          soln_cell_avg,
        const std::array<real, nstate>                          next_cell_avg,
        const std::array<real, nstate>                          M,
        const double                                            h);

    /// Function to obtain the neighbour cell average
    std::array<real, nstate> get_neighbour_cell_avg(
        const dealii::LinearAlgebra::distributed::Vector<double>&       solution,
        const dealii::hp::FECollection<dim>&                            fe_collection,
        const dealii::hp::QCollection<dim>&                             volume_quadrature_collection,
        OPERATOR::basis_functions<dim, 2 * dim, real>                         soln_basis,
        const int                                                       poly_degree,
        const std::vector<dealii::types::global_dof_index>&             neigh_dofs_indices,
        const unsigned int                                              n_dofs_neigh_cell);

    /// Function to obtain the current cell average
    std::array<real, nstate> get_current_cell_avg(
        std::array<std::vector<real>, nstate> soln_at_q,
        const unsigned int n_quad_pts,
        const std::vector<real>& quad_weights);

    /// Function to apply modified_minmod using Thm3.7 in Chen, Shu 2017
    real apply_modified_minmod(
        const double        a_state,
        const double        M_state,
        const double        h,
        const double        diff_next_state,
        const double        diff_prev_state,
        const double        cell_avg_state,
        const bool          left_face);
public:
    /// Function to obtain the solution cell average
    using BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg;

    /// Applies total variation bounded limiter to the solution.
    /// Using Chen,Shu September 2017 Thm3.7 we apply a limiter on the solution
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

}; // End of TVBLimiter Class
} // PHiLiP namespace

#endif

