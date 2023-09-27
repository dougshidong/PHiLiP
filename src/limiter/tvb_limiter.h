#ifndef __TVB_LIMITER__
#define __TVB_LIMITER__

#include "bound_preserving_limiter.h"

namespace PHiLiP {
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
    TVBLimiter(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~TVBLimiter() {};

private:
    /// Function to limit cell - apply minmod function, obtain theta (linear scaling value) and apply limiter
    std::array<std::vector<real>, nstate> limit_cell(
        std::array<std::vector<real>, nstate>                   soln_at_q,
        const unsigned int                                      n_quad_pts,
        std::array<real, nstate>                                prev_cell_avg,
        std::array<real, nstate>                                soln_cell_avg,
        std::array<real, nstate>                                next_cell_avg,
        std::array<real, nstate>                                M,
        double                                                  h);

    /// Function to obtain the neighbour cell average
    std::array<real, nstate> get_neighbour_cell_avg(
        dealii::LinearAlgebra::distributed::Vector<double>      solution,
        const dealii::hp::FECollection<dim>&                    fe_collection,
        dealii::hp::QCollection<dim>                            volume_quadrature_collection,
        OPERATOR::basis_functions<dim, 2 * dim>                 soln_basis,
        const int                                               poly_degree,
        std::vector<dealii::types::global_dof_index>            neigh_dofs_indices,
        const unsigned int                                      n_dofs_neigh_cell);

    /// Function to obtain the current cell average
    std::array<real, nstate> get_current_cell_avg(
        std::array<std::vector<real>, nstate> soln_at_q,
        const unsigned int n_quad_pts,
        const std::vector<real>& quad_weights);

public:
    /// Function to obtain the solution cell average
    using BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg;

    /// Applies total variation bounded limiter to the solution.
    /** Using Chen,Shu September 2017 Thm3.7 we apply a limiter on the solution
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

}; // End of TVBLimiter Class
} // PHiLiP namespace

#endif

