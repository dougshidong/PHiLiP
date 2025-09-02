#ifndef __POSITIVITY_PRESERVING_LIMITER__
#define __POSITIVITY_PRESERVING_LIMITER__

#include "bound_preserving_limiter.h"

namespace PHiLiP {
/// Class for implementation of two forms of the Positivity-Preserving limiter derived from BoundPreservingLimiterState class
/**********************************
* Zhang, Xiangxiong, and Chi-Wang Shu. 
* "On positivity-preserving high order discontinuous Galerkin schemes 
* for compressible Euler equations on rectangular meshes." 
* Journal of Computational Physics 229.23 (2010): 8918-8934.
**********************************/
/**********************************
* Wang, Cheng, et al.
* "Robust high order discontinuous Galerkin schemes for two-dimensional gaseous detonations."
* Journal of Computational Physics 231.2 (2012): 653-665.
**********************************/
template<int dim, int nstate, typename real>
class PositivityPreservingLimiter : public BoundPreservingLimiterState <dim, nstate, real>
{
public:
    /// Constructor
    explicit PositivityPreservingLimiter(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~PositivityPreservingLimiter() = default;

    /// Flow solver parameters
    const Parameters::FlowSolverParam flow_solver_param; 

    /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
    std::shared_ptr<BoundPreservingLimiterState<dim, nstate, real>> tvbLimiter;

    /// Euler physics pointer. Used to compute pressure.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

    /// Function to obtain the solution cell average
    using BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg;

    /// Applies positivity-preserving limiter to the solution.
    /// Using Zhang,Shu November 2010 Eq 3.14-3.19 or Wang, Shu 2012 Eq 3.7
    /// we apply a limiter on the global solution
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
protected:

    /// Obtain the solution cell average using tensored quadrature rules for dim >= 2
    /// Using 3.11 from Zhang, Shu Nov 2010
    /// Call get_soln_cell_avg if dim == 1
    std::array<real, nstate> get_soln_cell_avg_PPL(
        const std::array<std::array<std::vector<real>, nstate>, dim>&        soln_at_q,
        const unsigned int                                                   n_quad_pts,
        const std::vector<real>&                                             quad_weights_GLL,
        const std::vector<real>&                                             quad_weights_GL,
        double&                                                              dt);

    /// Obtain the theta value used to scale all states and enforce positivity of pressure 
    /// Using 3.16-3.18 in Zhang, Shu Nov 2010
    std::vector<real> get_theta2_Zhang2010(
        const std::vector< real >&                      p_lim,
        const std::array<real, nstate>&                 soln_cell_avg,
        const std::array<std::vector<real>, nstate>&    soln_at_q,
        const unsigned int                              n_quad_pts,
        const double                                    eps,
        const double                                    gamma);

    /// Obtain the theta value used to scale all states and enforce positivity of pressure
    /// Using 3.7 in Wang, Shu 2012
    real get_theta2_Wang2012(
        const std::array<std::vector<real>, nstate>&    soln_at_q,
        const unsigned int                              n_quad_pts,
        const double                                    p_avg);

    /// Obtain the value used to scale density and enforce positivity of density
    /// Using 3.15 from Zhang, Shu Nov 2010
    real get_density_scaling_value(
        const double    density_avg,
        const double    density_min,
        const double    pos_eps,
        const double    p_avg);

    /// Function to verify the limited solution preserves positivity of density and pressure
    /// and write back limited solution
    void write_limited_solution(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const std::array<std::vector<real>, nstate>&            soln_coeff,
        const unsigned int                                      n_shape_fns,
        const std::vector<dealii::types::global_dof_index>&     current_dofs_indices);

    // Values required to compute solution cell average in 2D/3D
    real dx; ///< Value required to compute solution cell average in 2D/3D, calculated using xmax and xmin parameters
    real dy; ///< Value required to compute solution cell average in 2D/3D, calculated using ymax and ymin parameters
    real dz; ///< Value required to compute solution cell average in 2D/3D, calculated using zmax and zmin parameters
}; // End of PositivityPreservingLimiter Class
} // PHiLiP namespace

#endif

