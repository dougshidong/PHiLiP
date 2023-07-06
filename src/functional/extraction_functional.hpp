#ifndef __PHILIP_EXTRACTION_FUNCTIONAL_H__
#define __PHILIP_EXTRACTION_FUNCTIONAL_H__

#include "functional.h"
#include "physics/model.h"
#include "physics/reynolds_averaged_navier_stokes.h"
#include "physics/negative_spalart_allmaras_rans_model.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class ExtractionFunctional : public Functional<dim, nstate, real, MeshType>
{
public:
    /// @brief Switch between displacement and momentum thickness.
    enum Functional_types { displacement_thickness, momentum_thickness };
private:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real,MeshType>::evaluate_volume_integrand;

    /// @brief Extraction start point on solid boundary surface.
    dealii::Point<dim,real> start_point;

    /// @brief Normal vector (points into fluid domain) on the solid boundary surface at extraction start point.
    dealii::Tensor<1,dim,real> start_point_normal_vector;

    /// @brief Tangential vector (points to flow direction) on the solid boundary surface at extraction start point. 
    dealii::Tensor<1,dim,real> start_point_tangential_vector;

    /// @brief Interpolated solutions at extraction start point.
    std::array<real,nstate> soln_at_start_point;

    /// @brief Interpolated solution gradients at extraction start point.
    std::array<dealii::Tensor<1,dim,real>,nstate> soln_grad_at_start_point;

    /// @brief Extraction end point. 
    dealii::Point<dim,real> end_point;

    /// @brief Interpolated solutions at extraction end point.
    std::array<real,nstate> soln_at_end_point;

    /// @brief Interpolated solution gradients at extraction end point.
    std::array<dealii::Tensor<1,dim,real>,nstate> soln_grad_at_end_point;

    /// @brief Number of sampling points over the extraction line.
    const int number_of_sampling;

    /// @brief Coordinate of sampling points over extraction line.
    std::vector<dealii::Point<dim,real>> coord_of_sampling;

    /// @brief Interpolated solutions at each sampling points over extraction line.
    std::vector<std::array<real,nstate>> soln_of_sampling;

    /// @brief Interpolated solution gradients at each sampling points over extraction line.
    std::vector<std::array<dealii::Tensor<1,dim,real>,nstate>> soln_grad_of_sampling;

    /// @brief Length of extraction line 
    real length_of_sampling;

public:
    /// @brief Converged speed of free-stream flow over extraction line.
    real U_inf;

    /// @brief Converged density of free-stream flow over extraction line.
    real density_inf;

    /// @brief Navier-Stokes physics reference.
    const Physics::NavierStokes<dim,dim+2,real> &navier_stokes_real;

    /// @brief Reynolds-averaged Navier-Stokes negative SA model pointer.
    std::shared_ptr <Physics::ReynoldsAveragedNavierStokes_SAneg<dim,dim+3,real>> rans_sa_neg_real;

public:
    /// Constructor
    ExtractionFunctional(
        std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
        const dealii::Point<dim,real> start_point_input,
        const int number_of_sampling);
    /// Destructor
    ~ExtractionFunctional(){};

    real evaluate_functional( const bool compute_dIdW = false, const bool compute_dIdX = false, const bool compute_d2I = false) override;

public:
    /// Function to interpolate solutions and solution gradients on sampling points over extraction line.
    void evaluate_straight_line_sampling_point_soln();

    /// Function to interpolate solutions and solution gradients on extraction start and end points.
    void evaluate_start_end_point_soln();

    /// Function to evaluate straight line integral.
    real evaluate_straight_line_integral(
        const Functional_types &functional_type,
        const dealii::Point<dim,real> &start_point,
        const dealii::Point<dim,real> &end_point) const;

    /// Function to evaluate integrand of straight line integral.
    real evaluate_straight_line_integrand(
        const Functional_types &functional_type,
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const;

    /// Function to evaluate integrand for displacement thickness.
    real evaluate_displacement_thickness_integrand(
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q) const;

    /// Function to evaluate integrand for momentum thickness.
    real evaluate_momentum_thickness_integrand(
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q) const;

    /// Function to evaluate displacement thickness.
    real evaluate_displacement_thickness() const;

    /// Function to evaluate momentum thickness.
    real evaluate_momentum_thickness() const;

    /// Function to evaluate edge velocity of boundary layer.
    real evaluate_edge_velocity() const;

    /// Function to evaluate wall shear stress at extraction location.
    real evaluate_wall_shear_stress() const;

    /// Function to evaluate maximum shear stress over extraction line.
    real evaluate_maximum_shear_stress() const;

    /// Function to evaluate friction velocity at extraction location.
    real evaluate_friction_velocity() const;

    /// Function to evaluate boundary layer thickness at extraction location.
    real evaluate_boundary_layer_thickness() const;

    /// Function to evaluate pressure gradient over tangential direction on solid body at extraction location.
    real evaluate_pressure_gradient_tangential() const;

    /// Function to evaluate converged free-stream values, i.e. free-stream velocity and density, over extraction line.
    void evaluate_converged_free_stream_value();
};

} // PHiLiP namespace

#endif