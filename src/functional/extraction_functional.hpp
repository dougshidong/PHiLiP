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

    /// @brief Switches between displacement and momentum thickness.
    Functional_types functional_type;

    const dealii::Point<dim,real> start_point;

    dealii::Tensor<1,dim,real> start_point_normal_vector;

    dealii::Tensor<1,dim,real> start_point_tangential_vector;

    std::array<real,nstate> soln_at_start_point;

    std::array<dealii::Tensor<1,dim,real>,nstate> soln_grad_at_start_point;

    dealii::Point<dim,real> end_point;

    std::array<real,nstate> soln_at_end_point;

    std::array<dealii::Tensor<1,dim,real>,nstate> soln_grad_at_end_point;

    const int number_of_sampling;

    std::vector<dealii::Point<dim,real>> coord_of_sampling;

    std::vector<std::array<real,nstate>> soln_of_sampling;

    std::vector<std::array<dealii::Tensor<1,dim,real>,nstate>> soln_grad_of_sampling;

    real length_of_sampling;

    real U_inf;

    real density_inf;

    /// @brief Casts DG's physics into an Euler physics reference.
    //const Physics::Euler<dim,dim+2,FadFadType> &euler_fad_fad;

    const Physics::NavierStokes<dim,dim+2,real> &navier_stokes_real;
    //std::shared_ptr< Physics::PhysicsBase<dim,dim+2,real> > navier_stokes_real;

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
    void evaluate_straight_line_sampling_point_soln();

    void evaluate_start_end_point_soln();

    /// Function to evaluate a straight line integral.
    real evaluate_straight_line_integral(
        const dealii::Point<dim,real> &start_point,
        const dealii::Point<dim,real> &end_point) const override;

    /// Function for computation of line functional term
    /** Used only in the computation of evaluate_straight_line_integral(). If not overriden returns 0. */
    real evaluate_straight_line_integrand(
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const override;

    real evaluate_displacement_thickness_integrand(
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const;

    real evaluate_momentum_thickness_integrand(
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const;

    real evaluate_displacement_thickness();

    real evaluate_momentum_thickness();

    real evaluate_edge_velocity();

    real evaluate_wall_shear_stress();

    real evaluate_friction_velocity();

    real evaluate_boundary_layer_thickness();

    void evaluate_converged_free_stream_value();
};

} // PHiLiP namespace

#endif