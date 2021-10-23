#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
PhysicsModel<dim, nstate, real>::PhysicsModel( 
    Parameters::AllParameters::PartialDifferentialEquation       baseline_physics_type,
    const int                                                    nstate_baseline_physics,
    std::unique_ptr< ModelBase<dim,nstate,real> >                physics_model_input,
    const dealii::Tensor<2,3,double>                             input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> >    manufactured_solution_function)
    : PhysicsBase<dim,nstate,real>(input_diffusion_tensor,manufactured_solution_function)
    , nstate_baseline_physics(nstate_baseline_physics)
    , n_model_equations(nstate-nstate_baseline_physics)
    , model(model_input)
{
    // Creates the baseline physics
    physics_baseline = PhysicsFactory<dim,real>::create_Physics(parameters_input, baseline_physics_type);
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsModel<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;

    // Get baseline conservative solution with nstate_baseline_physics
    std::array<real,nstate_baseline_physics> baseline_conservative_soln;
    for(int s=0; s<nstate_baseline_physics; ++s){
        baseline_conservative_soln[s] = conservative_soln[s];
    }

    // Get baseline convective flux
    std::array<dealii::Tensor<1,dim,real>,nstate_baseline_physics> baseline_conv_flux
        = this->physics_baseline->convective_flux(baseline_conservative_soln);

    // Initialize conv_flux as the model convective flux
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux
        = this->model->convective_flux(conservative_soln);

    // Add the baseline_conv_flux terms to conv_flux
    for(int s=0; s<nstate_baseline_physics; ++s){
        for (int d=0; d<dim; ++d) {
            conv_flux[s][d] += baseline_conv_flux[s][d];
        }
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsModel<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;

    // Get baseline conservative solution with nstate_baseline_physics
    std::array<real,nstate_baseline_physics> baseline_conservative_soln;
    for(int s=0; s<nstate_baseline_physics; ++s){
        baseline_conservative_soln[s] = conservative_soln[s];
    }

    // Get baseline conservative solution gradient with nstate_baseline_physics
    std::array<dealii::Tensor<1,dim,real>,nstate_baseline_physics> baseline_solution_gradient;
    for(int s=0; s<nstate_baseline_physics; ++s){
        for (int d=0; d<dim; ++d) {
            baseline_solution_gradient[s][d] = solution_gradient[s][d];
        }
    }

    // Get baseline dissipative flux
    std::array<dealii::Tensor<1,dim,real>,nstate_baseline_physics> baseline_diss_flux
        = this->physics_baseline->dissipative_flux(baseline_conservative_soln, baseline_solution_gradient);

    // Initialize diss_flux as the model dissipative flux
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux
        = this->model->dissipative_flux(conservative_soln, solution_gradient);

    // Add the baseline_diss_flux terms to diss_flux
    for(int s=0; s<nstate_baseline_physics; ++s){
        for (int d=0; d<dim; ++d) {
            diss_flux[s][d] += baseline_diss_flux[s][d];
        }
    }
    return diss_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> PhysicsModel<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &/*conservative_soln*/) const
{
    // Initialize source_term as the model source term
    std::array<real,nstate> source_term = this->model->source_term(pos);
    
    // Get the baseline physics source term
    std::array<real,nstate_baseline_physics> baseline_source_term = this->physics_baseline->source_term(pos);
    
    // Add the baseline_source_term terms to source_term
    for(int s=0; s<nstate_baseline_physics; ++s){
        source_term[s] += baseline_source_term[s];
    }

    return source_term;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsModel<dim, nstate, real>
::convective_numerical_split_flux(const std::array<real,nstate> &conservative_soln1,
                                  const std::array<real,nstate> &conservative_soln2) const
{
    // TO DO: Update for when nstate > nstate_baseline_physics
    // also, is keyword "this" required here??
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    if(nstate==nstate_baseline_physics) {
        conv_num_split_flux = this->physics_baseline->convective_numerical_split_flux(conservative_soln1,conservative_soln2);
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
    }
    return conv_num_split_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> PhysicsModel<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // TO DO: Update for when nstate > nstate_baseline_physics
    // also, is keyword "this" required here??
    std::array<real,nstate> eig;
    if(nstate==nstate_baseline_physics) {
        eig = this->physics_baseline->convective_eigenvalues(conservative_soln, normal);
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
    }
    return eig;
}

template <int dim, int nstate, typename real>
real PhysicsModel<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    // TO DO: Update for when nstate > nstate_baseline_physics
    // also, is keyword "this" required here??
    real max_eig;
    if(nstate==nstate_baseline_physics) {
        max_eig = this->physics_baseline->max_convective_eigenvalue(conservative_soln);
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
    }
    return max_eig;
}

template <int dim, int nstate, typename real>
void PhysicsModel<dim,nstate,real>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // TO DO: Update for when nstate > nstate_baseline_physics
    // also, is keyword "this" required here??
    if(nstate==nstate_baseline_physics) {
        this->physics_baseline->boundary_face_values(
                boundary_type, pos, normal_int, soln_int, soln_grad_int, 
                soln_bc, soln_grad_bc);
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
    }
}

template <int dim, int nstate, typename real>
dealii::Vector<double> PhysicsModel<dim,nstate,real>::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &duh,
    const std::vector<dealii::Tensor<2,dim> > &dduh,
    const dealii::Tensor<1,dim>               &normals,
    const dealii::Point<dim>                  &evaluation_points) const
{
    // TO DO: Update for when nstate > nstate_baseline_physics
    // also, is keyword "this" required here??
    dealii::Vector<double> computed_quantities;
    if(nstate==nstate_baseline_physics) {
        computed_quantities = this->physics_baseline->post_compute_derived_quantities_vector(
                                        uh, duh, dduh, normals, evaluation_points);
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
    }
    return computed_quantities;
}

template <int dim, int nstate, typename real>
std::vector<std::string> PhysicsModel<dim,nstate,real>
::post_get_names () const
{
    // TO DO: Update for when nstate > nstate_baseline_physics
    // also, is keyword "this" required here??
    std::vector<std::string> names;
    if(nstate==nstate_baseline_physics) {
        names = this->physics_baseline->post_get_names();
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
    }
    return names;
}

template <int dim, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> PhysicsModel<dim,nstate,real>
::post_get_data_component_interpretation () const
{
    // TO DO: Update for when nstate > nstate_baseline_physics
    // also, is keyword "this" required here??
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation;
    if(nstate==nstate_baseline_physics) {
        interpretation = this->physics_baseline->post_get_data_component_interpretation();
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
    }
    return interpretation;
}

template <int dim, int nstate, typename real>
dealii::UpdateFlags PhysicsModel<dim,nstate,real>
::post_get_needed_update_flags () const
{
    // Note: This is the exact same function as in the class PhysicsBase::Euler
    //return update_values | update_gradients;
    return dealii::update_values
           | dealii::update_quadrature_points
           ;
}

// Instantiate explicitly
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, double >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace