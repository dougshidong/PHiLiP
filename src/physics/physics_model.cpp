#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "navier_stokes.h"

#include "physics_model.h"
#include "physics_factory.h"
#include "model.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real, int nstate_baseline_physics>
PhysicsModel<dim,nstate,real,nstate_baseline_physics>::PhysicsModel( 
    const Parameters::AllParameters                              *const parameters_input,
    Parameters::AllParameters::PartialDifferentialEquation       baseline_physics_type,
    std::shared_ptr< ModelBase<dim,nstate,real> >                model_input,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> >    manufactured_solution_function,
    const bool                                                   has_nonzero_diffusion,
    const bool                                                   has_nonzero_physical_source)
    : PhysicsBase<dim,nstate,real>(parameters_input, has_nonzero_diffusion, has_nonzero_physical_source, manufactured_solution_function)
    , n_model_equations(nstate-nstate_baseline_physics)
    , physics_baseline(PhysicsFactory<dim,nstate_baseline_physics,real>::create_Physics(parameters_input, baseline_physics_type))
    , model(model_input)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , n_mpi(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank==0)
{ }

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::convective_flux (const std::array<real,nstate> &conservative_soln) const
{
    // Get baseline conservative solution with nstate_baseline_physics
    std::array<real,nstate_baseline_physics> baseline_conservative_soln;
    for(int s=0; s<nstate_baseline_physics; ++s){
        baseline_conservative_soln[s] = conservative_soln[s];
    }

    // Get baseline convective flux
    std::array<dealii::Tensor<1,dim,real>,nstate_baseline_physics> baseline_conv_flux
         = physics_baseline->convective_flux(baseline_conservative_soln);

    // Initialize conv_flux as the model convective flux
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux = model->convective_flux(conservative_soln);

    // Add the baseline_conv_flux terms to conv_flux
    for(int s=0; s<nstate_baseline_physics; ++s){
        for (int d=0; d<dim; ++d) {
            conv_flux[s][d] += baseline_conv_flux[s][d];
        }
    }
    return conv_flux;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::types::global_dof_index cell_index) const
{
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
    /* Note: Even though the physics baseline dissipative flux does not depend on cell_index, we pass it 
             anyways to accomodate the pure virtual member function defined in the PhysicsBase class */
    std::array<dealii::Tensor<1,dim,real>,nstate_baseline_physics> baseline_diss_flux
        = physics_baseline->dissipative_flux(baseline_conservative_soln, baseline_solution_gradient, cell_index);

    // Initialize diss_flux as the model dissipative flux
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux = model->dissipative_flux(conservative_soln, solution_gradient, cell_index);

    // Add the baseline_diss_flux terms to diss_flux
    for(int s=0; s<nstate_baseline_physics; ++s){
        for (int d=0; d<dim; ++d) {
            diss_flux[s][d] += baseline_diss_flux[s][d];
        }
    }
    return diss_flux;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::array<real,nstate> PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::physical_source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    // Initialize physical_source_term as the model source term
    std::array<real,nstate> physical_source_term = model->physical_source_term(pos, conservative_soln, solution_gradient, cell_index);

    // Get baseline conservative solution with nstate_baseline_physics
    std::array<real,nstate_baseline_physics> baseline_conservative_soln;
    std::array<dealii::Tensor<1,dim,real>,nstate_baseline_physics> baseline_solution_gradient;
    for(int s=0; s<nstate_baseline_physics; ++s){
        baseline_conservative_soln[s] = conservative_soln[s];
        baseline_solution_gradient[s] = solution_gradient[s];
    }

    // Get the baseline physics physical source term
    /* Note: Even though the physics baseline source term does not depend on cell_index, we pass it 
             anyways to accomodate the pure virtual member function defined in the PhysicsBase class */
    std::array<real,nstate_baseline_physics> baseline_physical_source_term = physics_baseline->physical_source_term(pos,baseline_conservative_soln,baseline_solution_gradient,cell_index);

    // Add the baseline_physical_source_term terms to source_term
    for(int s=0; s<nstate_baseline_physics; ++s){
        physical_source_term[s] += baseline_physical_source_term[s];
    }

    return physical_source_term;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::array<real,nstate> PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &conservative_soln,
    const real current_time,
    const dealii::types::global_dof_index cell_index) const
{
    // Initialize source_term as the model source term
    std::array<real,nstate> source_term = model->source_term(
        pos,
        conservative_soln,
        current_time,
        cell_index);
    
    // Get baseline conservative solution with nstate_baseline_physics
    std::array<real,nstate_baseline_physics> baseline_conservative_soln;
    for(int s=0; s<nstate_baseline_physics; ++s){
        baseline_conservative_soln[s] = conservative_soln[s];
    }

    // Get the baseline physics source term
    /* Note: Even though the physics baseline source term does not depend on cell_index, we pass it 
             anyways to accomodate the pure virtual member function defined in the PhysicsBase class */
    std::array<real,nstate_baseline_physics> baseline_source_term = physics_baseline->source_term(
        pos,
        baseline_conservative_soln,
        current_time,
        cell_index);

    // Add the baseline_source_term terms to source_term
    for(int s=0; s<nstate_baseline_physics; ++s){
        source_term[s] += baseline_source_term[s];
    }

    return source_term;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::convective_numerical_split_flux(const std::array<real,nstate> &conservative_soln1,
                                  const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    if constexpr(nstate==nstate_baseline_physics) {
        conv_num_split_flux = physics_baseline->convective_numerical_split_flux(conservative_soln1,conservative_soln2);
    } else {
        pcout << "Error: convective_numerical_split_flux() not implemented for nstate!=nstate_baseline_physics." << std::endl;
        pcout << "Aborting..." << std::endl;
        std::abort();
    }    
    return conv_num_split_flux;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::array<real,nstate> PhysicsModel<dim, nstate, real, nstate_baseline_physics>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,nstate> entropy_var;
    if constexpr(nstate==nstate_baseline_physics) {
        entropy_var = physics_baseline->compute_entropy_variables(conservative_soln);
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
        std::abort();
    }
    return entropy_var;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::array<real,nstate> PhysicsModel<dim, nstate, real, nstate_baseline_physics>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    std::array<real,nstate> conservative_soln;
    if constexpr(nstate==nstate_baseline_physics) {
        conservative_soln = physics_baseline->compute_conservative_variables_from_entropy_variables(entropy_var);
    } else {
        // TO DO, make use of the physics_model object for nstate>nstate_baseline_physics
        std::abort();
    }
    return conservative_soln;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::array<real,nstate> PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::convective_eigenvalues (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    std::array<real,nstate> eig;
    if constexpr(nstate==nstate_baseline_physics) {
        eig = physics_baseline->convective_eigenvalues(conservative_soln, normal);
    } else {
        eig = model->convective_eigenvalues(conservative_soln, normal);
        std::array<real,nstate_baseline_physics> baseline_conservative_soln;
        for(int s=0; s<nstate_baseline_physics; ++s){
            baseline_conservative_soln[s] = conservative_soln[s];
        }
        std::array<real,nstate_baseline_physics> baseline_eig = physics_baseline->convective_eigenvalues(baseline_conservative_soln, normal);
        for(int s=0; s<nstate_baseline_physics; ++s){
            if(eig[s]!=0.0){
                pcout << "Error: PhysicsModel does not currently support additional convective flux terms." << std::endl; 
                pcout << "Aborting..." << std::endl;
                std::abort();
            } else {
                eig[s] += baseline_eig[s];
            }
        }  
    }

    return eig;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
real PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    real max_eig;
    if constexpr(nstate==nstate_baseline_physics) {
        max_eig = physics_baseline->max_convective_eigenvalue(conservative_soln);
    } else {
        max_eig = model->max_convective_eigenvalue(conservative_soln);
        std::array<real,nstate_baseline_physics> baseline_conservative_soln;
        for(int s=0; s<nstate_baseline_physics; ++s){
            baseline_conservative_soln[s] = conservative_soln[s];
        }
        real baseline_max_eig = physics_baseline->max_convective_eigenvalue(baseline_conservative_soln);
        max_eig = max_eig > baseline_max_eig ? max_eig : baseline_max_eig;
    }
    return max_eig;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
real PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::max_convective_normal_eigenvalue (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    real max_eig;
    if constexpr(nstate==nstate_baseline_physics) {
        max_eig = physics_baseline->max_convective_normal_eigenvalue(conservative_soln,normal);
    } else {
        max_eig = model->max_convective_normal_eigenvalue(conservative_soln,normal);
        std::array<real,nstate_baseline_physics> baseline_conservative_soln;
        for(int s=0; s<nstate_baseline_physics; ++s){
            baseline_conservative_soln[s] = conservative_soln[s];
        }
        real baseline_max_eig = physics_baseline->max_convective_normal_eigenvalue(baseline_conservative_soln,normal);
        max_eig = max_eig > baseline_max_eig ? max_eig : baseline_max_eig;
    }
    return max_eig;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
real PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::max_viscous_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    return 0.0;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
void PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    if constexpr(nstate==nstate_baseline_physics) {
        physics_baseline->boundary_face_values(
                boundary_type, pos, normal_int, soln_int, soln_grad_int, 
                soln_bc, soln_grad_bc);
    } else {
        std::array<real,nstate_baseline_physics> baseline_soln_int;
        std::array<dealii::Tensor<1,dim,real>,nstate_baseline_physics> baseline_soln_grad_int;
        for(int s=0; s<nstate_baseline_physics; ++s){
            baseline_soln_int[s] = soln_int[s];
            baseline_soln_grad_int[s] = soln_grad_int[s];
        }

        std::array<real,nstate_baseline_physics> baseline_soln_bc;
        std::array<dealii::Tensor<1,dim,real>,nstate_baseline_physics> baseline_soln_grad_bc;

        for (int istate=0; istate<nstate_baseline_physics; istate++) {
            baseline_soln_bc[istate]      = 0;
            baseline_soln_grad_bc[istate] = 0;
        }

        physics_baseline->boundary_face_values(
                boundary_type, pos, normal_int, baseline_soln_int, baseline_soln_grad_int, 
                baseline_soln_bc, baseline_soln_grad_bc);
        
        model->boundary_face_values(
                boundary_type, pos, normal_int, soln_int, soln_grad_int, 
                soln_bc, soln_grad_bc);

        for(int s=0; s<nstate_baseline_physics; ++s){
            soln_bc[s] += baseline_soln_bc[s];
            soln_grad_bc[s] += baseline_soln_grad_bc[s];
        }
    }
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
dealii::Vector<double> PhysicsModel<dim,nstate,real,nstate_baseline_physics>::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &duh,
    const std::vector<dealii::Tensor<2,dim> > &dduh,
    const dealii::Tensor<1,dim>               &normals,
    const dealii::Point<dim>                  &evaluation_points) const
{
    dealii::Vector<double> computed_quantities;
    if constexpr(nstate==nstate_baseline_physics) {
        computed_quantities = physics_baseline->post_compute_derived_quantities_vector(
                                        uh, duh, dduh, normals, evaluation_points);
    } else {
        dealii::Vector<double> computed_quantities_model;
        computed_quantities_model = model->post_compute_derived_quantities_vector(
                                        uh, duh, dduh, normals, evaluation_points);
        dealii::Vector<double> computed_quantities_base;
        computed_quantities_base = physics_baseline->post_compute_derived_quantities_vector(
                                        uh, duh, dduh, normals, evaluation_points);

        dealii::Vector<double> computed_quantities_total(computed_quantities_base.size()+computed_quantities_model.size());
        for (unsigned int i=0;i<computed_quantities_base.size();i++){
            computed_quantities_total(i) = computed_quantities_base(i);
        }
        for (unsigned int i=0;i<computed_quantities_model.size();i++){
            computed_quantities_total(i+computed_quantities_base.size()) = computed_quantities_model(i);
        }
        computed_quantities = computed_quantities_total;
    }
    return computed_quantities;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::vector<std::string> PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::post_get_names () const
{
    std::vector<std::string> names;
    if constexpr(nstate==nstate_baseline_physics) {
        names = physics_baseline->post_get_names();
    } else {
        std::vector<std::string> names_model;
        names = physics_baseline->post_get_names();
        names_model = model->post_get_names();
        names.insert(names.end(),names_model.begin(),names_model.end()); 
    }
    return names;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation;
    if constexpr(nstate==nstate_baseline_physics) {
        interpretation = physics_baseline->post_get_data_component_interpretation();
    } else {
        std::vector<DCI::DataComponentInterpretation> interpretation_model;
        interpretation = physics_baseline->post_get_data_component_interpretation();
        interpretation_model = model->post_get_data_component_interpretation();
        interpretation.insert(interpretation.end(),interpretation_model.begin(),interpretation_model.end());
    }
    return interpretation;
}

template <int dim, int nstate, typename real, int nstate_baseline_physics>
dealii::UpdateFlags PhysicsModel<dim,nstate,real,nstate_baseline_physics>
::post_get_needed_update_flags () const
{
    // Note: This is the exact same function as in the class PhysicsBase::Euler
    //return update_values | update_gradients;
    return dealii::update_values
           | dealii::update_quadrature_points
           ;
}

// Instantiate explicitly
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, double    , PHILIP_DIM+2 >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, FadType   , PHILIP_DIM+2 >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, RadType   , PHILIP_DIM+2 >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, FadFadType, PHILIP_DIM+2 >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+2, RadFadType, PHILIP_DIM+2 >;

template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+3, double    , PHILIP_DIM+2 >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+3, FadType   , PHILIP_DIM+2 >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+3, RadType   , PHILIP_DIM+2 >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+3, FadFadType, PHILIP_DIM+2 >;
template class PhysicsModel < PHILIP_DIM, PHILIP_DIM+3, RadFadType, PHILIP_DIM+2 >;

} // Physics namespace
} // PHiLiP namespace
