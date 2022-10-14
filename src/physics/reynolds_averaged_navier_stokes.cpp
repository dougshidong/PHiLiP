#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "model.h"
#include "reynolds_averaged_navier_stokes.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Reynolds Averaged Navier Stokes (RANS) Base Class
//================================================================
template <int dim, int nstate, typename real>
ReynoldsAveragedNavierStokesBase<dim, nstate, real>::ReynoldsAveragedNavierStokesBase(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function)
    : ModelBase<dim,nstate,real>(manufactured_solution_function) 
    , turbulent_prandtl_number(turbulent_prandtl_number)
    , navier_stokes_physics(std::make_unique < NavierStokes<dim,dim+2,real> > (
            ref_length,
            gamma_gas,
            mach_inf,
            angle_of_attack,
            side_slip_angle,
            prandtl_number,
            reynolds_number_inf,
            isothermal_wall_temperature,
            thermal_boundary_condition_type,
            manufactured_solution_function))
{
    static_assert(nstate>=dim+3, "ModelBase::ReynoldsAveragedNavierStokesBase() should be created with nstate>=dim+3");
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::get_vector_magnitude_sqr (
    const dealii::Tensor<1,3,real2> &vector) const
{
    real2 vector_magnitude_sqr; // complex initializes it as 0+0i
    if(std::is_same<real2,real>::value){
        vector_magnitude_sqr = 0.0;
    }
    for (int i=0; i<3; ++i) {
        vector_magnitude_sqr += vector[i]*vector[i];
    }
    return vector_magnitude_sqr;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::get_tensor_magnitude_sqr (
    const dealii::Tensor<2,dim,real2> &tensor) const
{
    real2 tensor_magnitude_sqr; // complex initializes it as 0+0i
    if(std::is_same<real2,real>::value){
        tensor_magnitude_sqr = 0.0;
    }
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            tensor_magnitude_sqr += tensor[i][j]*tensor[i][j];
        }
    }
    return tensor_magnitude_sqr;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convective_flux (
    const std::array<real,nstate> &conservative_soln) const
{
    return convective_flux_templated<real>(conservative_soln);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convective_flux_templated (
    const std::array<real2,nstate> &conservative_soln) const
{
    const std::array<real2,dim+2> conservative_soln_rans = extract_rans_conservative_solution(conservative_soln);
    const dealii::Tensor<1,dim,real2> vel = this->navier_stokes_physics->template compute_velocities<real2>(conservative_soln_rans); // from Euler
    std::array<dealii::Tensor<1,dim,real2>,nstate> conv_flux;

    for (int flux_dim=0; flux_dim<dim+2; ++flux_dim) {
        conv_flux[flux_dim] = 0.0; // No additional convective terms for RANS
    }
    // convective flux of additional RANS turbulence model
    for (int flux_dim=dim+2; flux_dim<nstate; ++flux_dim) {
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim) {
            conv_flux[flux_dim][velocity_dim] = conservative_soln[flux_dim]*vel[velocity_dim]; // Convective terms for turbulence model
        }
    }
    return conv_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::types::global_dof_index cell_index) const
{   
    return dissipative_flux_templated<real>(conservative_soln,solution_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::dissipative_flux_templated (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
    const dealii::types::global_dof_index /*cell_index*/) const
{   

    const std::array<real2,dim+2> conservative_soln_rans = extract_rans_conservative_solution(conservative_soln);
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> solution_gradient_rans = extract_rans_solution_gradient(solution_gradient);

    // Step 1,2: Primitive solution and Gradient of primitive solution
    const std::array<real2,dim+2> primitive_soln_rans = this->navier_stokes_physics->convert_conservative_to_primitive(conservative_soln_rans); // from Euler
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> primitive_soln_gradient_rans = this->navier_stokes_physics->convert_conservative_gradient_to_primitive_gradient(conservative_soln_rans, solution_gradient_rans);
    const std::array<real2,nstate-(dim+2)> primitive_soln_turbulence_model = this->convert_conservative_to_primitive_turbulence_model(conservative_soln); 
    const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> primitive_soln_gradient_turbulence_model = this->convert_conservative_gradient_to_primitive_gradient_turbulence_model(conservative_soln, solution_gradient);

    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const dealii::Tensor<1,dim,real2> vel = this->navier_stokes_physics->extract_velocities_from_primitive(primitive_soln_rans); // from Euler
    // Templated virtual member functions
    dealii::Tensor<2,dim,real2> viscous_stress_tensor;
    dealii::Tensor<1,dim,real2> heat_flux;
    if constexpr(std::is_same<real2,real>::value){ 
        viscous_stress_tensor = compute_Reynolds_stress_tensor(primitive_soln_rans, primitive_soln_gradient_rans,primitive_soln_turbulence_model);
        heat_flux = compute_Reynolds_heat_flux(primitive_soln_rans, primitive_soln_gradient_rans,primitive_soln_turbulence_model);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        viscous_stress_tensor = compute_Reynolds_stress_tensor_fad(primitive_soln_rans, primitive_soln_gradient_rans,primitive_soln_turbulence_model);
        heat_flux = compute_Reynolds_heat_flux_fad(primitive_soln_rans, primitive_soln_gradient_rans,primitive_soln_turbulence_model);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> dissipative_flux_templated(): real2!=real || real2!=FadType)" << std::endl;
        std::abort();
    }
    
    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    std::array<dealii::Tensor<1,dim,real2>,dim+2> viscous_flux_rans
        = this->navier_stokes_physics->dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux(vel,viscous_stress_tensor,heat_flux);

    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> viscous_flux_turbulence_model
        = this->dissipative_flux_turbulence_model(primitive_soln_rans,primitive_soln_turbulence_model,primitive_soln_gradient_turbulence_model);

    std::array<dealii::Tensor<1,dim,real2>,nstate> viscous_flux;
    for(int flux_dim=0; flux_dim<dim+2; ++flux_dim)
    {
        viscous_flux[flux_dim] = viscous_flux_rans[flux_dim];
    }
    for(int flux_dim=dim+2; flux_dim<nstate; ++flux_dim)
    {
        viscous_flux[flux_dim] = viscous_flux_turbulence_model[flux_dim-(dim+2)];
    }
    
    return viscous_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<real2,dim+2> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::extract_rans_conservative_solution (
    const std::array<real2,nstate> &conservative_soln) const
{   
    std::array<real2,dim+2> conservative_soln_rans;
    for(int i=0; i<dim+2; ++i){
        conservative_soln_rans[i] = conservative_soln[i];
    }
 
    return conservative_soln_rans;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,dim+2> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::extract_rans_solution_gradient (
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const
{   
    std::array<dealii::Tensor<1,dim,real2>,dim+2> solution_gradient_rans;
    for(int i=0; i<dim+2; ++i){
        solution_gradient_rans[i] = solution_gradient[i];
    }
 
    return solution_gradient_rans;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::dissipative_flux_turbulence_model (
    const std::array<real2,dim+2> &primitive_soln_rans,
    const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model,
    const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> &primitive_solution_gradient_turbulence_model) const
{   
    std::array<real2,nstate-(dim+2)> effective_viscosity_turbulence_model; 

    if constexpr(std::is_same<real2,real>::value){ 
        effective_viscosity_turbulence_model = compute_effective_viscosity_turbulence_model(primitive_soln_rans, primitive_soln_turbulence_model);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        effective_viscosity_turbulence_model = compute_effective_viscosity_turbulence_model_fad(primitive_soln_rans, primitive_soln_turbulence_model);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> dissipative_flux_turbulence_model(): real2!=real || real2!=FadType)" << std::endl;
        std::abort();
    }

    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> viscous_flux_turbulence_model;

    for(int i=0; i<nstate-(dim+2); ++i){
        for(int j=0; j<dim; ++j){
            viscous_flux_turbulence_model[i][j] = -effective_viscosity_turbulence_model[i]*primitive_solution_gradient_turbulence_model[i][j];
        }
    }
    
    return viscous_flux_turbulence_model;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<real2,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convert_conservative_to_primitive_turbulence_model (
    const std::array<real2,nstate> &conservative_soln) const
{   
    std::array<real2,nstate-(dim+2)> primitive_soln_turbulence_model;
    for(int i=0; i<nstate-(dim+2); ++i){
        primitive_soln_turbulence_model[i] = conservative_soln[dim+2+i]/conservative_soln[0];
    }
 
    return primitive_soln_turbulence_model;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convert_conservative_gradient_to_primitive_gradient_turbulence_model (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const
{   
    const std::array<real2,nstate-(dim+2)> primitive_soln_turbulence_model = this->convert_conservative_to_primitive_turbulence_model(conservative_soln); 
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> primitive_soln_gradient_turbulence_model;

    for(int i=0; i<nstate-(dim+2); ++i){
        for(int j=0; j<dim; ++j){
            primitive_soln_gradient_turbulence_model[i][j] = (solution_gradient[dim+2+i][j]-primitive_soln_turbulence_model[i]*solution_gradient[0][j])/conservative_soln[0];
        }
    }
 
    return primitive_soln_gradient_turbulence_model;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convective_numerical_split_flux(const std::array<real,nstate> &conservative_soln1,
                                  const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    std::array<real,dim+2> conservative_soln1_rans = extract_rans_conservative_solution(conservative_soln1);
    std::array<real,dim+2> conservative_soln2_rans = extract_rans_conservative_solution(conservative_soln2);

    const real mean_density = this->navier_stokes_physics->compute_mean_density(conservative_soln1_rans, conservative_soln2_rans);
    const dealii::Tensor<1,dim,real> mean_velocities = this->navier_stokes_physics->compute_mean_velocities(conservative_soln1_rans,conservative_soln2_rans);
    const std::array<real,nstate-(dim+2)> mean_turbulence_property = compute_mean_turbulence_property(conservative_soln1, conservative_soln2);

    for (int i=0;i<dim+2;++i){
        conv_num_split_flux[i] = 0.0;
    }
    for (int i=dim+2;i<nstate;++i)
    {
        for (int flux_dim = 0; flux_dim < dim; ++flux_dim){
            conv_num_split_flux[i][flux_dim] = mean_density*mean_velocities[flux_dim]*mean_turbulence_property[i-(dim+2)];
        }
    }

    return conv_num_split_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::compute_mean_turbulence_property(const std::array<real,nstate> &conservative_soln1,
                                   const std::array<real,nstate> &conservative_soln2) const
{
    std::array<real,nstate-(dim+2)> mean_turbulence_property;

    const std::array<real,nstate-(dim+2)> primitive_soln1_turbulence_model = convert_conservative_to_primitive_turbulence_model(conservative_soln1); 
    const std::array<real,nstate-(dim+2)> primitive_soln2_turbulence_model = convert_conservative_to_primitive_turbulence_model(conservative_soln2); 

    for (int i=0;i<nstate-(dim+2);++i){
        mean_turbulence_property[i] = (primitive_soln1_turbulence_model[i]+primitive_soln2_turbulence_model[i])/2.0;
    }

    return mean_turbulence_property;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    const std::array<real,dim+2> conservative_soln_rans = extract_rans_conservative_solution(conservative_soln);
    std::array<real,nstate> eig;
    const real vel_dot_n = this->navier_stokes_physics->convective_eigenvalues(conservative_soln_rans,normal)[0];
    for (int i=0; i<dim+2; ++i) {
        eig[i] = 0.0;
    }
    for (int i=dim+2; i<nstate; ++i) {
        eig[i] = vel_dot_n;
    }
    return eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    const std::array<real,dim+2> conservative_soln_rans = extract_rans_conservative_solution(conservative_soln);

    const dealii::Tensor<1,dim,real> vel = this->navier_stokes_physics->template compute_velocities<real>(conservative_soln_rans);

    const real vel2 = this->navier_stokes_physics->template compute_velocity_squared<real>(vel);

    const real max_eig = sqrt(vel2);

    return max_eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index /*cell_index*/) const
{
    std::array<real,nstate> physical_source;
    physical_source = this->compute_production_dissipation_cross_term(pos, conservative_soln, solution_gradient);

    return physical_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &/*solution*/,
        const dealii::types::global_dof_index cell_index) const
{
    std::array<real,nstate> conv_source_term = convective_source_term(pos);
    std::array<real,nstate> diss_source_term = dissipative_source_term(pos,cell_index);
    std::array<real,nstate> phys_source_source_term = physical_source_source_term(pos,cell_index);
    std::array<real,nstate> source_term;
    for (int s=0; s<nstate; ++s)
    {
        source_term[s] = conv_source_term[s] + diss_source_term[s] - phys_source_source_term[s];
    }
    return source_term;
}

//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convective_dissipative_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &/*solution*/,
        const dealii::types::global_dof_index cell_index) const
{
    std::array<real,nstate> conv_source_term = convective_source_term(pos);
    std::array<real,nstate> diss_source_term = dissipative_source_term(pos,cell_index);
    std::array<real,nstate> convective_dissipative_source_term;
    for (int s=0; s<nstate; ++s)
    {
        convective_dissipative_source_term[s] = conv_source_term[s] + diss_source_term[s];
    }
    return convective_dissipative_source_term;
}

// Returns the value from a CoDiPack or Sacado variable.
template<typename real>
double getValue(const real &x) {
    if constexpr(std::is_same<real,double>::value) {
        return x;
    }
    else if constexpr(std::is_same<real,FadType>::value) {
        return x.val(); // sacado
    } 
    else if constexpr(std::is_same<real,FadFadType>::value) {
        return x.val().val(); // sacado
    }
    else if constexpr(std::is_same<real,RadType>::value) {
      return x.value(); // CoDiPack
    } 
    else if(std::is_same<real,RadFadType>::value) {
        return x.value().value(); // CoDiPack
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convective_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    for (int s=0; s<nstate; ++s) {
        adtype ADvar(nstate, s, getValue<real>(conservative_soln[s])); // create AD variable
        AD_conservative_soln[s] = ADvar;
    }

    // Compute AD convective flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_convective_flux = convective_flux_templated<adtype>(AD_conservative_soln);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; ++sp) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; ++s) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;++d) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_convective_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::dissipative_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::Tensor<1,dim,real> &normal,
    const dealii::types::global_dof_index cell_index) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_solution_gradient;
    for (int s=0; s<nstate; ++s) {
        adtype ADvar(nstate, s, getValue<real>(conservative_soln[s])); // create AD variable
        AD_conservative_soln[s] = ADvar;
        for (int d=0;d<dim;++d) {
            AD_solution_gradient[s][d] = getValue<real>(solution_gradient[s][d]);
        }
    }

    // Compute AD dissipative flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_dissipative_flux = dissipative_flux_templated<adtype>(AD_conservative_soln, AD_solution_gradient, cell_index);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; ++sp) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; ++s) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;++d) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_dissipative_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::dissipative_flux_directional_jacobian_wrt_gradient_component (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::Tensor<1,dim,real> &normal,
    const int d_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_solution_gradient;
    for (int s=0; s<nstate; ++s) {
        AD_conservative_soln[s] = getValue<real>(conservative_soln[s]);
        for (int d=0;d<dim;++d) {
            if(d == d_gradient){
                adtype ADvar(nstate, s, getValue<real>(solution_gradient[s][d])); // create AD variable
                AD_solution_gradient[s][d] = ADvar;
            }
            else {
                AD_solution_gradient[s][d] = getValue<real>(solution_gradient[s][d]);
            }
        }
    }

    // Compute AD dissipative flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_dissipative_flux = dissipative_flux_templated<adtype>(AD_conservative_soln, AD_solution_gradient, cell_index);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; ++sp) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; ++s) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;++d) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_dissipative_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::get_manufactured_solution_value (
    const dealii::Point<dim,real> &pos) const
{
    std::array<real,nstate> manufactured_solution;
    for (int s=0; s<nstate; ++s) {
        manufactured_solution[s] = this->manufactured_solution_function->value (pos, s);
        if (s==0) {
            assert(manufactured_solution[s] > 0);
        }
    }
    return manufactured_solution;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::get_manufactured_solution_gradient (
    const dealii::Point<dim,real> &pos) const
{
    std::vector<dealii::Tensor<1,dim,real>> manufactured_solution_gradient_dealii(nstate);
    this->manufactured_solution_function->vector_gradient(pos,manufactured_solution_gradient_dealii);
    std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient;
    for (int d=0;d<dim;++d) {
        for (int s=0; s<nstate; ++s) {
            manufactured_solution_gradient[s][d] = manufactured_solution_gradient_dealii[s][d];
        }
    }
    return manufactured_solution_gradient;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convective_source_term (
    const dealii::Point<dim,real> &pos) const
{    
    // Get Manufactured Solution values
    const std::array<real,nstate> manufactured_solution = get_manufactured_solution_value(pos);
    
    // Get Manufactured Solution gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient = get_manufactured_solution_gradient(pos);

    dealii::Tensor<1,nstate,real> convective_flux_divergence;
    for (int d=0;d<dim;++d) {
        dealii::Tensor<1,dim,real> normal;
        normal[d] = 1.0;
        const dealii::Tensor<2,nstate,real> jacobian = convective_flux_directional_jacobian(manufactured_solution, normal);

        //convective_flux_divergence += jacobian*manufactured_solution_gradient[d]; <-- needs second term! (jac wrt gradient)
        for (int sr = 0; sr < nstate; ++sr) {
            real jac_grad_row = 0.0;
            for (int sc = 0; sc < nstate; ++sc) {
                jac_grad_row += jacobian[sr][sc]*manufactured_solution_gradient[sc][d];
            }
            convective_flux_divergence[sr] += jac_grad_row;
        }
    }
    std::array<real,nstate> convective_source_term;
    for (int s=0; s<nstate; ++s) {
        convective_source_term[s] = convective_flux_divergence[s];
    }

    return convective_source_term;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::dissipative_source_term (
    const dealii::Point<dim,real> &pos,
    const dealii::types::global_dof_index cell_index) const
{    
    /** This function is same as the one in NavierStokes; 
     *  TO DO: Reduce this code repetition in the future by moving it elsewhere
     * */

    // Get Manufactured Solution values
    const std::array<real,nstate> manufactured_solution = get_manufactured_solution_value(pos); // from Euler
    
    // Get Manufactured Solution gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient = get_manufactured_solution_gradient(pos); // from Euler
    
    // Get Manufactured Solution hessian
    std::array<dealii::SymmetricTensor<2,dim,real>,nstate> manufactured_solution_hessian;
    for (int s=0; s<nstate; ++s) {
        dealii::SymmetricTensor<2,dim,real> hessian = this->manufactured_solution_function->hessian(pos,s);
        for (int dr=0;dr<dim;++dr) {
            for (int dc=0;dc<dim;++dc) {
                manufactured_solution_hessian[s][dr][dc] = hessian[dr][dc];
            }
        }
    }

    // First term -- wrt to the conservative variables
    // This is similar, should simply provide this function a flux_directional_jacobian() -- could restructure later
    dealii::Tensor<1,nstate,real> dissipative_flux_divergence;
    for (int d=0;d<dim;++d) {
        dealii::Tensor<1,dim,real> normal;
        normal[d] = 1.0;
        const dealii::Tensor<2,nstate,real> jacobian = dissipative_flux_directional_jacobian(manufactured_solution, manufactured_solution_gradient, normal, cell_index);
        
        // get the directional jacobian wrt gradient
        std::array<dealii::Tensor<2,nstate,real>,dim> jacobian_wrt_gradient;
        for (int d_gradient=0;d_gradient<dim;++d_gradient) {
            
            // get the directional jacobian wrt gradient component (x,y,z)
            const dealii::Tensor<2,nstate,real> jacobian_wrt_gradient_component = dissipative_flux_directional_jacobian_wrt_gradient_component(manufactured_solution, manufactured_solution_gradient, normal, d_gradient, cell_index);
            
            // store each component in jacobian_wrt_gradient -- could do this in the function used above
            for (int sr = 0; sr < nstate; ++sr) {
                for (int sc = 0; sc < nstate; ++sc) {
                    jacobian_wrt_gradient[d_gradient][sr][sc] = jacobian_wrt_gradient_component[sr][sc];
                }
            }
        }

        //dissipative_flux_divergence += jacobian*manufactured_solution_gradient[d]; <-- needs second term! (jac wrt gradient)
        for (int sr = 0; sr < nstate; ++sr) {
            real jac_grad_row = 0.0;
            for (int sc = 0; sc < nstate; ++sc) {
                jac_grad_row += jacobian[sr][sc]*manufactured_solution_gradient[sc][d]; // Euler is the same as this
                // Second term -- wrt to the gradient of conservative variables
                // -- add the contribution of each gradient component (e.g. x,y,z for dim==3)
                for (int d_gradient=0;d_gradient<dim;++d_gradient) {
                    jac_grad_row += jacobian_wrt_gradient[d_gradient][sr][sc]*manufactured_solution_hessian[sc][d_gradient][d]; // symmetric so d indexing works both ways
                }
            }
            dissipative_flux_divergence[sr] += jac_grad_row;
        }
    }
    std::array<real,nstate> dissipative_source_term;
    for (int s=0; s<nstate; ++s) {
        dissipative_source_term[s] = dissipative_flux_divergence[s];
    }

    return dissipative_source_term;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::physical_source_source_term (
    const dealii::Point<dim,real> &pos,
    const dealii::types::global_dof_index cell_index) const
{    
    // Get Manufactured Solution values
    const std::array<real,nstate> manufactured_solution = get_manufactured_solution_value(pos); // from Euler
    
    // Get Manufactured Solution gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient = get_manufactured_solution_gradient(pos); // from Euler
    
    std::array<real,nstate> physical_source_source_term;
    for (int i=0;i<nstate;++i){
        physical_source_source_term = physical_source_term(pos, manufactured_solution, manufactured_solution_gradient, cell_index);
    }

    return physical_source_source_term;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::boundary_manufactured_solution (
    const dealii::Point<dim, real> &pos,
    const dealii::Tensor<1,dim,real> &normal_int,
    const std::array<real,nstate> &soln_int,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
    std::array<real,nstate> &soln_bc,
    std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // Manufactured solution
    std::array<real,nstate> conservative_boundary_values;
    std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients;
    for (int s=0; s<nstate; ++s) {
        conservative_boundary_values[s] = this->manufactured_solution_function->value (pos, s);
        boundary_gradients[s] = this->manufactured_solution_function->gradient (pos, s);
    }

    for (int istate=dim+2; istate<nstate; ++istate) {

        std::array<real,nstate> characteristic_dot_n = convective_eigenvalues(conservative_boundary_values, normal_int);
        const bool inflow = (characteristic_dot_n[istate] <= 0.);

        if (inflow) { // Dirichlet boundary condition
            soln_bc[istate] = conservative_boundary_values[istate];
            soln_grad_bc[istate] = soln_grad_int[istate];
        } else { // Neumann boundary condition
            soln_bc[istate] = soln_int[istate];
            soln_grad_bc[istate] = soln_grad_int[istate];
        }
    }
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- ReynoldsAveragedNavierStokesBase
template class ReynoldsAveragedNavierStokesBase         < PHILIP_DIM, PHILIP_DIM+3, double >;
template class ReynoldsAveragedNavierStokesBase         < PHILIP_DIM, PHILIP_DIM+3, FadType  >;
template class ReynoldsAveragedNavierStokesBase         < PHILIP_DIM, PHILIP_DIM+3, RadType  >;
template class ReynoldsAveragedNavierStokesBase         < PHILIP_DIM, PHILIP_DIM+3, FadFadType >;
template class ReynoldsAveragedNavierStokesBase         < PHILIP_DIM, PHILIP_DIM+3, RadFadType >;
//-------------------------------------------------------------------------------------
// Templated members used by derived classes, defined in respective parent classes
//-------------------------------------------------------------------------------------
// -- get_tensor_magnitude_sqr()
template double     ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::get_tensor_magnitude_sqr< double     >(const dealii::Tensor<2,PHILIP_DIM,double    > &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template RadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::get_tensor_magnitude_sqr< RadType    >(const dealii::Tensor<2,PHILIP_DIM,RadType   > &tensor) const;
template FadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::get_tensor_magnitude_sqr< FadFadType >(const dealii::Tensor<2,PHILIP_DIM,FadFadType> &tensor) const;
template RadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::get_tensor_magnitude_sqr< RadFadType >(const dealii::Tensor<2,PHILIP_DIM,RadFadType> &tensor) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::get_tensor_magnitude_sqr< FadType >(const dealii::Tensor<2,PHILIP_DIM,FadType> &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::get_tensor_magnitude_sqr< FadType >(const dealii::Tensor<2,PHILIP_DIM,FadType> &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::get_tensor_magnitude_sqr< FadType >(const dealii::Tensor<2,PHILIP_DIM,FadType> &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::get_tensor_magnitude_sqr< FadType >(const dealii::Tensor<2,PHILIP_DIM,FadType> &tensor) const;

// -- get_vector_magnitude_sqr()
template double     ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::get_vector_magnitude_sqr< double     >(const dealii::Tensor<1,3,double    > &vector) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::get_vector_magnitude_sqr< FadType    >(const dealii::Tensor<1,3,FadType   > &vector) const;
template RadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::get_vector_magnitude_sqr< RadType    >(const dealii::Tensor<1,3,RadType   > &vector) const;
template FadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::get_vector_magnitude_sqr< FadFadType >(const dealii::Tensor<1,3,FadFadType> &vector) const;
template RadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::get_vector_magnitude_sqr< RadFadType >(const dealii::Tensor<1,3,RadFadType> &vector) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::get_vector_magnitude_sqr< FadType >(const dealii::Tensor<1,3,FadType> &vector) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::get_vector_magnitude_sqr< FadType >(const dealii::Tensor<1,3,FadType> &vector) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::get_vector_magnitude_sqr< FadType >(const dealii::Tensor<1,3,FadType> &vector) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::get_vector_magnitude_sqr< FadType >(const dealii::Tensor<1,3,FadType> &vector) const;

// -- extract_rans_conservative_solution()
template std::array<double,     PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::extract_rans_conservative_solution< double     >(const std::array<double,     PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadType,    PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::extract_rans_conservative_solution< FadType    >(const std::array<FadType,    PHILIP_DIM+3> &conservative_soln) const;
template std::array<RadType,    PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::extract_rans_conservative_solution< RadType    >(const std::array<RadType,    PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadFadType, PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::extract_rans_conservative_solution< FadFadType >(const std::array<FadFadType, PHILIP_DIM+3> &conservative_soln) const;
template std::array<RadFadType, PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::extract_rans_conservative_solution< RadFadType >(const std::array<RadFadType, PHILIP_DIM+3> &conservative_soln) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template std::array<FadType, PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::extract_rans_conservative_solution< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadType, PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::extract_rans_conservative_solution< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadType, PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::extract_rans_conservative_solution< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadType, PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::extract_rans_conservative_solution< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln) const;

// -- extract_rans_solution_gradient()
template std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::extract_rans_solution_gradient< double     >(const std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::extract_rans_solution_gradient< FadType    >(const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::extract_rans_solution_gradient< RadType    >(const std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::extract_rans_solution_gradient< FadFadType >(const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::extract_rans_solution_gradient< RadFadType >(const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+3> &solution_gradient) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::extract_rans_solution_gradient< FadType >(const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::extract_rans_solution_gradient< FadType >(const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::extract_rans_solution_gradient< FadType >(const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+2> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::extract_rans_solution_gradient< FadType >(const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3> &solution_gradient) const;

// -- convert_conservative_to_primitive_turbulence_model()
template std::array<double,     1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::convert_conservative_to_primitive_turbulence_model< double     >(const std::array<double,     PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadType,    1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::convert_conservative_to_primitive_turbulence_model< FadType    >(const std::array<FadType,    PHILIP_DIM+3> &conservative_soln) const;
template std::array<RadType,    1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::convert_conservative_to_primitive_turbulence_model< RadType    >(const std::array<RadType,    PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadFadType, 1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::convert_conservative_to_primitive_turbulence_model< FadFadType >(const std::array<FadFadType, PHILIP_DIM+3> &conservative_soln) const;
template std::array<RadFadType, 1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::convert_conservative_to_primitive_turbulence_model< RadFadType >(const std::array<RadFadType, PHILIP_DIM+3> &conservative_soln) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template std::array<FadType, 1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::convert_conservative_to_primitive_turbulence_model< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadType, 1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::convert_conservative_to_primitive_turbulence_model< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadType, 1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::convert_conservative_to_primitive_turbulence_model< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln) const;
template std::array<FadType, 1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::convert_conservative_to_primitive_turbulence_model< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln) const;

// -- convert_conservative_gradient_to_primitive_gradient_turbulence_model()
template std::array<dealii::Tensor<1,PHILIP_DIM,double    >,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< double     >(const std::array<double,     PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,double    >,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType    >(const std::array<FadType,    PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType   >,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< RadType    >(const std::array<RadType,    PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadType   >,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadFadType >(const std::array<FadFadType, PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< RadFadType >(const std::array<RadFadType, PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,PHILIP_DIM+3> &solution_gradient) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3> &solution_gradient) const;
template std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,1> ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType >(const std::array<FadType, PHILIP_DIM+3> &conservative_soln, const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,PHILIP_DIM+3> &solution_gradient) const;


} // Physics namespace
} // PHiLiP namespace