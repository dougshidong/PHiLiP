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
    static_assert(nstate>=dim+2, "ModelBase::ReynoldsAveragedNavierStokesBase() should be created with nstate>=dim+2");
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
    const std::array<real2,dim+2> primitive_soln_rans = this->navier_stokes_physics->convert_conservative_to_primitive(conservative_soln_rans);
    const dealii::Tensor<1,dim,real2> vel = this->navier_stokes_physics->extract_velocities_from_primitive(primitive_soln_rans); // from Euler
    std::array<dealii::Tensor<1,dim,real2>,nstate> conv_flux;

    for (int flux_dim=0; flux_dim<dim+2; flux_dim++) {
        conv_flux[flux_dim] = 0.0; // No additional convective terms for RANS
    }
    // convective flux of additional RANS turbulence model
    for (int flux_dim=dim+2; flux_dim<nstate; flux_dim++) {
        for (int velocity_dim=0; velocity_dim<dim; velocity_dim++) {
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
    const dealii::types::global_dof_index cell_index) const
{   

    //To do later
    //do something to trick compiler
    //should be removed later  
    int cell_poly_degree = this->cellwise_poly_degree[cell_index];
    cell_poly_degree++;

    const std::array<real2,dim+2> conservative_soln_rans = extract_rans_conservative_solution(conservative_soln);
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> solution_gradient_rans = extract_rans_solution_gradient(solution_gradient);

    //const std::array<real2,nstate-(dim+2)> conservative_soln_turbulence_model = extract_turbulence_model_conservative_solution(conservative_soln);
    //const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> solution_gradient_turbulence_model = extract_turbulence_model_solution_gradient(solution_gradient);

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
    for(int flux_dim=0; flux_dim<dim+2; flux_dim++)
    {
        viscous_flux[flux_dim] = viscous_flux_rans[flux_dim];
    }
    for(int flux_dim=dim+2; flux_dim<nstate; flux_dim++)
    {
        viscous_flux[flux_dim] = viscous_flux_turbulence_model[flux_dim-(dim+2)];
    }
    
    return viscous_flux;
}
//To do later
//Introducing new template to build general extract function for RANS and turbulence model
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<real2,dim+2> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::extract_rans_conservative_solution (
    const std::array<real2,nstate> &conservative_soln) const
{   
    std::array<real2,dim+2> conservative_soln_rans;
    for(int i=0; i<dim+2; i++){
        conservative_soln_rans[i] = conservative_soln[i];
    }
 
    return conservative_soln_rans;
}
//To do later
//Introducing new template to build general extract function for RANS and turbulence model
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,dim+2> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::extract_rans_solution_gradient (
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const
{   
    std::array<dealii::Tensor<1,dim,real2>,dim+2> solution_gradient_rans;
    for(int i=0; i<dim+2; i++){
        solution_gradient_rans[i] = solution_gradient[i];
    }
 
    return solution_gradient_rans;
}
/*
//may not needed
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<real2,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::extract_turbulence_model_conservative_solution (
    const std::array<real2,nstate> &conservative_soln) const
{   
    std::array<real2,nstate-(dim+2)> conservative_soln_turbulence_model;
    for(int i=0; i<nstate-(dim+2); i++)
        conservative_soln_turbulence_model[i] = conservative_soln[dim+2+i];
 
    return conservative_soln_turbulence_model;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::extract_turbulence_model_solution_gradient (
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const
{   
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> solution_gradient_turbulence_model;
    for(int i=0; i<nstate-(dim+2); i++)
        solution_gradient_turbulence_model[i] = solution_gradient[dim+2+i];
 
    return solution_gradient_turbulence_model;
}
*/
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

    for(int i=0; i<nstate-(dim+2); i++){
        for(int j=0; j<dim; j++){
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
    for(int i=0; i<nstate-(dim+2); i++){
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

    for(int i=0; i<nstate-(dim+2); i++){
        for(int j=0; j<dim; j++){
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

    for (int i=0;i<dim+2;i++){
        conv_num_split_flux[i] = 0.0;
    }
    for (int i=dim+2;i<nstate;i++)
    {
        for (int flux_dim = 0; flux_dim < dim; flux_dim++){
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

    for (int i=0;i<nstate-(dim+2);i++){
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
    std::array<real,dim+2> conservative_soln_rans = extract_rans_conservative_solution(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = this->navier_stokes_physics->template compute_velocities<real>(conservative_soln_rans);
    std::array<real,nstate> eig;
    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    for (int i=0; i<dim+2; i++) {
        eig[i] = 0.0;
    }
    for (int i=dim+2; i<nstate; i++) {
        eig[i] = vel_dot_n;
    }
    return eig;
}
//----------------------------------------------------------------
//check latter
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,dim+2> conservative_soln_rans = extract_rans_conservative_solution(conservative_soln);

    const dealii::Tensor<1,dim,real> vel = this->navier_stokes_physics->template compute_velocities<real>(conservative_soln_rans);

    real vel2 = this->navier_stokes_physics->template compute_velocity_squared<real>(vel);

    const real max_eig = sqrt(vel2);

    return max_eig;
}
//adding physical source 
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const
{
    return physical_source_term_templated<real>(pos,conservative_soln,solution_gradient,cell_index);
}
//adding physical source 
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<real2,nstate> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::physical_source_term_templated (
        const dealii::Point<dim,real2> &pos,
        const std::array<real2,nstate> &conservative_solution,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const
{
    //To do later
    //do something to trick compiler 
    int cell_poly_degree = this->cellwise_poly_degree[cell_index];
    cell_poly_degree++;

    std::array<real,nstate> physical_source;
    if constexpr(std::is_same<real2,real>::value){ 
        physical_source = this->compute_production_dissipation_cross_term(pos, conservative_solution, solution_gradient);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        physical_source = this->compute_production_dissipation_cross_term_fad(pos, conservative_solution, solution_gradient);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> physical_source_term_templated(): real2!=real || real2!=FadType)" << std::endl;
        std::abort();
    }
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
    /* TO DO Note: Since this is only used for the manufactured solution source term, 
             the grid spacing is fixed --> No AD wrt grid --> Can use same computation as NavierStokes
             Acceptable if we can ensure that filter_width is the same everywhere in the domain
             for the manufacture solution cases chosen
     */
    std::array<real,nstate> conv_source_term = convective_source_term(pos);
    std::array<real,nstate> diss_source_term = dissipative_source_term(pos,cell_index);
    std::array<real,nstate> phys_source_source_term = physical_source_source_term(pos,cell_index);
    std::array<real,nstate> source_term;
    for (int s=0; s<nstate; s++)
    {
        source_term[s] = conv_source_term[s] + diss_source_term[s] - phys_source_source_term[s];
    }
    return source_term;
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
    for (int s=0; s<nstate; s++) {
        adtype ADvar(nstate, s, getValue<real>(conservative_soln[s])); // create AD variable
        AD_conservative_soln[s] = ADvar;
    }

    // Compute AD convective flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_convective_flux = convective_flux_templated<adtype>(AD_conservative_soln);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
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
    for (int s=0; s<nstate; s++) {
        adtype ADvar(nstate, s, getValue<real>(conservative_soln[s])); // create AD variable
        AD_conservative_soln[s] = ADvar;
        for (int d=0;d<dim;d++) {
            AD_solution_gradient[s][d] = getValue<real>(solution_gradient[s][d]);
        }
    }

    // Compute AD dissipative flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_dissipative_flux = dissipative_flux_templated<adtype>(AD_conservative_soln, AD_solution_gradient, cell_index);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
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
    for (int s=0; s<nstate; s++) {
        AD_conservative_soln[s] = getValue<real>(conservative_soln[s]);
        for (int d=0;d<dim;d++) {
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
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
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
    for (int s=0; s<nstate; s++) {
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
    for (int d=0;d<dim;d++) {
        for (int s=0; s<nstate; s++) {
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
    for (int d=0;d<dim;d++) {
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
    for (int s=0; s<nstate; s++) {
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
    for (int s=0; s<nstate; s++) {
        dealii::SymmetricTensor<2,dim,real> hessian = this->manufactured_solution_function->hessian(pos,s);
        for (int dr=0;dr<dim;dr++) {
            for (int dc=0;dc<dim;dc++) {
                manufactured_solution_hessian[s][dr][dc] = hessian[dr][dc];
            }
        }
    }

    // First term -- wrt to the conservative variables
    // This is similar, should simply provide this function a flux_directional_jacobian() -- could restructure later
    dealii::Tensor<1,nstate,real> dissipative_flux_divergence;
    for (int d=0;d<dim;d++) {
        dealii::Tensor<1,dim,real> normal;
        normal[d] = 1.0;
        const dealii::Tensor<2,nstate,real> jacobian = dissipative_flux_directional_jacobian(manufactured_solution, manufactured_solution_gradient, normal, cell_index);
        
        // get the directional jacobian wrt gradient
        std::array<dealii::Tensor<2,nstate,real>,dim> jacobian_wrt_gradient;
        for (int d_gradient=0;d_gradient<dim;d_gradient++) {
            
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
                for (int d_gradient=0;d_gradient<dim;d_gradient++) {
                    jac_grad_row += jacobian_wrt_gradient[d_gradient][sr][sc]*manufactured_solution_hessian[sc][d_gradient][d]; // symmetric so d indexing works both ways
                }
            }
            dissipative_flux_divergence[sr] += jac_grad_row;
        }
    }
    std::array<real,nstate> dissipative_source_term;
    for (int s=0; s<nstate; s++) {
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
    for (int i=0;i<nstate;i++){
        physical_source_source_term = physical_source_term(pos, manufactured_solution, manufactured_solution_gradient, cell_index);
    }

    return physical_source_source_term;
}
//----------------------------------------------------------------
//================================================================
// Negative Spalart-Allmaras model
//================================================================
template <int dim, int nstate, typename real>
ReynoldsAveragedNavierStokes_SAneg<dim, nstate, real>::ReynoldsAveragedNavierStokes_SAneg(
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
    : ReynoldsAveragedNavierStokesBase<dim,nstate,real>(ref_length,
                                                        gamma_gas,
                                                        mach_inf,
                                                        angle_of_attack,
                                                        side_slip_angle,
                                                        prandtl_number,
                                                        reynolds_number_inf,
                                                        turbulent_prandtl_number,
                                                        isothermal_wall_temperature,
                                                        thermal_boundary_condition_type,
                                                        manufactured_solution_function)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,dim+2> &primitive_soln_rans,
    const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
FadType ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity_fad (
    const std::array<FadType,dim+2> &primitive_soln_rans,
    const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    return compute_eddy_viscosity_templated<FadType>(primitive_soln_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,dim+2> &primitive_soln_rans,
    const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    real2 eddy_viscosity;
    if (primitive_soln_turbulence_model[0]>=0.0)
    {
        // Compute needed coefficients
        const real2 laminar_dynamic_viscosity = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln_rans);
        const real2 laminar_kinematic_viscosity = laminar_dynamic_viscosity/primitive_soln_rans[0];
        const real2 Chi = this->compute_coefficient_Chi(primitive_soln_turbulence_model[0],laminar_kinematic_viscosity);
        const real2 f_v1 = this->compute_coefficient_f_v1(Chi);
        eddy_viscosity = primitive_soln_rans[0]*primitive_soln_turbulence_model[0]*f_v1;
    } else {
        eddy_viscosity = 0.0;
    }

    return eddy_viscosity;
}
////----------------------------------------------------------------
//template <int dim, int nstate, typename real>
//template<typename real2>
//real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
//::scale_eddy_viscosity_templated (
//    const std::array<real2,dim+2> &primitive_soln_rans,
//    const real2 eddy_viscosity) const
//{
//    // Scaled non-dimensional eddy viscosity; 
//    const real2 scaled_eddy_viscosity = eddy_viscosity/reynolds_number_inf;
//
//    return scaled_eddy_viscosity;
//}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::scale_coefficient (
    const real2 coefficient) const
{
    real2 scaled_coefficient;
    if constexpr(std::is_same<real2,real>::value){
        scaled_coefficient = coefficient/this->navier_stokes_physics->reynolds_number_inf;
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        const FadType reynolds_number_inf_fad = this->navier_stokes_physics->reynolds_number_inf;
        scaled_coefficient = coefficient/reynolds_number_inf_fad;
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> scale_coefficient(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return scaled_coefficient;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate-(dim+2)> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_effective_viscosity_turbulence_model (
    const std::array<real,dim+2> &primitive_soln_rans,
    const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{   
    return compute_effective_viscosity_turbulence_model_templated<real>(primitive_soln_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<FadType,nstate-(dim+2)> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_effective_viscosity_turbulence_model_fad (
    const std::array<FadType,dim+2> &primitive_soln_rans,
    const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{   
    return compute_effective_viscosity_turbulence_model_templated<FadType>(primitive_soln_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<real2,nstate-(dim+2)> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_effective_viscosity_turbulence_model_templated (
    const std::array<real2,dim+2> &primitive_soln_rans,
    const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{   
    const real2 laminar_dynamic_viscosity = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln_rans);
    const real2 laminar_kinematic_viscosity = laminar_dynamic_viscosity/primitive_soln_rans[0];

    const real2 coefficient_f_n = this->compute_coefficient_f_n(primitive_soln_turbulence_model[0],laminar_kinematic_viscosity);

    std::array<real2,nstate-(dim+2)> effective_viscosity_turbulence_model;

    for(int i=0; i<nstate-(dim+2); i++){
        if constexpr(std::is_same<real2,real>::value){
            effective_viscosity_turbulence_model[i] = (laminar_dynamic_viscosity+coefficient_f_n*primitive_soln_rans[0]*primitive_soln_turbulence_model[0])/sigma;
        }
        else if constexpr(std::is_same<real2,FadType>::value){
            effective_viscosity_turbulence_model[i] = (laminar_dynamic_viscosity+coefficient_f_n*primitive_soln_rans[0]*primitive_soln_turbulence_model[0])/sigma_fad;
        }
        else{
            std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_effective_viscosity_turbulence_model_templated(): real2 != real or FadType" << std::endl;
            std::abort();
        }
        effective_viscosity_turbulence_model[i] = scale_coefficient(effective_viscosity_turbulence_model[i]);
    }
    
    return effective_viscosity_turbulence_model;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_Chi (
    const real2 &nu_tilde,
    const real2 &laminar_kinematic_viscosity) const
{
    // Compute coefficient Chi
    const real2 Chi = nu_tilde/laminar_kinematic_viscosity;

    return Chi;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_v1 (
    const real2 &coefficient_Chi) const
{
    // Compute coefficient f_v1
    real2 coefficient_f_v1;

    if constexpr(std::is_same<real2,real>::value){ 
        coefficient_f_v1 = coefficient_Chi*coefficient_Chi*coefficient_Chi/(coefficient_Chi*coefficient_Chi*coefficient_Chi+c_v1*c_v1*c_v1);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        coefficient_f_v1 = coefficient_Chi*coefficient_Chi*coefficient_Chi/(coefficient_Chi*coefficient_Chi*coefficient_Chi+c_v1_fad*c_v1_fad*c_v1_fad);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_coefficient_f_v1(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_f_v1;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_v2 (
    const real2 &coefficient_Chi) const
{
    // Compute coefficient f_v2
    real2 coefficient_f_v2;
    const real2 coefficient_f_v1 = this->compute_coefficient_f_v1(coefficient_Chi);

    if constexpr(std::is_same<real2,real>::value){ 
        coefficient_f_v2 = 1.0-coefficient_Chi/(1.0+coefficient_Chi*coefficient_f_v1);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        const FadType const_one_fad = 1.0; 
        coefficient_f_v2 = const_one_fad-coefficient_Chi/(const_one_fad+coefficient_Chi*coefficient_f_v1);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_coefficient_f_v2(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_f_v2;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_n (
    const real2 &nu_tilde,
    const real2 &laminar_kinematic_viscosity) const
{
    // Compute coefficient f_n
    real2 coefficient_f_n;
    const real2 coefficient_Chi = this->compute_coefficient_Chi(nu_tilde,laminar_kinematic_viscosity);

    if constexpr(std::is_same<real2,real>::value){ 
        if (nu_tilde>=0.0)
            coefficient_f_n = 1.0;
        else
            coefficient_f_n = (c_n1+coefficient_Chi*coefficient_Chi*coefficient_Chi)/(c_n1-coefficient_Chi*coefficient_Chi*coefficient_Chi);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        const FadType const_one_fad = 1.0; 
        if (nu_tilde>=0.0)
            coefficient_f_n = const_one_fad;
        else
            coefficient_f_n = (c_n1_fad+coefficient_Chi*coefficient_Chi*coefficient_Chi)/(c_n1_fad-coefficient_Chi*coefficient_Chi*coefficient_Chi);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_coefficient_f_n(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_f_n;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_t2 (
    const real2 &coefficient_Chi) const
{
    // Compute coefficient f_t2
    real2 coefficient_f_t2;
    if constexpr(std::is_same<real2,real>::value){ 
        coefficient_f_t2 = c_t3*exp(-c_t4*coefficient_Chi*coefficient_Chi);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        coefficient_f_t2 = c_t3_fad*exp(-c_t4_fad*coefficient_Chi*coefficient_Chi);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_coefficient_f_t2(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_f_t2;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_f_w (
    const real2 &coefficient_g) const
{
    // Compute coefficient f_w
    real2 coefficient_f_w;
    if constexpr(std::is_same<real2,real>::value){ 
        coefficient_f_w = coefficient_g*pow((1.0+pow(c_w3,6.0))/(pow(coefficient_g,6.0)+pow(c_w3,6.0)),1.0/6.0);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        const FadType const_one_fad = 1.0;
        coefficient_f_w = coefficient_g*pow((const_one_fad+pow(c_w3_fad,6.0))/(pow(coefficient_g,6.0)+pow(c_w3_fad,6.0)),1.0/6.0);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_coefficient_f_w(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_f_w;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_r (
    const real2 &nu_tilde,
    const real2 &d_wall,
    const real2 &s_tilde) const
{
    // Compute coefficient r
    real2 coefficient_r;
    if constexpr(std::is_same<real2,real>::value){ 
        coefficient_r = nu_tilde/(s_tilde*kappa*kappa*d_wall*d_wall); 
        coefficient_r = scale_coefficient(coefficient_r);
        coefficient_r = coefficient_r <= r_lim ? coefficient_r : r_lim; 
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        coefficient_r = nu_tilde/(s_tilde*kappa_fad*kappa_fad*d_wall*d_wall); 
        coefficient_r = scale_coefficient(coefficient_r);
        coefficient_r = coefficient_r <= r_lim_fad ? coefficient_r : r_lim_fad; 
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_coefficient_r(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_r;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_coefficient_g (
    const real2 &coefficient_r) const
{
    // Compute coefficient g
    real2 coefficient_g;
    if constexpr(std::is_same<real2,real>::value){ 
        coefficient_g = coefficient_r+c_w2*(pow(coefficient_r,6)-coefficient_r);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        coefficient_g = coefficient_r+c_w2_fad*(pow(coefficient_r,6)-coefficient_r);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_coefficient_g(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return coefficient_g;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_s (
    const std::array<real2,dim+2> &conservative_soln_rans,
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> &conservative_soln_gradient_rans) const
{
    // Compute s
    real2 s;

    // Get vorticity
    const dealii::Tensor<1,3,real2> vorticity 
        = this->navier_stokes_physics->compute_vorticity(conservative_soln_rans,conservative_soln_gradient_rans);

    s = sqrt(this->get_vector_magnitude_sqr(vorticity));

    return s;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_s_bar (
    const real2 &coefficient_Chi,
    const real2 &nu_tilde,
    const real2 &d_wall) const
{
    // Compute s_bar
    real2 s_bar;
    const real2 f_v2 = this->compute_coefficient_f_v2(coefficient_Chi);

    if constexpr(std::is_same<real2,real>::value){
        s_bar = nu_tilde*f_v2/(kappa*kappa*d_wall*d_wall);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        s_bar = nu_tilde*f_v2/(kappa_fad*kappa_fad*d_wall*d_wall);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_coefficient_f_v2(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return s_bar;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_s_tilde (
    const real2 &coefficient_Chi,
    const real2 &nu_tilde,
    const real2 &d_wall,
    const real2 &s) const
{
    // Compute s_bar
    real2 s_tilde;
    const real2 s_bar = this->compute_s_bar(coefficient_Chi,nu_tilde,d_wall);
    const real2 scaled_s_bar = scale_coefficient(s_bar);

    if constexpr(std::is_same<real2,real>::value){
        const real dimensional_s = s*this->navier_stokes_physics->mach_inf/this->navier_stokes_physics->ref_length;
        const real dimensional_s_bar = s_bar*this->navier_stokes_physics->reynolds_number_inf*this->navier_stokes_physics->mach_inf/this->navier_stokes_physics->ref_length;
        if(dimensional_s_bar>=-c_v2*dimensional_s) 
            s_tilde = s+scaled_s_bar;
        else
            s_tilde = s+s*(c_v2*c_v2*s+c_v3*scaled_s_bar)/((c_v3-2.0*c_v2)*s-scaled_s_bar);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        const FadType mach_inf_fad = this->navier_stokes_physics->mach_inf;
        const FadType ref_length_fad = this->navier_stokes_physics->ref_length;
        const FadType reynolds_number_inf_fad = this->navier_stokes_physics->reynolds_number_inf;
        const FadType dimensional_s = s*mach_inf_fad/ref_length_fad;
        const FadType dimensional_s_bar = s_bar*reynolds_number_inf_fad*mach_inf_fad/ref_length_fad;
        if(dimensional_s_bar>=-c_v2_fad*dimensional_s) 
            s_tilde = s+scaled_s_bar;
        else
            s_tilde = s+s*(c_v2_fad*c_v2_fad*s+c_v3_fad*scaled_s_bar)/((c_v3_fad-2.0*c_v2_fad)*s-scaled_s_bar);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_s_tilde(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return s_tilde;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
std::array<real2,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_production_source (
    const real2 &coefficient_f_t2,
    const real2 &density,
    const real2 &nu_tilde,
    const real2 &s,
    const real2 &s_tilde) const
{
    std::array<real2,nstate> production_source;

    for (int i=0;i<dim+2;i++){
        production_source[i] = 0.0;
    }

    if constexpr(std::is_same<real2,real>::value){
        if(nu_tilde>=0.0)
            production_source[dim+2] = c_b1*(1.0-coefficient_f_t2)*s_tilde*nu_tilde;
        else
            production_source[dim+2] = c_b1*(1.0-c_t3)*s*nu_tilde;

        production_source[dim+2] *= density;
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        const FadType const_one_fad = 1.0; 
        if(nu_tilde>=0.0)
            production_source[dim+2] = c_b1_fad*(const_one_fad-coefficient_f_t2)*s_tilde*nu_tilde;
        else
            production_source[dim+2] = c_b1_fad*(const_one_fad-c_t3_fad)*s*nu_tilde;

        production_source[dim+2] *= density;
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_production_source(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return production_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
std::array<real2,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_dissipation_source (
    const real2 &coefficient_f_t2,
    const real2 &density,
    const real2 &nu_tilde,
    const real2 &d_wall,
    const real2 &s_tilde) const
{
    const real2 coefficient_r = this->compute_coefficient_r(nu_tilde,d_wall,s_tilde);
    const real2 coefficient_g = this->compute_coefficient_g(coefficient_r);
    const real2 coefficient_f_w = this->compute_coefficient_f_w(coefficient_g);
    std::array<real2,nstate> dissipation_source;

    for (int i=0;i<dim+2;i++){
        dissipation_source[i] = 0.0;
    }

    if constexpr(std::is_same<real2,real>::value){
        if(nu_tilde>=0.0)
            dissipation_source[dim+2] = (c_w1*coefficient_f_w-c_b1*coefficient_f_t2/(kappa*kappa))*nu_tilde*nu_tilde/(d_wall*d_wall);
        else
            dissipation_source[dim+2] = -c_w1*nu_tilde*nu_tilde/(d_wall*d_wall);

        dissipation_source[dim+2] *= density;
        dissipation_source[dim+2] = scale_coefficient(dissipation_source[dim+2]);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        if(nu_tilde>=0.0)
            dissipation_source[dim+2] = (c_w1_fad*coefficient_f_w-c_b1_fad*coefficient_f_t2/(kappa_fad*kappa_fad))*nu_tilde*nu_tilde/(d_wall*d_wall);
        else
            dissipation_source[dim+2] = -c_w1_fad*nu_tilde*nu_tilde/(d_wall*d_wall);

        dissipation_source[dim+2] *= density;
        dissipation_source[dim+2] = scale_coefficient(dissipation_source[dim+2]);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_dissipation_source(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return dissipation_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
std::array<real2,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_cross_source (
    const real2 &density,
    const real2 &nu_tilde,
    const real2 &laminar_kinematic_viscosity,
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> &primitive_soln_gradient_rans,
    const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> &primitive_solution_gradient_turbulence_model) const
{
    real2 cross_nu_tilde_nu_tilde = 0.0;
    real2 cross_rho_nu_tilde = 0.0;
    std::array<real2,nstate> cross_source;
    const real2 coefficient_f_n = this->compute_coefficient_f_n(nu_tilde,laminar_kinematic_viscosity);

    for (int i=0;i<dim+2;i++){
        cross_source[i] = 0.0;
    }
    for (int i=0;i<dim;i++){
        cross_nu_tilde_nu_tilde += primitive_solution_gradient_turbulence_model[0][i]*primitive_solution_gradient_turbulence_model[0][i];
        cross_rho_nu_tilde += primitive_soln_gradient_rans[0][i]*primitive_solution_gradient_turbulence_model[0][i];
    }

    if constexpr(std::is_same<real2,real>::value){
        cross_source[dim+2] = (c_b2*density*cross_nu_tilde_nu_tilde-(laminar_kinematic_viscosity+nu_tilde*coefficient_f_n)*cross_rho_nu_tilde)/sigma;
        cross_source[dim+2] = scale_coefficient(cross_source[dim+2]);
    }
    else if constexpr(std::is_same<real2,FadType>::value){
        cross_source[dim+2] = (c_b2_fad*density*cross_nu_tilde_nu_tilde-(laminar_kinematic_viscosity+nu_tilde*coefficient_f_n)*cross_rho_nu_tilde)/sigma_fad;
        cross_source[dim+2] = scale_coefficient(cross_source[dim+2]);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_cross_source(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    return cross_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux (
    const std::array<real,dim+2> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
    const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    return compute_Reynolds_heat_flux_templated<real>(primitive_soln_rans,primitive_soln_gradient_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,FadType> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux_fad (
    const std::array<FadType,dim+2> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,FadType>,dim+2> &primitive_soln_gradient_rans,
    const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    return compute_Reynolds_heat_flux_templated<FadType>(primitive_soln_rans,primitive_soln_gradient_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux_templated (
    const std::array<real2,dim+2> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> &primitive_soln_gradient_rans,
    const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{   
    // Compute non-dimensional eddy viscosity;
    real2 eddy_viscosity;
    if constexpr(std::is_same<real2,real>::value){ 
        eddy_viscosity = compute_eddy_viscosity(primitive_soln_rans,primitive_soln_turbulence_model);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        eddy_viscosity = compute_eddy_viscosity_fad(primitive_soln_rans,primitive_soln_turbulence_model);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_Reynolds_heat_flux_templated(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    // Scaled non-dimensional eddy viscosity; See Plata 2019, Computers and Fluids, Eq.(12)
    //const real2 scaled_eddy_viscosity = this->navier_stokes_physics->template scale_viscosity_coefficient<real2>(eddy_viscosity);
    const real2 scaled_eddy_viscosity = scale_coefficient(eddy_viscosity);

    // Compute scaled heat conductivity
    const real2 scaled_heat_conductivity = this->navier_stokes_physics->compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number(scaled_eddy_viscosity,this->turbulent_prandtl_number);

    // Get temperature gradient
    const dealii::Tensor<1,dim,real2> temperature_gradient = this->navier_stokes_physics->compute_temperature_gradient(primitive_soln_rans, primitive_soln_gradient_rans);

    // Compute the Reynolds stress tensor via the eddy_viscosity and the strain rate tensor
    dealii::Tensor<1,dim,real2> heat_flux_Reynolds = this->navier_stokes_physics->compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient(scaled_heat_conductivity,temperature_gradient);

    return heat_flux_Reynolds;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,real> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor (
    const std::array<real,dim+2> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,real>,dim+2> &primitive_soln_gradient_rans,
    const std::array<real,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    return compute_Reynolds_stress_tensor_templated<real>(primitive_soln_rans,primitive_soln_gradient_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,FadType> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor_fad (
    const std::array<FadType,dim+2> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,FadType>,dim+2> &primitive_soln_gradient_rans,
    const std::array<FadType,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    return compute_Reynolds_stress_tensor_templated<FadType>(primitive_soln_rans,primitive_soln_gradient_rans,primitive_soln_turbulence_model);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor_templated (
    const std::array<real2,dim+2> &primitive_soln_rans,
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> &primitive_soln_gradient_rans,
    const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    // Compute non-dimensional eddy viscosity;
    real2 eddy_viscosity;
    if constexpr(std::is_same<real2,real>::value){ 
        eddy_viscosity = compute_eddy_viscosity(primitive_soln_rans,primitive_soln_turbulence_model);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        eddy_viscosity = compute_eddy_viscosity_fad(primitive_soln_rans,primitive_soln_turbulence_model);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_Reynolds_stress_tensor_templated(): real2 != real or FadType" << std::endl;
        std::abort();
    }
    
    // Scaled non-dimensional eddy viscosity; 
    //const real2 scaled_eddy_viscosity = this->navier_stokes_physics->template scale_viscosity_coefficient<real2>(eddy_viscosity);
    const real2 scaled_eddy_viscosity = scale_coefficient(eddy_viscosity);

    // Get velocity gradients
    const dealii::Tensor<2,dim,real2> vel_gradient 
        = this->navier_stokes_physics->extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient_rans);
    
    // Get strain rate tensor
    const dealii::Tensor<2,dim,real2> strain_rate_tensor 
        = this->navier_stokes_physics->compute_strain_rate_tensor(vel_gradient);

    // Compute the Reynolds stress tensor via the eddy_viscosity and the strain rate tensor
    dealii::Tensor<2,dim,real2> Reynolds_stress_tensor;
    Reynolds_stress_tensor = this->navier_stokes_physics->compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor(scaled_eddy_viscosity,strain_rate_tensor);

    return Reynolds_stress_tensor;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_production_dissipation_cross_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_gradient) const
{
    return compute_production_dissipation_cross_term_templated<real>(pos,conservative_soln,soln_gradient);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<FadType,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_production_dissipation_cross_term_fad (
    const dealii::Point<dim,FadType> &pos,
    const std::array<FadType,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,FadType>,nstate> &soln_gradient) const
{
    return compute_production_dissipation_cross_term_templated<FadType>(pos,conservative_soln,soln_gradient);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
std::array<real2,nstate> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_production_dissipation_cross_term_templated (
    const dealii::Point<dim,real2> &pos,
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_gradient) const
{

    const std::array<real2,dim+2> conservative_soln_rans = this->extract_rans_conservative_solution(conservative_soln);
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> conservative_soln_gradient_rans = this->extract_rans_solution_gradient(soln_gradient);
    const std::array<real2,dim+2> primitive_soln_rans = this->navier_stokes_physics->convert_conservative_to_primitive(conservative_soln_rans); // from Euler
    const std::array<dealii::Tensor<1,dim,real2>,dim+2> primitive_soln_gradient_rans = this->navier_stokes_physics->convert_conservative_gradient_to_primitive_gradient(conservative_soln_rans, conservative_soln_gradient_rans);
    const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> primitive_soln_gradient_turbulence_model = this->convert_conservative_gradient_to_primitive_gradient_turbulence_model(conservative_soln, soln_gradient);


    const real2 density = conservative_soln_rans[0];
    const real2 nu_tilde = conservative_soln[nstate-1]/conservative_soln_rans[0];
    const real2 laminar_dynamic_viscosity = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln_rans);
    const real2 laminar_kinematic_viscosity = laminar_dynamic_viscosity/density;

    const real2 coefficient_Chi = compute_coefficient_Chi(nu_tilde,laminar_kinematic_viscosity);
    const real2 coefficient_f_t2 = compute_coefficient_f_t2(coefficient_Chi); 

    const real2 d_wall = pos[1]+1.0;

    const real2 s = compute_s(conservative_soln_rans, conservative_soln_gradient_rans);
    const real2 s_tilde = compute_s_tilde(coefficient_Chi, nu_tilde, d_wall, s);

    const std::array<real2,nstate> production = compute_production_source(coefficient_f_t2, density, nu_tilde, s, s_tilde);
    const std::array<real2,nstate> dissipation = compute_dissipation_source(coefficient_f_t2, density, nu_tilde, d_wall, s_tilde);
    const std::array<real2,nstate> cross = compute_cross_source(density, nu_tilde, laminar_kinematic_viscosity, primitive_soln_gradient_rans, primitive_soln_gradient_turbulence_model);

    std::array<real2,nstate> physical_source_term;
    for (int i=0;i<nstate;i++){
        physical_source_term[i] = production[i]-dissipation[i]+cross[i];
    }

    return physical_source_term;

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
// -- ReynoldsAveragedNavierStokes_SAneg
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, double >;
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, FadType  >;
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, RadType  >;
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, FadFadType >;
template class ReynoldsAveragedNavierStokes_SAneg < PHILIP_DIM, PHILIP_DIM+3, RadFadType >;
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
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;

/*
// -- convert_conservative_to_primitive_turbulence_model()
template double     ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::convert_conservative_to_primitive_turbulence_model< double     >(const std::array<double,    PHILIP_DIM+3> &conservative_soln) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::convert_conservative_to_primitive_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln) const;
template RadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::convert_conservative_to_primitive_turbulence_model< RadType    >(const std::array<RadType,   PHILIP_DIM+3> &conservative_soln) const;
template FadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::convert_conservative_to_primitive_turbulence_model< FadFadType >(const std::array<FadFadType,PHILIP_DIM+3> &conservative_soln) const;
template RadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::convert_conservative_to_primitive_turbulence_model< RadFadType >(const std::array<RadFadType,PHILIP_DIM+3> &conservative_soln) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::convert_conservative_to_primitive_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::convert_conservative_to_primitive_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::convert_conservative_to_primitive_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::convert_conservative_to_primitive_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln) const;
// -- convert_conservative_to_primitive_turbulence_model()
template double     ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< double     >(const std::array<double,    PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,double>,nstate>     &solution_gradient) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,FadType>,nstate>    &solution_gradient) const;
template RadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< RadType    >(const std::array<RadType,   PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,RadType>,nstate>    &solution_gradient) const;
template FadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadFadType >(const std::array<FadFadType,PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &solution_gradient) const;
template RadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< RadFadType >(const std::array<RadFadType,PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,RadFadType>,nstate> &solution_gradient) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::convert_conservative_gradient_to_primitive_gradient_turbulence_model< FadType    >(const std::array<FadType,   PHILIP_DIM+3> &conservative_soln,const std::array<dealii::Tensor<1,dim,FadType>,nstate> &solution_gradient) const;
// -- dissipative_flux_turbulence_model()
template double     ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::dissipative_flux_turbulence_model< double     >(const std::array<double,    1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,double>,    1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadType    >::dissipative_flux_turbulence_model< FadType    >(const std::array<FadType,   1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,   1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
template RadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::dissipative_flux_turbulence_model< RadType    >(const std::array<RadType,   1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,RadType>,   1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
template FadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::dissipative_flux_turbulence_model< FadFadType >(const std::array<FadFadType,1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,FadFadType>,1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
template RadFadType ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::dissipative_flux_turbulence_model< RadFadType >(const std::array<RadFadType,1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,RadFadType>,1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
// -- instantiate all the real types with real2 = FadType for automatic differentiation
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::dissipative_flux_turbulence_model< FadType     >(const std::array<FadType,    1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,    1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::dissipative_flux_turbulence_model< FadType     >(const std::array<FadType,    1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,    1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::dissipative_flux_turbulence_model< FadType     >(const std::array<FadType,    1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,    1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::dissipative_flux_turbulence_model< FadType     >(const std::array<FadType,    1> &primitive_soln_turbulence_model,const std::array<dealii::Tensor<1,PHILIP_DIM,FadType>,    1> &primitive_solution_gradient_turbulence_model,const dealii::types::global_dof_index cell_index) const;
*/

} // Physics namespace
} // PHiLiP namespace