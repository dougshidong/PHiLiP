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
    //const double                                              ratio_of_filter_width_to_cell_size,
    //no need for RANS models
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function)
    : ModelBase<dim,nstate,real>(manufactured_solution_function) 
    , turbulent_prandtl_number(turbulent_prandtl_number)
    //, ratio_of_filter_width_to_cell_size(ratio_of_filter_width_to_cell_size)
    //no need for RANS models
    , navier_stokes_physics(std::make_unique < NavierStokes<dim,nstate,real> > (
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
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    for (int i=0; i<nstate-1; i++) {
        conv_flux[i] = 0.0; // No additional convective terms for RANS
    }
    // convective flux of additional RANS turbulence model
    conv_flux[nstate-1] = 0.0;
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
    // Step 1,2: Primitive solution and Gradient of primitive solution
    const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = this->navier_stokes_physics->convert_conservative_gradient_to_primitive_gradient(conservative_soln, solution_gradient);
    const std::array<real2,nstate> primitive_soln = this->navier_stokes_physics->convert_conservative_to_primitive(conservative_soln); // from Euler

    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const dealii::Tensor<1,dim,real2> vel = this->navier_stokes_physics->extract_velocities_from_primitive(primitive_soln); // from Euler
    // Templated virtual member functions
    dealii::Tensor<2,dim,real2> viscous_stress_tensor;
    dealii::Tensor<1,dim,real2> heat_flux;
    if constexpr(std::is_same<real2,real>::value){ 
        viscous_stress_tensor = compute_Reynolds_stress_tensor(primitive_soln, primitive_soln_gradient,cell_index);
        heat_flux = compute_Reynolds_heat_flux(primitive_soln, primitive_soln_gradient,cell_index);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        viscous_stress_tensor = compute_Reynolds_stress_tensor_fad(primitive_soln, primitive_soln_gradient,cell_index);
        heat_flux = compute_Reynolds_heat_flux_fad(primitive_soln, primitive_soln_gradient,cell_index);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> dissipative_flux_templated(): real2!=real || real2!=FadType)" << std::endl;
        std::abort();
    }
    
    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    std::array<dealii::Tensor<1,dim,real2>,nstate> viscous_flux
        = this->navier_stokes_physics->dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux(vel,viscous_stress_tensor,heat_flux);
    
    return viscous_flux;
}
//----------------------------------------------------------------
//not sure if it is needed for RANS
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
    std::array<real,nstate> source_term = dissipative_source_term(pos,cell_index);
    return source_term;
}
//----------------------------------------------------------------
//no need for RANS models
//template <int dim, int nstate, typename real>
//double ReynoldsAveragedNavierStokesBase<dim,nstate,real>
//::get_filter_width (const dealii::types::global_dof_index cell_index) const
//{ 
//    // Compute the LES filter width
//    /** Reference: Flad, David, and Gregor Gassner. "On the use of kinetic
//     *  energy preserving DG-schemes for large eddy simulation."
//     *  Journal of Computational Physics 350 (2017): 782-795.
//     * */
//    const int cell_poly_degree = this->cellwise_poly_degree[cell_index];
//    const double cell_volume = this->cellwise_volume[cell_index];
//    double filter_width = cell_volume;
//    for(int i=0; i<dim; ++i) {
//        filter_width /= (cell_poly_degree+1);
//    }
//    // Resize given the ratio of filter width to cell size
//    filter_width *= ratio_of_filter_width_to_cell_size;

//    return filter_width;
//}
//----------------------------------------------------------------
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
//not sure if it is needed for RANS
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
//not sure if it is needed for RANS
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
//not sure if it is needed for RANS
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
//not sure if it is needed for RANS
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
//not sure if it is needed for RANS
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
    //const double                                              ratio_of_filter_width_to_cell_size,
    //no need for RANS models
    //const double                                              model_constant,
    //no need for RANS models
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
                                                        //ratio_of_filter_width_to_cell_size,
                                                        //no need for RANS models
                                                        isothermal_wall_temperature,
                                                        thermal_boundary_condition_type,
                                                        manufactured_solution_function)
    //, model_constant(model_constant)
    //no need for RANS models
{ }
//----------------------------------------------------------------
//no need for RANS models
//template <int dim, int nstate, typename real>
//double ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
//::get_model_constant_times_filter_width (
//    const dealii::types::global_dof_index cell_index) const
//{
//    // Compute the filter width for the cell
//    const double filter_width = this->get_filter_width(cell_index);
//    // Product of the model constant (Cs) and the filter width (delta)
//    const double model_constant_times_filter_width = model_constant*filter_width;
//    return model_constant_times_filter_width;
//}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
FadType ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity_fad (
    const std::array<FadType,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_eddy_viscosity_templated<FadType>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,nstate> &/*primitive_soln*/,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    // Get velocity gradient
    const dealii::Tensor<2,dim,real2> vel_gradient 
        = this->navier_stokes_physics->extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    // Get strain rate tensor
    const dealii::Tensor<2,dim,real2> strain_rate_tensor 
        = this->navier_stokes_physics->compute_strain_rate_tensor(vel_gradient);
    
    // Compute the eddy viscosity
    // constant for now to test the code 
    const real2 eddy_viscosity = 0.0;

    return eddy_viscosity;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::scale_eddy_viscosity_templated (
    const std::array<real2,nstate> &primitive_soln,
    const real2 eddy_viscosity) const
{
    // Scaled non-dimensional eddy viscosity; 
    const real2 scaled_eddy_viscosity = primitive_soln[0]*eddy_viscosity;

    return scaled_eddy_viscosity;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_Reynolds_heat_flux_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,FadType> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux_fad (
    const std::array<FadType,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_Reynolds_heat_flux_templated<FadType>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_heat_flux_templated (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{   
    // Compute non-dimensional eddy viscosity;
    real2 eddy_viscosity;
    if constexpr(std::is_same<real2,real>::value){ 
        eddy_viscosity = compute_eddy_viscosity(primitive_soln,primitive_soln_gradient,cell_index);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        eddy_viscosity = compute_eddy_viscosity_fad(primitive_soln,primitive_soln_gradient,cell_index);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_Reynolds_heat_flux_templated(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    // Scaled non-dimensional eddy viscosity; See Plata 2019, Computers and Fluids, Eq.(12)
    const real2 scaled_eddy_viscosity = scale_eddy_viscosity_templated<real2>(primitive_soln,eddy_viscosity);

    // Compute scaled heat conductivity
    const real2 scaled_heat_conductivity = this->navier_stokes_physics->compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number(scaled_eddy_viscosity,this->turbulent_prandtl_number);

    // Get temperature gradient
    const dealii::Tensor<1,dim,real2> temperature_gradient = this->navier_stokes_physics->compute_temperature_gradient(primitive_soln, primitive_soln_gradient);

    // Compute the Reynolds stress tensor via the eddy_viscosity and the strain rate tensor
    dealii::Tensor<1,dim,real2> heat_flux_Reynolds = this->navier_stokes_physics->compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient(scaled_heat_conductivity,temperature_gradient);

    return heat_flux_Reynolds;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,real> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_Reynolds_stress_tensor_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,FadType> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor_fad (
    const std::array<FadType,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_Reynolds_stress_tensor_templated<FadType>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_Reynolds_stress_tensor_templated (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    // Compute non-dimensional eddy viscosity;
    real2 eddy_viscosity;
    if constexpr(std::is_same<real2,real>::value){ 
        eddy_viscosity = compute_eddy_viscosity(primitive_soln,primitive_soln_gradient,cell_index);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        eddy_viscosity = compute_eddy_viscosity_fad(primitive_soln,primitive_soln_gradient,cell_index);
    }
    else{
        std::cout << "ERROR in physics/reynolds_averaged_navier_stokes.cpp --> compute_Reynolds_stress_tensor_templated(): real2 != real or FadType" << std::endl;
        std::abort();
    }
    
    // Scaled non-dimensional eddy viscosity; 
    const real2 scaled_eddy_viscosity = scale_eddy_viscosity_templated<real2>(primitive_soln,eddy_viscosity);

    // Get velocity gradients
    const dealii::Tensor<2,dim,real2> vel_gradient 
        = this->navier_stokes_physics->extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    
    // Get strain rate tensor
    const dealii::Tensor<2,dim,real2> strain_rate_tensor 
        = this->navier_stokes_physics->compute_strain_rate_tensor(vel_gradient);

    // Compute the Reynolds stress tensor via the eddy_viscosity and the strain rate tensor
    dealii::Tensor<2,dim,real2> Reynolds_stress_tensor;
    Reynolds_stress_tensor = this->navier_stokes_physics->compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor(scaled_eddy_viscosity,strain_rate_tensor);

    return Reynolds_stress_tensor;
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
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, double     >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadType    >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, FadFadType >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    ReynoldsAveragedNavierStokesBase < PHILIP_DIM, PHILIP_DIM+3, RadFadType >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;


} // Physics namespace
} // PHiLiP namespace