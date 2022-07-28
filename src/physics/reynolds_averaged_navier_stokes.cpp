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
    const std::array<real,nstate> &/*conservative_soln*/) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    for (int i=0; i<dim+2; i++) {
        conv_flux[i] = 0.0; // No additional convective terms for RANS
    }
    // convective flux of additional RANS turbulence model
    for (int i=dim+2; i<nstate-1; i++) {
        //To do 
        conv_flux[i] = 0.0; // Additional convective terms for turbulence model
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
        = this->dissipative_flux_turbulence_model(primitive_soln_turbulence_model,primitive_soln_gradient_turbulence_model);

    std::array<dealii::Tensor<1,dim,real2>,nstate> viscous_flux;
    for(int i=0; i<dim+2; i++)
    {
        viscous_flux[i] = viscous_flux_rans[i];
    }
    for(int i=dim+2; i<nstate-1; i++)
    {
        viscous_flux[i] = viscous_flux_turbulence_model[i-(dim+2)];
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
    for(int i=0; i<dim+2; i++)
        conservative_soln_rans[i] = conservative_soln[i];
 
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
    for(int i=0; i<dim+2; i++)
        solution_gradient_rans[i] = solution_gradient[i];
 
    return solution_gradient_rans;
}
/*
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
//To do later
//get the right viscous flux
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::dissipative_flux_turbulence_model (
    const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model,
    const std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> &primitive_solution_gradient_turbulence_model) const
{   
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> primitive_solution_gradient_turbulence_model_t;

    //do something to trick compiler
    primitive_solution_gradient_turbulence_model_t[0] = primitive_solution_gradient_turbulence_model[1];

    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> viscous_flux_turbulence_model;
    for(int i=0; i<nstate-(dim+2); i++)
    {
        //do something to trick compiler
        viscous_flux_turbulence_model[i] = primitive_soln_turbulence_model[i];
    }
    
    return viscous_flux_turbulence_model;
}
//To do later
//get right primitive variables for turbulence model
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<real2,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convert_conservative_to_primitive_turbulence_model (
    const std::array<real2,nstate> &conservative_soln) const
{   
    std::array<real2,nstate-(dim+2)> primitive_soln_turbulence_model;
    for(int i=0; i<nstate-(dim+2); i++)
        primitive_soln_turbulence_model[i] = conservative_soln[dim+2+i]/conservative_soln[0];
 
    return primitive_soln_turbulence_model;
}
//get right primitive variables gradients for turbulence model
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> ReynoldsAveragedNavierStokesBase<dim,nstate,real>
::convert_conservative_gradient_to_primitive_gradient_turbulence_model (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const
{   
    std::array<dealii::Tensor<1,dim,real2>,nstate-(dim+2)> primitive_soln_gradient_turbulence_model;

    //do something to trick compiler
    std::array<real2,nstate> conservative_solution;
    conservative_solution[0]=conservative_soln[1];

    for(int i=0; i<nstate-(dim+2); i++)
        //do something to trick compiler
        primitive_soln_gradient_turbulence_model[i] = solution_gradient[i];
 
    return primitive_soln_gradient_turbulence_model;
}
//not sure if it should be changed for RANS
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
    std::array<real,nstate> source_term = dissipative_source_term(pos,cell_index);
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
//not sure if it should be changed for RANS
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
//not sure if it should be changed for RANS
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
//not sure if it should be changed for RANS
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
//not sure if it should be changed for RANS
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
//not sure if it should be changed for RANS
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
//To do later
//get the right formula for eddy viscosity
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,dim+2> &primitive_soln_rans,
    const std::array<real2,nstate-(dim+2)> &primitive_soln_turbulence_model) const
{
    // Compute the eddy viscosity
    //do something to trick compiler 
    const real2 eddy_viscosity = primitive_soln_rans[0]+primitive_soln_turbulence_model[0];

    return eddy_viscosity;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 ReynoldsAveragedNavierStokes_SAneg<dim,nstate,real>
::scale_eddy_viscosity_templated (
    const std::array<real2,dim+2> &primitive_soln_rans,
    const real2 eddy_viscosity) const
{
    // Scaled non-dimensional eddy viscosity; 
    const real2 scaled_eddy_viscosity = primitive_soln_rans[0]*eddy_viscosity;

    return scaled_eddy_viscosity;
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
    const real2 scaled_eddy_viscosity = scale_eddy_viscosity_templated<real2>(primitive_soln_rans,eddy_viscosity);

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
    const real2 scaled_eddy_viscosity = scale_eddy_viscosity_templated<real2>(primitive_soln_rans,eddy_viscosity);

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