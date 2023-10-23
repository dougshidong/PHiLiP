#include <cmath>
#include <vector>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "model.h"
#include "large_eddy_simulation.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Large Eddy Simulation (LES) Base Class
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulationBase<dim, nstate, real>::LargeEddySimulationBase(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : ModelBase<dim,nstate,real>(manufactured_solution_function) 
    , turbulent_prandtl_number(turbulent_prandtl_number)
    , ratio_of_filter_width_to_cell_size(ratio_of_filter_width_to_cell_size)
    , navier_stokes_physics(std::make_unique < NavierStokes<dim,nstate,real> > (
            ref_length,
            gamma_gas,
            mach_inf,
            angle_of_attack,
            side_slip_angle,
            prandtl_number,
            reynolds_number_inf,
            use_constant_viscosity,
            constant_viscosity,
            temperature_inf,
            isothermal_wall_temperature,
            thermal_boundary_condition_type,
            manufactured_solution_function,
            two_point_num_flux_type))
{
    static_assert(nstate==dim+2, "ModelBase::LargeEddySimulationBase() should be created with nstate=dim+2");
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 LargeEddySimulationBase<dim,nstate,real>
::get_tensor_magnitude_sqr (
    const dealii::Tensor<2,dim,real2> &tensor) const
{
    real2 tensor_magnitude_sqr = 0.0; // complex initializes it as 0+0i
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
template<typename real2>
real2 LargeEddySimulationBase<dim,nstate,real>
::get_tensor_magnitude (
    const dealii::Tensor<2,dim,real2> &tensor) const
{
    const real2 tensor_magnitude_sqr = this->template get_tensor_magnitude_sqr<real2>(tensor);
    return sqrt(2.0*tensor_magnitude_sqr);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> LargeEddySimulationBase<dim,nstate,real>
::convective_flux (
    const std::array<real,nstate> &/*conservative_soln*/) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    for (int i=0; i<nstate; i++) {
        conv_flux[i] = 0; // No additional convective terms for Large Eddy Simulation
    }
    return conv_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> LargeEddySimulationBase<dim,nstate,real>
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
std::array<dealii::Tensor<1,dim,real2>,nstate> LargeEddySimulationBase<dim,nstate,real>
::dissipative_flux_templated (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
    const dealii::types::global_dof_index cell_index) const
{   
    // Step 1,2: Primitive solution and Gradient of primitive solution
    const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient = this->navier_stokes_physics->convert_conservative_gradient_to_primitive_gradient_templated(conservative_soln, solution_gradient);
    const std::array<real2,nstate> primitive_soln = this->navier_stokes_physics->convert_conservative_to_primitive_templated(conservative_soln); // from Euler

    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const dealii::Tensor<1,dim,real2> vel = this->navier_stokes_physics->extract_velocities_from_primitive(primitive_soln); // from Euler
    // Templated virtual member functions
    dealii::Tensor<2,dim,real2> viscous_stress_tensor;
    dealii::Tensor<1,dim,real2> heat_flux;
    if constexpr(std::is_same<real2,real>::value){ 
        viscous_stress_tensor = compute_SGS_stress_tensor(primitive_soln, primitive_soln_gradient,cell_index);
        heat_flux = compute_SGS_heat_flux(primitive_soln, primitive_soln_gradient,cell_index);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        viscous_stress_tensor = compute_SGS_stress_tensor_fad(primitive_soln, primitive_soln_gradient,cell_index);
        heat_flux = compute_SGS_heat_flux_fad(primitive_soln, primitive_soln_gradient,cell_index);
    }
    else{
        std::cout << "ERROR in physics/large_eddy_simulation.cpp --> dissipative_flux_templated(): real2!=real || real2!=FadType)" << std::endl;
        std::abort();
    }
    
    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    std::array<dealii::Tensor<1,dim,real2>,nstate> viscous_flux
        = this->navier_stokes_physics->dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux(vel,viscous_stress_tensor,heat_flux);
    
    return viscous_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> LargeEddySimulationBase<dim,nstate,real>
::dissipative_flux_dot_normal (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const std::array<real,nstate> &/*filtered_solution*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*filtered_solution_gradient*/,
        const bool on_boundary,
        const dealii::types::global_dof_index cell_index,
        const dealii::Tensor<1,dim,real> &normal,
        const int boundary_type) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux;
    std::array<real,nstate> dissipative_flux_dot_normal;
    dissipative_flux_dot_normal.fill(0.0); // initialize
    // Associated thermal boundary condition
    if((on_boundary && (navier_stokes_physics->thermal_boundary_condition_type == thermal_boundary_condition_enum::adiabatic))
        && ((boundary_type == 1001) || (boundary_type == 1006))) { 

        /** If adiabatic on either slip (1001) or no-slip (1006) wall BCs */
        // adiabatic boundary
        // --> Modify viscous flux such that normal_vector dot gradient of temperature must be zero

        // REFERENCES:
        /* (1) Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
         * (2) For the boundary condition case, refer to the equation above equation 458 of the following paper:
         *  Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
         */

        // Step 1,2: Primitive solution and Gradient of primitive solution
        const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = this->navier_stokes_physics->convert_conservative_gradient_to_primitive_gradient_templated(solution, solution_gradient);
        const std::array<real,nstate> primitive_soln = this->navier_stokes_physics->convert_conservative_to_primitive_templated(solution); // from Euler

        // Step 3: Viscous stress tensor, Velocities, Heat flux
        const dealii::Tensor<1,dim,real> vel = this->navier_stokes_physics->extract_velocities_from_primitive(primitive_soln); // from Euler
        dealii::Tensor<2,dim,real> viscous_stress_tensor = compute_SGS_stress_tensor(primitive_soln, primitive_soln_gradient,cell_index);
        dealii::Tensor<1,dim,real> heat_flux;
        for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
            // set the heat flux to zero since we want the normal dot gradient of temperature to be zero for an adiabatic boundary
            heat_flux[flux_dim] = 0.0;
        }
        // Step 4: Construct viscous flux; Note: sign corresponds to LHS
        dissipative_flux = this->navier_stokes_physics->dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux(vel,viscous_stress_tensor,heat_flux);
    } else {
        // if not on boundary and for all other types of boundary conditions (including isothermal) --> no change to dissipative flux
        // no change to dissipative flux for BCs that do not impose a condition on the gradient at the boundary
        dissipative_flux = dissipative_flux_templated<real>(solution,solution_gradient,cell_index);
    }

    // compute the dot product with the normal vector
    for (int s=0; s<nstate; s++) {
        for (int d=0; d<dim; ++d) {
            dissipative_flux_dot_normal[s] += dissipative_flux[s][d] * normal[d];//compute dot product
        }
    }

    return dissipative_flux_dot_normal;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> LargeEddySimulationBase<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    std::array<real,nstate> eig;
    eig.fill(0.0);
    return eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real LargeEddySimulationBase<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    const real max_eig = 0.0;
    return max_eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real LargeEddySimulationBase<dim,nstate,real>
::max_convective_normal_eigenvalue (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    const real max_eig = 0.0;
    return max_eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> LargeEddySimulationBase<dim,nstate,real>
::source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &/*solution*/,
        const real /*current_time*/,
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
template <int dim, int nstate, typename real>
std::array<real,nstate> LargeEddySimulationBase<dim,nstate,real>
::physical_source_term (
        const dealii::Point<dim,real> &/*pos*/,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
        const dealii::types::global_dof_index /*cell_index*/) const
{
    std::array<real,nstate> physical_source;
    physical_source = this->channel_flow_source_term(conservative_soln);

    return physical_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> LargeEddySimulationBase<dim,nstate,real>
::channel_flow_source_term (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,nstate> source_term;
    std::fill(source_term.begin(), source_term.end(), 0.0);

    // Get nondimensional (w.r.t. freestream) bulk velocity
    // const real density = conservative_soln[0];
    const std::array<real,nstate> primitive_soln = this->navier_stokes_physics->convert_conservative_to_primitive_templated(conservative_soln);
    // const real viscosity_coefficient = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln);
    // const real bulk_velocity = viscosity_coefficient*(this->channel_bulk_velocity_reynolds_number)/(density*this->half_channel_height*this->navier_stokes_physics->reynolds_number_inf);
    const real bulk_velocity = 1.0; // since we nondimensionalize w.r.t. freestream values, which are set at the bulk values, this value is simply 1.0
    
    // x-momentum term
    source_term[1] = (this->bulk_density*bulk_velocity - conservative_soln[1])/this->time_step;
    
    // energy term
    const real x_velocity = primitive_soln[1];
    source_term[nstate-1] = x_velocity*source_term[1];
    
    return source_term;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
double LargeEddySimulationBase<dim,nstate,real>
::get_filter_width (const dealii::types::global_dof_index cell_index) const
{ 
    // Compute the LES filter width
    /** Reference: Marta de la Llave Plata, et al. "On the performance of a 
     *  high-order multiscale DG approach to LES at increasing Reynolds number."
     *  Computers and Fluids 194 (2019), Page 4, Eq.(14).
     * */
    const int cell_poly_degree = this->cellwise_poly_degree[cell_index];
    return get_filter_width_from_poly_degree(cell_index,cell_poly_degree);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
double LargeEddySimulationBase<dim,nstate,real>
::get_filter_width_from_poly_degree (
    const dealii::types::global_dof_index cell_index, 
    const int cell_poly_degree) const
{ 
    // Compute the LES filter width
    /** Reference: Marta de la Llave Plata, et al. "On the performance of a 
     *  high-order multiscale DG approach to LES at increasing Reynolds number."
     *  Computers and Fluids 194 (2019), Page 4, Eq.(14).
     * */
    const double cell_volume = this->cellwise_volume[cell_index];
    double filter_width = pow(cell_volume, (1.0/3.0))/(cell_poly_degree+1);
    // Resize given the ratio of filter width to cell size
    filter_width *= ratio_of_filter_width_to_cell_size;

    return filter_width;
}
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
template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> LargeEddySimulationBase<dim,nstate,real>
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
dealii::Tensor<2,nstate,real> LargeEddySimulationBase<dim,nstate,real>
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
std::array<real,nstate> LargeEddySimulationBase<dim,nstate,real>
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
std::array<dealii::Tensor<1,dim,real>,nstate> LargeEddySimulationBase<dim,nstate,real>
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
std::array<real,nstate> LargeEddySimulationBase<dim,nstate,real>
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
// Smagorinsky eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_Smagorinsky<dim, nstate, real>::LargeEddySimulation_Smagorinsky(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              model_constant,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : LargeEddySimulationBase<dim,nstate,real>(ref_length,
                                               gamma_gas,
                                               mach_inf,
                                               angle_of_attack,
                                               side_slip_angle,
                                               prandtl_number,
                                               reynolds_number_inf,
                                               use_constant_viscosity,
                                               constant_viscosity,
                                               temperature_inf,
                                               turbulent_prandtl_number,
                                               ratio_of_filter_width_to_cell_size,
                                               isothermal_wall_temperature,
                                               thermal_boundary_condition_type,
                                               manufactured_solution_function,
                                               two_point_num_flux_type)
    , model_constant(model_constant)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
double LargeEddySimulation_Smagorinsky<dim,nstate,real>
::get_model_constant_times_filter_width (
    const dealii::types::global_dof_index cell_index) const
{
    // Compute the filter width for the cell
    const double filter_width = this->get_filter_width(cell_index);
    // Product of the model constant (Cs) and the filter width (delta)
    const double model_constant_times_filter_width = model_constant*filter_width;
    return model_constant_times_filter_width;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
double LargeEddySimulation_Smagorinsky<dim,nstate,real>
::get_model_constant_times_filter_width_squared (
    const dealii::types::global_dof_index cell_index) const
{
    // Product of the model constant (Cs) and the filter width (delta) all squared
    const double model_constant_times_filter_width = get_model_constant_times_filter_width(cell_index);
    return model_constant_times_filter_width*model_constant_times_filter_width;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
FadType LargeEddySimulation_Smagorinsky<dim,nstate,real>
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
real2 LargeEddySimulation_Smagorinsky<dim,nstate,real>
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
    
    // Product of the model constant (Cs) and the filter width (delta)
    const real2 model_constant_times_filter_width_squared = get_model_constant_times_filter_width_squared(cell_index);
    // Get magnitude of strain_rate_tensor
    const real2 strain_rate_tensor_magnitude = this->template get_tensor_magnitude<real2>(strain_rate_tensor);
    // Compute the eddy viscosity
    const real2 eddy_viscosity = model_constant_times_filter_width_squared*strain_rate_tensor_magnitude;

    return eddy_viscosity;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
real2 LargeEddySimulation_Smagorinsky<dim,nstate,real>
::scale_eddy_viscosity_templated (
    const std::array<real2,nstate> &primitive_soln,
    const real2 eddy_viscosity) const
{
    // Scaled non-dimensional eddy viscosity; See Plata 2019, Computers and Fluids, Eq.(12)
    const real2 scaled_eddy_viscosity = primitive_soln[0]*eddy_viscosity;

    return scaled_eddy_viscosity;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_heat_flux (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_SGS_heat_flux_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,FadType> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_heat_flux_fad (
    const std::array<FadType,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_SGS_heat_flux_templated<FadType>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<1,dim,real2> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_heat_flux_templated (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{   
    // Compute non-dimensional eddy viscosity; See Plata 2019, Computers and Fluids, Eq.(12)
    real2 eddy_viscosity;
    if constexpr(std::is_same<real2,real>::value){ 
        eddy_viscosity = compute_eddy_viscosity(primitive_soln,primitive_soln_gradient,cell_index);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        eddy_viscosity = compute_eddy_viscosity_fad(primitive_soln,primitive_soln_gradient,cell_index);
    }
    else{
        std::cout << "ERROR in physics/large_eddy_simulation.cpp --> compute_SGS_heat_flux_templated(): real2 != real or FadType" << std::endl;
        std::abort();
    }

    // Scaled non-dimensional eddy viscosity; See Plata 2019, Computers and Fluids, Eq.(12)
    const real2 scaled_eddy_viscosity = scale_eddy_viscosity_templated<real2>(primitive_soln,eddy_viscosity);

    // Compute scaled heat conductivity
    const real2 scaled_heat_conductivity = this->navier_stokes_physics->compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number(scaled_eddy_viscosity,this->turbulent_prandtl_number);

    // Get temperature gradient
    const dealii::Tensor<1,dim,real2> temperature_gradient = this->navier_stokes_physics->compute_temperature_gradient(primitive_soln, primitive_soln_gradient);

    // Compute the SGS stress tensor via the eddy_viscosity and the strain rate tensor
    dealii::Tensor<1,dim,real2> heat_flux_SGS = this->navier_stokes_physics->compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient(scaled_heat_conductivity,temperature_gradient);

    return heat_flux_SGS;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,real> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_stress_tensor (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_SGS_stress_tensor_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,dim,FadType> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_stress_tensor_fad (
    const std::array<FadType,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,FadType>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_SGS_stress_tensor_templated<FadType>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template<typename real2>
dealii::Tensor<2,dim,real2> LargeEddySimulation_Smagorinsky<dim,nstate,real>
::compute_SGS_stress_tensor_templated (
    const std::array<real2,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    // Compute non-dimensional eddy viscosity; See Plata 2019, Computers and Fluids, Eq.(12)
    real2 eddy_viscosity;
    if constexpr(std::is_same<real2,real>::value){ 
        eddy_viscosity = compute_eddy_viscosity(primitive_soln,primitive_soln_gradient,cell_index);
    }
    else if constexpr(std::is_same<real2,FadType>::value){ 
        eddy_viscosity = compute_eddy_viscosity_fad(primitive_soln,primitive_soln_gradient,cell_index);
    }
    else{
        std::cout << "ERROR in physics/large_eddy_simulation.cpp --> compute_SGS_stress_tensor_templated(): real2 != real or FadType" << std::endl;
        std::abort();
    }
    
    // Scaled non-dimensional eddy viscosity; See Plata 2019, Computers and Fluids, Eq.(12)
    const real2 scaled_eddy_viscosity = scale_eddy_viscosity_templated<real2>(primitive_soln,eddy_viscosity);

    // Get velocity gradients
    const dealii::Tensor<2,dim,real2> vel_gradient 
        = this->navier_stokes_physics->extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    
    // Get strain rate tensor
    const dealii::Tensor<2,dim,real2> strain_rate_tensor 
        = this->navier_stokes_physics->compute_strain_rate_tensor(vel_gradient);

    // Compute the SGS stress tensor via the eddy_viscosity and the strain rate tensor
    dealii::Tensor<2,dim,real2> SGS_stress_tensor;
    SGS_stress_tensor = this->navier_stokes_physics->compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor(scaled_eddy_viscosity,strain_rate_tensor);

    return SGS_stress_tensor;
}
//----------------------------------------------------------------
//================================================================
// WALE (Wall-Adapting Local Eddy-viscosity) eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_WALE<dim, nstate, real>::LargeEddySimulation_WALE(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              model_constant,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : LargeEddySimulation_Smagorinsky<dim,nstate,real>(ref_length,
                                                       gamma_gas,
                                                       mach_inf,
                                                       angle_of_attack,
                                                       side_slip_angle,
                                                       prandtl_number,
                                                       reynolds_number_inf,
                                                       use_constant_viscosity,
                                                       constant_viscosity,
                                                       temperature_inf,
                                                       turbulent_prandtl_number,
                                                       ratio_of_filter_width_to_cell_size,
                                                       model_constant,
                                                       isothermal_wall_temperature,
                                                       thermal_boundary_condition_type,
                                                       manufactured_solution_function,
                                                       two_point_num_flux_type)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real LargeEddySimulation_WALE<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
FadType LargeEddySimulation_WALE<dim,nstate,real>
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
real2 LargeEddySimulation_WALE<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,nstate> &/*primitive_soln*/,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    const dealii::Tensor<2,dim,real2> vel_gradient 
        = this->navier_stokes_physics->extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    const dealii::Tensor<2,dim,real2> strain_rate_tensor 
        = this->navier_stokes_physics->compute_strain_rate_tensor(vel_gradient);
    
    // Product of the model constant (Cs) and the filter width (delta) squared
    const real2 model_constant_times_filter_width_squared = this->get_model_constant_times_filter_width_squared(cell_index);

    /** Get traceless symmetric square of velocity gradient tensor, i.e. $\bm{S}^{d}$
     *  Reference: Nicoud and Ducros (1999) - Equation (10)
     */
    // -- Compute $\bm{g}^{2}$
    dealii::Tensor<2,dim,real2> g_sqr; // $g_{ij}^{2}$
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            
            real2 val;if(std::is_same<real2,real>::value){val = 0.0;}

            for (int k=0; k<dim; ++k) {
                val += vel_gradient[i][k]*vel_gradient[k][j];
            }
            g_sqr[i][j] = val;
        }
    }
    real2 trace_g_sqr;if(std::is_same<real2,real>::value){trace_g_sqr = 0.0;}
    for (int k=0; k<dim; ++k) {
        trace_g_sqr += g_sqr[k][k];
    }
    dealii::Tensor<2,dim,real2> traceless_symmetric_square_of_velocity_gradient_tensor;
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            traceless_symmetric_square_of_velocity_gradient_tensor[i][j] = 0.5*(g_sqr[i][j]+g_sqr[j][i]);
        }
    }
    for (int k=0; k<dim; ++k) {
        traceless_symmetric_square_of_velocity_gradient_tensor[k][k] += -(1.0/3.0)*trace_g_sqr;
    }
    
    // Get magnitude of strain_rate_tensor and ducros_strain_rate_tensor
    const real2 strain_rate_tensor_magnitude_sqr                                     = this->template get_tensor_magnitude_sqr<real2>(strain_rate_tensor);
    const real2 traceless_symmetric_square_of_velocity_gradient_tensor_magnitude_sqr = this->template get_tensor_magnitude_sqr<real2>(traceless_symmetric_square_of_velocity_gradient_tensor);
    // Compute the eddy viscosity
    // -- Initialize as zero
    real2 eddy_viscosity;if(std::is_same<real2,real>::value){eddy_viscosity = 0.0;}
    if((strain_rate_tensor_magnitude_sqr != 0.0) &&
       (traceless_symmetric_square_of_velocity_gradient_tensor_magnitude_sqr != 0.0)) {
        /** Eddy viscosity is zero in the absence of turbulent fluctuations, 
         *  i.e. zero strain rate and zero rotation rate. See Nicoud and Ducros (1999). 
         *  Since the denominator in this eddy viscosity model will go to zero, 
         *  we must explicitly set the eddy viscosity to zero to avoid a division by zero.
         *  Or equivalently, update it from its zero initialization only if there is turbulence.
        */
        eddy_viscosity = model_constant_times_filter_width_squared*pow(traceless_symmetric_square_of_velocity_gradient_tensor_magnitude_sqr,1.5)/(pow(strain_rate_tensor_magnitude_sqr,2.5) + pow(traceless_symmetric_square_of_velocity_gradient_tensor_magnitude_sqr,1.25));
    }

    return eddy_viscosity;
}
//----------------------------------------------------------------
//================================================================
// Vreman eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_Vreman<dim, nstate, real>::LargeEddySimulation_Vreman(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              model_constant,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : LargeEddySimulation_Smagorinsky<dim,nstate,real>(ref_length,
                                                       gamma_gas,
                                                       mach_inf,
                                                       angle_of_attack,
                                                       side_slip_angle,
                                                       prandtl_number,
                                                       reynolds_number_inf,
                                                       use_constant_viscosity,
                                                       constant_viscosity,
                                                       temperature_inf,
                                                       turbulent_prandtl_number,
                                                       ratio_of_filter_width_to_cell_size,
                                                       model_constant,
                                                       isothermal_wall_temperature,
                                                       thermal_boundary_condition_type,
                                                       manufactured_solution_function,
                                                       two_point_num_flux_type)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real LargeEddySimulation_Vreman<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
FadType LargeEddySimulation_Vreman<dim,nstate,real>
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
real2 LargeEddySimulation_Vreman<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,nstate> &/*primitive_soln*/,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    const dealii::Tensor<2,dim,real2> vel_gradient 
        = this->navier_stokes_physics->extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);

    // Compute the filter width for the cell
    const double filter_width = this->get_filter_width(cell_index);
    
    /** Reference: Vreman (2004) - Equation (7)
     */
    // -- Compute $\bm{beta}$
    dealii::Tensor<2,dim,real2> beta_tensor;
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            
            real2 val;if(std::is_same<real2,real>::value){val = 0.0;}

            for (int k=0; k<dim; ++k) {
                val += vel_gradient[i][k]*vel_gradient[j][k];
            }
            beta_tensor[i][j] = filter_width*filter_width*val; // for isotropic filter width
        }
    }
    // Reference: Vreman (2004) - Equation (8) - $B_{\beta}$ (determinant of beta tensor -- symmetrical)
    real2 beta_tensor_determinant;if(std::is_same<real2,real>::value){beta_tensor_determinant = 0.0;}
    if constexpr(dim>1){
        beta_tensor_determinant = beta_tensor[0][0]*beta_tensor[1][1] - beta_tensor[0][1]*beta_tensor[0][1];
    }
    if constexpr(dim==3){
        for (int i=0; i<2; ++i) {
            beta_tensor_determinant += beta_tensor[i][i]*beta_tensor[2][2] - beta_tensor[i][2]*beta_tensor[i][2];
        }
    }
    
    // Get magnitude of velocity gradient tensor squared
    const real2 velocity_gradient_tensor_magnitude_sqr = this->template get_tensor_magnitude_sqr<real2>(vel_gradient);
    // Compute the eddy viscosity
    // -- Initialize as zero
    real2 eddy_viscosity;if(std::is_same<real2,real>::value){eddy_viscosity = 0.0;}
    if((velocity_gradient_tensor_magnitude_sqr !=0.0) && (beta_tensor_determinant >= 0.0)) {
        /** Eddy viscosity is zero in the absence of turbulent fluctuations, 
         *  i.e. zero strain rate and zero rotation rate, 
         *  also B is positive-semidefinite (i.e. beta_tensor_determinant>=0). See Vreman (2004). 
         *  Since the denominator in this eddy viscosity model will go to zero, 
         *  we must explicitly set the eddy viscosity to zero to avoid a division by zero.
         *  Or equivalently, update it from its zero initialization only if there is turbulence.
        */
        // Reference: Vreman (2004) - Equation (5)
        eddy_viscosity = this->model_constant*sqrt(beta_tensor_determinant/velocity_gradient_tensor_magnitude_sqr);
    }

    return eddy_viscosity;
}
//----------------------------------------------------------------
//================================================================
// Shear-improved Smagorinsky eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_ShearImprovedSmagorinsky<dim, nstate, real>::LargeEddySimulation_ShearImprovedSmagorinsky(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              model_constant,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : LargeEddySimulation_Smagorinsky<dim,nstate,real>(ref_length,
                                                       gamma_gas,
                                                       mach_inf,
                                                       angle_of_attack,
                                                       side_slip_angle,
                                                       prandtl_number,
                                                       reynolds_number_inf,
                                                       use_constant_viscosity,
                                                       constant_viscosity,
                                                       temperature_inf,
                                                       turbulent_prandtl_number,
                                                       ratio_of_filter_width_to_cell_size,
                                                       model_constant,
                                                       isothermal_wall_temperature,
                                                       thermal_boundary_condition_type,
                                                       manufactured_solution_function,
                                                       two_point_num_flux_type)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real LargeEddySimulation_ShearImprovedSmagorinsky<dim,nstate,real>
::compute_eddy_viscosity (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return compute_eddy_viscosity_templated<real>(primitive_soln,primitive_soln_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
FadType LargeEddySimulation_ShearImprovedSmagorinsky<dim,nstate,real>
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
real2 LargeEddySimulation_ShearImprovedSmagorinsky<dim,nstate,real>
::compute_eddy_viscosity_templated (
    const std::array<real2,nstate> &/*primitive_soln*/,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &primitive_soln_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    /* This SGS model modifies the original Smagorinsky model, therefore two references are provided:
     *  - Reference 1: de la Llave Plata et al. (2019). "On the performance of a high-order multiscale DG approach to LES at increasing Reynolds number."
     *  - Reference 2: E. Leveque, F. Toschi, L. Shao and J.-P. Bertoglio (2007, J. Fluid Mech.) "Shear-improved Smagorinsky model for large-eddy simulation of wall-bounded turbulent flows"
     * This implementation uses Equation (14) in reference 1 for the original Smagorinsky model,
     * and the modification provided by equation (2.4) in reference 2.
    */
    // Get velocity gradient
    const dealii::Tensor<2,dim,real2> vel_gradient 
        = this->navier_stokes_physics->extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    // Get strain rate tensor
    const dealii::Tensor<2,dim,real2> strain_rate_tensor 
        = this->navier_stokes_physics->compute_strain_rate_tensor(vel_gradient);
    
    // Product of the model constant (Cs) and the filter width (delta) squared
    const real2 model_constant_times_filter_width_squared = this->get_model_constant_times_filter_width_squared(cell_index);
    // Get magnitude of strain_rate_tensor
    const real2 strain_rate_tensor_magnitude = this->template get_tensor_magnitude<real2>(strain_rate_tensor);
    // Compute the eddy viscosity; Eq.(14) in reference 1 with modification by Eq.(2.4) in reference 2
    const real2 eddy_viscosity = model_constant_times_filter_width_squared*(
                                    strain_rate_tensor_magnitude - this->cellwise_mean_strain_rate_tensor_magnitude[cell_index]);

    return eddy_viscosity;
}
//----------------------------------------------------------------
//================================================================
// Variational multiscale (VMS) eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_VMS<dim, nstate, real>::LargeEddySimulation_VMS(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              model_constant,
    const unsigned int                                        poly_degree,
    const unsigned int                                        poly_degree_large_scales,
    const double                                              mesh_size,
    const double                                              curve_fit_constant,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : LargeEddySimulation_Smagorinsky<dim,nstate,real>(ref_length,
                                                       gamma_gas,
                                                       mach_inf,
                                                       angle_of_attack,
                                                       side_slip_angle,
                                                       prandtl_number,
                                                       reynolds_number_inf,
                                                       use_constant_viscosity,
                                                       constant_viscosity,
                                                       temperature_inf,
                                                       turbulent_prandtl_number,
                                                       ratio_of_filter_width_to_cell_size,
                                                       model_constant,
                                                       isothermal_wall_temperature,
                                                       thermal_boundary_condition_type,
                                                       manufactured_solution_function,
                                                       two_point_num_flux_type)
    , poly_degree((double)poly_degree)
    , poly_degree_large_scales((double)poly_degree_large_scales)
    , mesh_size(mesh_size)
    , curve_fit_constant(curve_fit_constant)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
double LargeEddySimulation_VMS<dim,nstate,real>
::get_model_constant_times_filter_width (
    const dealii::types::global_dof_index /*cell_index*/) const
{
    // Smagorinsky constant for a given DG discretization; equation 8 in reference
    const double discontinuous_galerkin_smagorinsky_model_constant = 0.172*mesh_size/poly_degree;

    // Model constant times filter width; equations 18 and 14 in reference
    double model_constant_times_filter_width = 1.0 - pow(curve_fit_constant*(poly_degree_large_scales/poly_degree), 4.0/3.0);
    model_constant_times_filter_width = pow(model_constant_times_filter_width, -3.0/4.0);
    model_constant_times_filter_width *= discontinuous_galerkin_smagorinsky_model_constant;

    return model_constant_times_filter_width;
}
//----------------------------------------------------------------
//================================================================
// Small-Small Variational multiscale (VMS) eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_SmallSmallVMS<dim, nstate, real>::LargeEddySimulation_SmallSmallVMS(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              model_constant,
    const unsigned int                                        poly_degree,
    const unsigned int                                        poly_degree_large_scales,
    const double                                              mesh_size,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : LargeEddySimulation_VMS<dim,nstate,real>(ref_length,
                                               gamma_gas,
                                               mach_inf,
                                               angle_of_attack,
                                               side_slip_angle,
                                               prandtl_number,
                                               reynolds_number_inf,
                                               use_constant_viscosity,
                                               constant_viscosity,
                                               temperature_inf,
                                               turbulent_prandtl_number,
                                               ratio_of_filter_width_to_cell_size,
                                               model_constant,
                                               poly_degree,
                                               poly_degree_large_scales,
                                               mesh_size,
                                               1.174, // Equation 18 in reference
                                               isothermal_wall_temperature,
                                               thermal_boundary_condition_type,
                                               manufactured_solution_function,
                                               two_point_num_flux_type)
{ }
//----------------------------------------------------------------
//================================================================
// All-All Variational multiscale (VMS) eddy viscosity model
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_AllAllVMS<dim, nstate, real>::LargeEddySimulation_AllAllVMS(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              model_constant,
    const unsigned int                                        poly_degree,
    const unsigned int                                        poly_degree_large_scales,
    const double                                              mesh_size,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : LargeEddySimulation_VMS<dim,nstate,real>(ref_length,
                                               gamma_gas,
                                               mach_inf,
                                               angle_of_attack,
                                               side_slip_angle,
                                               prandtl_number,
                                               reynolds_number_inf,
                                               use_constant_viscosity,
                                               constant_viscosity,
                                               temperature_inf,
                                               turbulent_prandtl_number,
                                               ratio_of_filter_width_to_cell_size,
                                               model_constant,
                                               poly_degree,
                                               poly_degree_large_scales,
                                               mesh_size,
                                               1.082, // Equation 14 in reference
                                               isothermal_wall_temperature,
                                               thermal_boundary_condition_type,
                                               manufactured_solution_function,
                                               two_point_num_flux_type)
{ }
//----------------------------------------------------------------
//================================================================
// Dynamic Smagorinsky Model (DSM)
//================================================================
template <int dim, int nstate, typename real>
LargeEddySimulation_DSM<dim, nstate, real>::LargeEddySimulation_DSM(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              turbulent_prandtl_number,
    const double                                              ratio_of_filter_width_to_cell_size,
    const double                                              model_constant,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : LargeEddySimulation_Smagorinsky<dim,nstate,real>(ref_length,
                                                       gamma_gas,
                                                       mach_inf,
                                                       angle_of_attack,
                                                       side_slip_angle,
                                                       prandtl_number,
                                                       reynolds_number_inf,
                                                       use_constant_viscosity,
                                                       constant_viscosity,
                                                       temperature_inf,
                                                       turbulent_prandtl_number,
                                                       ratio_of_filter_width_to_cell_size,
                                                       model_constant,
                                                       isothermal_wall_temperature,
                                                       thermal_boundary_condition_type,
                                                       manufactured_solution_function,
                                                       two_point_num_flux_type)
{ }
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
double LargeEddySimulation_DSM<dim,nstate,real>
::get_model_constant_times_filter_width_squared (
    const dealii::types::global_dof_index cell_index) const
{
    // Model constant times filter width squared
    return this->dynamic_smagorinsky_model_constant_times_filter_width_sqr[cell_index];
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- LargeEddySimulationBase
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulationBase         < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_Smagorinsky
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_Smagorinsky < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_WALE
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_WALE        < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_Vreman
template class LargeEddySimulation_Vreman      < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_Vreman      < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_Vreman      < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_Vreman      < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_Vreman      < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_ShearImprovedSmagorinsky
template class LargeEddySimulation_ShearImprovedSmagorinsky < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_ShearImprovedSmagorinsky < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_ShearImprovedSmagorinsky < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_ShearImprovedSmagorinsky < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_ShearImprovedSmagorinsky < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_VMS
template class LargeEddySimulation_VMS < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_VMS < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_VMS < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_VMS < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_VMS < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_SmallSmallVMS
template class LargeEddySimulation_SmallSmallVMS < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_SmallSmallVMS < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_SmallSmallVMS < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_SmallSmallVMS < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_SmallSmallVMS < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_AllAllVMS
template class LargeEddySimulation_AllAllVMS < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_AllAllVMS < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_AllAllVMS < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_AllAllVMS < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_AllAllVMS < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
// -- LargeEddySimulation_DSM
template class LargeEddySimulation_DSM < PHILIP_DIM, PHILIP_DIM+2, double >;
template class LargeEddySimulation_DSM < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class LargeEddySimulation_DSM < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class LargeEddySimulation_DSM < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class LargeEddySimulation_DSM < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
//-------------------------------------------------------------------------------------
// Templated members used by derived classes, defined in respective parent classes
//-------------------------------------------------------------------------------------
// -- get_tensor_magnitude_sqr()
template double     LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, double     >::get_tensor_magnitude_sqr< double     >(const dealii::Tensor<2,PHILIP_DIM,double    > &tensor) const;
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, FadType    >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template RadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadType    >::get_tensor_magnitude_sqr< RadType    >(const dealii::Tensor<2,PHILIP_DIM,RadType   > &tensor) const;
template FadFadType LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::get_tensor_magnitude_sqr< FadFadType >(const dealii::Tensor<2,PHILIP_DIM,FadFadType> &tensor) const;
template RadFadType LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::get_tensor_magnitude_sqr< RadFadType >(const dealii::Tensor<2,PHILIP_DIM,RadFadType> &tensor) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, double     >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadType    >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::get_tensor_magnitude_sqr< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
// -- get_tensor_magnitude()
template double     LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, double     >::get_tensor_magnitude< double     >(const dealii::Tensor<2,PHILIP_DIM,double    > &tensor) const;
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, FadType    >::get_tensor_magnitude< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template RadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadType    >::get_tensor_magnitude< RadType    >(const dealii::Tensor<2,PHILIP_DIM,RadType   > &tensor) const;
template FadFadType LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::get_tensor_magnitude< FadFadType >(const dealii::Tensor<2,PHILIP_DIM,FadFadType> &tensor) const;
template RadFadType LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::get_tensor_magnitude< RadFadType >(const dealii::Tensor<2,PHILIP_DIM,RadFadType> &tensor) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, double     >::get_tensor_magnitude< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadType    >::get_tensor_magnitude< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::get_tensor_magnitude< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;
template FadType    LargeEddySimulationBase < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::get_tensor_magnitude< FadType    >(const dealii::Tensor<2,PHILIP_DIM,FadType   > &tensor) const;


} // Physics namespace
} // PHiLiP namespace
