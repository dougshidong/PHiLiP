#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
Euler<dim,nstate,real>::Euler ( 
    const Parameters::AllParameters *const                    parameters_input,
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type_input,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source)
    : PhysicsBase<dim,nstate,real>(parameters_input, has_nonzero_diffusion,has_nonzero_physical_source,manufactured_solution_function)
    , ref_length(ref_length)
    , gam(gamma_gas)
    , gamm1(gam-1.0)
    , density_inf(1.0) // Nondimensional - Free stream values
    , mach_inf(mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
    , angle_of_attack(angle_of_attack)
    , side_slip_angle(side_slip_angle)
    , sound_inf(1.0/(mach_inf))
    , pressure_inf(1.0/(gam*mach_inf_sqr))
    , entropy_inf(pressure_inf*pow(density_inf,-gam))
    , two_point_num_flux_type(two_point_num_flux_type_input)
    //, internal_energy_inf(1.0/(gam*(gam-1.0)*mach_inf_sqr)) 
    // Note: Eq.(3.11.18) has a typo in internal_energy_inf expression, mach_inf_sqr should be in denominator. 
{
    static_assert(nstate==dim+2, "Physics::Euler() should be created with nstate=dim+2");

    // Nondimensional temperature at infinity
    temperature_inf = gam*pressure_inf/density_inf * mach_inf_sqr; // Note by JB: this can simply be set = 1

    // For now, don't allow side-slip angle
    if (std::abs(side_slip_angle) >= 1e-14) {
        this->pcout << "Side slip angle = " << side_slip_angle << ". Side_slip_angle must be zero. " << std::endl;
        this->pcout << "I have not figured out the side slip angles just yet." << std::endl;
        std::abort();
    }
    if(dim==1) {
        velocities_inf[0] = 1.0;
    } else if(dim==2) {
        velocities_inf[0] = cos(angle_of_attack);
        velocities_inf[1] = sin(angle_of_attack); // Maybe minus?? -- Clarify with Doug
    } else if (dim==3) {
        velocities_inf[0] = cos(angle_of_attack)*cos(side_slip_angle);
        velocities_inf[1] = sin(angle_of_attack)*cos(side_slip_angle);
        velocities_inf[2] = sin(side_slip_angle);
    }

    assert(std::abs(velocities_inf.norm() - 1.0) < 1e-14);

    double velocity_inf_sqr = 1.0;
    dynamic_pressure_inf = 0.5 * density_inf * velocity_inf_sqr;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &conservative_soln,
    const real current_time,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    return source_term(pos,conservative_soln,current_time);
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &/*conservative_soln*/,
    const real /*current_time*/) const
{
    std::array<real,nstate> source_term = convective_source_term(pos);
    return source_term;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
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

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim,nstate,real>
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

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::convective_source_term (
    const dealii::Point<dim,real> &pos) const
{
    const std::array<real,nstate> manufactured_solution = get_manufactured_solution_value(pos);
    const std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient = get_manufactured_solution_gradient(pos);

    dealii::Tensor<1,nstate,real> convective_flux_divergence;
    for (int d=0;d<dim;d++) {
        dealii::Tensor<1,dim,real> normal;
        normal[d] = 1.0;
        const dealii::Tensor<2,nstate,real> jacobian = convective_flux_directional_jacobian(manufactured_solution, normal);

        //convective_flux_divergence += jacobian*manufactured_solution_gradient[d];
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

template <int dim, int nstate, typename real>
template<typename real2>
bool Euler<dim,nstate,real>::check_positive_quantity(real2 &qty, const std::string qty_name) const {
    using limiter_enum = Parameters::LimiterParam::LimiterType;
    bool qty_is_positive;

    if (this->all_parameters->limiter_param.bound_preserving_limiter != limiter_enum::positivity_preservingZhang2010
        && this->all_parameters->limiter_param.bound_preserving_limiter != limiter_enum::positivity_preservingWang2012) {
        if (qty < 0.0) {
            // Refer to base class for non-physical results handling
            qty = this->template handle_non_physical_result<real2>(qty_name + " is negative.");
            qty_is_positive = false;
        }
        else {
            qty_is_positive = true;
        }
    } else {
        qty_is_positive = true;
    }

    return qty_is_positive;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline std::array<real2,nstate> Euler<dim,nstate,real>
::convert_conservative_to_primitive ( const std::array<real2,nstate> &conservative_soln ) const
{
    std::array<real2, nstate> primitive_soln;

    real2 density = conservative_soln[0];
    dealii::Tensor<1,dim,real2> vel = compute_velocities<real2>(conservative_soln);
    real2 pressure = compute_pressure<real2>(conservative_soln);

    check_positive_quantity<real2>(density, "density");
    check_positive_quantity<real2>(pressure, "pressure");
    primitive_soln[0] = density;
    for (int d=0; d<dim; ++d) {
        primitive_soln[1+d] = vel[d];
    }
    primitive_soln[nstate-1] = pressure;

    return primitive_soln;
}

template <int dim, int nstate, typename real>
inline std::array<real,nstate> Euler<dim,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const
{

    const real density = primitive_soln[0];
    const dealii::Tensor<1,dim,real> velocities = extract_velocities_from_primitive<real>(primitive_soln);

    std::array<real, nstate> conservative_soln;
    conservative_soln[0] = density;
    for (int d=0; d<dim; ++d) {
        conservative_soln[1+d] = density*velocities[d];
    }
    conservative_soln[nstate-1] = compute_total_energy(primitive_soln);

    return conservative_soln;
}

//template <int dim, int nstate, typename real>
//inline dealii::Tensor<1,dim,double> Euler<dim,nstate,real>::compute_velocities_inf() const
//{
//    dealii::Tensor<1,dim,double> velocities;
//    return velocities;
//}

template <int dim, int nstate, typename real>
template<typename real2>
inline dealii::Tensor<1,dim,real2> Euler<dim,nstate,real>
::compute_velocities ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 density = conservative_soln[0];
    dealii::Tensor<1,dim,real2> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/density; }
    return vel;
}

template <int dim, int nstate, typename real>
template <typename real2>
inline real2 Euler<dim,nstate,real>
::compute_velocity_squared ( const dealii::Tensor<1,dim,real2> &velocities ) const
{
    real2 vel2 = 0.0;
    for (int d=0; d<dim; d++) { 
        vel2 = vel2 + velocities[d]*velocities[d]; 
    }    
    
    return vel2;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline dealii::Tensor<1,dim,real2> Euler<dim,nstate,real>
::extract_velocities_from_primitive ( const std::array<real2,nstate> &primitive_soln ) const
{
    dealii::Tensor<1,dim,real2> velocities;
    for (int d=0; d<dim; d++) { velocities[d] = primitive_soln[1+d]; }
    return velocities;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_total_energy ( const std::array<real,nstate> &primitive_soln ) const
{
    const real pressure = primitive_soln[nstate-1];
    const real kinetic_energy = compute_kinetic_energy_from_primitive_solution(primitive_soln);
    const real tot_energy = pressure / this->gamm1 + kinetic_energy;
    return tot_energy;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_kinetic_energy_from_primitive_solution ( const std::array<real,nstate> &primitive_soln ) const
{
    const real density = primitive_soln[0];
    const dealii::Tensor<1,dim,real> velocities = extract_velocities_from_primitive<real>(primitive_soln);
    const real vel2 = compute_velocity_squared<real>(velocities);
    const real kinetic_energy = 0.5*density*vel2;
    return kinetic_energy;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_kinetic_energy_from_conservative_solution ( const std::array<real,nstate> &conservative_soln ) const
{
    const std::array<real,nstate> primitive_soln = convert_conservative_to_primitive<real>(conservative_soln);
    const real kinetic_energy = compute_kinetic_energy_from_primitive_solution(primitive_soln);
    return kinetic_energy;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_entropy_measure ( const std::array<real,nstate> &conservative_soln ) const
{
    real density = conservative_soln[0];
    const real pressure = compute_pressure<real>(conservative_soln);
    return compute_entropy_measure(density, pressure);
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_entropy_measure ( const real density, const real pressure ) const
{
    //Copy such that we don't modify the original density that is passed
    real density_check = density;
    const bool density_is_positive = check_positive_quantity<real>(density_check, "density");
    if (density_is_positive)     return pressure*pow(density,-gam);
    else                         return (real)this->BIG_NUMBER;
}


template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_specific_enthalpy ( const std::array<real,nstate> &conservative_soln, const real pressure ) const
{
    const real density = conservative_soln[0];
    const real total_energy = conservative_soln[nstate-1];
    const real specific_enthalpy = (total_energy+pressure)/density;
    return specific_enthalpy;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_numerical_entropy_function ( const std::array<real,nstate> &conservative_soln ) const
{
    const real pressure = compute_pressure<real>(conservative_soln);
    const real density = conservative_soln[0];

    const real entropy = compute_entropy<real>(density, pressure);

    const real numerical_entropy_function = - density * entropy;

    return numerical_entropy_function;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 Euler<dim,nstate,real>
::compute_temperature ( const std::array<real2,nstate> &primitive_soln ) const
{
    const real2 density = primitive_soln[0];
    const real2 pressure = primitive_soln[nstate-1];
    const real2 temperature = gam*mach_inf_sqr*(pressure/density);
    return temperature;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_density_from_pressure_temperature ( const real pressure, const real temperature ) const
{
    const real density = gam*mach_inf_sqr*(pressure/temperature);
    return density;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_temperature_from_density_pressure ( const real density, const real pressure ) const
{
    const real temperature = gam*mach_inf_sqr*(pressure/density);
    return temperature;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_pressure_from_density_temperature ( const real density, const real temperature ) const
{
    const real pressure = density*temperature/(gam*mach_inf_sqr);
    return pressure;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline real2 Euler<dim,nstate,real>
::compute_pressure ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 density = conservative_soln[0];

    const real2 tot_energy  = conservative_soln[nstate-1];

    const dealii::Tensor<1,dim,real2> vel = compute_velocities<real2>(conservative_soln);

    const real2 vel2 = compute_velocity_squared<real2>(vel);
    real2 pressure = gamm1*(tot_energy - 0.5*density*vel2);
    
    check_positive_quantity<real2>(pressure, "pressure");
    return pressure;
}


template <int dim, int nstate, typename real>
template<typename real2>
inline real2 Euler<dim,nstate,real>
:: compute_entropy (const real2 density, const real2 pressure) const
{
    // copy density and pressure such that the check will not modify originals
    real2 density_check = density; 
    real2 pressure_check = pressure;
    const bool density_is_positive = check_positive_quantity(density_check, "density");
    const bool pressure_is_positive = check_positive_quantity(pressure_check, "pressure");
    if (density_is_positive && pressure_is_positive) {
        real2 entropy = pressure * pow(density, -gam);
        entropy = log(entropy);
        return entropy;
    } else {
        return (real2)this->BIG_NUMBER;
    }

}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    real density = conservative_soln[0];
    check_positive_quantity<real>(density, "density");
    const real pressure = compute_pressure<real>(conservative_soln);
    const real sound = sqrt(pressure*gam/density);
    return sound;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_sound ( const real density, const real pressure ) const
{
    //assert(density > 0);
    const real sound = sqrt(pressure*gam/density);
    return sound;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>
::compute_mach_number ( const std::array<real,nstate> &conservative_soln ) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);
    const real velocity = sqrt(compute_velocity_squared<real>(vel));
    const real sound = compute_sound (conservative_soln);
    const real mach_number = velocity/sound;
    return mach_number;
}

// Split form functions:

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim, nstate, real>
::convective_numerical_split_flux(const std::array<real,nstate> &conservative_soln1,
                                  const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    if(two_point_num_flux_type == two_point_num_flux_enum::KG) {
        conv_num_split_flux = convective_numerical_split_flux_kennedy_gruber(conservative_soln1, conservative_soln2);
    } else if(two_point_num_flux_type == two_point_num_flux_enum::IR) {
        conv_num_split_flux = convective_numerical_split_flux_ismail_roe(conservative_soln1, conservative_soln2);
    } else if(two_point_num_flux_type == two_point_num_flux_enum::CH) {
        conv_num_split_flux = convective_numerical_split_flux_chandrashekar(conservative_soln1, conservative_soln2);
    } else if(two_point_num_flux_type == two_point_num_flux_enum::Ra) {
        conv_num_split_flux = convective_numerical_split_flux_ranocha(conservative_soln1, conservative_soln2);
    }

    return conv_num_split_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim, nstate, real>
::convective_numerical_split_flux_kennedy_gruber(const std::array<real,nstate> &conservative_soln1,
                                                 const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    const real mean_density = compute_mean_density(conservative_soln1, conservative_soln2);
    const real mean_pressure = compute_mean_pressure(conservative_soln1, conservative_soln2);
    const dealii::Tensor<1,dim,real> mean_velocities = compute_mean_velocities(conservative_soln1,conservative_soln2);
    const real mean_specific_total_energy = compute_mean_specific_total_energy(conservative_soln1, conservative_soln2);

    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = mean_density * mean_velocities[flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = mean_density*mean_velocities[flux_dim]*mean_velocities[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += mean_pressure; // Add diagonal of pressure
        // Energy equation
        conv_num_split_flux[nstate-1][flux_dim] = mean_density*mean_velocities[flux_dim]*mean_specific_total_energy + mean_pressure * mean_velocities[flux_dim];
    }

    return conv_num_split_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim, nstate, real>
::compute_ismail_roe_parameter_vector_from_primitive(const std::array<real,nstate> &primitive_soln) const
{
    // Ismail-Roe parameter vector; Eq (3.14) [Gassner, Winters, and Kopriva, 2016, SBP]
    std::array<real,nstate> ismail_roe_parameter_vector;
    ismail_roe_parameter_vector[0] = sqrt(primitive_soln[0]/primitive_soln[nstate-1]);
    for(int d=0; d<dim; ++d){
        ismail_roe_parameter_vector[1+d] = ismail_roe_parameter_vector[0]*primitive_soln[1+d];
    }
    ismail_roe_parameter_vector[nstate-1] = sqrt(primitive_soln[0]*primitive_soln[nstate-1]);

    return ismail_roe_parameter_vector;
}

template <int dim, int nstate, typename real>
real Euler<dim, nstate, real>
::compute_ismail_roe_logarithmic_mean(const real val1, const real val2) const
{
    // See Appendix B [Ismail and Roe, 2009, Entropy-Consistent Euler Flux Functions II]
    // -- Numerically stable algorithm for computing the logarithmic mean
    const real zeta = val1/val2;
    const real f = (zeta-1.0)/(zeta+1.0);
    const real u = f*f;
    
    real F;
    if(u<1.0e-2){ F = 1.0 + u/3.0 + u*u/5.0 + u*u*u/7.0; } 
    else { 
        if constexpr(std::is_same<real,double>::value) F = std::log(zeta)/2.0/f; 
    }
    
    const real log_mean_val = (val1+val2)/(2.0*F);

    return log_mean_val;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim, nstate, real>
::convective_numerical_split_flux_ismail_roe(const std::array<real,nstate> &conservative_soln1,
                                             const std::array<real,nstate> &conservative_soln2) const
{
    // Get Ismail Roe parameter vectors
    const std::array<real,nstate> parameter_vector1 = compute_ismail_roe_parameter_vector_from_primitive(
                                                        convert_conservative_to_primitive<real>(conservative_soln1));
    const std::array<real,nstate> parameter_vector2 = compute_ismail_roe_parameter_vector_from_primitive(
                                                        convert_conservative_to_primitive<real>(conservative_soln2));

    // Compute mean (average) parameter vector
    std::array<real,nstate> avg_parameter_vector;
    for(int s=0; s<nstate; ++s){
        avg_parameter_vector[s] = 0.5*(parameter_vector1[s] + parameter_vector2[s]);
    }

    // Compute logarithmic mean parameter vector
    std::array<real,nstate> log_mean_parameter_vector;
    for(int s=0; s<nstate; ++s){
        log_mean_parameter_vector[s] = compute_ismail_roe_logarithmic_mean(parameter_vector1[s], parameter_vector2[s]);
    }

    // Compute Ismail Roe mean primitive variables; Eq (3.15) [Gassner, Winters, and Kopriva, 2016, SBP]
    std::array<real,dim> mean_velocities;
    const real mean_density = avg_parameter_vector[0]*log_mean_parameter_vector[nstate-1];
    for(int d=0; d<dim; ++d){
        mean_velocities[d] = avg_parameter_vector[1+d]/avg_parameter_vector[0];
    }
    const real mean_pressure = avg_parameter_vector[nstate-1]/avg_parameter_vector[0];
    // -- enthalpy
    real mean_enthalpy = (gam+1.0)*(log_mean_parameter_vector[nstate-1]/log_mean_parameter_vector[0]) + gamm1*mean_pressure;
    mean_enthalpy /= 2.0*gam;
    mean_enthalpy *= gam/(mean_density*gamm1);
    // -- get sum of mean velocities squared
    real mean_velocities_sqr_sum = 0.0;
    for(int d=0; d<dim; ++d){
        mean_velocities_sqr_sum += mean_velocities[d]*mean_velocities[d];
    }
    // -- add to enthalpy
    mean_enthalpy += 0.5*mean_velocities_sqr_sum;

    // Compute Ismail Roe convective numerical split flux
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = mean_density * mean_velocities[flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = mean_density*mean_velocities[flux_dim]*mean_velocities[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += mean_pressure; // Add diagonal of pressure
        // Energy equation
        conv_num_split_flux[nstate-1][flux_dim] = mean_density*mean_velocities[flux_dim]*mean_enthalpy;
    }

    return conv_num_split_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim, nstate, real>
::convective_numerical_split_flux_chandrashekar(const std::array<real,nstate> &conservative_soln1,
                                                const std::array<real,nstate> &conservative_soln2) const
{

    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    const real rho_log = compute_ismail_roe_logarithmic_mean(conservative_soln1[0], conservative_soln2[0]);
    const real pressure1 = compute_pressure<real>(conservative_soln1);
    const real pressure2 = compute_pressure<real>(conservative_soln2);

    const real beta1 = conservative_soln1[0]/(2.0*pressure1);
    const real beta2 = conservative_soln2[0]/(2.0*pressure2);

    const real beta_log = compute_ismail_roe_logarithmic_mean(beta1, beta2);
    const dealii::Tensor<1,dim,real> vel1 = compute_velocities<real>(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = compute_velocities<real>(conservative_soln2);

    const real pressure_hat = 0.5*(conservative_soln1[0] + conservative_soln2[0])/(2.0*0.5*(beta1+beta2));

    dealii::Tensor<1,dim,real> vel_avg;
    real vel_square_avg = 0.0;;
    for(int idim=0; idim<dim; idim++){
        vel_avg[idim] = 0.5*(vel1[idim]+vel2[idim]);
        vel_square_avg += (0.5 *(vel1[idim]+vel2[idim])) * (0.5 *(vel1[idim]+vel2[idim]));
    }

    real enthalpy_hat = 1.0/(2.0*beta_log*gamm1) + vel_square_avg + pressure_hat/rho_log;

    for(int idim=0; idim<dim; idim++){
        enthalpy_hat -= 0.5*(0.5*(vel1[idim]*vel1[idim] + vel2[idim]*vel2[idim]));
    }

    for(int flux_dim=0; flux_dim<dim; flux_dim++){
        // Density equation
        conv_num_split_flux[0][flux_dim] = rho_log * vel_avg[flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = rho_log*vel_avg[flux_dim]*vel_avg[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += pressure_hat; // Add diagonal of pressure

        // Energy equation
        conv_num_split_flux[nstate-1][flux_dim] = rho_log * vel_avg[flux_dim] * enthalpy_hat;
    }

   return conv_num_split_flux; 
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim, nstate, real>
::convective_numerical_split_flux_ranocha(const std::array<real,nstate> &conservative_soln1,
                                                const std::array<real,nstate> &conservative_soln2) const
{

    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    const real rho_log = compute_ismail_roe_logarithmic_mean(conservative_soln1[0], conservative_soln2[0]);
    const real pressure1 = compute_pressure<real>(conservative_soln1);
    const real pressure2 = compute_pressure<real>(conservative_soln2);

    const real beta1 = conservative_soln1[0]/(pressure1);
    const real beta2 = conservative_soln2[0]/(pressure2);

    const real beta_log = compute_ismail_roe_logarithmic_mean(beta1, beta2);
    const dealii::Tensor<1,dim,real> vel1 = compute_velocities<real>(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = compute_velocities<real>(conservative_soln2);

    const real pressure_hat = 0.5*(pressure1+pressure2);

    dealii::Tensor<1,dim,real> vel_avg;
    real vel_square_avg = 0.0;;
    for(int idim=0; idim<dim; idim++){
        vel_avg[idim] = 0.5*(vel1[idim]+vel2[idim]);
        vel_square_avg += (0.5 *(vel1[idim]+vel2[idim])) * (0.5 *(vel1[idim]+vel2[idim]));
    }

    real enthalpy_hat = 1.0/(beta_log*gamm1) + vel_square_avg + 2.0*pressure_hat/rho_log;

    for(int idim=0; idim<dim; idim++){
        enthalpy_hat -= 0.5*(0.5*(vel1[idim]*vel1[idim] + vel2[idim]*vel2[idim]));
    }

    for(int flux_dim=0; flux_dim<dim; flux_dim++){
        // Density equation
        conv_num_split_flux[0][flux_dim] = rho_log * vel_avg[flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = rho_log*vel_avg[flux_dim]*vel_avg[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += pressure_hat; // Add diagonal of pressure

        // Energy equation
        conv_num_split_flux[nstate-1][flux_dim] = rho_log * vel_avg[flux_dim] * enthalpy_hat;
        conv_num_split_flux[nstate-1][flux_dim] -= ( 0.5 *(pressure1*vel1[flux_dim] + pressure2*vel2[flux_dim]));
    }

   return conv_num_split_flux; 

}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim, nstate, real>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,nstate> entropy_var;
    const real density = conservative_soln[0];
    const real pressure = compute_pressure<real>(conservative_soln);
    
    const real entropy = compute_entropy<real>(density, pressure);

    const real rho_theta = pressure / gamm1;

    entropy_var[0] = (rho_theta *(gam + 1.0 - entropy) - conservative_soln[nstate-1])/rho_theta;
    for(int idim=0; idim<dim; idim++){
        entropy_var[idim+1] = conservative_soln[idim+1] / rho_theta;
    }
    entropy_var[nstate-1] = - density / rho_theta;

    return entropy_var;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim, nstate, real>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    //Eq. 119 and 120 from Chan, Jesse. "On discretely entropy conservative and entropy stable discontinuous Galerkin methods." Journal of Computational Physics 362 (2018): 346-374.
    //Extrapolated for 3D
    std::array<real,nstate> conservative_var;
    real entropy_var_vel_squared = 0.0;
    for(int idim=0; idim<dim; idim++){
        entropy_var_vel_squared += entropy_var[idim + 1] * entropy_var[idim + 1];
    }
    const real entropy = gam - entropy_var[0] + 0.5 * entropy_var_vel_squared / entropy_var[nstate-1];
    const real rho_theta = pow( (gamm1/ pow(- entropy_var[nstate-1], gam)), 1.0 /gamm1)
                         * exp( - entropy / gamm1);

    conservative_var[0] = - rho_theta * entropy_var[nstate-1];
    for(int idim=0; idim<dim; idim++){
        conservative_var[idim+1] = rho_theta * entropy_var[idim+1];
    }
    conservative_var[nstate-1] = rho_theta * (1.0 - 0.5 * entropy_var_vel_squared / entropy_var[nstate-1]);
    return conservative_var;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim, nstate, real>
::compute_kinetic_energy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,nstate> kin_energy_var;
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);
    const real vel2 = compute_velocity_squared<real>(vel);

    kin_energy_var[0] = - 0.5 * vel2;
    for(int idim=0; idim<dim; idim++){
        kin_energy_var[idim+1] = vel[idim];
    }
    kin_energy_var[nstate-1] = 0;

    return kin_energy_var;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>::
compute_mean_density(const std::array<real,nstate> &conservative_soln1,
                     const std::array<real,nstate> &conservative_soln2) const
{
    return (conservative_soln1[0] + conservative_soln2[0])/2.;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>::
compute_mean_pressure(const std::array<real,nstate> &conservative_soln1,
                      const std::array<real,nstate> &conservative_soln2) const
{
    real pressure_1 = compute_pressure<real>(conservative_soln1);
    real pressure_2 = compute_pressure<real>(conservative_soln2);
    return (pressure_1 + pressure_2)/2.;
}

template <int dim, int nstate, typename real>
inline dealii::Tensor<1,dim,real> Euler<dim,nstate,real>::
compute_mean_velocities(const std::array<real,nstate> &conservative_soln1,
                        const std::array<real,nstate> &conservative_soln2) const
{
    dealii::Tensor<1,dim,real> vel_1 = compute_velocities<real>(conservative_soln1);
    dealii::Tensor<1,dim,real> vel_2 = compute_velocities<real>(conservative_soln2);
    dealii::Tensor<1,dim,real> mean_vel;
    for (int d=0; d<dim; ++d) {
        mean_vel[d] = 0.5*(vel_1[d]+vel_2[d]);
    }
    return mean_vel;
}

template <int dim, int nstate, typename real>
inline real Euler<dim,nstate,real>::
compute_mean_specific_total_energy(const std::array<real,nstate> &conservative_soln1,
                             const std::array<real,nstate> &conservative_soln2) const
{
    return ((conservative_soln1[nstate-1]/conservative_soln1[0]) + (conservative_soln2[nstate-1]/conservative_soln2[0]))/2.;
}


template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const real density = conservative_soln[0];
    const real pressure = compute_pressure<real>(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);
    const real specific_total_energy = conservative_soln[nstate-1]/conservative_soln[0];
    const real specific_total_enthalpy = specific_total_energy + pressure/density;

    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Density equation
        conv_flux[0][flux_dim] = conservative_soln[1+flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_flux[1+velocity_dim][flux_dim] = density*vel[flux_dim]*vel[velocity_dim];
        }
        conv_flux[1+flux_dim][flux_dim] += pressure; // Add diagonal of pressure
        // Energy equation
        conv_flux[nstate-1][flux_dim] = density*vel[flux_dim]*specific_total_enthalpy;
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::convective_normal_flux (const std::array<real,nstate> &conservative_soln, const dealii::Tensor<1,dim,real> &normal) const
{
    std::array<real, nstate> conv_normal_flux;
    const real density = conservative_soln[0];
    const real pressure = compute_pressure<real>(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);
    real normal_vel = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel += vel[d]*normal[d];
    }
    const real total_energy = conservative_soln[nstate-1];
    const real specific_total_enthalpy = (total_energy + pressure) / density;

    const real rhoV = density*normal_vel;
    // Density equation
    conv_normal_flux[0] = rhoV;
    // Momentum equation
    for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
        conv_normal_flux[1+velocity_dim] = rhoV*vel[velocity_dim] + normal[velocity_dim] * pressure;
    }
    // Energy equation
    conv_normal_flux[nstate-1] = rhoV*specific_total_enthalpy;
    return conv_normal_flux;
}

template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> Euler<dim,nstate,real>
::convective_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // See Blazek (year?) Appendix A.9 p. 429-430
    // For Blazek (2001), see Appendix A.7 p. 419-420
    // Alternatively, see Masatsuka 2018 "I do like CFD", p.77, eq.(3.6.8)
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);
    real vel_normal = 0.0;
    for (int d=0;d<dim;d++) { vel_normal += vel[d] * normal[d]; }

    const real vel2 = compute_velocity_squared<real>(vel);
    const real phi = 0.5*gamm1 * vel2;

    const real density = conservative_soln[0];
    const real tot_energy = conservative_soln[nstate-1];
    const real E = tot_energy / density;
    const real a1 = gam*E-phi;
    const real a2 = gam-1.0;
    const real a3 = gam-2.0;

    dealii::Tensor<2,nstate,real> jacobian;
    for (int d=0; d<dim; ++d) {
        jacobian[0][1+d] = normal[d];
    }
    for (int row_dim=0; row_dim<dim; ++row_dim) {
        jacobian[1+row_dim][0] = normal[row_dim]*phi - vel[row_dim] * vel_normal;
        for (int col_dim=0; col_dim<dim; ++col_dim){
            if (row_dim == col_dim) {
                jacobian[1+row_dim][1+col_dim] = vel_normal - a3*normal[row_dim]*vel[row_dim];
            } else {
                jacobian[1+row_dim][1+col_dim] = normal[col_dim]*vel[row_dim] - a2*normal[row_dim]*vel[col_dim];
            }
        }
        jacobian[1+row_dim][nstate-1] = normal[row_dim]*a2;
    }
    jacobian[nstate-1][0] = vel_normal*(phi-a1);
    for (int d=0; d<dim; ++d){
        jacobian[nstate-1][1+d] = normal[d]*a1 - a2*vel[d]*vel_normal;
    }
    jacobian[nstate-1][nstate-1] = gam*vel_normal;

    return jacobian;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Euler<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);
    std::array<real,nstate> eig;
    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    for (int i=0; i<nstate; i++) {
        eig[i] = vel_dot_n;
        //eig[i] = advection_speed*normal;

        //eig[i] = 1.0;
        //eig[i] = -1.0;
    }
    return eig;
}

template <int dim, int nstate, typename real>
real Euler<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);

    const real sound = compute_sound (conservative_soln);

    real vel2 = compute_velocity_squared<real>(vel);

    const real max_eig = sqrt(vel2) + sound;

    return max_eig;
}

template <int dim, int nstate, typename real>
real Euler<dim,nstate,real>
::max_convective_normal_eigenvalue (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities<real>(conservative_soln);

    const real sound = compute_sound (conservative_soln);
    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    const real max_normal_eig = abs(vel_dot_n) + sound;

    return max_normal_eig;
}

template <int dim, int nstate, typename real>
real Euler<dim,nstate,real>
::max_viscous_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    return 0.0;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    return dissipative_flux(conservative_soln, solution_gradient);
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Euler<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    // No dissipation for Euler
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = 0;
    }
    return diss_flux;
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::boundary_riemann (
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   std::array<real,nstate> &soln_bc) const
{
    std::array<real,nstate> primitive_int = convert_conservative_to_primitive<real>(soln_int);
    std::array<real,nstate> primitive_ext;
    primitive_ext[0] = density_inf;
    for (int d=0;d<dim;d++) { primitive_ext[1+d] = velocities_inf[d]; }
    primitive_ext[nstate-1] = pressure_inf;

    const dealii::Tensor<1,dim,real> velocities_int = extract_velocities_from_primitive<real>(primitive_int);
    const dealii::Tensor<1,dim,real> velocities_ext = extract_velocities_from_primitive<real>(primitive_ext);

    const real sound_int  = compute_sound ( primitive_int[0], primitive_int[nstate-1] );
    const real sound_ext  = compute_sound ( primitive_ext[0], primitive_ext[nstate-1] );

    real vel_int_dot_normal = 0.0;
    real vel_ext_dot_normal = 0.0;
    for (int d=0; d<dim; d++) {
        vel_int_dot_normal = vel_int_dot_normal + velocities_int[d]*normal_int[d];
        vel_ext_dot_normal = vel_ext_dot_normal + velocities_ext[d]*normal_int[d];
    }

    // Riemann invariants
    const real out_riemann_invariant = vel_int_dot_normal + 2.0/gamm1*sound_int, // Outgoing
               inc_riemann_invariant = vel_ext_dot_normal - 2.0/gamm1*sound_ext; // Incoming

    const real normal_velocity_bc = 0.5*(out_riemann_invariant+inc_riemann_invariant),
               sound_bc  = 0.25*gamm1*(out_riemann_invariant-inc_riemann_invariant);

    std::array<real,nstate> primitive_bc;
    if (abs(normal_velocity_bc) >= abs(sound_bc)) { // Supersonic
        if (normal_velocity_bc < 0.0) { // Inlet
            primitive_bc = primitive_ext;
        } else { // Outlet
            primitive_bc = primitive_int;
        }
    } else { // Subsonic

        real density_bc;
        dealii::Tensor<1,dim,real> velocities_bc;
        real pressure_bc;

        dealii::Tensor<1,dim,real> velocities_tangential;
        if (normal_velocity_bc < 0.0) { // Inlet
            const real entropy_ext = compute_entropy_measure(primitive_ext[0], primitive_ext[nstate-1]);
            density_bc = pow( 1.0/gam * sound_bc * sound_bc / entropy_ext, 1.0/gamm1 );
            for (int d=0; d<dim; ++d) {
                velocities_tangential[d] = velocities_ext[d] - vel_ext_dot_normal * normal_int[d];
            }
        } else { // Outlet
            const real entropy_int = compute_entropy_measure(primitive_int[0], primitive_int[nstate-1]);
            density_bc = pow( 1.0/gam * sound_bc * sound_bc / entropy_int, 1.0/gamm1 );
            for (int d=0; d<dim; ++d) {
                velocities_tangential[d] = velocities_int[d] - vel_int_dot_normal * normal_int[d];
            }
        }
        for (int d=0; d<dim; ++d) {
            velocities_bc[d] = velocities_tangential[d] + normal_velocity_bc*normal_int[d];
        }

        pressure_bc = 1.0/gam * sound_bc * sound_bc * density_bc;

        primitive_bc[0] = density_bc;
        for (int d=0;d<dim;d++) { primitive_bc[1+d] = velocities_bc[d]; }
        primitive_bc[nstate-1] = pressure_bc;
    }

    soln_bc = convert_primitive_to_conservative(primitive_bc);
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::boundary_slip_wall (
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // Slip wall boundary conditions (No penetration)
    // Given by Algorithm II of the following paper
    // Krivodonova, L., and Berger, M.,
    // “High-order accurate implementation of solid wall boundary conditions in curved geometries,”
    // Journal of Computational Physics, vol. 211, 2006, pp. 492–512.
    const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive<real>(soln_int);

    // Copy density and pressure
    std::array<real,nstate> primitive_boundary_values;
    primitive_boundary_values[0] = primitive_interior_values[0];
    primitive_boundary_values[nstate-1] = primitive_interior_values[nstate-1];

    const dealii::Tensor<1,dim,real> surface_normal = -normal_int;
    const dealii::Tensor<1,dim,real> velocities_int = extract_velocities_from_primitive<real>(primitive_interior_values);
    //const dealii::Tensor<1,dim,real> velocities_bc = velocities_int - 2.0*(velocities_int*surface_normal)*surface_normal;
    real vel_int_dot_normal = 0.0;
    for (int d=0; d<dim; d++) {
        vel_int_dot_normal = vel_int_dot_normal + velocities_int[d]*surface_normal[d];
    }
    dealii::Tensor<1,dim,real> velocities_bc;
    for (int d=0; d<dim; d++) {
        velocities_bc[d] = velocities_int[d] - 2.0*(vel_int_dot_normal)*surface_normal[d];
        //velocities_bc[d] = velocities_int[d] - (vel_int_dot_normal)*surface_normal[d];
        //velocities_bc[d] += velocities_int[d] * surface_normal.norm_square();
    }
    for (int d=0; d<dim; ++d) {
        primitive_boundary_values[1+d] = velocities_bc[d];
    }

    const std::array<real,nstate> modified_conservative_boundary_values = convert_primitive_to_conservative(primitive_boundary_values);
    for (int istate=0; istate<nstate; ++istate) {
        soln_bc[istate] = modified_conservative_boundary_values[istate];
    }

    for (int istate=0; istate<nstate; ++istate) {
        soln_grad_bc[istate] = -soln_grad_int[istate];
    }
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::boundary_wall (
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // Slip wall boundary for Euler
    boundary_slip_wall(normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
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
    for (int s=0; s<nstate; s++) {
        conservative_boundary_values[s] = this->manufactured_solution_function->value (pos, s);
        boundary_gradients[s] = this->manufactured_solution_function->gradient (pos, s);
    }
    std::array<real,nstate> primitive_boundary_values = convert_conservative_to_primitive<real>(conservative_boundary_values);
    for (int istate=0; istate<nstate; ++istate) {

        std::array<real,nstate> characteristic_dot_n = convective_eigenvalues(conservative_boundary_values, normal_int);
        const bool inflow = (characteristic_dot_n[istate] <= 0.);

        if (inflow) { // Dirichlet boundary condition

            soln_bc[istate] = conservative_boundary_values[istate];
            soln_grad_bc[istate] = soln_grad_int[istate];

            // Only set the pressure and velocity
            // primitive_boundary_values[0] = soln_int[0];;
            // for(int d=0;d<dim;d++){
            //    primitive_boundary_values[1+d] = soln_int[1+d]/soln_int[0];;
            //}
            const std::array<real,nstate> modified_conservative_boundary_values = convert_primitive_to_conservative(primitive_boundary_values);
            (void) modified_conservative_boundary_values;
            //conservative_boundary_values[nstate-1] = soln_int[nstate-1];
            soln_bc[istate] = conservative_boundary_values[istate];

        } else { // Neumann boundary condition
            // soln_bc[istate] = -soln_int[istate]+2*conservative_boundary_values[istate];
            soln_bc[istate] = soln_int[istate];

            // **************************************************************************************************************
            // Note I don't know how to properly impose the soln_grad_bc to obtain an adjoint consistent scheme
            // Currently, Neumann boundary conditions are only imposed for the linear advection
            // Therefore, soln_grad_bc does not affect the solution
            // **************************************************************************************************************
            soln_grad_bc[istate] = soln_grad_int[istate];
            //soln_grad_bc[istate] = boundary_gradients[istate];
            //soln_grad_bc[istate] = -soln_grad_int[istate]+2*boundary_gradients[istate];
        }

        // HARDCODE DIRICHLET BC
        soln_bc[istate] = conservative_boundary_values[istate];
    }
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::boundary_pressure_outflow (
   const real total_inlet_pressure,
   const real back_pressure,
   const std::array<real,nstate> &soln_int,
   std::array<real,nstate> &soln_bc) const
{
    // Pressure Outflow Boundary Condition (back pressure)
    // Reference: Carlson 2011, sec. 2.4

    const real mach_int = compute_mach_number(soln_int);
    if (mach_int > 1.0) {
        // Supersonic, simply extrapolate
        for (int istate=0; istate<nstate; ++istate) {
            soln_bc[istate] = soln_int[istate];
        }
    } 
    else {
        const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive<real>(soln_int);
        const real pressure_int = primitive_interior_values[nstate-1];

        const real radicant = 1.0+0.5*gamm1*mach_inf_sqr;
        const real pressure_inlet = total_inlet_pressure * pow(radicant, -gam/gamm1);
        const real pressure_bc = (mach_int >= 1) * pressure_int + (1-(mach_int >= 1)) * back_pressure*pressure_inlet;
        const real temperature_int = compute_temperature<real>(primitive_interior_values);

        // Assign primitive boundary values
        std::array<real,nstate> primitive_boundary_values;
        primitive_boundary_values[0] = compute_density_from_pressure_temperature(pressure_bc, temperature_int);
        for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = primitive_interior_values[1+d]; }
        primitive_boundary_values[nstate-1] = pressure_bc;

        const std::array<real,nstate> conservative_bc = convert_primitive_to_conservative(primitive_boundary_values);
        for (int istate=0; istate<nstate; ++istate) {
            soln_bc[istate] = conservative_bc[istate];
        }
    }
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::boundary_inflow (
   const real total_inlet_pressure,
   const real total_inlet_temperature,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   std::array<real,nstate> &soln_bc) const
{
   // Inflow boundary conditions (both subsonic and supersonic)
   // Carlson 2011, sec. 2.2 & sec 2.9

   const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive<real>(soln_int);

   const dealii::Tensor<1,dim,real> normal = -normal_int;

   const real                       density_i    = primitive_interior_values[0];
   const dealii::Tensor<1,dim,real> velocities_i = extract_velocities_from_primitive<real>(primitive_interior_values);
   const real                       pressure_i   = primitive_interior_values[nstate-1];

   //const real                       normal_vel_i = velocities_i*normal;
   real                       normal_vel_i = 0.0;
   for (int d=0; d<dim; ++d) {
   normal_vel_i += velocities_i[d]*normal[d];
   }
   const real                       sound_i      = compute_sound(soln_int);
   //const real                       mach_i       = std::abs(normal_vel_i)/sound_i;

   //const dealii::Tensor<1,dim,real> velocities_o = velocities_inf;
   //const real                       normal_vel_o = velocities_o*normal;
   //const real                       sound_o      = sound_inf;
   //const real                       mach_o       = mach_inf;

   if(mach_inf < 1.0) {
      // Subsonic inflow, sec 2.7

      //this->pcout << "Subsonic inflow, mach=" << mach_i << std::endl;

      // Want to solve for c_b (sound_bc), to then solve for U (velocity_magnitude_bc) and M_b (mach_bc)
      // Eq. 37
      const real riemann_pos = normal_vel_i + 2.0*sound_i/gamm1;
      // Could evaluate enthalpy from primitive like eq.36, but easier to use the following
      const real specific_total_energy = soln_int[nstate-1]/density_i;
      const real specific_total_enthalpy = specific_total_energy + pressure_i/density_i;
      // Eq. 43
      const real a = 1.0+2.0/gamm1;
      const real b = -2.0*riemann_pos;
      const real c = 0.5*gamm1 * (riemann_pos*riemann_pos - 2.0*specific_total_enthalpy);
      // Eq. 42
      const real term1 = -0.5*b/a;
      const real term2= 0.5*sqrt(b*b-4.0*a*c)/a;
      const real sound_bc1 = term1 + term2;
      const real sound_bc2 = term1 - term2;
      // Eq. 44
      const real sound_bc  = std::max(sound_bc1, sound_bc2);
      // Eq. 45
      //const real velocity_magnitude_bc = 2.0*sound_bc/gamm1 - riemann_pos;
      const real velocity_magnitude_bc = riemann_pos - 2.0*sound_bc/gamm1;
      const real mach_bc = velocity_magnitude_bc/sound_bc;
      // Eq. 46
      const real radicant = 1.0+0.5*gamm1*mach_bc*mach_bc;
      const real pressure_bc = total_inlet_pressure * pow(radicant, -gam/gamm1);
      const real temperature_bc = total_inlet_temperature * pow(radicant, -1.0);
      //this->pcout << " pressure_bc " << pressure_bc << "pressure_inf" << pressure_inf << std::endl;
      //this->pcout << " temperature_bc " << temperature_bc << "temperature_inf" << temperature_inf << std::endl;

      const real density_bc  = compute_density_from_pressure_temperature(pressure_bc, temperature_bc);
      std::array<real,nstate> primitive_boundary_values;
      primitive_boundary_values[0] = density_bc;
      for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = velocity_magnitude_bc*normal[d]; }
      primitive_boundary_values[nstate-1] = pressure_bc;
      const std::array<real,nstate> conservative_bc = convert_primitive_to_conservative(primitive_boundary_values);
      for (int istate=0; istate<nstate; ++istate) {
          soln_bc[istate] = conservative_bc[istate];
      }

      //this->pcout << " entropy_bc " << compute_entropy_measure(soln_bc) << "entropy_inf" << entropy_inf << std::endl;

   } 
   else {
      // Supersonic inflow, sec 2.9

      // Specify all quantities through
      // total_inlet_pressure, total_inlet_temperature, mach_inf & angle_of_attack
      //this->pcout << "Supersonic inflow, mach=" << mach_i << std::endl;
      const real radicant = 1.0+0.5*gamm1*mach_inf_sqr;
      const real static_inlet_pressure    = total_inlet_pressure * pow(radicant, -gam/gamm1);
      const real static_inlet_temperature = total_inlet_temperature * pow(radicant, -1.0);

      const real pressure_bc = static_inlet_pressure;
      const real temperature_bc = static_inlet_temperature;
      const real density_bc  = compute_density_from_pressure_temperature(pressure_bc, temperature_bc);
      const real sound_bc = sqrt(gam * pressure_bc / density_bc);
      const real velocity_magnitude_bc = mach_inf * sound_bc;

      // Assign primitive boundary values
      std::array<real,nstate> primitive_boundary_values;
      primitive_boundary_values[0] = density_bc;
      for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = -velocity_magnitude_bc*normal_int[d]; } // minus since it's inflow
      primitive_boundary_values[nstate-1] = pressure_bc;
      const std::array<real,nstate> conservative_bc = convert_primitive_to_conservative(primitive_boundary_values);
      for (int istate=0; istate<nstate; ++istate) {
         soln_bc[istate] = conservative_bc[istate];
      }
   }
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::boundary_farfield (
   std::array<real,nstate> &soln_bc) const
{
   const real density_bc = density_inf;
   const real pressure_bc = 1.0/(gam*mach_inf_sqr);
   std::array<real,nstate> primitive_boundary_values;
   primitive_boundary_values[0] = density_bc;
   for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = velocities_inf[d]; } // minus since it's inflow
   primitive_boundary_values[nstate-1] = pressure_bc;
   const std::array<real,nstate> conservative_bc = convert_primitive_to_conservative(primitive_boundary_values);
   for (int istate=0; istate<nstate; ++istate) {
      soln_bc[istate] = conservative_bc[istate];
   }
}

template <int dim, int nstate, typename real>
void Euler<dim, nstate, real>
::boundary_do_nothing(
    const std::array<real, nstate>& soln_int,
    std::array<real, nstate>& soln_bc,
    std::array<dealii::Tensor<1, dim, real>, nstate>& soln_grad_bc) const
{
    for (int istate = 0; istate < nstate; ++istate) {
            soln_bc[istate] = soln_int[istate];
            soln_grad_bc[istate] = 0;
    }
    
}

template <int dim, int nstate, typename real>
void Euler<dim, nstate, real>
::boundary_custom(
    std::array<real, nstate>& soln_bc) const
{
    std::array<real, nstate> primitive_boundary_values;
    for (int istate = 0; istate < nstate; ++istate) {
            primitive_boundary_values[istate] = this->all_parameters->euler_param.custom_boundary_for_each_state[istate];
    }

    const std::array<real, nstate> conservative_bc = convert_primitive_to_conservative(primitive_boundary_values);
    for (int istate = 0; istate < nstate; ++istate) {
        soln_bc[istate] = conservative_bc[istate];
    }
}

template <int dim, int nstate, typename real>
void Euler<dim, nstate, real>
::boundary_astrophysical_inflow(
    std::array<real, nstate>& soln_bc) const
{
    std::array<real, nstate> primitive_boundary_values;
    for (int istate = 0; istate < nstate; ++istate) {
        if(istate == 0)
            primitive_boundary_values[istate] = 0.5;
        if(istate == 1)
            primitive_boundary_values[istate] = 0.0;
        if(istate == 2)
            primitive_boundary_values[istate] = 0.0;
        if(istate == 3)
            primitive_boundary_values[istate] = 0.4127;
        if(istate == 4)
            primitive_boundary_values[istate] = 0.0;
    }

    const std::array<real, nstate> conservative_bc = convert_primitive_to_conservative(primitive_boundary_values);
    for (int istate = 0; istate < nstate; ++istate) {
        soln_bc[istate] = conservative_bc[istate];
    }
}

template <int dim, int nstate, typename real>
void Euler<dim,nstate,real>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // NEED TO PROVIDE AS INPUT ************************************** (ask Doug where this should be moved to, protected member?)
    const real total_inlet_pressure = pressure_inf*pow(1.0+0.5*gamm1*mach_inf_sqr, gam/gamm1);
    const real total_inlet_temperature = temperature_inf*pow(total_inlet_pressure/pressure_inf, gamm1/gam);

    if (boundary_type == 1000) {
        // Manufactured solution boundary condition
        boundary_manufactured_solution (pos, normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } 
    else if (boundary_type == 1001) {
        // Wall boundary condition (slip for Euler, no-slip for Navier-Stokes; done through polymorphism)
        boundary_wall (normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } 
    else if (boundary_type == 1002) {
        // Pressure outflow boundary condition (back pressure)
        const real back_pressure = 0.99;
        boundary_pressure_outflow (total_inlet_pressure, back_pressure, soln_int, soln_bc);
    } 
    else if (boundary_type == 1003) {
        // Inflow boundary condition
        boundary_inflow (total_inlet_pressure, total_inlet_temperature, normal_int, soln_int, soln_bc);
    } 
    else if (boundary_type == 1004) {
        // Riemann-based farfield boundary condition
        boundary_riemann (normal_int, soln_int, soln_bc);
    } 
    else if (boundary_type == 1005) {
        // Simple farfield boundary condition
        boundary_farfield(soln_bc);
    } 
    else if (boundary_type == 1006) {
        // Slip wall boundary condition
        boundary_slip_wall (normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    }
    else if (boundary_type == 1007) {
        // Do nothing bc, p0 interpolation
        boundary_do_nothing (soln_int, soln_bc, soln_grad_bc);
    }
    else if (boundary_type == 1008) {
        // Custom boundary condition, user defined in parameters
        boundary_custom (soln_bc);
    }
    else if (boundary_type == 1009) {
        // Boundary specific to the the astrophysical jet case
        boundary_astrophysical_inflow (soln_bc);
    }
    else {
        this->pcout << "Invalid boundary_type: " << boundary_type << std::endl;
        std::abort();
    }
}

template <int dim, int nstate, typename real>
dealii::Vector<double> Euler<dim,nstate,real>::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &duh,
    const std::vector<dealii::Tensor<2,dim> > &dduh,
    const dealii::Tensor<1,dim>               &normals,
    const dealii::Point<dim>                  &evaluation_points) const
{
    std::vector<std::string> names = post_get_names ();
    dealii::Vector<double> computed_quantities = PhysicsBase<dim,nstate,real>::post_compute_derived_quantities_vector ( uh, duh, dduh, normals, evaluation_points);
    unsigned int current_data_index = computed_quantities.size() - 1;
    computed_quantities.grow_or_shrink(names.size());
    if constexpr (std::is_same<real,double>::value) {

        std::array<double, nstate> conservative_soln;
        for (unsigned int s=0; s<nstate; ++s) {
            conservative_soln[s] = uh(s);
        }
        const std::array<double, nstate> primitive_soln = convert_conservative_to_primitive<real>(conservative_soln);
        // if (primitive_soln[0] < 0) this->pcout << evaluation_points << std::endl;

        // Density
        computed_quantities(++current_data_index) = primitive_soln[0];
        // Velocities
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = primitive_soln[1+d];
        }
        // Momentum
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = conservative_soln[1+d];
        }
        // Energy
        computed_quantities(++current_data_index) = conservative_soln[nstate-1];
        // Pressure
        computed_quantities(++current_data_index) = primitive_soln[nstate-1];
        // Pressure coefficient
        computed_quantities(++current_data_index) = (primitive_soln[nstate-1] - pressure_inf) / dynamic_pressure_inf;
        // Temperature
        computed_quantities(++current_data_index) = compute_temperature<real>(primitive_soln);
        // Entropy generation
        computed_quantities(++current_data_index) = compute_entropy_measure(conservative_soln) - entropy_inf;
        // Mach Number
        computed_quantities(++current_data_index) = compute_mach_number(conservative_soln);

    }
    if (computed_quantities.size()-1 != current_data_index) {
        this->pcout << " Did not assign a value to all the data. Missing " << computed_quantities.size() - current_data_index << " variables."
                  << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
                  << std::endl;
    }

    return computed_quantities;
}

template <int dim, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> Euler<dim,nstate,real>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation = PhysicsBase<dim,nstate,real>::post_get_data_component_interpretation (); // state variables
    interpretation.push_back (DCI::component_is_scalar); // Density
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Velocity
    }
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Momentum
    }
    interpretation.push_back (DCI::component_is_scalar); // Energy
    interpretation.push_back (DCI::component_is_scalar); // Pressure
    interpretation.push_back (DCI::component_is_scalar); // Pressure coefficient
    interpretation.push_back (DCI::component_is_scalar); // Temperature
    interpretation.push_back (DCI::component_is_scalar); // Entropy generation
    interpretation.push_back (DCI::component_is_scalar); // Mach number

    std::vector<std::string> names = post_get_names();
    if (names.size() != interpretation.size()) {
        this->pcout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
    }
    return interpretation;
}


template <int dim, int nstate, typename real>
std::vector<std::string> Euler<dim,nstate,real>
::post_get_names () const
{
    std::vector<std::string> names = PhysicsBase<dim,nstate,real>::post_get_names ();
    names.push_back ("density");
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("velocity");
    }
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("momentum");
    }
    names.push_back ("energy");
    names.push_back ("pressure");
    names.push_back ("pressure_coeffcient");
    names.push_back ("temperature");

    names.push_back ("entropy_generation");
    names.push_back ("mach_number");
    return names;
}

template <int dim, int nstate, typename real>
dealii::UpdateFlags Euler<dim,nstate,real>
::post_get_needed_update_flags () const
{
    //return update_values | update_gradients;
    return dealii::update_values
           | dealii::update_quadrature_points
           ;
}

// Instantiate explicitly
template class Euler < PHILIP_DIM, PHILIP_DIM+2, double     >;
template class Euler < PHILIP_DIM, PHILIP_DIM+2, FadType    >;
template class Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >;
template class Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

//==============================================================================
// -> Templated member functions: // could be automated later on using Boost MPL
//------------------------------------------------------------------------------
// -- check_positive_quantity
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::check_positive_quantity< double     >(double     &qty, const std::string qty_name) const;
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, FadType    >::check_positive_quantity< FadType    >(FadType    &qty, const std::string qty_name) const;
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::check_positive_quantity< RadType    >(RadType    &qty, const std::string qty_name) const;
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::check_positive_quantity< FadFadType >(FadFadType &qty, const std::string qty_name) const;
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::check_positive_quantity< RadFadType >(RadFadType &qty, const std::string qty_name) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in NavierStokes::dissipative_flux_directional_jacobian()
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::check_positive_quantity< FadType    >(FadType    &qty, const std::string qty_name) const;
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::check_positive_quantity< FadType    >(FadType    &qty, const std::string qty_name) const;
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::check_positive_quantity< FadType    >(FadType    &qty, const std::string qty_name) const;
template bool Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::check_positive_quantity< FadType    >(FadType    &qty, const std::string qty_name) const;
// -- compute_pressure()
template double     Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::compute_pressure< double     >(const std::array<double,    PHILIP_DIM+2> &conservative_soln) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, FadType    >::compute_pressure< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template RadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::compute_pressure< RadType    >(const std::array<RadType,   PHILIP_DIM+2> &conservative_soln) const;
template FadFadType Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::compute_pressure< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &conservative_soln) const;
template RadFadType Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::compute_pressure< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &conservative_soln) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in NavierStokes::dissipative_flux_directional_jacobian()
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::compute_pressure< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::compute_pressure< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::compute_pressure< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::compute_pressure< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
// -- compute_temperature()
template double     Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::compute_temperature< double     >(const std::array<double,    PHILIP_DIM+2> &primitive_soln) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, FadType    >::compute_temperature< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
template RadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::compute_temperature< RadType    >(const std::array<RadType,   PHILIP_DIM+2> &primitive_soln) const;
template FadFadType Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::compute_temperature< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &primitive_soln) const;
template RadFadType Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::compute_temperature< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &primitive_soln) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in NavierStokes::dissipative_flux_directional_jacobian()
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::compute_temperature< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::compute_temperature< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::compute_temperature< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::compute_temperature< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
// -- compute_velocity_squared()
template double     Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::compute_velocity_squared< double     >(const dealii::Tensor<1,PHILIP_DIM,double    > &velocities) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, FadType    >::compute_velocity_squared< FadType    >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &velocities) const;
template RadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::compute_velocity_squared< RadType    >(const dealii::Tensor<1,PHILIP_DIM,RadType   > &velocities) const;
template FadFadType Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::compute_velocity_squared< FadFadType >(const dealii::Tensor<1,PHILIP_DIM,FadFadType> &velocities) const;
template RadFadType Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::compute_velocity_squared< RadFadType >(const dealii::Tensor<1,PHILIP_DIM,RadFadType> &velocities) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in NavierStokes::dissipative_flux_directional_jacobian()
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::compute_velocity_squared< FadType    >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &velocities) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::compute_velocity_squared< FadType    >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &velocities) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::compute_velocity_squared< FadType    >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &velocities) const;
template FadType    Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::compute_velocity_squared< FadType    >(const dealii::Tensor<1,PHILIP_DIM,FadType   > &velocities) const;
// -- convert_conservative_to_primitive()
template std::array<double,    PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::convert_conservative_to_primitive< double     >(const std::array<double,    PHILIP_DIM+2> &conservative_soln) const;
template std::array<FadType,   PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, FadType    >::convert_conservative_to_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template std::array<RadType,   PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::convert_conservative_to_primitive< RadType    >(const std::array<RadType,   PHILIP_DIM+2> &conservative_soln) const;
template std::array<FadFadType,PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::convert_conservative_to_primitive< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &conservative_soln) const;
template std::array<RadFadType,PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::convert_conservative_to_primitive< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &conservative_soln) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in NavierStokes::dissipative_flux_directional_jacobian()
template std::array<FadType,   PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::convert_conservative_to_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template std::array<FadType,   PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::convert_conservative_to_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template std::array<FadType,   PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::convert_conservative_to_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template std::array<FadType,   PHILIP_DIM+2> Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::convert_conservative_to_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
// -- extract_velocities_from_primitive()
template dealii::Tensor<1,PHILIP_DIM,double    > Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::extract_velocities_from_primitive< double     >(const std::array<double,    PHILIP_DIM+2> &primitive_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, FadType    >::extract_velocities_from_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
template dealii::Tensor<1,PHILIP_DIM,RadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::extract_velocities_from_primitive< RadType    >(const std::array<RadType,   PHILIP_DIM+2> &primitive_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadFadType> Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::extract_velocities_from_primitive< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &primitive_soln) const;
template dealii::Tensor<1,PHILIP_DIM,RadFadType> Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::extract_velocities_from_primitive< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &primitive_soln) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in NavierStokes::dissipative_flux_directional_jacobian()
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::extract_velocities_from_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::extract_velocities_from_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::extract_velocities_from_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::extract_velocities_from_primitive< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &primitive_soln) const;
// -- compute_velocities()
template dealii::Tensor<1,PHILIP_DIM,double    > Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::compute_velocities< double     >(const std::array<double,    PHILIP_DIM+2> &conservative_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, FadType    >::compute_velocities< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template dealii::Tensor<1,PHILIP_DIM,RadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::compute_velocities< RadType    >(const std::array<RadType,   PHILIP_DIM+2> &conservative_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadFadType> Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::compute_velocities< FadFadType >(const std::array<FadFadType,PHILIP_DIM+2> &conservative_soln) const;
template dealii::Tensor<1,PHILIP_DIM,RadFadType> Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::compute_velocities< RadFadType >(const std::array<RadFadType,PHILIP_DIM+2> &conservative_soln) const;
// -- -- instantiate all the real types with real2 = FadType for automatic differentiation in NavierStokes::dissipative_flux_directional_jacobian()
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, double     >::compute_velocities< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, RadType    >::compute_velocities< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, FadFadType >::compute_velocities< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
template dealii::Tensor<1,PHILIP_DIM,FadType   > Euler < PHILIP_DIM, PHILIP_DIM+2, RadFadType >::compute_velocities< FadType    >(const std::array<FadType,   PHILIP_DIM+2> &conservative_soln) const;
//==============================================================================

} // Physics namespace
} // PHiLiP namespace

