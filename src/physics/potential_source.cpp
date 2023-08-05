#include <cmath>
#include <vector>
#include <tuple>
#include <complex> // for the jacobian

#include "ADTypes.hpp"

#include "model.h"
#include "potential_source.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Potential Flow Addition to the Navier Stokes (RANS) model
//================================================================
template <int dim, int nstate, typename real>
PotentialFlowBase<dim, nstate, real>::PotentialFlowBase(
    const Parameters::AllParameters *const                    parameters_input,
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              /*mach_inf*/,
    const double                                              angle_of_attack,
    const double                                              /*side_slip_angle*/,
    const double                                              /*prandtl_number*/,
    const double                                              reynolds_number_inf,
    const bool                                                /*use_constant_viscosity*/,
    const double                                              constant_viscosity,
    const double                                              /*temperature_inf*/,
    const double                                              /*isothermal_wall_temperature*/,
    const thermal_boundary_condition_enum                     /*thermal_boundary_condition_type*/,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             /*two_point_num_flux_type*/)
    : ModelBase<dim,nstate,real>(manufactured_solution_function)
    , potential_source_param(parameters_input->potential_source_param)
    , potential_source_geometry(potential_source_param.potential_source_geometry)
    , ref_length(ref_length)
    , gamma_gas(gamma_gas)
    , density_inf(1.0) // Nondimensional - Free stream values
    , angle_of_attack(angle_of_attack)
    , reynolds_number_inf(reynolds_number_inf)
    , const_viscosity(constant_viscosity) // Nondimensional - Free stream values
    , lift_vector(initialize_lift_vector())
    , drag_vector(initialize_drag_vector())
{
    static_assert(nstate>=dim+2, "ModelBase::PotentialFlowBase() should be created with nstate>=dim+2");
    // static_assert(dim>=2, "ModelBase::PotentialFlowBase() should be created with dim>=2");
    if constexpr(dim==1) {
        std::cout << "ModelBase::PotentialFlowBase() should be created with dim>=2.";
    }
    // Initialize zero arrays / tensors
    for (int s=0; s<nstate; ++s) 
    {
        zero_array[s] = 0.0;
        for (int d=0;d<dim;++d) 
        {
            zero_tensor_array[s][d] = 0.0;
        }
    }
}

template <int dim, int nstate, typename real>
template<typename real2>
inline std::tuple<real2, real2> PotentialFlowBase<dim,nstate,real>
::TES_geometry () const
{
    // geometric parameters
    const real2 TES_h = this->potential_source_param.TES_h;
    const real2 TES_frequency = this->potential_source_param.TES_frequency;
    const real2 TES_thickness = this->potential_source_param.TES_thickness;

    // area
    const real2 TES_area = (TES_h * TES_frequency);    // TES_h = 1/2 triangle height

    // volume
    const real2 TES_volume = TES_area * TES_thickness;

    // geometry tuple
    const std::tuple<real2, real2> TES_parameters = std::make_tuple(TES_area, TES_volume);

    return TES_parameters;
}

template <int dim, int nstate, typename real>
inline double PotentialFlowBase<dim,nstate,real>
::freestream_speed () const
{
    const double U = this->const_viscosity * this->reynolds_number_inf / (this->density_inf * this->ref_length);
    return U;
}

// Lift and Drag Vectors
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,double> PotentialFlowBase<dim,nstate,real>
::initialize_lift_vector () const
{
    dealii::Tensor<1,dim,double> lift_direction;
    lift_direction[0] = 0.0;
    lift_direction[1] = 1.0;

    if constexpr (dim == 1) {
        assert(false);
    }
    if constexpr (dim == 3) {
        lift_direction[2] = 0.0;
    }

    /// The chord is aligned with the x-axis, lift is [0, 1, 0]^T
    return lift_direction;
}

template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,double> PotentialFlowBase<dim,nstate,real>
::initialize_drag_vector () const
{
    dealii::Tensor<1,dim,double> drag_direction;
    drag_direction[0] = 1.0;
    drag_direction[1] = 0.0;

    if constexpr (dim == 1) {
        assert(false);
    }
    if constexpr (dim == 3) {
        drag_direction[2] = 0.0;
    }

    /// The chord is aligned with the x-axis, drag is [1, 0, 0]^T
    return drag_direction;
}


//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,double> PotentialFlowBase<dim,nstate,real>
::compute_body_force (
        const dealii::Point<dim,real> &/*pos*/,
        const std::array<real,nstate> &/*conservative_soln*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
        const dealii::types::global_dof_index cell_index) const
{   
    dealii::Tensor<1,dim,double> body_force;
    body_force[0] = 0.0;
    body_force[1] = 0.0;
    if constexpr(dim==3) {
        body_force[3] = 0.0; }

    // LES body force:
    if (potential_source_geometry == PS_geometry_enum::trailing_edge_serrations)
    {
        const double pi = atan(1.0) * 4.0;

        // TES trailing flap angle
        const double TES_flap_angle = this->potential_source_param.TES_flap_angle;

        // geometric parameters
        const double TES_h = this->potential_source_param.TES_h;
        const double TES_effective_length_factor = this->potential_source_param.TES_effective_length_factor;

        // TES area and volume
        const auto [TES_area, TES_volume] = this->TES_geometry<double>();

        const double cell_volume = this->cellwise_volume[cell_index];

        // effective length
        const double TES_effective_length = TES_effective_length_factor * (2 * TES_h);

        // flow parameters
        const double freestream_U = this->freestream_speed();

        // lift and drag coefficients -> Flat Plate assumption: See [Cao et al, 2021]
        const double pressure_force_coeff = ((2 * pi * (this->angle_of_attack + TES_flap_angle) * TES_effective_length) / this->ref_length);
        double friction_coeff = 0.0;

        if (potential_source_param.use_viscous_drag) {
            friction_coeff = 0.072 * pow(this->reynolds_number_inf / this->density_inf, -1 / 5); }

        const double lift_coeff = pressure_force_coeff * cos(TES_flap_angle);
        const double drag_coeff = 2 * friction_coeff + pressure_force_coeff * sin(TES_flap_angle);

        // force computation
        body_force = (0.5 * TES_area * (cell_volume / TES_volume) * (this->density_inf * freestream_U * freestream_U)
                     * (lift_coeff * this->lift_vector + drag_coeff * this->drag_vector));

    }
    else if (potential_source_geometry == PS_geometry_enum::circular_test)
    {
        body_force[0] = 1;
        body_force[1] = 0.0;

        if constexpr(dim!=2)
        {
            std::cout << "Circular test currently only applicable for dim == 2." << std::endl;
            body_force[2] = 0.0;  
        }
    }

    return body_force;
}

//----------------------------------------------------------------
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> PotentialFlowBase<dim,nstate,real>
::physical_source_term (
        const dealii::Point<dim,real> &/*pos*/,
        const std::array<real,nstate> &/*conservative_soln*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
        const dealii::types::global_dof_index /*cell_index*/) const
{
    return zero_array;

    // std::array<real,nstate> physical_source;
    // std::fill(physical_source.begin(), physical_source.end(), 0.0);

    // if (this->cellwise_geometry_condition[cell_index])
    // {
    //     // // density
    //     // physical_source[0] = 0;
    //     std::cout << "About to compute_body_force" << std::endl;

    //     // // momentum
    //     dealii::Tensor<1,dim,double> body_force = this->compute_body_force(pos, conservative_soln, solution_gradient, cell_index);
    //     for (unsigned int i=0;i<dim;++i)
    //     {
    //         physical_source[i+1] = body_force[i];
    //     }

    //     // // energy
    //     // physical_source[nstate - 1] = 0;
    // }
    // return physical_source;
}

//// Overwriting virtual methods ////
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PotentialFlowBase<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &/*conservative_soln*/) const
{
    // No additional convective terms, simply using baseline physics
    return this->zero_tensor_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PotentialFlowBase<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{   
    // No additional dissipative terms, simply using baseline physics
    return this->zero_tensor_array;
}

//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> PotentialFlowBase<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    return this->zero_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real PotentialFlowBase<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    const real max_eig = 0.0;
    return max_eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real PotentialFlowBase<dim,nstate,real>
::max_convective_normal_eigenvalue (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    const real max_eig = 0.0;
    return max_eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> PotentialFlowBase<dim,nstate,real>
::source_term (
        const dealii::Point<dim,real> &/*pos*/,
        const std::array<real,nstate> &/*solution*/,
        const real /*current_time*/,
        const dealii::types::global_dof_index /*cell_index*/) const
{
    // No additional source terms, simply using baseline physics
    return this->zero_array;
}

//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- PotentialFlowBase
template class PotentialFlowBase         < PHILIP_DIM, PHILIP_DIM+2, double >;
template class PotentialFlowBase         < PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class PotentialFlowBase         < PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class PotentialFlowBase         < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class PotentialFlowBase         < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace