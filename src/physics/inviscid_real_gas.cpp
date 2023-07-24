#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "inviscid_real_gas.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
InviscidRealGas<dim,nstate,real>::InviscidRealGas ( 
    const Parameters::AllParameters *const                    parameters_input,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source)
    : PhysicsBase<dim,nstate,real>(parameters_input, has_nonzero_diffusion,has_nonzero_physical_source,manufactured_solution_function)
    , mach_inf(parameters_input->euler_param.mach_inf)
    , mach_inf_sqr(mach_inf*mach_inf)
    , two_point_num_flux_type(parameters_input->two_point_num_flux_type)
{
    static_assert(nstate==dim+2, "Physics::InviscidRealGas() should be created with nstate=dim+2"); // TO DO: UPDATE THIS with nspecies
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> InviscidRealGas<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &/*conservative_soln*/) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;

    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Density equation
        conv_flux[0][flux_dim] = 0.0;
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_flux[1+velocity_dim][flux_dim] = 0.0;
        }
        conv_flux[1+flux_dim][flux_dim] += 0.0;
        // Energy equation
        conv_flux[nstate-1][flux_dim] = 0.0;
        // TO DO: now loop over nspecies
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> InviscidRealGas<dim, nstate, real>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::cout<<"Entropy variables for InviscidRealGas hasn't been done yet."<<std::endl;
    std::abort();
    return conservative_soln;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> InviscidRealGas<dim, nstate, real>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    std::cout<<"Entropy variables for InviscidRealGas hasn't been done yet."<<std::endl;
    std::abort();
    return entropy_var;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> InviscidRealGas<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    // TO DO: define this
    std::array<real,nstate> eig;
    eig.fill(0.0);
    return eig;
}

template <int dim, int nstate, typename real>
real InviscidRealGas<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    // TO DO: define this
    const real max_eig = 0.0;
    return max_eig;
}

template <int dim, int nstate, typename real>
real InviscidRealGas<dim,nstate,real>
::max_viscous_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    // zero because inviscid
    const real max_eig = 0.0;
    return max_eig;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> InviscidRealGas<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
     std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    // No dissipative flux (i.e. viscous terms) for InviscidRealGas
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = 0;
    }
    return diss_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> InviscidRealGas<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &/*pos*/,
    const std::array<real,nstate> &/*conservative_soln*/,
    const real /*current_time*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    // nothing to add here
    std::array<real,nstate> source_term;
    source_term.fill(0.0);
    return source_term;
}

// TO DO: Provide required definition for this
// template <int dim, int nstate, typename real>
// template<typename real2>
// bool InviscidRealGas<dim,nstate,real>::check_positive_quantity(real2 &qty, const std::string qty_name) const {
//     bool qty_is_positive;
//     if (qty < 0.0) {
//         // Refer to base class for non-physical results handling
//         qty = this->template handle_non_physical_result<real2>(qty_name + " is negative.");
//         qty_is_positive = false;
//     } else {
//         qty_is_positive = true;
//     }

//     return qty_is_positive;
// }


template <int dim, int nstate, typename real>
void InviscidRealGas<dim,nstate,real>
::boundary_face_values (
   const int /*boundary_type*/,
   const dealii::Point<dim, real> &/*pos*/,
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
   std::array<real,nstate> &/*soln_bc*/,
   std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // TO DO: Update this you are using any kind of BC that is not periodic
}

template <int dim, int nstate, typename real>
inline real InviscidRealGas<dim,nstate,real>
:: compute_density ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];
    return density;
}

template <int dim, int nstate, typename real>
template<typename real2>
inline dealii::Tensor<1,dim,real2> InviscidRealGas<dim,nstate,real>
::compute_velocities ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 density = compute_density(conservative_soln);
    dealii::Tensor<1,dim,real2> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/density; }
    return vel;
}


// Instantiate explicitly
template class InviscidRealGas < PHILIP_DIM, PHILIP_DIM+2, double     >;
template class InviscidRealGas < PHILIP_DIM, PHILIP_DIM+2, FadType    >;
template class InviscidRealGas < PHILIP_DIM, PHILIP_DIM+2, RadType    >;
template class InviscidRealGas < PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class InviscidRealGas < PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace

