#ifndef __MULTI_SPECIES_CALORICALLY_PERFECT__
#define __MULTI_SPECIES_CALORICALLY_PERFECT__

#include <deal.II/base/tensor.h>
#include "real_gas.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// MultiSpeciesCaloricallyPerfect equations. Derived from PhysicsBase
template <int dim, int nstate, typename real> // TO DO: TEMPLATE for nspecies -- see how the LES class has nstate_baseline_physics
class MultiSpeciesCaloricallyPerfect : public RealGas <dim, nstate, real>
{
public:
    // using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    MultiSpeciesCaloricallyPerfect ( 
        const Parameters::AllParameters *const                    parameters_input,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const bool                                                has_nonzero_diffusion = false,
        const bool                                                has_nonzero_physical_source = false);

private:
    const std::array<real,nstate-dim-1> Cp;
    const std::array<real,nstate-dim-1> Cv;

public:
    /// Destructor
    ~MultiSpeciesCaloricallyPerfect() {};

    /// Pointer to all real gas constants object for accessing the coefficients and properties (CAP)
    std::shared_ptr< PHiLiP::RealGasConstants::AllRealGasConstants > real_gas_cap;
// protected:
//     /// f_M18: Compute convective flux from conservative_soln
//     std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux ( 
//         const std::array<real,nstate> &conservative_soln) const override;

    /// f_S19: Convert primitive solutions to conservative solutions // TO DO: delete new and delete the original function
    virtual std::array<real,nstate> convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const override; 

// /// Mian functions
protected:
    /// f_M14: Compute temperature from conservative_soln
    real compute_temperature ( const std::array<real,nstate> &conservative_soln ) const override;

    /// f_M16: Compute mixture pressure from conservative_soln
    real compute_mixture_pressure ( const std::array<real,nstate> &conservative_soln ) const override;    

/// Suporting functions
protected:
    /// f_S20: Compute species specific heat ratio from conservative_soln
    std::array<real,nstate-dim-1> compute_species_specific_heat_ratio ( const std::array<real,nstate> &conservative_soln ) const override;

private:
    std::array<real,nstate-dim-1> gamma;
};

} // Physics namespace
} // PHiLiP namespace

#endif
