#ifndef __MULTI_SPECIES_CALORICALLY_PERFECT__
#define __MULTI_SPECIES_CALORICALLY_PERFECT__

#include <deal.II/base/tensor.h>
#include "physics.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"
#include "real_gas_file_reader_and_variables/all_real_gas_constants.h"

namespace PHiLiP {
namespace Physics {

/// MultiSpeciesCaloricallyPerfect equations. Derived from PhysicsBase
template <int dim, int nstate, typename real> // TO DO: TEMPLATE for nspecies -- see how the LES class has nstate_baseline_physics
class MultiSpeciesCaloricallyPerfect : public PhysicsBase <dim, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following: */
    using PhysicsBase<dim,nstate,real>::dissipative_flux;
    using PhysicsBase<dim,nstate,real>::source_term;
public:
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    MultiSpeciesCaloricallyPerfect ( 
        const Parameters::AllParameters *const                    parameters_input,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const bool                                                has_nonzero_diffusion = false,
        const bool                                                has_nonzero_physical_source = false);

    /// Destructor
    ~MultiSpeciesCaloricallyPerfect() {};
public:
    /// Pointer to all real gas constants object for accessing the coefficients and properties (CAP)
    std::shared_ptr< PHiLiP::RealGasConstants::AllRealGasConstants > real_gas_cap;
protected:
    /// f_M18: Compute convective flux from conservative_soln
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux ( 

};

} // Physics namespace
} // PHiLiP namespace

#endif
