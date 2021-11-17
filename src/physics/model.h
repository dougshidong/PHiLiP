#ifndef __MODEL__
#define __MODEL__

#include <deal.II/base/tensor.h>

#include "parameters/parameters_manufactured_solution.h"
#include "physics/manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// Physics model additional terms and equations to the baseline physics. 
template <int dim, int nstate, typename real>
class ModelBase
{
public:
	/// Constructor
	ModelBase(
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input = nullptr);

    /// Virtual destructor required for abstract classes.
    virtual ~ModelBase() = 0;

    /// Manufactured solution function
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function;

    /// Convective flux terms additional to the baseline physics
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
    convective_flux (
        const std::array<real,nstate> &conservative_soln) const = 0;

    /// Dissipative flux terms additional to the baseline physics
	virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
	dissipative_flux (
    	const std::array<real,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const = 0;

    /// Source terms additional to the baseline physics
    virtual std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution) const = 0;
};

} // Physics namespace
} // PHiLiP namespace

#endif
