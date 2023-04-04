#include <assert.h>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include "ADTypes.hpp"

#include "physics.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
PhysicsBase<dim,nstate,real>::PhysicsBase(
    const Parameters::AllParameters *const                    parameters_input,
    const bool                                                has_nonzero_diffusion_input,
    const bool                                                has_nonzero_physical_source_input,
    const dealii::Tensor<2,3,double>                          input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input)
    : has_nonzero_diffusion(has_nonzero_diffusion_input)
    , has_nonzero_physical_source(has_nonzero_physical_source_input)
    , manufactured_solution_function(manufactured_solution_function_input)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    , all_parameters(parameters_input)
    , non_physical_behavior_type(all_parameters->non_physical_behavior_type)
{
    // if provided with a null ptr, give it the default manufactured solution
    // currently only necessary for the unit test
    if(!manufactured_solution_function)
        manufactured_solution_function = std::make_shared<ManufacturedSolutionSine<dim,real>>(nstate);

    // anisotropic diffusion matrix
    diffusion_tensor[0][0] = input_diffusion_tensor[0][0];
    if constexpr(dim >= 2) {
        diffusion_tensor[0][1] = input_diffusion_tensor[0][1];
        diffusion_tensor[1][0] = input_diffusion_tensor[1][0];
        diffusion_tensor[1][1] = input_diffusion_tensor[1][1];
    }
    if constexpr(dim >= 3) {
        diffusion_tensor[0][2] = input_diffusion_tensor[0][2];
        diffusion_tensor[2][0] = input_diffusion_tensor[2][0];
        diffusion_tensor[1][2] = input_diffusion_tensor[1][2];
        diffusion_tensor[2][1] = input_diffusion_tensor[2][1];
        diffusion_tensor[2][2] = input_diffusion_tensor[2][2];
    }
}

template <int dim, int nstate, typename real>
PhysicsBase<dim,nstate,real>::PhysicsBase(
    const Parameters::AllParameters *const                    parameters_input,
    const bool                                                has_nonzero_diffusion_input,
    const bool                                                has_nonzero_physical_source_input,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input)
    : PhysicsBase<dim,nstate,real>(
        parameters_input,
        has_nonzero_diffusion_input,
        has_nonzero_physical_source_input,
        Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
        manufactured_solution_function_input)
{ }

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsBase<dim,nstate,real>::convective_numerical_split_flux (
    const std::array<real,nstate> &/*conservative_soln1*/,
    const std::array<real,nstate> &/*conservative_soln2*/) const
{
    pcout << "ERROR: convective_numerical_split_flux() has not yet been implemented for (overridden by) the selected PDE. Aborting..." <<std::flush;
    std::abort();
    std::array<dealii::Tensor<1,dim,real>,nstate> dummy;
    return dummy;
}

/*
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsBase<dim,nstate,real>
::artificial_dissipative_flux (
    const real viscosity_coefficient,
    const std::array<real,nstate> &,//solution,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    for (int i=0; i<nstate; i++) {
        for (int d=0; d<dim; d++) {
            diss_flux[i][d] = -viscosity_coefficient*(solution_gradient[i][d]);
        }
    }
    return diss_flux;
}
*/
template <int dim, int nstate, typename real>
std::array<real,nstate> PhysicsBase<dim,nstate,real>
::artificial_source_term (
    const real viscosity_coefficient,
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;
    
    dealii::Tensor<2,dim,double> artificial_diffusion_tensor;
    for (int i=0;i<dim;i++)
        for (int j=0;j<dim;j++)
            artificial_diffusion_tensor[i][j] = (i==j) ? 1.0 : 0.0;

    for (int istate=0; istate<nstate; istate++) {
        dealii::SymmetricTensor<2,dim,real> manufactured_hessian = this->manufactured_solution_function->hessian (pos, istate);
        //source[istate] = -viscosity_coefficient*scalar_product(artificial_diffusion_tensor,manufactured_hessian);
        source[istate] = 0.0;
        for (int dr=0; dr<dim; ++dr) {
            for (int dc=0; dc<dim; ++dc) {
                source[istate] += artificial_diffusion_tensor[dr][dc] * manufactured_hessian[dr][dc];
            }
        }
        source[istate] *= -viscosity_coefficient;
    }
    return source;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> PhysicsBase<dim,nstate,real>
::physical_source_term (
    const dealii::Point<dim,real> &/*pos*/,
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    std::array<real,nstate> physical_source;
    for (int i=0; i<nstate; i++) {
        physical_source[i] = 0;
    }
    return physical_source;
}

template <int dim, int nstate, typename real>
dealii::Vector<double> PhysicsBase<dim,nstate,real>::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &/*duh*/,
    const std::vector<dealii::Tensor<2,dim> > &/*dduh*/,
    const dealii::Tensor<1,dim>               &/*normals*/,
    const dealii::Point<dim>                  &/*evaluation_points*/) const
{
    dealii::Vector<double> computed_quantities(nstate);
    for (unsigned int s=0; s<nstate; ++s) {
        computed_quantities(s) = uh(s);
    }
    return computed_quantities;
}

template <int dim, int nstate, typename real>
dealii::Vector<double> PhysicsBase<dim,nstate,real>::post_compute_derived_quantities_scalar (
    const double              &uh,
    const dealii::Tensor<1,dim> &/*duh*/,
    const dealii::Tensor<2,dim> &/*dduh*/,
    const dealii::Tensor<1,dim> &/*normals*/,
    const dealii::Point<dim>    &/*evaluation_points*/) const
{
    assert(nstate == 1);
    dealii::Vector<double> computed_quantities(nstate);
    for (unsigned int s=0; s<nstate; ++s) {
        computed_quantities(s) = uh;
    }
    return computed_quantities;
}

template <int dim, int nstate, typename real>
std::vector<std::string> PhysicsBase<dim,nstate,real> ::post_get_names () const
{
    std::vector<std::string> names;
    for (unsigned int s=0; s<nstate; ++s) {
        std::string varname = "state" + dealii::Utilities::int_to_string(s,1);
        names.push_back(varname);
    }
    return names;
}

template <int dim, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> PhysicsBase<dim,nstate,real>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation;
    for (unsigned int s=0; s<nstate; ++s) {
        interpretation.push_back (DCI::component_is_scalar);
    }
    return interpretation;
}

template <int dim, int nstate, typename real>
dealii::UpdateFlags PhysicsBase<dim,nstate,real>
::post_get_needed_update_flags () const
{
    return dealii::update_values;
}

template <int dim, int nstate, typename real>
template<typename real2>
real2 PhysicsBase<dim,nstate,real>
::handle_non_physical_result() const
{
    if (this->non_physical_behavior_type == NonPhysicalBehaviorEnum::abort_run) {
        this->pcout << "ERROR: Non-physical result has been detected. Aborting... " << std::endl << std::flush;
        std::abort();
    } else if (this->non_physical_behavior_type == NonPhysicalBehaviorEnum::print_warning) {
        this->pcout << "WARNING: Non-physical result has been detected at a node." << std::endl;
    } else if (this->non_physical_behavior_type == NonPhysicalBehaviorEnum::do_nothing) {
        // do nothing -- assume that the test or iterative solver can handle this.
    }
        
    const real2 BIG_NUMBER = 1e100;
    return BIG_NUMBER;
}

template class PhysicsBase < PHILIP_DIM, 1, double >;
template class PhysicsBase < PHILIP_DIM, 2, double >;
template class PhysicsBase < PHILIP_DIM, 3, double >;
template class PhysicsBase < PHILIP_DIM, 4, double >;
template class PhysicsBase < PHILIP_DIM, 5, double >;
template class PhysicsBase < PHILIP_DIM, 6, double >;
template class PhysicsBase < PHILIP_DIM, 8, double >;

template class PhysicsBase < PHILIP_DIM, 1, FadType >;
template class PhysicsBase < PHILIP_DIM, 2, FadType >;
template class PhysicsBase < PHILIP_DIM, 3, FadType >;
template class PhysicsBase < PHILIP_DIM, 4, FadType >;
template class PhysicsBase < PHILIP_DIM, 5, FadType >;
template class PhysicsBase < PHILIP_DIM, 6, FadType >;
template class PhysicsBase < PHILIP_DIM, 8, FadType >;

template class PhysicsBase < PHILIP_DIM, 1, RadType >;
template class PhysicsBase < PHILIP_DIM, 2, RadType >;
template class PhysicsBase < PHILIP_DIM, 3, RadType >;
template class PhysicsBase < PHILIP_DIM, 4, RadType >;
template class PhysicsBase < PHILIP_DIM, 5, RadType >;
template class PhysicsBase < PHILIP_DIM, 6, RadType >;
template class PhysicsBase < PHILIP_DIM, 8, RadType >;

template class PhysicsBase < PHILIP_DIM, 1, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 2, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 3, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 4, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 5, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 6, FadFadType >;
template class PhysicsBase < PHILIP_DIM, 8, FadFadType >;

template class PhysicsBase < PHILIP_DIM, 1, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 2, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 3, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 4, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 5, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 6, RadFadType >;
template class PhysicsBase < PHILIP_DIM, 8, RadFadType >;

//==============================================================================
// -> Templated member functions: // could be automated later on using Boost MPL
//------------------------------------------------------------------------------
// -- handle_non_physical_result
template double PhysicsBase < PHILIP_DIM, 1, double >::handle_non_physical_result<double>() const;
template double PhysicsBase < PHILIP_DIM, 2, double >::handle_non_physical_result<double>() const;
template double PhysicsBase < PHILIP_DIM, 3, double >::handle_non_physical_result<double>() const;
template double PhysicsBase < PHILIP_DIM, 4, double >::handle_non_physical_result<double>() const;
template double PhysicsBase < PHILIP_DIM, 5, double >::handle_non_physical_result<double>() const;
template double PhysicsBase < PHILIP_DIM, 6, double >::handle_non_physical_result<double>() const;
template double PhysicsBase < PHILIP_DIM, 8, double >::handle_non_physical_result<double>() const;

template FadType PhysicsBase < PHILIP_DIM, 1, FadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 2, FadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 3, FadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 4, FadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 5, FadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 6, FadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 8, FadType >::handle_non_physical_result<FadType>() const;

template RadType PhysicsBase < PHILIP_DIM, 1, RadType >::handle_non_physical_result<RadType>() const;
template RadType PhysicsBase < PHILIP_DIM, 2, RadType >::handle_non_physical_result<RadType>() const;
template RadType PhysicsBase < PHILIP_DIM, 3, RadType >::handle_non_physical_result<RadType>() const;
template RadType PhysicsBase < PHILIP_DIM, 4, RadType >::handle_non_physical_result<RadType>() const;
template RadType PhysicsBase < PHILIP_DIM, 5, RadType >::handle_non_physical_result<RadType>() const;
template RadType PhysicsBase < PHILIP_DIM, 6, RadType >::handle_non_physical_result<RadType>() const;
template RadType PhysicsBase < PHILIP_DIM, 8, RadType >::handle_non_physical_result<RadType>() const;

template FadFadType PhysicsBase < PHILIP_DIM, 1, FadFadType >::handle_non_physical_result<FadFadType>() const;
template FadFadType PhysicsBase < PHILIP_DIM, 2, FadFadType >::handle_non_physical_result<FadFadType>() const;
template FadFadType PhysicsBase < PHILIP_DIM, 3, FadFadType >::handle_non_physical_result<FadFadType>() const;
template FadFadType PhysicsBase < PHILIP_DIM, 4, FadFadType >::handle_non_physical_result<FadFadType>() const;
template FadFadType PhysicsBase < PHILIP_DIM, 5, FadFadType >::handle_non_physical_result<FadFadType>() const;
template FadFadType PhysicsBase < PHILIP_DIM, 6, FadFadType >::handle_non_physical_result<FadFadType>() const;
template FadFadType PhysicsBase < PHILIP_DIM, 8, FadFadType >::handle_non_physical_result<FadFadType>() const;

template RadFadType PhysicsBase < PHILIP_DIM, 1, RadFadType >::handle_non_physical_result<RadFadType>() const;
template RadFadType PhysicsBase < PHILIP_DIM, 2, RadFadType >::handle_non_physical_result<RadFadType>() const;
template RadFadType PhysicsBase < PHILIP_DIM, 3, RadFadType >::handle_non_physical_result<RadFadType>() const;
template RadFadType PhysicsBase < PHILIP_DIM, 4, RadFadType >::handle_non_physical_result<RadFadType>() const;
template RadFadType PhysicsBase < PHILIP_DIM, 5, RadFadType >::handle_non_physical_result<RadFadType>() const;
template RadFadType PhysicsBase < PHILIP_DIM, 6, RadFadType >::handle_non_physical_result<RadFadType>() const;
template RadFadType PhysicsBase < PHILIP_DIM, 8, RadFadType >::handle_non_physical_result<RadFadType>() const;
 // -- -- instantiate all the real types with real2 = FadType for automatic differentiation in NavierStokes::dissipative_flux_directional_jacobian() 
template FadType PhysicsBase < PHILIP_DIM, 1, double >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 2, double >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 3, double >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 4, double >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 5, double >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 6, double >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 8, double >::handle_non_physical_result<FadType>() const;

template FadType PhysicsBase < PHILIP_DIM, 1, RadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 2, RadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 3, RadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 4, RadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 5, RadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 6, RadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 8, RadType >::handle_non_physical_result<FadType>() const;

template FadType PhysicsBase < PHILIP_DIM, 1, FadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 2, FadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 3, FadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 4, FadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 5, FadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 6, FadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 8, FadFadType >::handle_non_physical_result<FadType>() const;

template FadType PhysicsBase < PHILIP_DIM, 1, RadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 2, RadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 3, RadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 4, RadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 5, RadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 6, RadFadType >::handle_non_physical_result<FadType>() const;
template FadType PhysicsBase < PHILIP_DIM, 8, RadFadType >::handle_non_physical_result<FadType>() const;

} // Physics namespace
} // PHiLiP namespace

