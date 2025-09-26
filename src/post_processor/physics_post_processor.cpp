#include "physics_post_processor.h"
#include "physics/physics_factory.h"
#include "physics/model_factory.h"
#include <boost/preprocessor/seq/for_each.hpp>

namespace PHiLiP {
namespace Postprocess {

template <int dim, int nspecies> 
std::unique_ptr< dealii::DataPostprocessor<dim> > PostprocessorFactory<dim,nspecies>
::create_Postprocessor(const Parameters::AllParameters *const parameters_input)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = parameters_input->pde_type;
    using Model_enum = Parameters::AllParameters::ModelType;
    const Model_enum model_type = parameters_input->model_type;
    using RANSModel_enum = Parameters::PhysicsModelParam::ReynoldsAveragedNavierStokesModel;
    const RANSModel_enum rans_model_type = parameters_input->physics_model_param.RANS_model_type;

    if (pde_type == PDE_enum::advection && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,1> >(parameters_input);
    } else if (pde_type == PDE_enum::advection_vector && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,2> >(parameters_input);
    } else if (pde_type == PDE_enum::diffusion && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,1> >(parameters_input);
    } else if (pde_type == PDE_enum::convection_diffusion && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,1> >(parameters_input);
    } else if (pde_type == PDE_enum::burgers_inviscid && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim> >(parameters_input);
    } else if (pde_type == PDE_enum::burgers_viscous && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim> >(parameters_input);
    } else if (pde_type == PDE_enum::burgers_rewienski && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim> >(parameters_input);
    } else if (pde_type == PDE_enum::euler && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim+2> >(parameters_input);
    } else if (pde_type == PDE_enum::navier_stokes && nspecies == 1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim+2> >(parameters_input);
    } else if (pde_type == PDE_enum::real_gas) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim+2+(nspecies-1)> >(parameters_input);
    } else if (pde_type == PDE_enum::multi_species_calorically_perfect_euler) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim+2+(nspecies-1)> >(parameters_input);    
    } else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::reynolds_averaged_navier_stokes) && (rans_model_type == RANSModel_enum::SA_negative) && nspecies==1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim+3> >(parameters_input);
    } 
#if PHILIP_DIM==3
    else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::large_eddy_simulation) && nspecies==1) {
        return std::make_unique< PhysicsPostprocessor<dim,nspecies,dim+2> >(parameters_input);
    } 
#endif
    else {
        std::cout << "Invalid PDE when creating post-processor" << std::endl;
        std::abort();
    }
}
template class PostprocessorFactory <PHILIP_DIM, PHILIP_SPECIES>;

template <int dim, int nspecies, int nstate> PhysicsPostprocessor<dim,nspecies,nstate>
::PhysicsPostprocessor (const Parameters::AllParameters *const parameters_input)
{
    if(nspecies==1) {
        this->model = Physics::ModelFactory<dim,nspecies,nstate,double>::create_Model(parameters_input);
        this->physics = Physics::PhysicsFactory<dim,nspecies,nstate,double>::create_Physics(parameters_input,model);
    } else {
        this->model = nullptr;
        this->physics = Physics::PhysicsFactory<dim,nspecies,nstate,double>::create_Physics(parameters_input);
    }
 }

template <int dim, int nspecies, int nstate> void PhysicsPostprocessor<dim,nspecies,nstate>
::evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &inputs, std::vector<dealii::Vector<double>> &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_values.size();
    Assert (computed_quantities.size() == n_quadrature_points, dealii::ExcInternalError());
    Assert (inputs.solution_values[0].size() == nstate, dealii::ExcInternalError());
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        computed_quantities[q] = this->physics->post_compute_derived_quantities_vector(
                inputs.solution_values[q],
                inputs.solution_gradients[q],
                inputs.solution_hessians[q],
                inputs.normals[q],
                inputs.evaluation_points[q]);
    }
}

template <int dim, int nspecies, int nstate> void PhysicsPostprocessor<dim,nspecies,nstate>
::evaluate_scalar_field (const dealii::DataPostprocessorInputs::Scalar<dim> &inputs, std::vector<dealii::Vector<double>> &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_values.size();
    Assert (computed_quantities.size() == n_quadrature_points, dealii::ExcInternalError());
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        computed_quantities[q] = this->physics->post_compute_derived_quantities_scalar(
                inputs.solution_values[q],
                inputs.solution_gradients[q],
                inputs.solution_hessians[q],
                inputs.normals[q],
                inputs.evaluation_points[q]);
    }
}


template <int dim, int nspecies, int nstate>
std::vector<std::string> PhysicsPostprocessor<dim,nspecies,nstate>::get_names () const
{
    return this->physics->post_get_names();
}
template <int dim, int nspecies, int nstate>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
PhysicsPostprocessor<dim,nspecies,nstate>::get_data_component_interpretation () const
{
    return this->physics->post_get_data_component_interpretation();
}
template <int dim, int nspecies, int nstate>
dealii::UpdateFlags PhysicsPostprocessor<dim,nspecies,nstate>::get_needed_update_flags () const
{
    return this->physics->post_get_needed_update_flags();
}

#if PHILIP_SPECIES==1
    // Define a sequence of indices representing the range [1, 6] - max is 6 because nstate=dim+2+(species-1)=6 when dim=3 species=2
    // Templated to allow compilation when NUMBER_OF_SPECIES > 1, but may not work.
    #define POSSIBLE_NSTATE (1)(2)(3)(4)(5)(6)(7)(8)

    // Define a macro to instantiate MyTemplate for a specific index
    #define INSTANTIATE_TEMPLATE(r, data, index) \
    template class PhysicsPostprocessor <PHILIP_DIM,PHILIP_SPECIES,index>;
    BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATE, _, POSSIBLE_NSTATE)
#endif
} // Postprocess namespace
} // PHiLiP namespace



