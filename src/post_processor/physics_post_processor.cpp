#include "physics_post_processor.h"
#include "physics/physics_factory.h"
#include "physics/model_factory.h"

namespace PHiLiP {
namespace Postprocess {

template <int dim> 
std::unique_ptr< dealii::DataPostprocessor<dim> > PostprocessorFactory<dim>
::create_Postprocessor(const Parameters::AllParameters *const parameters_input)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = parameters_input->pde_type;
    using Model_enum = Parameters::AllParameters::ModelType;
    const Model_enum model_type = parameters_input->model_type;
    using RANSModel_enum = Parameters::PhysicsModelParam::ReynoldsAveragedNavierStokesModel;
    const RANSModel_enum rans_model_type = parameters_input->physics_model_param.RANS_model_type;

    if (pde_type == PDE_enum::advection) {
        return std::make_unique< PhysicsPostprocessor<dim,1> >(parameters_input);
    } else if (pde_type == PDE_enum::advection_vector) {
        return std::make_unique< PhysicsPostprocessor<dim,2> >(parameters_input);
    } else if (pde_type == PDE_enum::diffusion) {
        return std::make_unique< PhysicsPostprocessor<dim,1> >(parameters_input);
    } else if (pde_type == PDE_enum::convection_diffusion) {
        return std::make_unique< PhysicsPostprocessor<dim,1> >(parameters_input);
    } else if (pde_type == PDE_enum::burgers_inviscid) {
        return std::make_unique< PhysicsPostprocessor<dim,dim> >(parameters_input);
    } else if (pde_type == PDE_enum::burgers_viscous) {
        return std::make_unique< PhysicsPostprocessor<dim,dim> >(parameters_input);
    } else if (pde_type == PDE_enum::burgers_rewienski) {
        return std::make_unique< PhysicsPostprocessor<dim,dim> >(parameters_input);
    } else if (pde_type == PDE_enum::euler) {
        return std::make_unique< PhysicsPostprocessor<dim,dim+2> >(parameters_input);
    } else if (pde_type == PDE_enum::navier_stokes) {
        return std::make_unique< PhysicsPostprocessor<dim,dim+2> >(parameters_input);
    } else if (pde_type == PDE_enum::inviscid_real_gas) {
        return std::make_unique< PhysicsPostprocessor<dim,dim+2> >(parameters_input);
    } else if (pde_type == PDE_enum::real_gas) {
        return std::make_unique< PhysicsPostprocessor<dim,PHILIP_DIM+2+(N_SPECIES-1)> >(parameters_input);
    } else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::reynolds_averaged_navier_stokes) && (rans_model_type == RANSModel_enum::SA_negative)) {
        return std::make_unique< PhysicsPostprocessor<dim,dim+3> >(parameters_input);
    } 
#if PHILIP_DIM==3
    else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::large_eddy_simulation)) {
        return std::make_unique< PhysicsPostprocessor<dim,dim+2> >(parameters_input);
    } 
#endif
    else {
        std::cout << "Invalid PDE when creating post-processor" << std::endl;
        std::abort();
    }
}
template class PostprocessorFactory <PHILIP_DIM>;

template <int dim, int nstate> PhysicsPostprocessor<dim,nstate>
::PhysicsPostprocessor (const Parameters::AllParameters *const parameters_input)
    : model(Physics::ModelFactory<dim,nstate,double>::create_Model(parameters_input)) 
    , physics(Physics::PhysicsFactory<dim,nstate,double>::create_Physics(parameters_input,model))
{ }

template <int dim, int nstate> void PhysicsPostprocessor<dim,nstate>
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

template <int dim, int nstate> void PhysicsPostprocessor<dim,nstate>
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


template <int dim, int nstate>
std::vector<std::string> PhysicsPostprocessor<dim,nstate>::get_names () const
{
    return this->physics->post_get_names();
}
template <int dim, int nstate>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
PhysicsPostprocessor<dim,nstate>::get_data_component_interpretation () const
{
    return this->physics->post_get_data_component_interpretation();
}
template <int dim, int nstate>
dealii::UpdateFlags PhysicsPostprocessor<dim,nstate>::get_needed_update_flags () const
{
    return this->physics->post_get_needed_update_flags();
}

template class PhysicsPostprocessor < PHILIP_DIM, 1 >;
template class PhysicsPostprocessor < PHILIP_DIM, 2 >;
// template class PhysicsPostprocessor < PHILIP_DIM, 3 >;
template class PhysicsPostprocessor < PHILIP_DIM, 4 >;
template class PhysicsPostprocessor < PHILIP_DIM, 5 >;
template class PhysicsPostprocessor < PHILIP_DIM, 6 >;
template class PhysicsPostprocessor < PHILIP_DIM, PHILIP_DIM+2+(N_SPECIES-1) >; // TO DO: (dim+2)+(nspecies-1)

} // Postprocess namespace
} // PHiLiP namespace



