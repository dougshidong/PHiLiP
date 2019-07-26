#ifndef __PHYSICS_POST__
#define __PHYSICS_POST__

#include <deal.II/numerics/data_postprocessor.h>

#include "physics/physics.h"

namespace PHiLiP {
namespace Postprocess {

/// Create the post-processor with the correct template parameter 
template <int dim>
class PostprocessorFactory
{
public:
    static std::unique_ptr< dealii::DataPostprocessor<dim> > create_Postprocessor(const Parameters::AllParameters *const parameters_input);
};

template <int dim, int nstate>
class PhysicsPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    /// Constructor
    PhysicsPostprocessor (const Parameters::AllParameters *const parameters_input);

    /// Physics that the post-processor will use to evaluate derived data types
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > physics;

    // virtual void compute_derived_quantities_vector (
    //     const std::vector<dealii::Vector<double> >              &uh,
    //     const std::vector<std::vector<dealii::Tensor<1,dim> > > &duh,
    //     const std::vector<std::vector<dealii::Tensor<2,dim> > > &dduh,
    //     const std::vector<dealii::Point<dim> >                  &normals,
    //     const std::vector<dealii::Point<dim> >                  &evaluation_points,
    //     std::vector<dealii::Vector<double> >                    &computed_quantities) const;
    virtual void evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &inputs, std::vector<dealii::Vector<double>> &computed_quantities) const override;
    virtual void evaluate_scalar_field (const dealii::DataPostprocessorInputs::Scalar<dim> &inputs, std::vector<dealii::Vector<double>> &computed_quantities) const override;
    virtual std::vector<std::string> get_names () const override;
    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> get_data_component_interpretation () const override;
    virtual dealii::UpdateFlags get_needed_update_flags () const override;
};

} // Postprocess namespace
} // PHiLiP namespace

#endif

