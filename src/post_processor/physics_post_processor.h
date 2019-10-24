#ifndef __PHYSICS_POST__
#define __PHYSICS_POST__

#include <deal.II/numerics/data_postprocessor.h>

#include "physics/physics.h"

namespace PHiLiP {
namespace Postprocess {

/// Postprocessor factory.
template <int dim>
class PostprocessorFactory
{
public:
    /// Create the post-processor with the correct template parameters.
    static std::unique_ptr< dealii::DataPostprocessor<dim> > create_Postprocessor(const Parameters::AllParameters *const parameters_input);
};

/// Postprocessor to output solution and other values computed by associated physics.
/** The functions in this class will call the Physics functions to query data.
 */
template <int dim, int nstate>
class PhysicsPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    /// Constructor.
    PhysicsPostprocessor (const Parameters::AllParameters *const parameters_input);

    /// Physics that the post-processor will use to evaluate derived data types.
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > physics;

    /// Queries the Physics to output data of a vector-valued problem.
    virtual void evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &inputs, std::vector<dealii::Vector<double>> &computed_quantities) const override;
    /// Queries the Physics to output data of a scalar-valued problem.
    virtual void evaluate_scalar_field (const dealii::DataPostprocessorInputs::Scalar<dim> &inputs, std::vector<dealii::Vector<double>> &computed_quantities) const override;
    /// Queries the Physics for the names of output data variables.
    virtual std::vector<std::string> get_names () const override;
    /// Queries the Physics for the type (scalar/vector) of output data variables.
    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> get_data_component_interpretation () const override;
    /// Queries the Physics for the required update flags to evaluate output data.
    virtual dealii::UpdateFlags get_needed_update_flags () const override;
};

} // Postprocess namespace
} // PHiLiP namespace

#endif

