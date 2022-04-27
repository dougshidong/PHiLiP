#ifndef __PARAMETERS_MESH_ADAPTATION_H__
#define __PARAMETERS_MESH_ADAPTATION_H__
 
#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {

/// Parameters for Mesh Adaptation
class MeshAdaptationParam 
{
public:

    /// Total/maximum number of refinement cycles while solving a problem.
    int total_refinement_cycles;
    
    /// Critical residual below which refinement begins.
    double critical_residual;

    /// Fraction of cells to be h-refined
    double h_refine_fraction;

    /// Fraction of cells to be h-coarsened
    double h_coarsen_fraction;
    
    /// Fraction of cells to be p-refined
    double p_refine_fraction;

    /// Fraction of cells to be p-coarsened
    double p_coarsen_fraction;

    /// Flag to use goal oriented mesh adaptation
    bool use_goal_oriented_mesh_adaptation;

    /// Constructor of mesh adaptation parameters.
    MeshAdaptationParam();

    /// Declare parameters
    static void declare_parameters (dealii::ParameterHandler &prm);
 
    /// Parse parameters
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // namespace Parameters
} // namespace PHiLiP
#endif
