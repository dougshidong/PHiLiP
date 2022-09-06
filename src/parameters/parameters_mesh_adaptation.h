#ifndef __PARAMETERS_MESH_ADAPTATION_H__
#define __PARAMETERS_MESH_ADAPTATION_H__
 
#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {

/// Parameters for Mesh Adaptation
class MeshAdaptationParam 
{
public:
    /// Choices for mesh adaptation to be used
    enum MeshAdaptationType{
        h_adaptation,
        p_adaptation,
        hp_adaptation
    };
    /// Selection of mesh adaptation type
    MeshAdaptationType mesh_adaptation_type;
    
    /// Total/maximum number of mesh adaptation cycles while solving a problem.
    int total_mesh_adaptation_cycles;
    
    /// Fraction of cells to be h-refined
    double h_refine_fraction;

    /// Fraction of cells to be h-coarsened
    double h_coarsen_fraction;
    
    /// Flag to use goal oriented mesh adaptation
    bool use_goal_oriented_mesh_adaptation;

    /// Tolerance to decide between h- or p-refinement
    double hp_smoothness_tolerance;

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
