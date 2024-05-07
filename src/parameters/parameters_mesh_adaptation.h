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
        hp_adaptation,
        anisotropic_adaptation
    };
    /// Selection of mesh adaptation type
    MeshAdaptationType mesh_adaptation_type;
    
    /// Total/maximum number of mesh adaptation cycles while solving a problem.
    int total_mesh_adaptation_cycles;
    
    /// Fraction of cells to be h or p-refined
    double refine_fraction;

    /// Fraction of cells to be h-coarsened
    double h_coarsen_fraction;
    
    /// Flag to use goal oriented mesh adaptation
    bool use_goal_oriented_mesh_adaptation;

    /// Tolerance to decide between h- or p-refinement
    double hp_smoothness_tolerance;

    /// Continuous equivalent of number of vertices/elements. Used in anisotropic mesh adaptation.
    double mesh_complexity_anisotropic_adaptation;

    /// Lp norm w.r.t. which the optimization is performed in the continuous mesh framework.
    double norm_Lp_anisotropic_adaptation;

    /// Declare parameters
    static void declare_parameters (dealii::ParameterHandler &prm);
 
    /// Parse parameters
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // namespace Parameters
} // namespace PHiLiP
#endif
