#ifndef __MESHADAPTATION_H__
#define __MESHADAPTATION_H__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "mesh_error_estimate.h"
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include "mesh_error_factory.h"

namespace PHiLiP {


#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

/** Contains functions for mesh adaptation. It supports residual based and goal-oriented hp-adaptation with fixed fraction coarsening and refinement. 
  * @note Mesh adaptation is currently implemented for steady state test cases. Tests using FlowSolver can use mesh adaptation directly by modifying the parameters file.
  */
class MeshAdaptation
{
public:

    /// Constructor to initialize the class with a pointer to DG.
    MeshAdaptation(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, const Parameters::MeshAdaptationParam *const mesh_adaptation_param_input);

    /// Destructor
    ~MeshAdaptation(){};

    /// Pointer to the error estimator class.
    std::unique_ptr<MeshErrorEstimateBase<dim, real, MeshType>> mesh_error;

    /// Pointer to DGBase.
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    /// Function to adapt the mesh based on input parameters.
    void adapt_mesh();

    /// Stores the current adaptation cycle.
    int current_mesh_adaptation_cycle;

    /// Holds parameters of mesh adaptation.
    const Parameters::MeshAdaptationParam *const mesh_adaptation_param;

protected:
    
    /// Performs fixed fraction refinement based on refinement and coarsening fractions.
    void fixed_fraction_isotropic_refinement_and_coarsening();
    
    /// Decide whether to perform h or p refinement based on a smoothness indicator.
    void smoothness_sensor_based_hp_refinement();
    
    /// Stores errors in each cell
    dealii::Vector<real> cellwise_errors;

    /// Parallel std::cout.
    dealii::ConditionalOStream pcout;

};

} // namespace PHiLiP

#endif
