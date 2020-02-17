#ifndef __GRID_REFINEMENT_STUDY_H__
#define __GRID_REFINEMENT_STUDY_H__

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {

namespace Tests {

/// Performs grid convergence for various polynomial degrees.
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinementStudy : public TestsBase
{
public:
    /// Constructor. Deleted the default constructor since it should not be used
    GridRefinementStudy() = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    GridRefinementStudy(
        const Parameters::AllParameters *const parameters_input);

    ~GridRefinementStudy() {}; ///< Destructor.

    int run_test() const;
};

// constructs the mesh for the test
template <typename MeshType>
class MeshFactory
{
public:
    static std::unique_ptr<MeshType>
    create_MeshType(const MPI_Comm mpi_communicator);
};

} // Tests namespace

} // PHiLiP namespace

#endif // __GRID_REFINEMENT_STUDY_H__
