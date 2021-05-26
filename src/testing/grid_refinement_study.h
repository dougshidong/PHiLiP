#ifndef __GRID_REFINEMENT_STUDY_H__
#define __GRID_REFINEMENT_STUDY_H__

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

#include "grid_refinement/gnu_out.h"

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

    /// gets the grid from the enum and reads file if neccesary
    void get_grid(
        const std::shared_ptr<MeshType>&            grid,
        const Parameters::GridRefinementStudyParam& grs_param) const;

    /// Approximates the exact functional using a uniformly refined grid
    double approximate_exact_functional(
        const std::shared_ptr<Physics::PhysicsBase<dim,nstate,double>>& physics_double,
        const std::shared_ptr<Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<double>>>& physics_adtype,
        const Parameters::AllParameters& param,
        const Parameters::GridRefinementStudyParam& grs_param) const;
};

// constructs the mesh for the test
template <typename MeshType>
class MeshFactory
{
public:
    static std::shared_ptr<MeshType>
    create_MeshType(const MPI_Comm mpi_communicator);
};

// function to perform the formatted output to gnuplot (of the solution error)
void output_gnufig_solution(
    PHiLiP::GridRefinement::GnuFig<double> &gf);

// function to perform the formatted output to gnuplot (of the funcitonal error)
void output_gnufig_functional(
    PHiLiP::GridRefinement::GnuFig<double> &gf);

} // Tests namespace

} // PHiLiP namespace

#endif // __GRID_REFINEMENT_STUDY_H__
