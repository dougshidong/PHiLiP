#ifndef __FLAT_PLATE_2D_H__
#define __FLAT_PLATE_2D_H__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class FlatPlate2D : public FlowSolverCaseBase<dim,nstate>
{
public:
    /// Constructor.
    FlatPlate2D(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~FlatPlate2D() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Will evaluate and print boundary layer informations
    void steady_state_postprocessing(std::shared_ptr <DGBase<dim, double>> dg) const override;

protected:
    const double free_length; ///< Length of free area upwind to the flat plate
    const double free_height; ///< Height of free area above the flat plate
    const double plate_length; ///< Length of the flat plate
    const double skewness_x_free; /// Skewness of the meshes in the x direction for the free area
    const double skewness_x_plate; /// Skewness of the meshes in the x direction for the plate area
    const double skewness_y; /// Skewness of the meshes in the y direction
    const int number_of_subdivisions_in_x_direction_free; ///< Number of cells per x direction for the grid in free area 
    const int number_of_subdivisions_in_x_direction_plate; ///< Number of cells per x direction for the grid in plate area
    const int number_of_subdivisions_in_y_direction; ///< Number of cells per y direction for the grid

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

};

} // FlowSolver namespace
} // PHiLiP namespace
#endif