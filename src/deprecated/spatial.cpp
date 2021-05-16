#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include "integrator.h"

#include "dg.h"
#include "advection_boundary.h"

#include "parameters.h"
namespace PHiLiP {
    using namespace dealii;

    template <int dim, typename real>
    void Spatial<dim, real>::assemble_right_hand_side (
        Vector<real> &solution,
        Vector<real> &right_hand_side
    )
    {
        system_matrix = 0;
        right_hand_side = 0;

        // For now assume same polynomial degree across domain
        const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
        std::vector<types::global_dof_index> current_dofs_indices (dofs_per_cell);
        std::vector<types::global_dof_index> neighbor_dofs_indices (dofs_per_cell);

        // Local vector contribution from each cell
        Vector<real> current_cell_rhs (dofs_per_cell);
        Vector<real> neighbor_cell_rhs (dofs_per_cell);


        // ACTIVE cells, therefore, no children
        typename DoFHandler<dim>::active_cell_iterator
        current_cell = dof_handler.begin_active(),
        endc = dof_handler.end();

        unsigned int n_cell_visited = 0;
        unsigned int n_face_visited = 0;
        for (; current_cell!=endc; ++current_cell) {
            n_cell_visited++;

            current_cell_rhs = 0;
            fe_values->reinit (current_cell);
            current_cell->get_dof_indices (current_dofs_indices);

            assemble_cell_terms_implicit (fe_values, current_dofs_indices, current_cell_rhs);

            for (unsigned int face_no=0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {

                typename DoFHandler<dim>::face_iterator current_face = current_cell->face(face_no);
                typename DoFHandler<dim>::cell_iterator neighbor_cell = current_cell->neighbor(face_no);

                // See tutorial step-30 for breakdown of 4 face cases

                // Case 1:
                // Face at boundary
                if (current_face->at_boundary()) {

                    n_face_visited++;

                    fe_values_face->reinit (current_cell, face_no);
                    assemble_boundary_term_implicit (fe_values_face, current_dofs_indices, current_cell_rhs);

                // Case 2:
                // Neighbour is finer occurs if the face has children
                // This is because we are looping over the current_cell's face, so 2, 4, and 8 faces.
                } else if (current_face->has_children()) {
                    neighbor_cell_rhs = 0;
                    Assert (current_cell->neighbor(face_no).state() == IteratorState::valid, ExcInternalError());

                    // Obtain cell neighbour
                    const unsigned int neighbor_face = current_cell->neighbor_face_no(face_no);

                    for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {

                        n_face_visited++;

                        typename DoFHandler<dim>::cell_iterator neighbor_child_cell = current_cell->neighbor_child_on_subface (face_no, subface_no);
                        Assert (!neighbor_child_cell->has_children(), ExcInternalError());

                        neighbor_child_cell->get_dof_indices (neighbor_dofs_indices);

                        fe_values_subface->reinit (current_cell, face_no, subface_no);
                        fe_values_face_neighbor->reinit (neighbor_child_cell, neighbor_face);
                        assemble_face_term_implicit (
                            fe_values_subface, fe_values_face_neighbor,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);

                        // Add local contribution from neighbor cell to global vector
                        for (unsigned int i=0; i<dofs_per_cell; ++i) {
                            right_hand_side(neighbor_dofs_indices[i]) -= neighbor_cell_rhs(i);
                        }
                    }

                // Case 3:
                // Neighbor cell is NOT coarser
                // Therefore, they have the same coarseness, and we need to choose one of them to do the work
                } else if (
                    !current_cell->neighbor_is_coarser(face_no) &&
                        // Cell with lower index does work
                        (neighbor_cell->index() > current_cell->index() || 
                        // If both cells have same index
                        // See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
                        // then cell at the lower level does the work
                            (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
                        ) )
                {
                    n_face_visited++;

                    neighbor_cell_rhs = 0;
                    Assert (current_cell->neighbor(face_no).state() == IteratorState::valid, ExcInternalError());
                    typename DoFHandler<dim>::cell_iterator neighbor_cell = current_cell->neighbor(face_no);

                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                    const unsigned int neighbor_face = current_cell->neighbor_of_neighbor(face_no);

                    fe_values_face->reinit (current_cell, face_no);
                    fe_values_face_neighbor->reinit (neighbor_cell, neighbor_face);
                    assemble_face_term_implicit (
                            fe_values_face, fe_values_face_neighbor,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);

                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<dofs_per_cell; ++i) {
                        right_hand_side(neighbor_dofs_indices[i]) -= neighbor_cell_rhs(i);
                    }
                } 
                // Case 4: Neighbor is coarser
                // Do nothing.
                // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces

            } // end of face loop

            for (unsigned int i=0; i<dofs_per_cell; ++i) {
                right_hand_side(current_dofs_indices[i]) -= current_cell_rhs(i);
            }

        } // end of cell loop
    } // end of assemble_right_hand_side ()


} // end of PHiLiP namespace

