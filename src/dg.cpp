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


// Finally, we take our exact solution from the library as well as quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>



#include "integrator.h"

#include "dg.h"

#include "parameters.h"

namespace PHiLiP
{
    using namespace dealii;


    // Constructors
    template <int dim, typename real>
    DiscontinuousGalerkin<dim, real>::DiscontinuousGalerkin(
        Parameters::AllParameters *parameters_input,
        Triangulation<dim>   *triangulation_input,
        const unsigned int degree)
        :
        //triangulation(triangulation_input)
        mapping(degree+1)
        , fe(degree)
        , parameters(parameters_input)
        , quadrature (degree+1)
        , face_quadrature (degree+1)
    {
        set_triangulation(triangulation_input);
    }
    template <int dim, typename real>
    DiscontinuousGalerkin<dim, real>::DiscontinuousGalerkin(
        Parameters::AllParameters *parameters_input,
        const unsigned int degree)
        :
        mapping(degree+1)
        , fe(degree)
        , parameters(parameters_input)
        , quadrature (degree+1)
        , face_quadrature (degree+1)
    {
    }
    // Destructor
    template <int dim, typename real>
    DiscontinuousGalerkin<dim, real>::~DiscontinuousGalerkin ()
    {
        std::cout << std::endl << "Destructing DG" << std::endl;
        delete_fe_values();
    }

    template <int dim, typename real>
    double DiscontinuousGalerkin<dim, real>::get_residual_l2norm ()
    {
        return right_hand_side.l2_norm();
    }
    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::assemble_system ()
    {
        assemble_system_implicit();
    }
    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::allocate_system ()
    {
        allocate_system_implicit();
    }

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::delete_fe_values ()
    {
        std::cout << std::endl << "Deallocating FEValues" << std::endl;

        if (fe_values_cell          != NULL) delete fe_values_cell;
        if (fe_values_face_int      != NULL) delete fe_values_face_int;
        if (fe_values_subface_int   != NULL) delete fe_values_subface_int;
        if (fe_values_face_ext      != NULL) delete fe_values_face_ext;
        fe_values_cell          = NULL; 
        fe_values_face_int      = NULL;
        fe_values_subface_int   = NULL;
        fe_values_face_ext      = NULL;
    }
  

    
    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::allocate_system_explicit ()
    {
        std::cout << std::endl << "Allocating DG system and initializing FEValues" << std::endl;
        // This function allocates all the necessary memory to the 
        // system matrices and vectors.

        dof_handler.initialize(*triangulation, fe);
        // Allocates memory from triangulation and finite element space
        dof_handler.distribute_dofs(fe);

        // Allocate vectors
        solution.reinit(dof_handler.n_dofs());
        right_hand_side.reinit(dof_handler.n_dofs());
        source_term.reinit(dof_handler.n_dofs());

        const UpdateFlags update_flags = update_values
                                         | update_gradients
                                         | update_quadrature_points
                                         | update_JxW_values;
        const UpdateFlags face_update_flags = update_values
                                              | update_gradients
                                              | update_quadrature_points
                                              | update_JxW_values
                                              | update_normal_vectors;
        const UpdateFlags neighbor_face_update_flags = update_values
                                                      | update_gradients;

        fe_values_cell          = new FEValues<dim,dim> (mapping, fe, quadrature, update_flags);
        fe_values_face_int      = new FEFaceValues<dim,dim> (mapping, fe, face_quadrature, face_update_flags);
        fe_values_subface_int   = new FESubfaceValues<dim,dim> (mapping, fe, face_quadrature, face_update_flags);
        fe_values_face_ext      = new FEFaceValues<dim,dim> (mapping, fe, face_quadrature, neighbor_face_update_flags);
    }

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::allocate_system_implicit ()
    {
        std::cout << std::endl << "Allocating DG system and initializing FEValues" << std::endl;
        // This function allocates all the necessary memory to the 
        // system matrices and vectors.

        dof_handler.initialize(*triangulation, fe);
        // Allocates memory from triangulation and finite element space
        dof_handler.distribute_dofs(fe);


        // Allocate matrix
        unsigned int n_dofs = dof_handler.n_dofs();
        //DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        sparsity_pattern.reinit(n_dofs, n_dofs);

        DoFTools::make_flux_sparsity_pattern(dof_handler, sparsity_pattern);

        //sparsity_pattern.print(std::cout);

        system_matrix.reinit(sparsity_pattern);

        // Allocate vectors
        solution.reinit(dof_handler.n_dofs());
        right_hand_side.reinit(dof_handler.n_dofs());
        source_term.reinit(dof_handler.n_dofs());

        newton_update.reinit(dof_handler.n_dofs());

        const UpdateFlags update_flags = update_values
                                         | update_gradients
                                         | update_quadrature_points
                                         | update_JxW_values;
        const UpdateFlags face_update_flags = update_values
                                              | update_gradients
                                              | update_quadrature_points
                                              | update_JxW_values
                                              | update_normal_vectors;
        const UpdateFlags neighbor_face_update_flags = update_values
                                                      | update_gradients;

        fe_values_cell          = new FEValues<dim,dim> (mapping, fe, quadrature, update_flags);
        fe_values_face_int      = new FEFaceValues<dim,dim> (mapping, fe, face_quadrature, face_update_flags);
        fe_values_subface_int   = new FESubfaceValues<dim,dim> (mapping, fe, face_quadrature, face_update_flags);
        fe_values_face_ext      = new FEFaceValues<dim,dim> (mapping, fe, face_quadrature, neighbor_face_update_flags);
    }

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::assemble_system_implicit ()
    {
        system_matrix = 0;
        right_hand_side = 0;

        // For now assume same polynomial degree across domain
        const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
        std::vector<types::global_dof_index> current_dofs_indices (dofs_per_cell);
        std::vector<types::global_dof_index> neighbor_dofs_indices (dofs_per_cell);

        // Local vector contribution from each cell
        Vector<double> current_cell_rhs (dofs_per_cell);
        Vector<double> neighbor_cell_rhs (dofs_per_cell);


        // ACTIVE cells, therefore, no children
        typename DoFHandler<dim>::active_cell_iterator
            current_cell = dof_handler.begin_active(),
            endc = dof_handler.end();

        unsigned int n_cell_visited = 0;
        unsigned int n_face_visited = 0;
        for (; current_cell!=endc; ++current_cell) {
            n_cell_visited++;

            current_cell_rhs = 0;
            fe_values_cell->reinit (current_cell);
            current_cell->get_dof_indices (current_dofs_indices);

            assemble_cell_terms_implicit (fe_values_cell, current_dofs_indices, current_cell_rhs);

            for (unsigned int iface=0; iface < GeometryInfo<dim>::faces_per_cell; ++iface) {

                typename DoFHandler<dim>::face_iterator current_face = current_cell->face(iface);
                typename DoFHandler<dim>::cell_iterator neighbor_cell = current_cell->neighbor(iface);

                // See tutorial step-30 for breakdown of 4 face cases

                // Case 1:
                // Face at boundary
                if (current_face->at_boundary()) {

                    n_face_visited++;

                    fe_values_face_int->reinit (current_cell, iface);
                    const unsigned int degree_current = fe.get_degree();
                    const unsigned int deg1sq = (degree_current == 0) ? 1 : degree_current * (degree_current+1);
                    const unsigned int normal_direction = GeometryInfo<dim>::unit_normal_direction[iface];
                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction);

                    real penalty = deg1sq / vol_div_facearea1;
                    //penalty = 1;//99;

                    assemble_boundary_term_implicit (fe_values_face_int, penalty, current_dofs_indices, current_cell_rhs);

                // Case 2:
                // Neighbour is finer occurs if the face has children
                // This is because we are looping over the current_cell's face, so 2, 4, and 6 faces.
                } else if (current_face->has_children()) {
                    std::cout << "SHOULD NOT HAPPEN!!!!!!!!!!!! I haven't put in adaptatation yet" << std::endl;
                    neighbor_cell_rhs = 0;
                    Assert (current_cell->neighbor(iface).state() == IteratorState::valid, ExcInternalError());

                    // Obtain cell neighbour
                    const unsigned int neighbor_face_no = current_cell->neighbor_face_no(iface);

                    for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {

                        n_face_visited++;

                        typename DoFHandler<dim>::cell_iterator neighbor_child_cell = current_cell->neighbor_child_on_subface (iface, subface_no);

                        Assert (!neighbor_child_cell->has_children(), ExcInternalError());

                        neighbor_child_cell->get_dof_indices (neighbor_dofs_indices);

                        fe_values_subface_int->reinit (current_cell, iface, subface_no);
                        fe_values_face_ext->reinit (neighbor_child_cell, neighbor_face_no);

                        const unsigned int normal_direction1 = GeometryInfo<dim>::unit_normal_direction[iface];
                        const unsigned int normal_direction2 = GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                        const unsigned int degree_current = fe.get_degree();
                        const unsigned int deg1sq = (degree_current == 0) ? 1 : degree_current * (degree_current+1);
                        const unsigned int deg2sq = (degree_current == 0) ? 1 : degree_current * (degree_current+1);

                        //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
                        const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                        const real vol_div_facearea2 = neighbor_child_cell->extent_in_direction(normal_direction2);

                        const real penalty1 = deg1sq / vol_div_facearea1;
                        const real penalty2 = deg2sq / vol_div_facearea2;
                        
                        real penalty = 0.5 * ( penalty1 + penalty2 );
                        //penalty = 1;

                        //std::cout
                        //    << "Degree: " << deg1sq
                        //    << "Length 1: " << vol_div_facearea1
                        //    << "Length 2: " << vol_div_facearea2
                        //    << "Penalty: " << penalty
                        //    << std::endl
                        //    << std::endl
                        //    ;


                        assemble_face_term_implicit (
                            fe_values_subface_int, fe_values_face_ext,
                            penalty,
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
                    !current_cell->neighbor_is_coarser(iface) &&
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
                    Assert (current_cell->neighbor(iface).state() == IteratorState::valid, ExcInternalError());
                    typename DoFHandler<dim>::cell_iterator neighbor_cell = current_cell->neighbor(iface);

                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                    const unsigned int neighbor_face_no = current_cell->neighbor_of_neighbor(iface);

                    const unsigned int normal_direction1 = GeometryInfo<dim>::unit_normal_direction[iface];
                    const unsigned int normal_direction2 = GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                    const unsigned int degree_current = fe.get_degree();
                    const unsigned int deg1sq = (degree_current == 0) ? 1 : degree_current * (degree_current+1);
                    const unsigned int deg2sq = (degree_current == 0) ? 1 : degree_current * (degree_current+1);

                    //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                    const real penalty1 = deg1sq / vol_div_facearea1;
                    const real penalty2 = deg2sq / vol_div_facearea2;
                    
                    real penalty = 0.5 * ( penalty1 + penalty2 );
                    //penalty = 1;//99;

                    //    std::cout
                    //        << "  Degree: " << deg1sq
                    //        << "  Length 1: " << vol_div_facearea1
                    //        << "  Length 2: " << vol_div_facearea2
                    //        << "  Penalty: " << penalty
                    //        << std::endl
                    //        << std::endl
                    //        ;

                    fe_values_face_int->reinit (current_cell, iface);
                    fe_values_face_ext->reinit (neighbor_cell, neighbor_face_no);
                    assemble_face_term_implicit (
                            fe_values_face_int, fe_values_face_ext,
                            penalty,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);

                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<dofs_per_cell; ++i) {
                        right_hand_side(neighbor_dofs_indices[i]) -= neighbor_cell_rhs(i);
                    }
                } else {
                // Do nothing
                }
                // Case 4: Neighbor is coarser
                // Do nothing.
                // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces

            } // end of face loop

            for (unsigned int i=0; i<dofs_per_cell; ++i) {
                right_hand_side(current_dofs_indices[i]) -= current_cell_rhs(i);
            }

        } // end of cell loop
    } // end of assemble_system_implicit ()
  
//    template <int dim, typename real>
//    void DiscontinuousGalerkin<dim, real>::assemble_system_explicit ()
//    {
//        right_hand_side = 0;
//        // For now assume same polynomial degree across domain
//        const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
//        std::vector<types::global_dof_index> current_dofs_indices (dofs_per_cell);
//        std::vector<types::global_dof_index> neighbor_dofs_indices (dofs_per_cell);
//
//        // Local vector contribution from each cell
//        Vector<double> current_cell_rhs (dofs_per_cell);
//        Vector<double> neighbor_cell_rhs (dofs_per_cell);
//
//
//        // ACTIVE cells, therefore, no children
//        typename DoFHandler<dim>::active_cell_iterator
//        current_cell = dof_handler.begin_active(),
//        endc = dof_handler.end();
//
//        unsigned int n_cell_visited = 0;
//        unsigned int n_face_visited = 0;
//        for (; current_cell!=endc; ++current_cell) {
//            n_cell_visited++;
//
//            current_cell_rhs = 0;
//            fe_values_cell->reinit (current_cell);
//            current_cell->get_dof_indices (current_dofs_indices);
//
//            assemble_cell_terms_explicit(fe_values_cell, current_dofs_indices, current_cell_rhs);
//
//            for (unsigned int iface=0; iface < GeometryInfo<dim>::faces_per_cell; ++iface) {
//
//                typename DoFHandler<dim>::face_iterator current_face = current_cell->face(iface);
//                typename DoFHandler<dim>::cell_iterator neighbor_cell = current_cell->neighbor(iface);
//
//                // See tutorial step-30 for breakdown of 4 face cases
//
//                // Case 1:
//                // Face at boundary
//                if (current_face->at_boundary()) {
//
//                    n_face_visited++;
//
//                    fe_values_face_int->reinit (current_cell, iface);
//                    assemble_boundary_term_explicit(fe_values_face_int, current_dofs_indices, current_cell_rhs);
//
//                // Case 2:
//                // Neighbour is finer occurs if the face has children
//                // This is because we are looping over the current_cell's face, so 2, 4, and 8 faces.
//                } else if (current_face->has_children()) {
//                    neighbor_cell_rhs = 0;
//                    Assert (current_cell->neighbor(iface).state() == IteratorState::valid, ExcInternalError());
//                    std::cout << "SHOULD NOT HAPPEN!!!!!!!!!!!! I haven't put in adaptatation yet" << std::endl;
//
//                    // Obtain cell neighbour
//                    const unsigned int neighbor_face_no = current_cell->neighbor_face_no(iface);
//
//                    for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {
//
//                        n_face_visited++;
//
//                        typename DoFHandler<dim>::cell_iterator neighbor_child_cell = current_cell->neighbor_child_on_subface (iface, subface_no);
//                        Assert (!neighbor_child_cell->has_children(), ExcInternalError());
//
//                        neighbor_child_cell->get_dof_indices (neighbor_dofs_indices);
//
//                        fe_values_subface_int->reinit (current_cell, iface, subface_no);
//                        fe_values_face_ext->reinit (neighbor_child_cell, neighbor_face_no);
//                        assemble_face_term_explicit(
//                            fe_values_subface_int, fe_values_face_ext,
//                            current_dofs_indices, neighbor_dofs_indices,
//                            current_cell_rhs, neighbor_cell_rhs);
//
//                        // Add local contribution from neighbor cell to global vector
//                        for (unsigned int i=0; i<dofs_per_cell; ++i) {
//                            right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
//                        }
//                    }
//
//                // Case 3:
//                // Neighbor cell is NOT coarser
//                // Therefore, they have the same coarseness, and we need to choose one of them to do the work
//                } else if (
//                    !current_cell->neighbor_is_coarser(iface) &&
//                        // Cell with lower index does work
//                        (neighbor_cell->index() > current_cell->index() || 
//                        // If both cells have same index
//                        // See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
//                        // then cell at the lower level does the work
//                            (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
//                        ) )
//                {
//                    n_face_visited++;
//
//                    neighbor_cell_rhs = 0;
//                    Assert (current_cell->neighbor(iface).state() == IteratorState::valid, ExcInternalError());
//                    typename DoFHandler<dim>::cell_iterator neighbor_cell = current_cell->neighbor(iface);
//
//                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);
//
//                    const unsigned int neighbor_face_no = current_cell->neighbor_of_neighbor(iface);
//
//                    fe_values_face_int->reinit (current_cell, iface);
//                    fe_values_face_ext->reinit (neighbor_cell, neighbor_face_no);
//                    assemble_face_term_explicit(
//                            fe_values_face_int, fe_values_face_ext,
//                            current_dofs_indices, neighbor_dofs_indices,
//                            current_cell_rhs, neighbor_cell_rhs);
//
//                    // Add local contribution from neighbor cell to global vector
//                    for (unsigned int i=0; i<dofs_per_cell; ++i) {
//                        right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
//                    }
//                } else {
//                // Do nothing
//                }
//                // Case 4: Neighbor is coarser
//                // Do nothing.
//                // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces
//
//            } // end of face loop
//
//            for (unsigned int i=0; i<dofs_per_cell; ++i) {
//                right_hand_side(current_dofs_indices[i]) += current_cell_rhs(i);
//            }
//
//        } // end of cell loop
//    } // end of assemble_system_explicit()


    template <int dim, typename real>
    void DiscontinuousGalerkin<dim,real>::output_results (const unsigned int ith_grid)// const
    {
      const std::string filename = "sol-" +
                                   Utilities::int_to_string(ith_grid,2) +
                                   ".gnuplot";

      std::cout << "Writing solution to <" << filename << ">..."
                << std::endl << std::endl;
      std::ofstream gnuplot_output (filename.c_str());

      DataOut<dim> data_out;
      data_out.attach_dof_handler (dof_handler);
      data_out.add_data_vector (solution, "u", DataOut<dim>::type_dof_data);
      //data_out.add_data_vector (estimates.block(0), "est");

      data_out.build_patches ();

      data_out.write_gnuplot(gnuplot_output);
    }



    template class DiscontinuousGalerkin <PHILIP_DIM, double>;

} // end of PHiLiP namespace
