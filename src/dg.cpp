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
namespace PHiLiP
{
    using namespace dealii;

    template class DiscontinuousGalerkin <PHILIP_DIM, double>;

    template <int dim, typename real>
    DiscontinuousGalerkin<dim, real>::DiscontinuousGalerkin(
        Parameters::AllParameters *parameters_input,
        Triangulation<dim>   *triangulation_input,
        const unsigned int degree)
        :
        triangulation(triangulation_input)
        , mapping(degree+1)
        , fe(degree)
        , quadrature (degree+1)
        , face_quadrature (degree+1)
        , parameters(parameters_input)
    {
    }

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::set_triangulation(Triangulation<dim> *triangulation_input)
    {
        triangulation = triangulation_input;
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

    template DiscontinuousGalerkin<1, double>::DiscontinuousGalerkin(
        Parameters::AllParameters *parameters_input,
        Triangulation<1>   *triangulation_input,
        const unsigned int degree);

    template <int dim, typename real>
    DiscontinuousGalerkin<dim, real>::DiscontinuousGalerkin(
        Parameters::AllParameters *parameters_input,
        const unsigned int degree)
        :
        mapping(degree+1)
        , fe(degree)
        , quadrature (degree+1)
        , face_quadrature (degree+1)
        , parameters(parameters_input)
    {
    }
    template DiscontinuousGalerkin<PHILIP_DIM, double>::DiscontinuousGalerkin(
        Parameters::AllParameters *parameters_input,
        const unsigned int degree);

    
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
                                              | update_quadrature_points
                                              | update_JxW_values
                                              | update_normal_vectors;
        const UpdateFlags neighbor_face_update_flags = update_values;

        fe_values               = new FEValues<dim,dim> (mapping, fe, quadrature, update_flags);
        fe_values_face          = new FEFaceValues<dim,dim> (mapping, fe, face_quadrature, face_update_flags);
        fe_values_subface       = new FESubfaceValues<dim,dim> (mapping, fe, face_quadrature, face_update_flags);
        fe_values_face_neighbor = new FEFaceValues<dim,dim> (mapping, fe, face_quadrature, neighbor_face_update_flags);
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
                                              | update_quadrature_points
                                              | update_JxW_values
                                              | update_normal_vectors;
        const UpdateFlags neighbor_face_update_flags = update_values;

        fe_values               = new FEValues<dim,dim> (mapping, fe, quadrature, update_flags);
        fe_values_face          = new FEFaceValues<dim,dim> (mapping, fe, face_quadrature, face_update_flags);
        fe_values_subface       = new FESubfaceValues<dim,dim> (mapping, fe, face_quadrature, face_update_flags);
        fe_values_face_neighbor = new FEFaceValues<dim,dim> (mapping, fe, face_quadrature, neighbor_face_update_flags);
    }

    template <int dim, typename real>
    DiscontinuousGalerkin<dim, real>::~DiscontinuousGalerkin ()
    {
        std::cout << std::endl << "Destructing DG" << std::endl;
        delete_fe_values();
    }

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::delete_fe_values ()
    {
        std::cout << std::endl << "Deallocating FEValues" << std::endl;

        if (fe_values               != NULL) delete fe_values;
        if (fe_values_face          != NULL) delete fe_values_face;
        if (fe_values_subface       != NULL) delete fe_values_subface;
        if (fe_values_face_neighbor != NULL) delete fe_values_face_neighbor;
        fe_values               = NULL; 
        fe_values_face          = NULL;
        fe_values_subface       = NULL;
        fe_values_face_neighbor = NULL;
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
    } // end of assemble_system_implicit ()
  
    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::assemble_system_explicit ()
    {
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
            fe_values->reinit (current_cell);
            current_cell->get_dof_indices (current_dofs_indices);

            assemble_cell_terms_explicit(fe_values, current_dofs_indices, current_cell_rhs);

            for (unsigned int face_no=0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {

                typename DoFHandler<dim>::face_iterator current_face = current_cell->face(face_no);
                typename DoFHandler<dim>::cell_iterator neighbor_cell = current_cell->neighbor(face_no);

                // See tutorial step-30 for breakdown of 4 face cases

                // Case 1:
                // Face at boundary
                if (current_face->at_boundary()) {

                    n_face_visited++;

                    fe_values_face->reinit (current_cell, face_no);
                    assemble_boundary_term_explicit(fe_values_face, current_dofs_indices, current_cell_rhs);

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
                        assemble_face_term_explicit(
                            fe_values_subface, fe_values_face_neighbor,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);

                        // Add local contribution from neighbor cell to global vector
                        for (unsigned int i=0; i<dofs_per_cell; ++i) {
                            right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
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
                    assemble_face_term_explicit(
                            fe_values_face, fe_values_face_neighbor,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);

                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<dofs_per_cell; ++i) {
                        right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
                    }
                } 
                // Case 4: Neighbor is coarser
                // Do nothing.
                // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces

            } // end of face loop

            for (unsigned int i=0; i<dofs_per_cell; ++i) {
                right_hand_side(current_dofs_indices[i]) += current_cell_rhs(i);
            }

        } // end of cell loop
    } // end of assemble_system_explicit()

    template <int dim, typename real>
    int DiscontinuousGalerkin<dim, real>::run_explicit () 
    {
        allocate_system_explicit ();
        assemble_system_explicit ();

        double residual_norm = right_hand_side.l2_norm();
        typename DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active(),
           endc = dof_handler.end();

        double CFL = 0.1;
        double dx = 1.0/pow(triangulation->n_active_cells(),(1.0/dim));
        double speed = sqrt(dim);
        double dt = CFL * dx / speed;

        int iteration = 0;
        int print = 1000;
        while (residual_norm > 1e-13 && iteration < 100000) {
            ++iteration;

            assemble_system_explicit ();
            residual_norm = right_hand_side.l2_norm();

            if ( (iteration%print) == 0)
            std::cout << " Iteration: " << iteration 
                      << " Residual norm: " << residual_norm
                      << std::endl;

            solution += (right_hand_side*=dt);
        }

        std::vector<unsigned int> dof_indices(fe.dofs_per_cell);

        QGauss<dim> quad_plus10(fe.degree+10);
        const unsigned int n_quad_pts =quad_plus10.size();
        FEValues<dim,dim> fe_values_plus10(mapping, fe,quad_plus10, update_values | update_JxW_values | update_quadrature_points);

        std::vector<double> solution_values(n_quad_pts);

        double l2error = 0;
        for (; cell!=endc; ++cell) {
            //const unsigned int icell = cell->user_index();

            fe_values_plus10.reinit (cell);
            fe_values_plus10.get_function_values (solution, solution_values);

            double uexact = 0;
            for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                const Point<dim> qpoint = (fe_values_plus10.quadrature_point(iquad));
                if (dim==1) uexact = sin(3.19/dim*qpoint(0));
                if (dim==2) uexact = sin(3.19/dim*qpoint(0))*sin(3.19/dim*qpoint(1));
                if (dim==3) uexact = sin(3.19/dim*qpoint(0))*sin(3.19/dim*qpoint(1))*sin(3.19/dim*qpoint(2));

                double u_at_q = solution_values[iquad];
                l2error += pow(u_at_q - uexact, 2) * fe_values_plus10.JxW(iquad);
            }

        }
        l2error = sqrt(l2error);

        return 0;

    }

    template <int dim, typename real>
    int DiscontinuousGalerkin<dim, real>::grid_convergence_explicit () 
    {

        unsigned int n_grids = 4;
        std::vector<double> error(n_grids);
        std::vector<double> grid_size(n_grids);
        std::vector<double> ncell(n_grids);

        ncell[0] = 2;
        ncell[1] = 4;
        ncell[2] = 6;
        ncell[3] = 8;

        triangulation = new Triangulation<dim>();
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {


            triangulation->clear();

            std::cout << "Generating hypercube for grid convergence... " << std::endl;
            GridGenerator::subdivided_hyper_cube(*triangulation, ncell[igrid]);
            //if (igrid == 0) {
            //    //GridGenerator::hyper_cube (*triangulation);
            //    GridGenerator::hyper_ball(*triangulation);
            //    //triangulation->refine_global (1);
            //} else {
            //    triangulation->refine_global (1);
            //}

            //IntegratorExplicit<dim,real> &integrator = new IntegratorExplicit<dim,real>();
            allocate_system_explicit ();

            assemble_system_explicit ();

            std::cout << "Cycle " << igrid 
                      << ". Number of active cells: " << triangulation->n_active_cells()
                      << ". Number of degrees of freedom: " << dof_handler.n_dofs()
                      << std::endl;

            double residual_norm = right_hand_side.l2_norm();
            typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end();

            double CFL = 0.1;
            double dx = 1.0/pow(triangulation->n_active_cells(),(1.0/dim));
            double speed = sqrt(dim);
            double dt = CFL * dx / speed;

            int iteration = 0;
            int print = 1000;
            while (residual_norm > 1e-13 && iteration < 100000) {
                ++iteration;

                assemble_system_explicit ();
                residual_norm = right_hand_side.l2_norm();

                if ( (iteration%print) == 0)
                std::cout << " Iteration: " << iteration 
                          << " Residual norm: " << residual_norm
                          << std::endl;

                solution += (right_hand_side*=dt);
            }
            delete_fe_values ();

            std::vector<unsigned int> dof_indices(fe.dofs_per_cell);

            QGauss<dim> quad_plus10(fe.degree+10);
            const unsigned int n_quad_pts =quad_plus10.size();
            FEValues<dim,dim> fe_values_plus10(mapping, fe,quad_plus10, update_values | update_JxW_values | update_quadrature_points);

            std::vector<double> solution_values(n_quad_pts);

            double l2error = 0;
            for (; cell!=endc; ++cell) {
                //const unsigned int icell = cell->user_index();

                fe_values_plus10.reinit (cell);
                fe_values_plus10.get_function_values (solution, solution_values);

                double uexact = 0;
                for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    const Point<dim> qpoint = (fe_values_plus10.quadrature_point(iquad));
                    if (dim==1) uexact = sin(3.19/dim*qpoint(0));
                    if (dim==2) uexact = sin(3.19/dim*qpoint(0))*sin(3.19/dim*qpoint(1));
                    if (dim==3) uexact = sin(3.19/dim*qpoint(0))*sin(3.19/dim*qpoint(1))*sin(3.19/dim*qpoint(2));

                    double u_at_q = solution_values[iquad];
                    l2error += pow(u_at_q - uexact, 2) * fe_values_plus10.JxW(iquad);
                }

            }
            l2error = sqrt(l2error);

            grid_size[igrid] = dx;
            error[igrid] = l2error;


            std::cout   << " dx: " << dx 
                        << " l2error: " << l2error
                        << " residual: " << residual_norm
                        << std::endl;

            if (igrid > 0)
            std::cout << "From grid " << igrid-1
                      << "  to grid " << igrid
                      << "  e1 " << error[igrid-1]
                      << "  e2 " << error[igrid]
                      << "  dimension: " << dim
                      << "  polynomial degree p: " << fe.get_degree()
                      << "  slope " << log((error[igrid]/error[igrid-1]))
                                      / log(grid_size[igrid]/grid_size[igrid-1])
                      << std::endl;

            //output_results (igrid);
        }
        delete triangulation;
        triangulation = NULL;

        std::cout << std::endl << std::endl;
        for (unsigned int igrid=0; igrid<n_grids-1; ++igrid) {

            const double slope = log(error[igrid+1]/error[igrid])
                                  / log(grid_size[igrid+1]/grid_size[igrid]);
            std::cout
                      << "From grid " << igrid
                      << "  to grid " << igrid+1
                      << "  e1 " << error[igrid]
                      << "  e2 " << error[igrid+1]
                      << "  dimension: " << dim
                      << "  polynomial degree p: " << fe.get_degree()
                      << "  slope " << slope
                      << std::endl;

        }
        std::cout << std::endl << std::endl;


        const double last_slope = log(error[n_grids-1]/error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        const double expected_slope = fe.get_degree()+1;
        const double slope_diff = std::abs(last_slope-expected_slope);
        const double slope_tolerance = 0.1;

        if (slope_diff > slope_tolerance) {
            std::cout << std::endl
                      << "Convergence order not achieved. Slope of "
                      << last_slope << " instead of expected "
                      << expected_slope << " within a tolerance of "
                      << slope_tolerance
                      << std::endl;
            return 1;
        }

        return 0;

    }

    template <int dim, typename real>
    std::pair<unsigned int, double>
    DiscontinuousGalerkin<dim, real>::solve_linear(Vector<real> &newton_update)
    {
        //SolverControl solver_control(1, 0);
        //TrilinosWrappers::SolverDirect::AdditionalData data(true);
        ////TrilinosWrappers::SolverDirect::AdditionalData data(parameters.output == Parameters::Solver::verbose);
        //TrilinosWrappers::SolverDirect direct(solver_control, data);

        ////system_matrix.print(std::cout, true);
        ////newton_update.print(std::cout);
        ////right_hand_side.print(std::cout);
        //direct.solve(system_matrix, newton_update, right_hand_side);
        //return {solver_control.last_step(), solver_control.last_value()};

        {
          Epetra_Vector x(View,
                          system_matrix.trilinos_matrix().DomainMap(),
                          newton_update.begin());
          Epetra_Vector b(View,
                          system_matrix.trilinos_matrix().RangeMap(),
                          right_hand_side.begin());
          AztecOO solver;
          solver.SetAztecOption(
            AZ_output,
            (true ? AZ_none : AZ_all));
          solver.SetAztecOption(AZ_solver, AZ_gmres);
          solver.SetRHS(&b);
          solver.SetLHS(&x);
          solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
          solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
          solver.SetAztecOption(AZ_overlap, 0);
          solver.SetAztecOption(AZ_reorder, 0);

          const double 
            ilut_drop = 1e-10,
            ilut_rtol = 1.1,
            ilut_atol = 1e-9,
            linear_residual = 1e-4;
          const int 
            ilut_fill = 2,
            max_iterations = 1000
            ;

          //solver.SetAztecParam(AZ_drop, parameters.ilut_drop);
          //solver.SetAztecParam(AZ_ilut_fill, parameters.ilut_fill);
          //solver.SetAztecParam(AZ_athresh, parameters.ilut_atol);
          //solver.SetAztecParam(AZ_rthresh, parameters.ilut_rtol);
          //solver.SetUserMatrix(
          //  const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix()));
          //solver.Iterate(parameters.max_iterations,
          //               parameters.linear_residual);
          solver.SetAztecParam(AZ_drop, ilut_drop);
          solver.SetAztecParam(AZ_ilut_fill, ilut_fill);
          solver.SetAztecParam(AZ_athresh, ilut_atol);
          solver.SetAztecParam(AZ_rthresh, ilut_rtol);
          solver.SetUserMatrix(
            const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix()));
          solver.Iterate(max_iterations,
                         linear_residual);
          return {solver.NumIters(), solver.TrueResidual()};
        }
    }

    template <int dim, typename real>
    int DiscontinuousGalerkin<dim, real>::grid_convergence_implicit () 
    {

        unsigned int n_grids = 5;
        std::vector<double> error(n_grids);
        std::vector<double> grid_size(n_grids);
        std::vector<double> ncell(n_grids);

        ncell[0] = 2;
        ncell[1] = 4;
        ncell[2] = 6;
        ncell[3] = 8;
        ncell[4] = 10;

        ncell[0] = 2;
        for (int i=1;i<n_grids;++i) {
            ncell[i] = ncell[i-1]*1.5;
        }


        triangulation = new Triangulation<dim>();
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {


            triangulation->clear();

            std::cout << "Generating hypercube for grid convergence... " << std::endl;
            GridGenerator::subdivided_hyper_cube(*triangulation, ncell[igrid]);
            //if (igrid == 0) {
            //    //GridGenerator::hyper_cube (*triangulation);
            //    GridGenerator::hyper_ball(*triangulation);
            //    //triangulation->refine_global (1);
            //} else {
            //    triangulation->refine_global (1);
            //}

            //IntegratorExplicit<dim,real> &integrator = new IntegratorExplicit<dim,real>();
            allocate_system_implicit ();

            assemble_system_implicit ();

            std::cout << "Cycle " << igrid 
                      << ". Number of active cells: " << triangulation->n_active_cells()
                      << ". Number of degrees of freedom: " << dof_handler.n_dofs()
                      << std::endl;

            double residual_norm = right_hand_side.l2_norm();
            typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end();

            double CFL = 0.1;
            double dx = 1.0/pow(triangulation->n_active_cells(),(1.0/dim));
            double speed = sqrt(dim);
            double dt = CFL * dx / speed;

            int iteration = 0;
            int print = 1;
            while (residual_norm > 1e-13 && iteration < 100000) {
                ++iteration;


                right_hand_side = 0;
                assemble_system_implicit ();
                residual_norm = right_hand_side.l2_norm();

                if ( (iteration%print) == 0)
                std::cout << " Iteration: " << iteration 
                          << " Residual norm: " << residual_norm
                          << std::endl;

                newton_update = 0;
                std::pair<unsigned int, double> convergence = solve_linear(newton_update);

                solution += newton_update;
            }
            delete_fe_values ();

            std::vector<unsigned int> dof_indices(fe.dofs_per_cell);

            QGauss<dim> quad_plus10(fe.degree+10);
            const unsigned int n_quad_pts =quad_plus10.size();
            FEValues<dim,dim> fe_values_plus10(mapping, fe,quad_plus10, update_values | update_JxW_values | update_quadrature_points);

            std::vector<double> solution_values(n_quad_pts);

            double l2error = 0;
            for (; cell!=endc; ++cell) {
                //const unsigned int icell = cell->user_index();

                fe_values_plus10.reinit (cell);
                fe_values_plus10.get_function_values (solution, solution_values);

                double uexact = 0;
                for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    const Point<dim> qpoint = (fe_values_plus10.quadrature_point(iquad));
                    if (dim==1) uexact = sin(3.19/dim*qpoint(0));
                    if (dim==2) uexact = sin(3.19/dim*qpoint(0))*sin(3.19/dim*qpoint(1));
                    if (dim==3) uexact = sin(3.19/dim*qpoint(0))*sin(3.19/dim*qpoint(1))*sin(3.19/dim*qpoint(2));

                    double u_at_q = solution_values[iquad];
                    l2error += pow(u_at_q - uexact, 2) * fe_values_plus10.JxW(iquad);
                }

            }
            l2error = sqrt(l2error);

            grid_size[igrid] = dx;
            error[igrid] = l2error;


            std::cout   << " dx: " << dx 
                        << " l2error: " << l2error
                        << " residual: " << residual_norm
                        << std::endl;

            if (igrid > 0)
            std::cout << "From grid " << igrid-1
                      << "  to grid " << igrid
                      << "  e1 " << error[igrid-1]
                      << "  e2 " << error[igrid]
                      << "  dimension: " << dim
                      << "  polynomial degree p: " << fe.get_degree()
                      << "  slope " << log((error[igrid]/error[igrid-1]))
                                      / log(grid_size[igrid]/grid_size[igrid-1])
                      << std::endl;

            //output_results (igrid);
        }
        delete triangulation;
        triangulation = NULL;

        std::cout << std::endl << std::endl;
        for (unsigned int igrid=0; igrid<n_grids-1; ++igrid) {

            const double slope = log(error[igrid+1]/error[igrid])
                                  / log(grid_size[igrid+1]/grid_size[igrid]);
            std::cout
                      << "From grid " << igrid
                      << "  to grid " << igrid+1
                      << "  e1 " << error[igrid]
                      << "  e2 " << error[igrid+1]
                      << "  dimension: " << dim
                      << "  polynomial degree p: " << fe.get_degree()
                      << "  slope " << slope
                      << std::endl;

        }
        std::cout << std::endl << std::endl;


        const double last_slope = log(error[n_grids-1]/error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        const double expected_slope = fe.get_degree()+1;
        const double slope_diff = std::abs(last_slope-expected_slope);
        const double slope_tolerance = 0.1;

        if (slope_diff > slope_tolerance) {
            std::cout << std::endl
                      << "Convergence order not achieved. Slope of "
                      << last_slope << " instead of expected "
                      << expected_slope << " within a tolerance of "
                      << slope_tolerance
                      << std::endl;
            return 1;
        }

        return 0;

    }

} // end of PHiLiP namespace
