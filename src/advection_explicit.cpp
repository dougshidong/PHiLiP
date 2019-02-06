#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include "advection_explicit.h"
#include "advection_boundary.h"
namespace PHiLiP
{
    using namespace dealii;

    // Constructors definition
    template <int dim, typename real>
    PDE<dim, real>::PDE()
        :
        mapping(1)
        , fe(1)
        , dof_handler(triangulation)
    {}
    template PDE<1, double>::PDE();
    template PDE<2, double>::PDE();
    template PDE<3, double>::PDE();


    template <int dim, typename real>
    PDE<dim, real>::PDE(const unsigned int polynomial_order)
        :
        mapping(polynomial_order+1)
        , fe(polynomial_order)
        , dof_handler(triangulation)
    {}
    template PDE<1, double>::PDE(const unsigned int);
    template PDE<2, double>::PDE(const unsigned int);
    template PDE<3, double>::PDE(const unsigned int);

    
    template <int dim, typename real>
    void PDE<dim, real>::compute_inv_mass_matrix()
    {
        unsigned int fe_degree = fe.get_degree();
        QGauss<dim> quadrature(fe_degree+1);
        unsigned int n_quad_pts = quadrature.size();
        FEValues<dim> fe_values(mapping, fe, quadrature, update_values | update_JxW_values);
        
        std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
        
        typename DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active(),
           endc = dof_handler.end();
        
        // Allocate inverse mass matrices
        inv_mass_matrix.resize(triangulation.n_active_cells(),
                               FullMatrix<real>(fe.dofs_per_cell));

        for (; cell!=endc; ++cell) {

            const unsigned int icell = cell->user_index();
            cell->get_dof_indices (dof_indices);
            fe_values.reinit(cell);

            for(unsigned int idof=0; idof<fe.dofs_per_cell; ++idof) {
            for(unsigned int jdof=0; jdof<fe.dofs_per_cell; ++jdof) {

                inv_mass_matrix[icell][idof][jdof] = 0.0;

                for(unsigned int q=0; q<n_quad_pts; ++q) {
                    inv_mass_matrix[icell][idof][jdof] +=
                        fe_values.shape_value(idof,q) *
                        fe_values.shape_value(jdof,q) *
                        fe_values.JxW(q);
                }
            }
            }

            // Invert mass matrix
            inv_mass_matrix[icell].gauss_jordan();
        }

    }
    template <int dim, typename real>
    void PDE<dim, real>::setup_system ()
    {
        // This function allocates all the necessary memory to the 
        // system matrices and vectors.


        //// Allocates memory from triangulation and finite element space
        dof_handler.distribute_dofs(fe);

        std::cout << "Creating mass matrices... \n";
        compute_inv_mass_matrix();

        //// Generate sparsity pattern
        //// This function is a variation of the make_sparsity_pattern()
        //// functions above in that it assumes that the bilinear form you
        //// want to use to generate the matrix also contains terms
        //// that integrate over the faces between cells
        //// (i.e., it contains "fluxes" between cells, 
        ////        explaining the name of the function).
        //// Basically, make_sparsity_pattern, but for DG
        //DynamicSparsityPattern dsp(dof_handler.n_dofs());
        //DoFTools::make_sparsity_pattern(dof_handler, dsp);
        //sparsity_pattern.copy_from(dsp);
        //system_matrix.reinit(sparsity_pattern);

        // Allocate vectors
        solution.reinit(dof_handler.n_dofs());
        right_hand_side.reinit(dof_handler.n_dofs());
    }
  
    // For now hard-code advection speed
    template <int dim>
    Tensor<1,dim> velocity_field (const Point<dim> &p)
    {
        Point<dim> v_field;
        //Assert (dim >= 2, ExcNotImplemented());
        v_field(0) = p(1);
        v_field(1) = p(0);
        //v_field /= v_field.norm();
        v_field(0) = 1.0;
        v_field(1) = 1.0;
        return v_field;
    }
    // For now hard-code source term
    template <int dim>
    double source_term (const Point<dim> &p)
    {
        double source;
        source = 1.0;
        return source;
    }


    template <int dim, typename real>
    void PDE<dim, real>::integrate_cell_terms(DoFInfo &dof_info, CellInfo &cell_info)
    {
        const FEValuesBase<dim> &fe_values = cell_info.fe_values();
        const std::vector<real> &JxW = fe_values.get_JxW_values ();

        Vector<real> &local_vector = dof_info.vector(0).block(0);
        std::vector<unsigned int> &dof_indices = dof_info.indices;

        const unsigned int n_quad_pts = fe_values.n_quadrature_points;
        const unsigned int n_dofs_cell     = fe_values.dofs_per_cell;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const Tensor<1,dim> vel_at_point =
                velocity_field (fe_values.quadrature_point(iquad));

            const double source_at_point =
                source_term (fe_values.quadrature_point(iquad));

            for (unsigned int i_test=0; i_test<n_dofs_cell; ++i_test) {

                const double adv_dot_grad_test = 
                    vel_at_point*fe_values.shape_grad(i_test, iquad);

                for (unsigned int i_trial=0; i_trial<n_dofs_cell; ++i_trial) {
                    // Stiffness matrix contibution
                    local_vector(i_test) += 
                        -adv_dot_grad_test *
                        solution(dof_indices[i_trial]) *
                        fe_values.shape_value(i_trial,iquad) *
                        JxW[iquad];
                }
                // Source term contribution
                local_vector(i_test) += 
                    -source_at_point *
                    fe_values.shape_value(i_test,iquad) *
                    JxW[iquad];
            }
        }
    }
    template void PDE<1, double>::integrate_cell_terms(DoFInfo &dof_info, CellInfo &cell_info);
    template void PDE<2, double>::integrate_cell_terms(DoFInfo &dof_info, CellInfo &cell_info);
    template void PDE<3, double>::integrate_cell_terms(DoFInfo &dof_info, CellInfo &cell_info);


    template <int dim, typename real>
    void PDE<dim, real>::integrate_boundary_terms(DoFInfo &dof_info, CellInfo &face_info)
    {
        const FEValuesBase<dim> &fe_face_values = face_info.fe_values();
        Vector<real> &local_vector = dof_info.vector(0).block(0);
        std::vector<unsigned int> &dof_indices = dof_info.indices;

        const std::vector<real> &JxW = fe_face_values.get_JxW_values ();
        const std::vector<Tensor<1,dim> > &normals = fe_face_values.get_normal_vectors ();

        std::vector<real> boundary_values(fe_face_values.n_quadrature_points);

        static AdvectionBoundary<dim> boundary_function;
        boundary_function.value_list (fe_face_values.get_quadrature_points(), boundary_values);

        const unsigned int n_quad_pts = fe_face_values.n_quadrature_points;
        const unsigned int n_dofs_cell = fe_face_values.dofs_per_cell;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const real vel_dot_normal = 
                velocity_field(fe_face_values.quadrature_point(iquad)) * normals[iquad];

            const bool inflow = (vel_dot_normal < 0.);
            if (inflow) {
                // Setting the boundary condition when inflow
                for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                    local_vector(itest) += -vel_dot_normal *
                                       boundary_values[iquad] *
                                       fe_face_values.shape_value(itest,iquad) *
                                       JxW[iquad];
                }
            } else {
                // "Numerical flux" at the boundary is the same as the analytical flux
                for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                    for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                        local_vector(itest) += vel_dot_normal *
                                             fe_face_values.shape_value(itest,iquad) *
                                             fe_face_values.shape_value(itrial,iquad) *
                                             solution(dof_indices[itrial]) *
                                             JxW[iquad];
                    }
                }
            }
         }
    }
    template void PDE<1, double>::integrate_boundary_terms(DoFInfo &dof_info, CellInfo &face_info);
    template void PDE<2, double>::integrate_boundary_terms(DoFInfo &dof_info, CellInfo &face_info);
    template void PDE<3, double>::integrate_boundary_terms(DoFInfo &dof_info, CellInfo &face_info);

    template <int dim, typename real>
    void PDE<dim, real>::integrate_face_terms(
        DoFInfo &dof_info1, DoFInfo &dof_info2, CellInfo &face_info1, CellInfo &face_info2)
    {

        Vector<real> &local_vector1              = dof_info1.vector(0).block(0);
        Vector<real> &local_vector2              = dof_info2.vector(0).block(0);

        std::vector<unsigned int> &dof_indices1  = dof_info1.indices;
        std::vector<unsigned int> &dof_indices2  = dof_info2.indices;

        const FEValuesBase<dim> &fe_face_values1 = face_info1.fe_values();
        const FEValuesBase<dim> &fe_face_values2 = face_info2.fe_values();

        const unsigned int n_dofs_cell1 = fe_face_values1.dofs_per_cell;
        const unsigned int n_dofs_cell2 = fe_face_values2.dofs_per_cell;

        // Jacobian and normal should always be consistent between two elements
        // even for non-conforming meshes?
        const std::vector<real> &JxW1               = fe_face_values1.get_JxW_values();
        const std::vector<Tensor<1,dim> > &normals1 = fe_face_values1.get_normal_vectors();

        // Use quadrature points of first cell
        // Might want to use the maximum n_quad_pts1 and n_quad_pts2
        const unsigned int n_quad_pts  = fe_face_values1.n_quadrature_points;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const Tensor<1,dim,real> velocity_at_q =
                velocity_field(fe_face_values1.quadrature_point(iquad));

            real w1 = 0;
            real w2 = 0;
            // Interpolate solution to face
            for (unsigned int itrial=0; itrial<n_dofs_cell1; itrial++) {
                w1 += solution(dof_indices1[itrial]) * fe_face_values1.shape_value(itrial, iquad);
            }
            for (unsigned int itrial=0; itrial<n_dofs_cell2; itrial++) {
                w2 += solution(dof_indices2[itrial]) * fe_face_values2.shape_value(itrial, iquad);
            }
            const Tensor<1,dim,real> analytical_flux1 = velocity_at_q*w1;
            const Tensor<1,dim,real> analytical_flux2 = velocity_at_q*w2;
            const Tensor<1,dim,real> n1 = normals1[iquad];
                
            const real lambda = velocity_at_q * n1;

            const Tensor<1,dim> numerical_flux =
                (analytical_flux1 + analytical_flux2 + (n1*w1 - n1*w2) * lambda) * 0.5;

            const real normal1_numerical_flux = numerical_flux*n1;

            for (unsigned int itest=0; itest<n_dofs_cell1; ++itest) {
                local_vector1(itest) += // plus
                    fe_face_values1.shape_value(itest,iquad) *
                    normal1_numerical_flux *
                    JxW1[iquad];
            }
            for (unsigned int itest=0; itest<n_dofs_cell2; ++itest) {
                local_vector2(itest) -= // minus
                    fe_face_values2.shape_value(itest,iquad) *
                    normal1_numerical_flux *
                    JxW1[iquad];
            }
        }
    }
    template void PDE<1, double>::integrate_face_terms(
        DoFInfo &dof_info1, DoFInfo &dof_info2, CellInfo &face_info1, CellInfo &face_info2);
    template void PDE<2, double>::integrate_face_terms(
        DoFInfo &dof_info1, DoFInfo &dof_info2, CellInfo &face_info1, CellInfo &face_info2);
    template void PDE<3, double>::integrate_face_terms(
        DoFInfo &dof_info1, DoFInfo &dof_info2, CellInfo &face_info1, CellInfo &face_info2);

    template <int dim, typename real>
    void PDE<dim, real>::assemble_system ()
    {
        MeshWorker::IntegrationInfoBox<dim> info_box;
        // Using p+1 integration points
        const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;

        info_box.initialize_gauss_quadrature(n_gauss_points, n_gauss_points, n_gauss_points);

        info_box.initialize_update_flags();
        UpdateFlags update_flags = update_quadrature_points |
                                   update_values            |
                                   update_gradients;
        const bool update_cell     = true;
        const bool update_boundary = true;
        const bool update_face     = true;
        const bool update_neighbor = true;
        info_box.add_update_flags(
            update_flags, update_cell, update_boundary, update_face, update_neighbor);

        info_box.initialize(fe, mapping);

        MeshWorker::DoFInfo<dim> dof_info(dof_handler);

        //MeshWorker::Assembler::SystemSimple< SparseMatrix<real>, Vector<real> > assembler;
        MeshWorker::Assembler::ResidualSimple< Vector<real> > assembler;

        //assembler.initialize(system_matrix, right_hand_side);
        AnyData rhs;
        Vector<real>* data = &right_hand_side;
        rhs.add<Vector<real>*> (data, "RHS");
        assembler.initialize(rhs);

        MeshWorker::loop<
            dim, dim, 
            MeshWorker::DoFInfo<dim>, 
            MeshWorker::IntegrationInfoBox<dim>
            >
            (dof_handler.begin_active(),
             dof_handler.end(),
             dof_info,
             info_box,
             &PDE<dim, real>::integrate_cell_terms,
             &PDE<dim, real>::integrate_boundary_terms,
             &PDE<dim, real>::integrate_face_terms,
             assembler);
    }
    template void PDE<1, double>::assemble_system ();
    template void PDE<2, double>::assemble_system ();
    template void PDE<3, double>::assemble_system ();


    template <int dim, typename real>
    void PDE<dim, real>::run ()
    {
        for (unsigned int igrid_size=0; igrid_size<5; ++igrid_size)
        {
            std::cout << "Cycle " << igrid_size << std::endl;

            if (igrid_size == 0) {
                GridGenerator::hyper_cube (triangulation);
                triangulation.refine_global (2);
            } else {
                triangulation.refine_global (1);
            }

            std::cout << "Number of active cells:       "
                    << triangulation.n_active_cells()
                    << std::endl;

            setup_system ();

            std::cout << "Number of degrees of freedom: "
                    << dof_handler.n_dofs()
                    << std::endl;

            assemble_system ();
            //solve (solution);

            //output_results (igrid_size);
        }
    }
    template void PDE<1, double>::run ();
    template void PDE<2, double>::run ();
    template void PDE<3, double>::run ();





} // end of PHiLiP namespace
