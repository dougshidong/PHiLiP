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

#include "advection.h"
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
    void PDE<dim, real>::setup_system ()
    {
        // This function allocates all the necessary memory to the 
        // system matrices and vectors.


        // Allocates memory from triangulation and finite element space
        dof_handler.distribute_dofs(fe);

        // Generate sparsity pattern
        // This function is a variation of the make_sparsity_pattern()
        // functions above in that it assumes that the bilinear form you
        // want to use to generate the matrix also contains terms
        // that integrate over the faces between cells
        // (i.e., it contains "fluxes" between cells, 
        //        explaining the name of the function).
        // Basically, make_sparsity_pattern, but for DG
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

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
        //v_field(0) = -p(1);
        //v_field(1) = p(0);
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

        FullMatrix<real> &local_matrix = dof_info.matrix(0).matrix;
        Vector<real> &local_vector = dof_info.vector(0).block(0);

        for (unsigned int point=0; point<fe_values.n_quadrature_points; ++point) {

            const Tensor<1,dim> vel_at_point =
                velocity_field (fe_values.quadrature_point(point));

            const double source_at_point =
                source_term (fe_values.quadrature_point(point));

            for (unsigned int i_test=0; i_test<fe_values.dofs_per_cell; ++i_test) {
                for (unsigned int i_trial=0; i_trial<fe_values.dofs_per_cell; ++i_trial) {
                    // Stiffness matrix
                    const double adv_dot_grad_test = 
                        vel_at_point*fe_values.shape_grad(i_test, point);
                    local_matrix(i_test,i_trial) += 
                        -adv_dot_grad_test *
                        fe_values.shape_value(i_trial,point) *
                        JxW[point];
                }
                // Source term
                local_vector(i_test) += 
                    -source_at_point *
                    fe_values.shape_value(i_test,point) *
                    JxW[point];
            }
        }
    }
    //template void PDE<1, double>::integrate_cell_terms(DoFInfo &dof_info, CellInfo &cell_info);
    //template void PDE<2, double>::integrate_cell_terms(DoFInfo &dof_info, CellInfo &cell_info);
    //template void PDE<3, double>::integrate_cell_terms(DoFInfo &dof_info, CellInfo &cell_info);


    template <int dim, typename real>
    void PDE<dim, real>::integrate_boundary_terms(DoFInfo &dof_info, CellInfo &cell_info)
    {
        const FEValuesBase<dim> &fe_face_values = cell_info.fe_values();
        FullMatrix<real> &local_matrix = dof_info.matrix(0).matrix;
        Vector<real> &local_vector = dof_info.vector(0).block(0);

        const std::vector<real> &JxW = fe_face_values.get_JxW_values ();
        const std::vector<Tensor<1,dim> > &normals = fe_face_values.get_normal_vectors ();

        std::vector<real> boundary_values(fe_face_values.n_quadrature_points);

        static AdvectionBoundary<dim> boundary_function;
        boundary_function.value_list (fe_face_values.get_quadrature_points(), boundary_values);

        for (unsigned int point=0; point<fe_face_values.n_quadrature_points; ++point) {
            const real vel_dot_normal = 
                velocity_field(fe_face_values.quadrature_point(point)) * normals[point];

            const bool inflow = (vel_dot_normal < 0.);
            if (inflow) {
                for (unsigned int itest=0; itest<fe_face_values.dofs_per_cell; ++itest) {
                    local_vector(itest) += -vel_dot_normal *
                                       boundary_values[point] *
                                       fe_face_values.shape_value(itest,point) *
                                       JxW[point];
                }
            } else {
                for (unsigned int itest=0; itest<fe_face_values.dofs_per_cell; ++itest) {
                    for (unsigned int itrial=0; itrial<fe_face_values.dofs_per_cell; ++itrial) {
                        local_matrix(itest,itrial) += vel_dot_normal *
                                             fe_face_values.shape_value(itrial,point) *
                                             fe_face_values.shape_value(itest,point) *
                                             JxW[point];
                    }
                }
            }
         }
    }
    //template void PDE<1, double>::integrate_boundary_terms
    //    (DoFInfo &dof_info, CellInfo &cell_info);
    //template void PDE<2, double>::integrate_boundary_terms
    //    (DoFInfo &dof_info, CellInfo &cell_info);
    //template void PDE<3, double>::integrate_boundary_terms
    //    (DoFInfo &dof_info, CellInfo &cell_info);


    //template <int dim, typename real>
    //void PDE<dim, real>::integrate_face_terms(
    //    DoFInfo &dinfo1, DoFInfo &dinfo2, CellInfo &info1, CellInfo &info2)
    //{
    //}
    //template void PDE<1, double>::integrate_face_terms
    //    (DoFInfo &dinfo1, DoFInfo &dinfo2, CellInfo &info1, CellInfo &info2);
    //template void PDE<2, double>::integrate_face_terms
    //    (DoFInfo &dinfo1, DoFInfo &dinfo2, CellInfo &info1, CellInfo &info2);
    //template void PDE<3, double>::integrate_face_terms
    //    (DoFInfo &dinfo1, DoFInfo &dinfo2, CellInfo &info1, CellInfo &info2);

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

        MeshWorker::Assembler::SystemSimple< SparseMatrix<real>, Vector<real> > assembler;

        assembler.initialize(system_matrix, right_hand_side);

        //MeshWorker::loop< dim, dim,
        //    MeshWorker ::DoFInfo<dim>, MeshWorker ::IntegrationInfoBox<dim> >
        //    (dof_handler.begin_active(), dof_handler.end(),
        //     dof_info, info_box,
        //     &PDE<dim, real>::integrate_cell_terms,
        //     &PDE<dim, real>::integrate_boundary_terms,
        //     &PDE<dim, real>::integrate_face_terms,
        //     assembler);
    }


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
