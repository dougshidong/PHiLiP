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

#include "integrator.h"

#include "advection_explicit.h"
#include "advection_boundary.h"
namespace PHiLiP {
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


        //// Allocates memory from triangulation and finite element space
        dof_handler.distribute_dofs(fe);

        //std::cout << "Creating mass matrices... \n";
        //compute_inv_mass_matrix();

        // Allocate vectors
        solution.reinit(dof_handler.n_dofs());
        right_hand_side.reinit(dof_handler.n_dofs());
        source_term.reinit(dof_handler.n_dofs());

    }
    template <int dim, typename real>
    void PDE<dim, real>::setup_meshworker (
        IntegratorExplicit<dim,real> &integrator)
    {

        // Integration info
        // Using p+1 integration points
        const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;

        const unsigned int n_integration_cell  = n_gauss_points,
                           n_integration_boundary = n_gauss_points,
                           n_integration_face  = n_gauss_points;
        integrator.info_box.initialize_gauss_quadrature(
            n_integration_cell, n_integration_boundary, n_integration_face);

        integrator.info_box.initialize_update_flags ();
        integrator.info_box.add_update_flags_all (update_values |
                                                  update_quadrature_points |
                                                  update_JxW_values);
        integrator.info_box.add_update_flags_cell (update_gradients);
        integrator.info_box.add_update_flags_boundary (update_normal_vectors);
        integrator.info_box.add_update_flags_face (update_normal_vectors);

        integrator.info_box.initialize(fe, mapping);

        AnyData rhs;
        Vector<real>* data = &right_hand_side;
        rhs.add<Vector<real>*> (data, "RHS");
        integrator.assembler.initialize(rhs);
    }
  
    // For now hard-code advection speed
    template <int dim>
    Tensor<1,dim> velocity_field ()
    {
        Tensor<1,dim> v_field;
        v_field[0] = 1.0;
        if(dim >= 2) v_field[1] = 1.0;
        if(dim >= 3) v_field[2] = 1.0;
        return v_field;
    }
    template <int dim>
    Tensor<1,dim> velocity_field (const Point<dim> &p)
    {
        Tensor<1,dim> v_field;
        //Assert (dim >= 2, ExcNotImplemented());
        //v_field(0) = p(1);
        //v_field(1) = p(0);
        //v_field /= v_field.norm();
        v_field = p;
        return v_field;
    }
    // For now hard-code source term
    template <int dim>
    double evaluate_source_term (const Point<dim> &p)
    {
        double source;
        source = 1.0;
        if (dim==1) source = cos(p(0));
        if (dim==2) source = cos(p(0))*sin(p(1)) + sin(p(0))*cos(p(1));
        if (dim==3) source =   cos(p(0))*sin(p(1))*sin(p(2))
                             + sin(p(0))*cos(p(1))*sin(p(2))
                             + sin(p(0))*sin(p(1))*cos(p(2));
        return source;
    }


    template <int dim, typename real>
    void PDE<dim, real>::integrate_cell_terms(
        DoFInfo &dof_info,
        CellInfo &cell_info)
    {
        const FEValuesBase<dim> &fe_values = cell_info.fe_values();
        const std::vector<real> &JxW = fe_values.get_JxW_values ();

        Vector<real> &local_vector = dof_info.vector(0).block(0);
        std::vector<unsigned int> &dof_indices = dof_info.indices;

        const unsigned int n_quad_pts      = fe_values.n_quadrature_points;
        const unsigned int n_dofs_cell     = fe_values.dofs_per_cell;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const Tensor<1,dim> vel_at_point =
                velocity_field<dim>();
                //velocity_field (fe_values.quadrature_point(iquad));

            const double source_at_point =
                evaluate_source_term (fe_values.quadrature_point(iquad));

            for (unsigned int i_test=0; i_test<n_dofs_cell; ++i_test) {

                const double adv_dot_grad_test = 
                    vel_at_point*fe_values.shape_grad(i_test, iquad);

                for (unsigned int i_trial=0; i_trial<n_dofs_cell; ++i_trial) {
                    // Stiffness matrix contibution
                    local_vector(i_test) += 
                        adv_dot_grad_test *
                        solution(dof_indices[i_trial]) *
                        fe_values.shape_value(i_trial,iquad) *
                        JxW[iquad];
                }
                // Source term contribution
                local_vector(i_test) += 
                    source_at_point *
                    fe_values.shape_value(i_test,iquad) *
                    JxW[iquad];
            }
        }
    }

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
        const unsigned int dummy = 0; // Virtual function that requires 3 arguments
        boundary_function.value_list (fe_face_values.get_quadrature_points(), boundary_values, dummy);

        const unsigned int n_quad_pts = fe_face_values.n_quadrature_points;
        const unsigned int n_dofs_cell = fe_face_values.dofs_per_cell;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const real vel_dot_normal = 
                velocity_field<dim> () * normals[iquad];
                //velocity_field(fe_face_values.quadrature_point(iquad)) * normals[iquad];

            const bool inflow = (vel_dot_normal < 0.);
            if (inflow) {
                // Setting the boundary condition when inflow
                for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                    local_vector(itest) += 
                        -vel_dot_normal *
                        boundary_values[iquad] *
                        fe_face_values.shape_value(itest,iquad) *
                        JxW[iquad];
                }
            } else {
                // "Numerical flux" at the boundary is the same as the analytical flux
                for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                    for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                        local_vector(itest) += 
                            -vel_dot_normal *
                            fe_face_values.shape_value(itest,iquad) *
                            fe_face_values.shape_value(itrial,iquad) *
                            solution(dof_indices[itrial]) *
                            JxW[iquad];
                    }
                }
            }
         }
    }

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
                velocity_field<dim>();
                //velocity_field(fe_face_values1.quadrature_point(iquad));

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
                local_vector1(itest) -= // plus
                    fe_face_values1.shape_value(itest,iquad) *
                    normal1_numerical_flux *
                    JxW1[iquad];
            }
            for (unsigned int itest=0; itest<n_dofs_cell2; ++itest) {
                local_vector2(itest) += // minus
                    fe_face_values2.shape_value(itest,iquad) *
                    normal1_numerical_flux *
                    JxW1[iquad];
            }
        }
    }

    template <int dim, typename real>
    void PDE<dim, real>::assemble_system (IntegratorExplicit<dim,real> &integrator)
    {
        MeshWorker::loop<
            dim, dim, 
            MeshWorker::DoFInfo<dim>, 
            MeshWorker::IntegrationInfoBox<dim>
            >
            (dof_handler.begin_active(),
             dof_handler.end(),
             integrator.dof_info,
             integrator.info_box,
             //&PDE<dim, real>::integrate_cell_terms,
             //&PDE<dim, real>::integrate_boundary_terms,
             //&PDE<dim, real>::integrate_face_terms,
             boost::bind(&PDE<dim, real>::integrate_cell_terms,
                    this, _1, _2),
             boost::bind(&PDE<dim, real>::integrate_boundary_terms,
                    this, _1, _2),
             boost::bind(&PDE<dim, real>::integrate_face_terms,
                    this, _1, _2, _3, _4),
             integrator.assembler);
    }
    //template void PDE<0, double>::assemble_system ();
    //template void PDE<2, double>::assemble_system ();
    //template void PDE<3, double>::assemble_system ();


    template <int dim, typename real>
    int PDE<dim, real>::run ()
    {
        unsigned int n_grids = 4;
        std::vector<double> error(n_grids);
        std::vector<double> grid_size(n_grids);
        std::vector<double> ncell(n_grids);

        ncell[0] = 2;
        ncell[1] = 4;
        ncell[2] = 6;
        ncell[3] = 8;

        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

            std::cout << "Cycle " << igrid << std::endl;

            triangulation.clear();
            GridGenerator::subdivided_hyper_cube(triangulation, ncell[igrid]);
            //if (igrid == 0) {
            //    //GridGenerator::hyper_cube (triangulation);
            //    GridGenerator::hyper_ball(triangulation);
            //    //triangulation.refine_global (1);
            //} else {
            //    triangulation.refine_global (1);
            //}

            //IntegratorExplicit<dim,real> &integrator = new IntegratorExplicit<dim,real>();
            setup_system ();

            IntegratorExplicit<dim,real> integrator(dof_handler);

            setup_meshworker (integrator);

            assemble_system(integrator);

            std::cout << "Number of active cells:       "
                      << triangulation.n_active_cells()
                      << std::endl;

            std::cout << "Number of degrees of freedom: "
                      << dof_handler.n_dofs()
                      << std::endl;

            double residual_norm = right_hand_side.l2_norm();
            typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end();

            double CFL = 0.1;
            double dx = 1.0/pow(triangulation.n_active_cells(),(1.0/dim));
            double speed = sqrt(dim);
            double dt = CFL * dx / speed;

            std::cout   << " Speed: " << speed 
                        << " dx: " << dx 
                        << std::endl;

            int iteration = 0;
            int print = 1000;
            while (residual_norm > 1e-13 && iteration < 100000) {
                ++iteration;

                assemble_system(integrator);
                residual_norm = right_hand_side.l2_norm();

                if ( (iteration%print) == 0)
                std::cout << " Iteration: " << iteration 
                          << " Residual norm: " << residual_norm
                          << std::endl;

                solution += (right_hand_side*=dt);
                //std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
                //for (; cell!=endc; ++cell)
                //{
                //    const unsigned int icell = cell->user_index();

                //    cell->get_dof_indices (dof_indices);
                //    solution += (right_hand_side*=dt);
                //}
            }

            std::vector<unsigned int> dof_indices(fe.dofs_per_cell);

            QGauss<dim>   quadrature(fe.degree+10);
            const unsigned int n_quad_pts = quadrature.size();
            FEValues<dim> fe_values(mapping, fe, quadrature, update_values | update_JxW_values | update_quadrature_points);

            std::vector<double> solution_values(n_quad_pts);

            double l2error = 0;
            for (; cell!=endc; ++cell) {
                //const unsigned int icell = cell->user_index();

                fe_values.reinit (cell);
                fe_values.get_function_values (solution, solution_values);

                double uexact = 0;
                for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    const Point<dim> qpoint = (fe_values.quadrature_point(iquad));
                    if (dim==1) uexact = sin(qpoint(0));
                    if (dim==2) uexact = sin(qpoint(0))*sin(qpoint(1));
                    if (dim==3) uexact = sin(qpoint(0))*sin(qpoint(1))*sin(qpoint(2));

                    double u_at_q = solution_values[iquad];
                    l2error += pow(u_at_q - uexact, 2) * fe_values.JxW(iquad);
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
        std::cout << std::endl << std::endl;
        for (unsigned int igrid=0; igrid<n_grids-1; ++igrid) {
            std::cout
                      << "From grid " << igrid
                      << "  to grid " << igrid+1
                      << "  e1 " << error[igrid]
                      << "  e2 " << error[igrid+1]
                      << "  dimension: " << dim
                      << "  polynomial degree p: " << fe.get_degree()
                      << "  slope " << log((error[igrid+1]/error[igrid]))
                                      / log(grid_size[igrid+1]/grid_size[igrid])
                      << std::endl;

        }
        std::cout << std::endl << std::endl;

        int error_tolerance = 0.1;
        if (std::abs(error[n_grids-1] - fe.get_degree()) < error_tolerance) return 1;

        return 0;

    }
    template int PDE<1, double>::run ();
    template int PDE<2, double>::run ();
    //template void PDE<3, double>::run ();





} // end of PHiLiP namespace
