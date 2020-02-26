#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>

#include "euler_bump_optimization.h"

#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "functional/target_functional.h"


namespace PHiLiP {
namespace Tests {

///** Target pressure along bottom boundary.
// *  Zero out the default volume contribution and evaluate L2-error of pressure on the bottom surface.
// */
//template <int dim, int nstate, typename real>
//class TargetBoundaryPressure : public TargetFunctional<dim, nstate, real>
//{
//    using TargetFunctional<dim,nstate,real>::dg;
//    using TargetFunctional<dim,nstate,real>::target_solution;
//    using TargetFunctional<dim,nstate,real>::dIdw;
//    using TargetFunctional<dim,nstate,real>::target_solution;
//public:
//    /// Constructor
//    TargetBoundaryPressure(
//        std::shared_ptr<DGBase<dim,real>> dg_input,
//		const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
//        const bool uses_solution_values = true,
//        const bool uses_solution_gradient = true)
//	: TargetFunctional<dim,nstate,real>(dg_input, target_solution, uses_solution_values, uses_solution_gradient)
//	{}
//
//    real evaluate_functional(
//        const bool compute_dIdW,
//        const bool compute_dIdX,
//        const bool compute_d2I)
//    {
//        using ADtype = Sacado::Fad::DFad<real>;
//        using ADADtype = Sacado::Fad::DFad<ADtype>;
//        // Returned value
//        real local_functional = 0.0;
//
//        const Parameters::AllParameters &param = dg->all_parameters;
//        Physics::Euler<dim,nstate,ADADtype> euler_physics
//            = Physics::Euler<dim, nstate, ADADtype>(
//                    param.euler_param.ref_length,
//                    param.euler_param.gamma_gas,
//                    param.euler_param.mach_inf,
//                    param.euler_param.angle_of_attack,
//                    param.euler_param.side_slip_angle);
//
//        // for taking the local derivatives
//        const dealii::FESystem<dim,dim> &fe_metric = dg->high_order_grid.fe_system;
//        const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
//        std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);
//
//        // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
//        const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
//        std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
//        std::vector<ADADtype> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
//        std::vector<real> target_soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
//        std::vector<real>   local_dIdw(max_dofs_per_cell);
//        std::vector<real>   local_dIdX(n_metric_dofs_cell);
//
//        const auto mapping = (*(dg->high_order_grid.mapping_fe_field));
//        dealii::hp::MappingCollection<dim> mapping_collection(mapping);
//
//        dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg->fe_collection, dg->face_quadrature_collection,   this->face_update_flags);
//
//        this->allocate_derivatives(compute_dIdW, compute_dIdX, compute_d2I);
//
//        dg->solution.update_ghost_values();
//        auto metric_cell = dg->high_order_grid.dof_handler_grid.begin_active();
//        auto soln_cell = dg->dof_handler.begin_active();
//        for( ; soln_cell != dg->dof_handler.end(); ++soln_cell, ++metric_cell) {
//            if(!soln_cell->is_locally_owned()) continue;
//
//            // setting up the volume integration
//            const unsigned int i_mapp = 0; // *** ask doug if this will ever be 
//            const unsigned int i_fele = soln_cell->active_fe_index();
//            const unsigned int i_quad = i_fele;
//            (void) i_mapp;
//
//            // Get solution coefficients
//            const dealii::FESystem<dim,dim> &fe_solution = dg->fe_collection[i_fele];
//            const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
//            cell_soln_dofs_indices.resize(n_soln_dofs_cell);
//            soln_cell->get_dof_indices(cell_soln_dofs_indices);
//            soln_coeff.resize(n_soln_dofs_cell);
//            target_soln_coeff.resize(n_soln_dofs_cell);
//
//            // Get metric coefficients
//            metric_cell->get_dof_indices (cell_metric_dofs_indices);
//            std::vector< ADADtype > coords_coeff(n_metric_dofs_cell);
//            for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
//                coords_coeff[idof] = dg->high_order_grid.nodes[cell_metric_dofs_indices[idof]];
//            }
//
//            // Setup automatic differentiation
//            unsigned int n_total_indep = 0;
//            if (compute_dIdW || compute_d2I) n_total_indep += n_soln_dofs_cell;
//            if (compute_dIdX || compute_d2I) n_total_indep += n_metric_dofs_cell;
//            unsigned int i_derivative = 0;
//            for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
//                const real val = dg->solution[cell_soln_dofs_indices[idof]];
//                soln_coeff[idof] = val;
//                if (compute_dIdW || compute_d2I) soln_coeff[idof].diff(i_derivative++, n_total_indep);
//            }
//            for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
//                const real val = target_solution[cell_soln_dofs_indices[idof]];
//                target_soln_coeff[idof] = val;
//            }
//            for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
//                const real val = dg->high_order_grid.nodes[cell_metric_dofs_indices[idof]];
//                coords_coeff[idof] = val;
//                if (compute_dIdX || compute_d2I) coords_coeff[idof].diff(i_derivative++, n_total_indep);
//            }
//            AssertDimension(i_derivative, n_total_indep);
//            if (compute_d2I) {
//                unsigned int i_derivative = 0;
//                for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
//                    const real val = dg->solution[cell_soln_dofs_indices[idof]];
//                    soln_coeff[idof].val() = val;
//                    soln_coeff[idof].val().diff(i_derivative++, n_total_indep);
//                }
//                for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
//                    const real val = dg->high_order_grid.nodes[cell_metric_dofs_indices[idof]];
//                    coords_coeff[idof].val() = val;
//                    coords_coeff[idof].val().diff(i_derivative++, n_total_indep);
//                }
//            }
//            AssertDimension(i_derivative, n_total_indep);
//
//            // Get quadrature point on reference cell
//            const dealii::Quadrature<dim> &volume_quadrature = dg->volume_quadrature_collection[i_quad];
//
//            // Evaluate integral on the cell volume
//            ADADtype volume_local_sum = evaluate_volume_cell_functional(*physics_fad_fad, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
//
//            // std::cout << "volume_local_sum.val().val() : " <<  volume_local_sum.val().val() << std::endl;
//
//            // next looping over the faces of the cell checking for boundary elements
//            for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
//                auto face = soln_cell->face(iface);
//                
//                if(face->at_boundary()){
//                    const dealii::Quadrature<dim-1> &used_face_quadrature = dg->face_quadrature_collection[i_quad]; // or i_quad
//                    const dealii::Quadrature<dim> face_quadrature = dealii::QProjector<dim>::project_to_face(used_face_quadrature,iface);
//
//                    volume_local_sum += evaluate_face_cell_functional(*physics_fad_fad, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
//
//                }
//
//            }
//
//            local_functional += volume_local_sum.val().val();
//            // now getting the values and adding them to the derivaitve vector
//
//            i_derivative = 0;
//            if (compute_dIdW) {
//                local_dIdw.resize(n_soln_dofs_cell);
//                for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
//                    local_dIdw[idof] = volume_local_sum.dx(i_derivative++).val();
//                }
//                dIdw.add(cell_soln_dofs_indices, local_dIdw);
//            }
//            if (compute_dIdX) {
//                local_dIdX.resize(n_metric_dofs_cell);
//                for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof){
//                    local_dIdX[idof] = volume_local_sum.dx(i_derivative++).val();
//                }
//                dIdX.add(cell_metric_dofs_indices, local_dIdX);
//            }
//            if (compute_dIdW || compute_dIdX) AssertDimension(i_derivative, n_total_indep);
//            if (compute_d2I) {
//                std::vector<real> dWidW(n_soln_dofs_cell);
//                std::vector<real> dWidX(n_metric_dofs_cell);
//                std::vector<real> dXidX(n_metric_dofs_cell);
//
//
//                i_derivative = 0;
//                for (unsigned int idof=0; idof<n_soln_dofs_cell; ++idof) {
//
//                    unsigned int j_derivative = 0;
//                    const ADtype dWi = volume_local_sum.dx(i_derivative++);
//
//                    for (unsigned int jdof=0; jdof<n_soln_dofs_cell; ++jdof) {
//                        dWidW[jdof] = dWi.dx(j_derivative++);
//                    }
//                    d2IdWdW.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices, dWidW);
//
//                    for (unsigned int jdof=0; jdof<n_metric_dofs_cell; ++jdof) {
//                        dWidX[jdof] = dWi.dx(j_derivative++);
//                    }
//                    d2IdWdX.add(cell_soln_dofs_indices[idof], cell_metric_dofs_indices, dWidX);
//                }
//
//                for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) {
//
//                    const ADtype dXi = volume_local_sum.dx(i_derivative++);
//
//                    unsigned int j_derivative = n_soln_dofs_cell;
//                    for (unsigned int jdof=0; jdof<n_metric_dofs_cell; ++jdof) {
//                        dXidX[jdof] = dXi.dx(j_derivative++);
//                    }
//                    d2IdXdX.add(cell_metric_dofs_indices[idof], cell_metric_dofs_indices, dXidX);
//                }
//            }
//            AssertDimension(i_derivative, n_total_indep);
//        }
//        current_functional_value = dealii::Utilities::MPI::sum(local_functional, MPI_COMM_WORLD);
//        // compress before the return
//        if (compute_dIdW) dIdw.compress(dealii::VectorOperation::add);
//        if (compute_dIdX) dIdX.compress(dealii::VectorOperation::add);
//        if (compute_d2I) {
//            d2IdWdW.compress(dealii::VectorOperation::add);
//            d2IdWdX.compress(dealii::VectorOperation::add);
//            d2IdXdX.compress(dealii::VectorOperation::add);
//        }
//
//        return current_functional_value;
//    }
//};

template <int dim, int nstate>
EulerBumpOptimization<dim,nstate>::EulerBumpOptimization(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

const double y_height = 0.8;
const double bump_height = 0.0625; // High-Order Prediction Workshop
const double coeff_expx = -25; // High-Order Prediction Workshop
const double coeff_expy = -30;
template <int dim, int nstate>
dealii::Point<dim> EulerBumpOptimization<dim,nstate>
::warp (const dealii::Point<dim> &p)
{
    const double x_ref = p[0];
    const double coeff = 1.0;
    const double y_ref = (exp(coeff*std::pow(p[1],1.25))-1.0)/(exp(coeff)-1.0);
    dealii::Point<dim> q = p;
    q[0] = x_ref;
    q[1] = 0.8*y_ref + bump_height*exp(coeff_expy*y_ref*y_ref)*exp(coeff_expx*q[0]*q[0]) * (1.0+0.7*q[0]);
    return q;
}


dealii::Point<2> BumpManifold2::pull_back(const dealii::Point<2> &space_point) const {
    double x_phys = space_point[0];
    double y_phys = space_point[1];
    double x_ref = x_phys;

    double y_ref = y_phys;

    for (int i=0; i<200; i++) {
        const double function = y_height*y_ref + bump_height*exp(coeff_expy*y_ref*y_ref)*exp(coeff_expx*x_phys*x_phys) * (1.0+0.7*x_phys) - y_phys;
        const double derivative = y_height + bump_height*coeff_expy*2*y_ref*exp(coeff_expy*y_ref*y_ref)*exp(coeff_expx*x_phys*x_phys) * (1.0+0.7*x_phys);
        //const double function = y_height*y_ref + bump_height*exp(coeff_expy*y_ref*y_ref)*exp(coeff_expx*x_phys*x_phys) - y_phys;
        //const double derivative = y_height + bump_height*coeff_expy*2*y_ref*exp(coeff_expy*y_ref*y_ref)*exp(coeff_expx*x_phys*x_phys);
        y_ref = y_ref - function/derivative;
        if(std::abs(function) < 1e-15) break;
    }
    const double function = y_height*y_ref + bump_height*exp(coeff_expy*y_ref*y_ref)*exp(coeff_expx*x_phys*x_phys) * (1.0+0.7*x_phys);
    const double error = std::abs(function - y_phys);
    if (error > 1e-13) {
        std::cout << "Large error " << error << std::endl;
        std::cout << "xref " << x_ref << "yref " << y_ref << "y_phys " << y_phys << " " << function << " " << error << std::endl;
    }

    dealii::Point<2> p(x_ref, y_ref);
    return p;
}

dealii::Point<2> BumpManifold2::push_forward(const dealii::Point<2> &chart_point) const 
{
    double x_ref = chart_point[0];
    double y_ref = chart_point[1];
    double x_phys = x_ref;//-1.5+x_ref*3.0;
    double y_phys = y_height*y_ref + exp(coeff_expy*y_ref*y_ref)*bump_height*exp(coeff_expx*x_phys*x_phys) * (1.0+0.7*x_phys);
    return dealii::Point<2> ( x_phys, y_phys); // Trigonometric
}

dealii::DerivativeForm<1,2,2> BumpManifold2::push_forward_gradient(const dealii::Point<2> &chart_point) const
{
    dealii::DerivativeForm<1, 2, 2> dphys_dref;
    double x_ref = chart_point[0];
    double y_ref = chart_point[1];
    double x_phys = x_ref;
    //double y_phys = y_height*y_ref + exp(coeff_expy*y_ref*y_ref)*bump_height*exp(coeff_expx*x_phys*x_phys);
    dphys_dref[0][0] = 1;
    dphys_dref[0][1] = 0;
    dphys_dref[1][0] = exp(coeff_expy*y_ref*y_ref)*bump_height*exp(coeff_expx*x_phys*x_phys) * coeff_expx*2*x_phys*dphys_dref[0][0] * (1.0+0.7*x_phys);
    dphys_dref[1][0] += exp(coeff_expy*y_ref*y_ref)*bump_height*exp(coeff_expx*x_phys*x_phys) * 0.7*dphys_dref[0][0];
    dphys_dref[1][1] = y_height + coeff_expy * 2*y_ref * exp(coeff_expy*y_ref*y_ref)*bump_height*exp(coeff_expx*x_phys*x_phys) * (1.0+0.7*x_phys);
    return dphys_dref;
}

std::unique_ptr<dealii::Manifold<2,2> > BumpManifold2::clone() const
{
    return std::make_unique<BumpManifold2>();
}


template<int dim, int nstate>
int EulerBumpOptimization<dim,nstate>
::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;


    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
    Physics::FreeStreamInitialConditions<dim,nstate> initial_conditions(euler_physics_double);
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }

    int poly_degree = 1;

    const int n_1d_cells = manu_grid_conv_param.initial_grid_size;

    std::vector<unsigned int> n_subdivisions(dim);
    //n_subdivisions[1] = n_1d_cells; // y-direction
    //n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction
    n_subdivisions[1] = n_1d_cells; // y-direction
    n_subdivisions[0] = 9*n_subdivisions[1]; // x-direction
    dealii::Point<2> p1(-1.5,0.0), p2(1.5,y_height);
    const bool colorize = true;
    dealii::parallel::distributed::Triangulation<dim> grid(this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    dealii::GridGenerator::subdivided_hyper_rectangle (grid, n_subdivisions, p1, p2, colorize);

    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 2 || current_id == 3) cell->face(face)->set_boundary_id (1001); // Bottom and top wall
                if (current_id == 1) cell->face(face)->set_boundary_id (1002); // Outflow with supersonic or back_pressure
                if (current_id == 0) cell->face(face)->set_boundary_id (1003); // Inflow

                if (current_id == 2) {
                    cell->face(face)->set_user_index(1); // Bottom wall
                } else {
                    cell->face(face)->set_user_index(-1); // All other boundaries.
                }
            }
        }
    }

    // Warp grid to be a gaussian bump
    dealii::GridTools::transform (&warp, grid);
    
    // Assign a manifold to have curved geometry
    const BumpManifold2 bump_manifold;
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid.reset_all_manifolds();
    grid.set_all_manifold_ids(manifold_id);
    grid.set_manifold ( manifold_id, bump_manifold );

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);

    // Initialize coarse grid solution with free-stream
    dg->allocate_system ();
    dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

    // Create ODE solver and ramp up the solution from p0
    std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->initialize_steady_polynomial_ramping (poly_degree);

    const unsigned int n_global_active_cells = grid.n_global_active_cells();
    const unsigned int n_dofs = dg->dof_handler.n_dofs();
    pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
         << ". Number of active cells: " << n_global_active_cells
         << ". Number of degrees of freedom: " << n_dofs
         << std::endl;

    // Solve the steady state problem
    ode_solver->steady_state();

	dg->output_results_vtk(9999);
    pcout << " Residual: " << ode_solver->residual_norm << std::endl;
    int ifail = 1;
    return ifail;
}


#if PHILIP_DIM==2
    template class EulerBumpOptimization <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

