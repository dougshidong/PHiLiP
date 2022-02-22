#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>

#include "euler_gaussian_bump_adjoint.h"

#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver_factory.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include <deal.II/distributed/solution_transfer.h>


#include "linear_solver/linear_solver.h"
//#include "template_instantiator.h"
#include "post_processor/physics_post_processor.h"

namespace PHiLiP {
namespace Tests {

// template <int dim, int nstate, typename real>
// class L2_Norm_Functional : public Functional<dim, nstate, real>
// {
//  public:
//   template <typename real2>
//   real2 evaluate_cell_volume(
//    const Physics::PhysicsBase<dim,nstate,real> &physics,
//    const dealii::FEValues<dim,dim> &fe_values_volume,
//    std::vector<real2> local_solution)
//   {
//    unsigned int n_quad_pts = fe_values_volume.n_quadrature_points;

//    std::array<real2,nstate> soln_at_q;

//    real2 l2error = 0;

//    // looping over the quadrature points
//    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
//     std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
//     for (unsigned int idof=0; idof<fe_values_volume.dofs_per_cell; ++idof) {
//      const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
//      soln_at_q[istate] += local_solution[idof] * fe_values_volume.shape_value_component(idof, iquad, istate);
//     }
   
//     const dealii::Point<dim> qpoint = (fe_values_volume.quadrature_point(iquad));

//     for (int istate=0; istate<nstate; ++istate) {
//      const double uexact = physics.manufactured_solution_function.value(qpoint, istate);
//      l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_volume.JxW(iquad);
//     }
//    }

//    return l2error;
//   }

//   // non-template functions to override the template classes
//   real evaluate_cell_volume(
//    const Physics::PhysicsBase<dim,nstate,real> &physics,
//    const dealii::FEValues<dim,dim> &fe_values_volume,
//    std::vector<real> local_solution) override
//   {
//    return evaluate_cell_volume<>(physics, fe_values_volume, local_solution);
//   }
//   Sacado::Fad::DFad<real> evaluate_cell_volume(
//    const Physics::PhysicsBase<dim,nstate,real> &physics,
//    const dealii::FEValues<dim,dim> &fe_values_volume,
//    std::vector<Sacado::Fad::DFad<real>> local_solution) override
//   {
//    return evaluate_cell_volume<>(physics, fe_values_volume, local_solution);
//   }
// };

/** Boundary integral for the Euler Gaussian bump.
 *  Pressure integral on the outlet.
 */
template <int dim, int nstate, typename real>
class BoundaryIntegral : public PHiLiP::Functional<dim, nstate, real>
{
 public:
        /// Templated function to evaluate exit pressure integral.
        template <typename real2>
        real2 evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
            const unsigned int boundary_id,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            std::vector<real2> local_solution){
              
            real2 l2error = 0.0;

            if(boundary_id == 1002){
                // casting it to a physics euler as it is needed for the pressure computation
                const Physics::Euler<dim,nstate,real2>& euler_physics = dynamic_cast<const Physics::Euler<dim,nstate,real2>&>(physics);

                unsigned int n_quad_pts = fe_values_boundary.n_quadrature_points;

                // looping over quadrature points one at a time and assembling the state vector
                std::array<real2,nstate> soln_at_q;
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for(unsigned int idof=0; idof<fe_values_boundary.dofs_per_cell; idof++){
                        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += local_solution[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
                    }
                    
                    // converting the state vector to the point pressure 
                    real2 pressure = euler_physics.compute_pressure(soln_at_q);

                    // integrating (with quadrature weights)
                    l2error += pressure * fe_values_boundary.JxW(iquad);
                }
            }
            // std::cout << boundary_id << " == 10002? " << (boundary_id == 1002) << std::endl;
            // if(boundary_id ==  1002){
            //     for(unsigned int idof=0; idof<fe_values_boundary.dofs_per_cell; idof++){
            //         l2error += local_solution[idof];
            //     }
            // }

            // unsigned int n_quad_pts = fe_values_boundary.n_quadrature_points;

            // std::array<real2,nstate> soln_at_q;

            // trying with the functional from the test case but only on the boundaray
            // // seeing if evaluating anything seems to work
            // for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            // for(unsigned int idof=0; idof<fe_values_boundary.dofs_per_cell; idof++){
            //     const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            //     soln_at_q[istate] += local_solution[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            // }
            // const dealii::Point<dim> qpoint = (fe_values_boundary.quadrature_point(iquad));
            // for (int istate=0; istate<nstate; ++istate) {
   //  const double uexact = physics.manufactured_solution_function.value(qpoint, istate);
   //  l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_boundary.JxW(iquad);
   // }

            // }


            // // only for the outflow conditions
            // if(boundary_id == 1002){
            //     const std::vector<real> &JxW       = fe_values_boundary.get_JxW_values();
            //     const unsigned int n_dofs_cell     = fe_values_boundary.dofs_per_cell;
            //     const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

            //     for(unsigned int itest=0; itest<n_dofs_cell; itest++){
            //         const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

            //         for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
            //             local_sum += std::pow(fe_values_boundary.shape_value_component(itest,iquad,istate) * local_solution[itest], 2.0) * JxW[iquad];
            //         }
            //     }
            // }
            return l2error;
        }

        /// Corresponding non-templated function for evaluate_cell_boundary.
  real evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
            const unsigned int boundary_id,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            std::vector<real> local_solution) override {return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);}

        /// Corresponding non-templated function for evaluate_cell_boundary.
  Sacado::Fad::DFad<real> evaluate_cell_boundary(
            const PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics,
            const unsigned int boundary_id,
            const dealii::FEFaceValues<dim,dim> &fe_values_boundary,
            std::vector<Sacado::Fad::DFad<real>> local_solution) override {return evaluate_cell_boundary<>(physics, boundary_id, fe_values_boundary, local_solution);}
};

/// Initial conditions to initialize our flow with.
template <int dim, int nstate>
class FreeStreamInitialConditionsAdjoint : public dealii::Function<dim>
{
public:
    /// Conservative freestream variables.
    std::array<double,nstate> far_field_conservative;

    /// Constructor
    FreeStreamInitialConditions (const Physics::Euler<dim,nstate,double> euler_physics)
    : dealii::Function<dim,double>(nstate)
    {
        const double density_bc = 2.33333*euler_physics.density_inf;
        const double pressure_bc = 1.0/(euler_physics.gam*euler_physics.mach_inf_sqr);
        std::array<double,nstate> primitive_boundary_values;
        primitive_boundary_values[0] = density_bc;
        for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = euler_physics.velocities_inf[d]; }
        primitive_boundary_values[nstate-1] = pressure_bc;
        farfield_conservative = euler_physics.convert_primitive_to_conservative(primitive_boundary_values);
    }

    /// Destructor
    ~FreeStreamInitialConditions() {};
  
    /// Corresponds to dealii::Function::value
    double value (const dealii::Point<dim> &/*point*/, const unsigned int istate) const override
    {
        return farfield_conservative[istate];
    }
};
template class FreeStreamInitialConditionsAdjoint <PHILIP_DIM, PHILIP_DIM+2>;

template <int dim, int nstate>
EulerGaussianBumpAdjoint<dim,nstate>::EulerGaussianBumpAdjoint(const Parameters::AllParameters *const parameters_input)
    :
    TestsBase::TestsBase(parameters_input)
{}

const double y_height = 0.8;
const double bump_height = 0.0625; // High-Order Prediction Workshop
const double coeff_expx = -25; // High-Order Prediction Workshop
const double coeff_expy = -30;
template <int dim, int nstate>
dealii::Point<dim> EulerGaussianBumpAdjoint<dim,nstate>
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


dealii::Point<2> BumpManifoldAdjoint::pull_back(const dealii::Point<2> &space_point) const {
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

dealii::Point<2> BumpManifoldAdjoint::push_forward(const dealii::Point<2> &chart_point) const 
{
    double x_ref = chart_point[0];
    double y_ref = chart_point[1];
    double x_phys = x_ref;//-1.5+x_ref*3.0;
    double y_phys = y_height*y_ref + exp(coeff_expy*y_ref*y_ref)*bump_height*exp(coeff_expx*x_phys*x_phys) * (1.0+0.7*x_phys);
    return dealii::Point<2> ( x_phys, y_phys); // Trigonometric
}

dealii::DerivativeForm<1,2,2> BumpManifoldAdjoint::push_forward_gradient(const dealii::Point<2> &chart_point) const
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

std::unique_ptr<dealii::Manifold<2,2> > BumpManifoldAdjoint::clone() const
{
    return std::make_unique<BumpManifoldAdjoint>();
}


template<int dim, int nstate>
int EulerGaussianBumpAdjoint<dim,nstate>
::run_test () const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    //Assert(param.pde_type != param.PartialDifferentialEquation::euler, dealii::ExcNotImplemented());
    //if (param.pde_type == param.PartialDifferentialEquation::euler) return 1;

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;

    const unsigned int n_grids_input       = manu_grid_conv_param.number_of_grids;

    // const unsigned int poly_max = p_end+1;

    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);

    Physics::Euler<dim,nstate,Sacado::Fad::DFad<double>> euler_physics_adtype
        = Physics::Euler<dim, nstate, Sacado::Fad::DFad<double>>(
            param.euler_param.ref_length,
            param.euler_param.gamma_gas,
            param.euler_param.mach_inf,
            param.euler_param.angle_of_attack,
            param.euler_param.side_slip_angle);

    FreeStreamInitialConditionsAdjoint<dim,nstate> initial_conditions(euler_physics_double);
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        // p0 tends to require a finer grid to reach asymptotic region
        unsigned int n_grids = n_grids_input;
        if (poly_degree <= 1) n_grids = n_grids_input;

        std::vector<double> entropy_error(n_grids);
        std::vector<double> grid_size(n_grids);

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

        std::vector<unsigned int> n_subdivisions(dim);
        //n_subdivisions[1] = n_1d_cells[0]; // y-direction
        //n_subdivisions[0] = 4*n_subdivisions[1]; // x-direction
        n_subdivisions[1] = n_1d_cells[0]; // y-direction
        n_subdivisions[0] = 9*n_subdivisions[1]; // x-direction
        dealii::Point<2> p1(-1.5,0.0), p2(1.5,y_height);
        const bool colorize = true;
        dealii::parallel::distributed::Triangulation<dim> grid(this->mpi_communicator);
            // typename dealii::Triangulation<dim>::MeshSmoothing(
            //     dealii::Triangulation<dim>::smoothing_on_refinement |
            //     dealii::Triangulation<dim>::smoothing_on_coarsening));
        dealii::GridGenerator::subdivided_hyper_rectangle (grid, n_subdivisions, p1, p2, colorize);

        for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int current_id = cell->face(face)->boundary_id();
                    if (current_id == 2 || current_id == 3) cell->face(face)->set_boundary_id (1001); // Bottom and top wall
                    if (current_id == 1) cell->face(face)->set_boundary_id (1002); // Outflow with supersonic or back_pressure
                    if (current_id == 0) cell->face(face)->set_boundary_id (1003); // Inflow
                }
            }
        }

        // Warp grid to be a gaussian bump
        dealii::GridTools::transform (&warp, grid);
        

        // Assign a manifold to have curved geometry
        const BumpManifoldAdjoint bump_manifold;
        unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
        grid.reset_all_manifolds();
        grid.set_all_manifold_ids(manifold_id);
        grid.set_manifold ( manifold_id, bump_manifold );

        // Create DG object
        // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);
        // std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_max/*poly_degree*/, &grid);
        std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, poly_degree+1, &grid);

        // Initialize coarse grid solution with free-stream
        dg->allocate_system ();
        dealii::VectorTools::interpolate(dg->dof_handler, initial_conditions, dg->solution);

        // Create ODE solver and ramp up the solution from p0
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver->initialize_steady_polynomial_ramping (poly_degree);

        // setting up the target functional (error reduction)
        BoundaryIntegral<dim, nstate, double> BoundaryIntegralFunctional;

        // initializing an adjoint for this case
        Adjoint<dim, nstate, double> adjoint(*dg, BoundaryIntegralFunctional, euler_physics_adtype);

        dealii::Vector<float> estimated_error_per_cell(grid.n_active_cells());
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {


            if (igrid!=0) {
                dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
                dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
                solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
                dg->high_order_grid.prepare_for_coarsening_and_refinement();
                // grid.refine_global (1);

                dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(grid,
                // dealii::GridRefinement::refine_and_coarsen_fixed_number(grid,
                                               estimated_error_per_cell,
                                               0.3,
                                               0.03);

                grid.execute_coarsening_and_refinement();
                dg->high_order_grid.execute_coarsening_and_refinement();
                dg->allocate_system ();
                dg->solution.zero_out_ghosts();
                solution_transfer.interpolate(dg->solution);
                dg->solution.update_ghost_values();

                estimated_error_per_cell.reinit(grid.n_active_cells());
            }

            const unsigned int n_global_active_cells = grid.n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim << "\t Polynomial degree p: " << poly_degree << std::endl
                 << "Grid number: " << igrid+1 << "/" << n_grids
                 << ". Number of active cells: " << n_global_active_cells
                 << ". Number of degrees of freedom: " << n_dofs
                 << std::endl;

            // Solve the steady state problem
            ode_solver->steady_state();
            //ode_solver->initialize_steady_polynomial_ramping(poly_degree);

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
            dealii::MappingQ<dim> mappingq(dg->max_degree+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(mappingq, dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0;

            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

            const double entropy_inf = euler_physics_double.entropy_inf;

            // Integrate solution error and output error
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;
                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }
                    const double entropy = euler_physics_double.compute_entropy_measure(soln_at_q);

                    const double uexact = entropy_inf;
                    l2error += pow(entropy - uexact, 2) * fe_values_extra.JxW(iquad);
                }
            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));

            // computing using Functional for comparison
            const double l2error_functional = BoundaryIntegralFunctional.evaluate_function(*dg, euler_physics_double);
            pcout << "Error computed by original loop: " << l2error_mpi_sum << std::endl << "Error computed by the functional: " << std::sqrt(l2error_functional) << std::endl; 

            // reinitializing the adjoint with the current values (from references)
            adjoint.reinit();

            // evaluating the derivatives and the adjoint on the fine grid
            adjoint.convert_to_state(PHiLiP::Adjoint<dim,nstate,double>::AdjointStateEnum::fine); // will do this automatically, but I prefer to repeat explicitly
            adjoint.fine_grid_adjoint();
            estimated_error_per_cell = adjoint.dual_weighted_residual(); // performing the error indicator computation

            // and outputing the fine properties
            adjoint.output_results_vtk(igrid);

            adjoint.convert_to_state(PHiLiP::Adjoint<dim,nstate,double>::AdjointStateEnum::coarse); // this one is necessary though
            adjoint.coarse_grid_adjoint();
            adjoint.output_results_vtk(igrid);

            // Convergence table
            double dx = 1.0/pow(n_dofs,(1.0/dim));
            //dx = dealii::GridTools::maximal_cell_diameter(grid);
            grid_size[igrid] = dx;
            entropy_error[igrid] = l2error_mpi_sum;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("L2_entropy_error", l2error_mpi_sum);


            pcout << " Grid size h: " << dx 
                 << " L2-entropy_error: " << l2error_mpi_sum
                 << " Residual: " << ode_solver->residual_norm
                 << std::endl;

            if (igrid > 0) {
                const double slope_soln_err = log(entropy_error[igrid]/entropy_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid-1
                     << "  to grid " << igrid
                     << "  dimension: " << dim
                     << "  polynomial degree p: " << poly_degree
                     << std::endl
                     << "  entropy_error1 " << entropy_error[igrid-1]
                     << "  entropy_error2 " << entropy_error[igrid]
                     << "  slope " << slope_soln_err
                     << std::endl;
            }

            //output_results (igrid);
        }
        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates("L2_entropy_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("L2_entropy_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = poly_degree+1;

        const double last_slope = log(entropy_error[n_grids-1]/entropy_error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        //double before_last_slope = last_slope;
        //if ( n_grids > 2 ) {
        //    before_last_slope = log(entropy_error[n_grids-2]/entropy_error[n_grids-3])
        //                        / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        //}
        //const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_avg = last_slope;
        const double slope_diff = slope_avg-expected_slope;

        double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);
        if(poly_degree == 0) slope_deficit_tolerance *= 2; // Otherwise, grid sizes need to be much bigger for p=0

        if (slope_diff < slope_deficit_tolerance) {
            pcout << std::endl
                 << "Convergence order not achieved. Average last 2 slopes of "
                 << slope_avg << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
            // p=0 just requires too many meshes to get into the asymptotic region.
            if(poly_degree!=0) fail_conv_poly.push_back(poly_degree);
            if(poly_degree!=0) fail_conv_slop.push_back(slope_avg);
        }

    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }
    int n_fail_poly = fail_conv_poly.size();
    if (n_fail_poly > 0) {
        for (int ifail=0; ifail < n_fail_poly; ++ifail) {
            const double expected_slope = fail_conv_poly[ifail]+1;
            const double slope_deficit_tolerance = -0.1;
            pcout << std::endl
                 << "Convergence order not achieved for polynomial p = "
                 << fail_conv_poly[ifail]
                 << ". Slope of "
                 << fail_conv_slop[ifail] << " instead of expected "
                 << expected_slope << " within a tolerance of "
                 << slope_deficit_tolerance
                 << std::endl;
        }
    }
    return n_fail_poly;
}


#if PHILIP_DIM==2
    template class EulerGaussianBumpAdjoint <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

