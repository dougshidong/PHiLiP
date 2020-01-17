#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

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


namespace PHiLiP {
namespace Tests {

/// Function used to evaluate farfield conservative solution
template <int dim, int nstate>
class FreeStreamInitialConditions2 : public dealii::Function<dim>
{
public:
    /// Farfield conservative solution
    std::array<double,nstate> farfield_conservative;

    /// Constructor.
    /** Evaluates the primary farfield solution and converts it into the store farfield_conservative solution
     */
    FreeStreamInitialConditions2 (const Physics::Euler<dim,nstate,double> euler_physics)
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
  
    /// Returns the istate-th farfield conservative value
    double value (const dealii::Point<dim> &/*point*/, const unsigned int istate) const
    {
        return farfield_conservative[istate];
    }
};
template class FreeStreamInitialConditions2 <PHILIP_DIM, PHILIP_DIM+2>;

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
    FreeStreamInitialConditions2<dim,nstate> initial_conditions(euler_physics_double);
    pcout << "Farfield conditions: "<< std::endl;
    for (int s=0;s<nstate;s++) {
        pcout << initial_conditions.farfield_conservative[s] << std::endl;
    }

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    int poly_degree = 1;


    const int n_1d_cells = manu_grid_conv_param.initial_grid_size;

    dealii::ConvergenceTable convergence_table;

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

