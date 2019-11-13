// includes
#include <stdlib.h>
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include "diffusion_exact_adjoint.h"

#include "parameters/all_parameters.h"

#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

namespace PHiLiP {
namespace Tests {
// need to build my own physics classes to override the source term in dg
// would be nice if there was a way to pass this directly to the dg class
// (otherwise this would need to be added to the physics enum)

// manufactured solution in u
template <int dim, typename real>
real ManufacturedSolutionU<dim,real>::value(const dealii::Point<dim> &pos, const unsigned int /*istate*/) const
{
    real val = 1;

    for(unsigned int d=0; d<dim; ++d){
        double x = pos[d];
        val *= std::pow(x,3)*std::pow(1-x,3);
    }

    return val;
}

// gradient of the solution in u
template <int dim, typename real>
dealii::Tensor<1,dim,real> ManufacturedSolutionU<dim,real>::gradient(const dealii::Point<dim> &pos, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;

    for(unsigned int d=0; d<dim; ++d){
        double x = pos[d];
        gradient[d] = (((-6.0*x+15.0)*x-12.0)*x+3.0)*x*x*this->value(pos,istate)/(((-30.0*x+60.0)*x-36.0)*x+6.0)*x;
    }

    return gradient;
}

// manufactured solution in v
template <int dim, typename real>
real ManufacturedSolutionV<dim,real>::value(const dealii::Point<dim> &pos, const unsigned int /*istate*/) const
{
    const double pi = std::acos(-1);

    real val = 1;

    for(unsigned int d=0; d<dim; ++d){
        double x = pos[d];
        val *= std::sin(pi*x);
    }

    return val;
}

// gradient of the solution in v
template <int dim, typename real>
dealii::Tensor<1,dim,real> ManufacturedSolutionV<dim,real>::gradient(const dealii::Point<dim> &pos, const unsigned int istate) const
{
    const double pi = std::acos(-1);

    dealii::Tensor<1,dim,real> gradient;

    for(unsigned int d=0; d<dim; ++d){
        // double x = pos[d];
        gradient[d] = -pi*pi*this->value(pos,istate);
    }

    return gradient;
}

/* Defining the physics objects to be used  */
template <int dim, int nstate, typename real>
std::array<real,nstate> diffusion_u<dim,nstate,real>::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;

    for (int istate=0; istate<nstate; istate++) {
        real val = 1;

        for(unsigned int d=0; d<dim; ++d){
            double x = pos[d];
            val *= (((-30.0*x+60.0)*x-36.0)*x+6.0)*x;
        }

        source[istate] = val;
    }

    return source;
}

template <int dim, int nstate, typename real>
real diffusion_u<dim,nstate,real>::objective_function (
    const dealii::Point<dim,double> &pos) const
{
    const double pi = std::acos(-1);

    real val = 1;

    for(unsigned int d=0; d<dim; ++d){
        double x = pos[d];
        val *= -(pi*pi)*std::sin(pi * x);
    }

    return val;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> diffusion_v<dim,nstate,real>::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    const double pi = std::acos(-1);

    std::array<real,nstate> source;

    for (int istate=0; istate<nstate; istate++) {
        real val = 1;

        for(unsigned int d=0; d<dim; ++d){
            double x = pos[d];
            val *= -(pi*pi)*std::sin(pi * x);
        }

        source[istate] = val;
    }

    return source;
}

template <int dim, int nstate, typename real>
real diffusion_v<dim,nstate,real>::objective_function (
    const dealii::Point<dim,double> &pos) const
{
    real val = 1;

    for(unsigned int d=0; d<dim; ++d){
        double x = pos[d];
        val *= (((-30.0*x+60.0)*x-36.0)*x+6.0)*x;
    }

    return val;
}

// Funcitonal that performs the inner product over the entire domain 
template <int dim, int nstate, typename real>
template <typename real2>
real2 DiffusionFunctional<dim,nstate,real>::evaluate_cell_volume(
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
    const dealii::FEValues<dim,dim> &fe_values_volume,
    std::vector<real2> local_solution)
{
    unsigned int n_quad_pts = fe_values_volume.n_quadrature_points;

    std::array<real2,nstate> soln_at_q;

    real2 val = 0;

    // casting our physics object into a diffusion_objective object 
    const diffusion_objective<dim,nstate,real2>& diff_physics = dynamic_cast<const diffusion_objective<dim,nstate,real2>&>(physics);
    
    // looping over the quadrature points
    for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad){
        std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
        for (unsigned int idof=0; idof<fe_values_volume.dofs_per_cell; ++idof) {
            const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
            soln_at_q[istate] += local_solution[idof] * fe_values_volume.shape_value_component(idof, iquad, istate);
        }

        const dealii::Point<dim> qpoint = (fe_values_volume.quadrature_point(iquad));

        // evaluating the associated objective function weighting at the quadrature point
        real2 objective_value = diff_physics.objective_function(qpoint);

        // integrating over the domain (adding istate loop but should always be 1)
        for (int istate=0; istate<nstate; ++istate) {
            val += soln_at_q[istate] * objective_value * fe_values_volume.JxW(iquad);
        }
    }

    return val;
}


template <int dim, int nstate>
DiffusionExactAdjoint<dim, nstate>::DiffusionExactAdjoint(const Parameters::AllParameters *const parameters_input): 
    TestsBase::TestsBase(parameters_input){}

template <int dim, int nstate>
int DiffusionExactAdjoint<dim,nstate>::run_test() const
{
    std::cout << "Running diffusion exact adjoint test case." << std::endl;

    // getting the problem parameters
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    using GridEnum = ManParam::GridEnum;
    using PdeEnum = Parameters::AllParameters::PartialDifferentialEquation;
    Parameters::AllParameters param = *(TestsBase::all_parameters);

    param.manufactured_convergence_study_param.use_manufactured_source_term = true;

    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));

    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start = manu_grid_conv_param.degree_start;
    const unsigned int p_end   = manu_grid_conv_param.degree_end;
    const unsigned int n_grids = manu_grid_conv_param.number_of_grids;

    // checking that the diffusion equation has been selected
    PdeEnum pde_type = param.pde_type;
    bool convection, diffusion;
    if(pde_type == PdeEnum::diffusion){
        convection = false;
        diffusion  = true;
    }else{
        std::cout << "Can't run diffusion_exact_adjoint test case with other PDE types." << std::endl;
    }

    // creating the physics objects
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, double> > physics_u_double 
          = std::make_shared< diffusion_u<dim, nstate, double> >(convection, diffusion);
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, double> > physics_v_double  
          = std::make_shared< diffusion_v<dim, nstate, double> >(convection, diffusion);

    // for adjoint, also need the AD'd physics
    using ADtype = Sacado::Fad::DFad<double>;
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, ADtype> > physics_u_adtype 
          = std::make_shared< diffusion_u<dim, nstate, ADtype> >(convection, diffusion);
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, ADtype> > physics_v_adtype 
          = std::make_shared< diffusion_v<dim, nstate, ADtype> >(convection, diffusion);
 
    // functional for computations
    DiffusionFunctional<dim,nstate,double> diffusion_functional;

    // checks 
    std::vector<int> fail_conv_poly;
    std::vector<double>  failt_conv_slope;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for(unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree){
        std::cout << "Starting polynomial order: " << poly_degree << std::endl;

        // grid_study refines P0 additinoally here (unused in current param file)
        std::vector<double> soln_error(n_grids);
        std::vector<double> output_error(n_grids);
        std::vector<double> grid_size(n_grids);

        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

        // constructing the grid (non-distrbuted type for 1D)
        dealii::Triangulation<dim> grid(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));

        // dimensions of the mesh
        const double left  = 0.0;
        const double right = 1.0;

        for(unsigned int igrid = 0; igrid < n_grids; ++igrid){
            // grid generation
            grid.clear();
            dealii::GridGenerator::subdivided_hyper_cube(grid, n_1d_cells[igrid], left, right);
            for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
                cell->set_material_id(9002);
                for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                    if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1000);
                }
            }

            // since a different grid is constructed each time, need to also generate a new DG
            // I don't think this would work outside loop since grid.clear() req's no subscriptors
            std::shared_ptr < DGBase<dim, double> > dg_u = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);
            std::shared_ptr < DGBase<dim, double> > dg_v = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, &grid);

            // casting to dg weak    
            std::shared_ptr< DGWeak<dim,nstate,double> > dg_weak_u = std::dynamic_pointer_cast< DGWeak<dim,nstate,double> >(dg_u);
            std::shared_ptr< DGWeak<dim,nstate,double> > dg_weak_v = std::dynamic_pointer_cast< DGWeak<dim,nstate,double> >(dg_v);

            // now overriding the original physics on each
            dg_weak_u->set_physics(physics_u_double);
            dg_weak_u->set_physics(physics_u_adtype);
            dg_weak_v->set_physics(physics_v_double);
            dg_weak_v->set_physics(physics_v_adtype);

            dg_u->allocate_system();
            dg_v->allocate_system();
            
            // Create ODE solvers using the factory and providing the DG object
            std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver_u = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg_u);
            std::shared_ptr<ODE::ODESolver<dim, double>> ode_solver_v = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg_v);

            // solving
            ode_solver_u->steady_state();
            ode_solver_v->steady_state();

            const std::string filename_u = "sol-u-" + std::to_string(igrid) + ".gnuplot";
            std::ofstream gnuplot_output_u(filename_u);
            dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> data_out_u;
            data_out_u.attach_dof_handler(dg_u->dof_handler);
            data_out_u.add_data_vector(dg_u->solution, "u");
            data_out_u.build_patches();
            data_out_u.write_gnuplot(gnuplot_output_u);

            const std::string filename_v = "sol-v-" + std::to_string(igrid) + ".gnuplot";
            std::ofstream gnuplot_output_v(filename_v);
            dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> data_out_v;
            data_out_v.attach_dof_handler(dg_v->dof_handler);
            data_out_v.add_data_vector(dg_v->solution, "u");
            data_out_v.build_patches();
            data_out_v.write_gnuplot(gnuplot_output_v);

            // evaluating functionals from both methods
            double val1 = diffusion_functional.evaluate_function(*dg_u, *physics_u_double);
            double val2 = diffusion_functional.evaluate_function(*dg_v, *physics_v_double);

            // comparison betweent the values, add these to the convergence table
            std::cout << std::endl << "Val1 = " << val1 << "\tVal2 = " << val2 << std::endl << std::endl; 

        }

    }

    return 0;
}

#if PHILIP_DIM==1
    template class DiffusionExactAdjoint <PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace