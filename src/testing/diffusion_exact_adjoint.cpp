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
// built own physics classes here for one time use and added a function to pass them directly to dg weak

// manufactured solution in u
template <int dim, typename real>
real ManufacturedSolutionU<dim,real>::value(const dealii::Point<dim> &pos, const unsigned int /*istate*/) const
{
    real val = 0;

    if(dim == 1){
        double x = pos[0];
        val = (-1.0*std::pow(x,6)+3.0*std::pow(x,5)-3.0*std::pow(x,4)+std::pow(x,3));
    }else if(dim == 2){
        double x = pos[0], y = pos[1];
        val = (-1.0*std::pow(x,6)+3.0*std::pow(x,5)-3.0*std::pow(x,4)+std::pow(x,3))
            * (-1.0*std::pow(y,6)+3.0*std::pow(y,5)-3.0*std::pow(y,4)+std::pow(y,3));
    }else if(dim == 3){
        double x = pos[0], y = pos[1], z = pos[2];
        val = (-1.0*std::pow(x,6)+3.0*std::pow(x,5)-3.0*std::pow(x,4)+std::pow(x,3))
            * (-1.0*std::pow(y,6)+3.0*std::pow(y,5)-3.0*std::pow(y,4)+std::pow(y,3))
            * (-1.0*std::pow(z,6)+3.0*std::pow(z,5)-3.0*std::pow(z,4)+std::pow(z,3));
    }

    return val;
}

// gradient of the solution in u
template <int dim, typename real>
dealii::Tensor<1,dim,real> ManufacturedSolutionU<dim,real>::gradient(const dealii::Point<dim> &pos, const unsigned int /*istate*/) const
{
    dealii::Tensor<1,dim,real> gradient;

    if(dim == 1){
        double x = pos[0];
        gradient[0] = (-6.0*std::pow(x,5)+15.0*std::pow(x,4)-12.0*std::pow(x,3)+3.0*std::pow(x,2));
    }else if(dim == 2){
        double x = pos[0], y = pos[1];
        gradient[0] = (-6.0*std::pow(x,5)+15.0*std::pow(x,4)-12.0*std::pow(x,3)+3.0*std::pow(x,2))
                    * (-1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3));
        gradient[1] = (-1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
                    * (-6.0*std::pow(y,5)+15.0*std::pow(y,4)-12.0*std::pow(y,3)+3.0*std::pow(y,2));
    }else if(dim == 3){
        double x = pos[0], y = pos[1], z = pos[2];
        gradient[0] = (-6.0*std::pow(x,5)+15.0*std::pow(x,4)-12.0*std::pow(x,3)+3.0*std::pow(x,2))
                    * (-1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3))
                    * (-1.0*std::pow(z,6)+ 3.0*std::pow(z,5)- 3.0*std::pow(z,4)+    std::pow(z,3));
        gradient[1] = (-1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
                    * (-6.0*std::pow(y,5)+15.0*std::pow(y,4)-12.0*std::pow(y,3)+3.0*std::pow(y,2))
                    * (-1.0*std::pow(z,6)+ 3.0*std::pow(z,5)- 3.0*std::pow(z,4)+    std::pow(z,3));
        gradient[2] = (-1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
                    * (-1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3))
                    * (-6.0*std::pow(z,5)+15.0*std::pow(z,4)-12.0*std::pow(z,3)+3.0*std::pow(z,2));
    }

    return gradient;
}

// manufactured solution in v
template <int dim, typename real>
real ManufacturedSolutionV<dim,real>::value(const dealii::Point<dim> &pos, const unsigned int /*istate*/) const
{
    const double pi = std::acos(-1);

    real val = 0;

    if(dim == 1){
        double x = pos[0];
        val = (std::sin(pi*x));
    }else if(dim == 2){
        double x = pos[0], y = pos[1];
        val = (std::sin(pi*x))
            * (std::sin(pi*y));
    }else if(dim == 3){
        double x = pos[0], y = pos[1], z = pos[2];
        val = (std::sin(pi*x))
            * (std::sin(pi*y))
            * (std::sin(pi*z));
    }

    return val;
}

// gradient of the solution in v
template <int dim, typename real>
dealii::Tensor<1,dim,real> ManufacturedSolutionV<dim,real>::gradient(const dealii::Point<dim> &pos, const unsigned int /*istate*/) const
{
    const double pi = std::acos(-1);

    dealii::Tensor<1,dim,real> gradient;

    if(dim == 1){
        double x = pos[0];
        gradient[0] = pi*std::cos(pi*x);
    }else if(dim == 2){
        double x = pos[0], y = pos[1];
        gradient[0] = (pi*std::cos(pi*x))
                    * (   std::sin(pi*y));
        gradient[1] = (   std::cos(pi*x))
                    * (pi*std::sin(pi*y));
    }else if(dim == 3){
        double x = pos[0], y = pos[1], z = pos[2];
        gradient[0] = (pi*std::cos(pi*x))
                    * (   std::sin(pi*y))
                    * (   std::sin(pi*z));
        gradient[1] = (   std::sin(pi*x))
                    * (pi*std::cos(pi*y))
                    * (   std::sin(pi*z));
        gradient[2] = (   std::sin(pi*x))
                    * (   std::sin(pi*y))
                    * (pi*std::cos(pi*z));
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

    double val = 0;

    if(dim == 1){
        double x = pos[0];
        val = (-30.0*std::pow(x,4)+60.0*std::pow(x,3)-36.0*std::pow(x,2)+6.0*x);
    }else if(dim == 2){
        double x = pos[0], y = pos[1];
        val = (-30.0*std::pow(x,4)+60.0*std::pow(x,3)-36.0*std::pow(x,2)+6.0*x)
            * ( -1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3))
            + ( -1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
            * (-30.0*std::pow(y,4)+60.0*std::pow(y,3)-36.0*std::pow(y,2)+6.0*y);
    }else if(dim == 3){
        double x = pos[0], y = pos[1], z = pos[2];
        val = (-30.0*std::pow(x,4)+60.0*std::pow(x,3)-36.0*std::pow(x,2)+6.0*x)
            * ( -1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3))
            * ( -1.0*std::pow(z,6)+ 3.0*std::pow(z,5)- 3.0*std::pow(z,4)+    std::pow(z,3))
            + ( -1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
            * (-30.0*std::pow(y,4)+60.0*std::pow(y,3)-36.0*std::pow(y,2)+6.0*y)
            * ( -1.0*std::pow(z,6)+ 3.0*std::pow(z,5)- 3.0*std::pow(z,4)+    std::pow(z,3))
            + ( -1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
            * ( -1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3))
            * (-30.0*std::pow(z,4)+60.0*std::pow(z,3)-36.0*std::pow(z,2)+6.0*z);
    }

    for(int istate = 0; istate < nstate; ++istate)
        source[istate] = val;

    return source;
}

template <int dim, int nstate, typename real>
real diffusion_u<dim,nstate,real>::objective_function (
    const dealii::Point<dim,double> &pos) const
{
    const double pi = std::acos(-1);

    real val = 0;

    if(dim == 1){
        double x = pos[0];
        val = -pi*pi*std::sin(pi*x);
    }else if(dim == 2){
        double x = pos[0], y = pos[1];
        val = -pi*pi*std::sin(pi*x)
            *        std::sin(pi*y)
            +        std::sin(pi*x)
            * -pi*pi*std::sin(pi*y);
    }else if(dim == 3){
        double x = pos[0], y = pos[1], z = pos[2];
        val = -pi*pi*std::sin(pi*x)
            *        std::sin(pi*y)
            *        std::sin(pi*z)
            +        std::sin(pi*x)
            * -pi*pi*std::sin(pi*y)
            *        std::sin(pi*z)
            +        std::sin(pi*x)
            *        std::sin(pi*y)
            * -pi*pi*std::sin(pi*z);
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

    double val = 0;

    if(dim == 1){
        double x = pos[0];
        val = -pi*pi*std::sin(pi*x);
    }else if(dim == 2){
        double x = pos[0], y = pos[1];
        val = -pi*pi*std::sin(pi*x)
            *        std::sin(pi*y)
            +        std::sin(pi*x)
            * -pi*pi*std::sin(pi*y);
    }else if(dim == 3){
        double x = pos[0], y = pos[1], z = pos[2];
        val = -pi*pi*std::sin(pi*x)
            *        std::sin(pi*y)
            *        std::sin(pi*z)
            +        std::sin(pi*x)
            * -pi*pi*std::sin(pi*y)
            *        std::sin(pi*z)
            +        std::sin(pi*x)
            *        std::sin(pi*y)
            * -pi*pi*std::sin(pi*z);
    }

    for(int istate = 0; istate < nstate; ++istate)
        source[istate] = val;

    return source;
}

template <int dim, int nstate, typename real>
real diffusion_v<dim,nstate,real>::objective_function (
    const dealii::Point<dim,double> &pos) const
{
    real val = 0;

    if(dim == 1){
        double x = pos[0];
        val = (-30.0*std::pow(x,4)+60.0*std::pow(x,3)-36.0*std::pow(x,2)+6.0*x);
    }else if(dim == 2){
        double x = pos[0], y = pos[1];
        val = (-30.0*std::pow(x,4)+60.0*std::pow(x,3)-36.0*std::pow(x,2)+6.0*x)
            * ( -1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3))
            + ( -1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
            * (-30.0*std::pow(y,4)+60.0*std::pow(y,3)-36.0*std::pow(y,2)+6.0*y);
    }else if(dim == 3){
        double x = pos[0], y = pos[1], z = pos[2];
        val = (-30.0*std::pow(x,4)+60.0*std::pow(x,3)-36.0*std::pow(x,2)+6.0*x)
            * ( -1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3))
            * ( -1.0*std::pow(z,6)+ 3.0*std::pow(z,5)- 3.0*std::pow(z,4)+    std::pow(z,3))
            + ( -1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
            * (-30.0*std::pow(y,4)+60.0*std::pow(y,3)-36.0*std::pow(y,2)+6.0*y)
            * ( -1.0*std::pow(z,6)+ 3.0*std::pow(z,5)- 3.0*std::pow(z,4)+    std::pow(z,3))
            + ( -1.0*std::pow(x,6)+ 3.0*std::pow(x,5)- 3.0*std::pow(x,4)+    std::pow(x,3))
            * ( -1.0*std::pow(y,6)+ 3.0*std::pow(y,5)- 3.0*std::pow(y,4)+    std::pow(y,3))
            * (-30.0*std::pow(z,4)+60.0*std::pow(z,3)-36.0*std::pow(z,2)+6.0*z);
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
    pcout << "Running diffusion exact adjoint test case." << std::endl;

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
        pcout << "Can't run diffusion_exact_adjoint test case with other PDE types." << std::endl;
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

    // exact value to be used in checks below
    const double pi = std::acos(-1);
    double exact_val = 0;
    
    if(dim == 1){
        exact_val = (144*(std::pow(pi,2)-10)/std::pow(pi,5));
    }else if(dim == 2){
        exact_val = 2 * (-144*(std::pow(pi,2)-10)/std::pow(pi,7)) * (144*(std::pow(pi,2)-10)/std::pow(pi,5));
    }else if(dim == 3){
        exact_val = 3 * std::pow(-144*(std::pow(pi,2)-10)/std::pow(pi,7), 2) * (144*(std::pow(pi,2)-10)/std::pow(pi,5));
    }

    // checks 
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    unsigned int n_fail = 0;

    for(unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree){
        pcout << "Starting polynomial order: " << poly_degree << std::endl;

        // grid_study refines P0 additinoally here (unused in current param file)
        std::vector<double> grid_size(n_grids);
        std::vector<double> output_error_u(n_grids);
        std::vector<double> output_error_v(n_grids);
        std::vector<double> soln_error_u(n_grids);
        std::vector<double> soln_error_v(n_grids);
        std::vector<double> adj_error_u(n_grids);
        std::vector<double> adj_error_v(n_grids);
        
        const std::vector<int> n_1d_cells = get_number_1d_cells(n_grids);

        dealii::ConvergenceTable convergence_table;

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
        dealii::Triangulation<dim> grid(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
        dealii::parallel::distributed::Triangulation<dim> grid(
            this->mpi_communicator,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::MeshSmoothing::smoothing_on_refinement));
                //dealii::Triangulation<dim>::smoothing_on_refinement |
                //dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

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

            // evaluating functionals from both methods
            double functional_val_u = diffusion_functional.evaluate_function(*dg_u, *physics_u_double);
            double functional_val_v = diffusion_functional.evaluate_function(*dg_v, *physics_v_double);

            // comparison betweent the values, add these to the convergence table
            pcout << std::endl << "Val1 = " << functional_val_u << "\tVal2 = " << functional_val_v << std::endl << std::endl; 

            // evaluating the error of this measure
            double error_functional_u = std::abs(functional_val_u-exact_val);
            double error_functional_v = std::abs(functional_val_v-exact_val);

            pcout << std::endl << "error_val1 = " << error_functional_u << "\terror_val2 = " << error_functional_v << std::endl << std::endl; 

            // // Initializing the adjoints for each problem
            Adjoint<dim, nstate, double> adj_u(*dg_u, diffusion_functional, *physics_u_adtype.get());
            Adjoint<dim, nstate, double> adj_v(*dg_v, diffusion_functional, *physics_v_adtype.get());

            // solving for each coarse adjoint
            dealii::LinearAlgebra::distributed::Vector<double> adjoint_u = adj_u.coarse_grid_adjoint();
            dealii::LinearAlgebra::distributed::Vector<double> adjoint_v = adj_v.coarse_grid_adjoint();

            // using overintegration for estimating the error in the adjoint
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(poly_degree+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg_u->high_order_grid.mapping_fe_field), dg_u->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;

            // examining the convergence of the soln compared to same manufactured solution
            double l2error_soln_u = 0;
            double l2error_soln_v = 0;

            // evaluate the error in the adjoints compared to the opposing manufactured solutions
            double l2error_adj_u = 0;
            double l2error_adj_v = 0;

            // for values at each quadrature point
            std::array<double,nstate> soln_at_q_u;
            std::array<double,nstate> soln_at_q_v;
            std::array<double,nstate> adj_at_q_u;
            std::array<double,nstate> adj_at_q_v;

            std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);
            for(auto cell = dg_u->dof_handler.begin_active(); cell != dg_u->dof_handler.end(); ++cell){
                if(!cell->is_locally_owned()) continue;

                fe_values_extra.reinit (cell);
                cell->get_dof_indices(dofs_indices);

                double cell_l2error_soln_u = 0;
                double cell_l2error_soln_v = 0;
                double cell_l2error_adj_u = 0;
                double cell_l2error_adj_v = 0;
                for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad){
                    std::fill(soln_at_q_u.begin(), soln_at_q_u.end(), 0);
                    std::fill(soln_at_q_v.begin(), soln_at_q_v.end(), 0);
                    std::fill(adj_at_q_u.begin(), adj_at_q_u.end(), 0);
                    std::fill(adj_at_q_v.begin(), adj_at_q_v.end(), 0);

                    for (unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q_u[istate] += dg_u->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                        soln_at_q_v[istate] += dg_v->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                        adj_at_q_u[istate]  += adjoint_u[dofs_indices[idof]]      * fe_values_extra.shape_value_component(idof, iquad, istate);
                        adj_at_q_v[istate]  += adjoint_v[dofs_indices[idof]]      * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }

                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                    for (int istate = 0; istate < nstate; ++istate){
                        const double soln_exact_u = physics_u_double->manufactured_solution_function->value(qpoint, istate);
                        const double soln_exact_v = physics_v_double->manufactured_solution_function->value(qpoint, istate);
                        
                        // comparing the converged solution to the manufactured solution
                        cell_l2error_soln_u += std::pow(soln_at_q_u[istate] - soln_exact_u, 2) * fe_values_extra.JxW(iquad);
                        cell_l2error_soln_v += std::pow(soln_at_q_v[istate] - soln_exact_v, 2) * fe_values_extra.JxW(iquad);

                        // adjoint should convert to the manufactured solution of the opposing case
                        cell_l2error_adj_u += std::pow(adj_at_q_u[istate] - soln_exact_v, 2) * fe_values_extra.JxW(iquad);
                        cell_l2error_adj_v += std::pow(adj_at_q_v[istate] - soln_exact_u, 2) * fe_values_extra.JxW(iquad);

                        // std::cout << "Adjoint value is = " << adj_at_q_u[istate] << std::endl << "and the exact value is = " << adj_exact_u << std::endl;
                    }
                }

                // std::cout << "Cell value is (for u) = " << cell_l2error_adj_u << std::endl;

                l2error_soln_u += cell_l2error_soln_u;
                l2error_soln_v += cell_l2error_soln_v;
                l2error_adj_u += cell_l2error_adj_u;
                l2error_adj_v += cell_l2error_adj_v;
            }
            const double l2error_soln_u_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error_soln_u, mpi_communicator));
            const double l2error_soln_v_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error_soln_v, mpi_communicator));
            const double l2error_adj_u_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error_adj_u, mpi_communicator));
            const double l2error_adj_v_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error_adj_v, mpi_communicator));

            // outputing the solutions obtained
            if(dim == 1){
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
            }else{
                // outputing the adjoint
                adj_u.output_results_vtk(10*poly_degree + igrid);
                adj_v.output_results_vtk(100 + 10*poly_degree + igrid);
            }

            // adding terms to the convergence table
            const int n_dofs = dg_u->dof_handler.n_dofs();
            const double dx = 1.0/pow(n_dofs,(1.0/dim));

            // adding terms to the table
            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", grid.n_global_active_cells());
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_u_val", functional_val_u);
            convergence_table.add_value("soln_v_val", functional_val_v);
            convergence_table.add_value("soln_u_err", error_functional_u);
            convergence_table.add_value("soln_v_err", error_functional_v);
            convergence_table.add_value("soln_u_L2_err", l2error_soln_u_mpi_sum);
            convergence_table.add_value("soln_v_L2_err", l2error_soln_v_mpi_sum);
            convergence_table.add_value("adj_u_L2_err", l2error_adj_u_mpi_sum);
            convergence_table.add_value("adj_v_L2_err", l2error_adj_v_mpi_sum);

            // storing in vectors for convergence checing
            grid_size[igrid] = dx;
            output_error_u[igrid] = error_functional_u;
            output_error_v[igrid] = error_functional_v;
            soln_error_u[igrid] = l2error_soln_u_mpi_sum;
            soln_error_v[igrid] = l2error_soln_v_mpi_sum;
            adj_error_u[igrid] = l2error_adj_u_mpi_sum;
            adj_error_v[igrid] = l2error_adj_v_mpi_sum;
        }

        // obtaining the convergence rates
        convergence_table.evaluate_convergence_rates("soln_u_err", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_v_err", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_u_L2_err", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("soln_v_L2_err", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("adj_u_L2_err", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates("adj_v_L2_err", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx",true);
        convergence_table.set_scientific("soln_u_val",true);
        convergence_table.set_scientific("soln_v_val",true);
        convergence_table.set_scientific("soln_u_err",true);
        convergence_table.set_scientific("soln_v_err",true);
        convergence_table.set_scientific("soln_u_L2_err",true);
        convergence_table.set_scientific("soln_v_L2_err",true);
        convergence_table.set_scientific("adj_u_L2_err",true);
        convergence_table.set_scientific("adj_v_L2_err",true);

        // adding it to the final list
        convergence_table_vector.push_back(convergence_table);

        // setting slope targets for convergence orders
        const double expected_slope_error_functional = 2*poly_degree + 1;
        const double expected_slope_l2error_soln     =   poly_degree + 1;
        const double expected_slope_l2error_adj      =   poly_degree + 1;

        // evaluating the average slopes from the last two steps
        const double avg_slope_error_functional_u   = eval_avg_slope(output_error_u, grid_size, n_grids);
        const double avg_slope_error_functional_v   = eval_avg_slope(output_error_v, grid_size, n_grids);
        const double avg_slope_error_l2error_soln_u = eval_avg_slope(  soln_error_u, grid_size, n_grids);
        const double avg_slope_error_l2error_soln_v = eval_avg_slope(  soln_error_v, grid_size, n_grids);
        const double avg_slope_error_l2error_adj_u  = eval_avg_slope(   adj_error_u, grid_size, n_grids);
        const double avg_slope_error_l2error_adj_v  = eval_avg_slope(   adj_error_v, grid_size, n_grids);

        // diffference from the expected
        const double diff_slope_error_functional_u   = avg_slope_error_functional_u - expected_slope_error_functional;
        const double diff_slope_error_functional_v   = avg_slope_error_functional_v - expected_slope_error_functional;
        const double diff_slope_error_l2error_soln_u = avg_slope_error_l2error_soln_u - expected_slope_l2error_soln;
        const double diff_slope_error_l2error_soln_v = avg_slope_error_l2error_soln_v - expected_slope_l2error_soln;
        const double diff_slope_error_l2error_adj_u  = avg_slope_error_l2error_adj_u - expected_slope_l2error_adj;
        const double diff_slope_error_l2error_adj_v  = avg_slope_error_l2error_adj_v - expected_slope_l2error_adj;

        // tolerance set from the input file
        double slope_deficit_tolerance = -std::abs(manu_grid_conv_param.slope_deficit_tolerance);

        // performing the actual checks
        if(diff_slope_error_functional_u < slope_deficit_tolerance){
            pcout << "Convergence order not achieved for functional_u." << std::endl
                  << "Average order of " << avg_slope_error_functional_u << " instead of expected " << expected_slope_error_functional << std::endl;
            n_fail++;
        }
        if(diff_slope_error_functional_v < slope_deficit_tolerance){
            pcout << "Convergence order not achieved for functional_v." << std::endl
                  << "Average order of " << avg_slope_error_functional_v << " instead of expected " << expected_slope_error_functional << std::endl;
            n_fail++;
        }
        if(diff_slope_error_l2error_soln_u < slope_deficit_tolerance){
            pcout << "Convergence order not achieved for l2error_soln_u." << std::endl
                  << "Average order of " << avg_slope_error_l2error_soln_u << " instead of expected " << expected_slope_l2error_soln << std::endl;
            n_fail++;
        }
        if(diff_slope_error_l2error_soln_v < slope_deficit_tolerance){
            pcout << "Convergence order not achieved for l2error_soln_v." << std::endl
                  << "Average order of " << avg_slope_error_l2error_soln_v << " instead of expected " << expected_slope_l2error_soln << std::endl;
            n_fail++;
        }
        if(diff_slope_error_l2error_adj_u < slope_deficit_tolerance){
            pcout << "Convergence order not achieved for l2error_adj_u." << std::endl
                  << "Average order of " << avg_slope_error_l2error_adj_u << " instead of expected " << expected_slope_l2error_adj << std::endl;
            n_fail++;
        }
        if(diff_slope_error_l2error_adj_v < slope_deficit_tolerance){
            pcout << "Convergence order not achieved for l2error_adj_v." << std::endl
                  << "Average order of " << avg_slope_error_l2error_adj_v << " instead of expected " << expected_slope_l2error_adj << std::endl;
            n_fail++;
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

    return n_fail;
}

// computes the average error of the last two slopes
double eval_avg_slope(std::vector<double> error, std::vector<double> grid_size, unsigned int n_grids){
    const double last_slope_error = log(error[n_grids-1]/error[n_grids-2])/(log(grid_size[n_grids-1]/grid_size[n_grids-2]));
    double prev_slope_error = last_slope_error;
    if(n_grids > 2){
        prev_slope_error = log(error[n_grids-2]/error[n_grids-3])/(log(grid_size[n_grids-2]/grid_size[n_grids-3]));
    }
    return 0.5*(last_slope_error+prev_slope_error);
}

template class DiffusionExactAdjoint <PHILIP_DIM,1>;

} // Tests namespace
} // PHiLiP namespace