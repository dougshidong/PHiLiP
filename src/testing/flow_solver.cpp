
// for the actual test:
#include "tests.h"
#include "flow_solver.h" // which includes all required for InitialConditionFunction

// for the grid:
#include "grid_refinement_study.h"
// -- which includes:
// #include "tests.h"
// #include "dg/dg.h"
// #include "physics/physics.h"
// #include "parameters/all_parameters.h"
// #include "grid_refinement/gnu_out.h"


namespace PHiLiP {
namespace Tests {

// ========================================================
// TAYLOR GREEN VORTEX -- Initial Condition
// ========================================================
// done
template <int dim, typename real>
InitialConditionFunction_TaylorGreenVortex<dim,real>
::InitialConditionFunction_TaylorGreenVortex (
    const unsigned int nstate,
    const double       gamma_gas,
    const double       mach_inf_sqr)
    : InitialConditionFunction_FlowSolver<dim,real>(nstate)
    , gamma_gas(gamma_gas)
    , mach_inf_sqr(mach_inf_sqr)    
{
    // casting `nstate` as `int` to avoid errors
    static_assert(((int)nstate)==dim+2, "Tests::InitialConditionFunction_TaylorGreenVortex() should be created with nstate=dim+2");
}
// done
template <int dim, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,real>
::primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    // Note: This is in non-dimensional form (free-stream values as reference)
    real value = 0.;
    if constexpr(dim == 3) {
        const real x = point[0], y = point[1], z = point[2];
        
        if(istate==0) {
            // density
            value = 1.0;
        }
        if(istate==1) {
            // x-velocity
            value = sin(x)*cos(y)*cos(z); 
        }
        if(istate==2) {  
            // y-velocity
            value = -cos(x)*sin(y)*cos(z);
        }
        if(istate==3) {
            // z-velocity
            value = 0.0;
        }
        if(istate==4) {
            // pressure
            value = 1.0/(gamma_gas*mach_inf_sqr) + (1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(cos(2.0*z)+2.0);
        }
    }
    return value;
}
// done
template <int dim, typename real>
real InitialConditionFunction_TaylorGreenVortex<dim,real>
::convert_primitive_to_conversative_value(
    const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    if (dim == 3) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        const real w   = primitive_value(point,3);
        const real p   = primitive_value(point,4);

        // convert primitive to conservative solution
        if(istate==0) value = rho; // density
        if(istate==1) value = rho*u; // x-momentum
        if(istate==2) value = rho*v; // y-momentum
        if(istate==3) value = rho*w; // z-momentum
        if(istate==4) value = p/(1.4-1.0) + 0.5*rho*(u*u + v*v + w*w); // total energy
    }

    return value;
}
// done
template <int dim, typename real>
inline real InitialConditionFunction_TaylorGreenVortex<dim,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real value = 0.0;
    value = convert_primitive_to_conversative_value(point,istate);
    return value;
}
// done
template <int dim, typename real>
dealii::Tensor<1,dim,real> InitialConditionFunction_TaylorGreenVortex<dim,real>
::primitive_gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    // Gradients of primitive variables 
    if (dim == 2) {
        const real x = point[0], y = point[1], z = point[2];

        if(istate==0) {
            // density
            gradient[0] = 0.0; // dx
            gradient[1] = 0.0; // dy
            gradient[2] = 0.0; // dz
        }
        if(istate==1) {
            // x-velocity
            gradient[0] =  cos(x)*cos(y)*cos(z); // dx
            gradient[1] = -sin(x)*sin(y)*cos(z); // dy
            gradient[2] = -sin(x)*cos(y)*sin(z); // dz
        }
        if(istate==2) {
            // y-velocity
            gradient[0] = -sin(x)*sin(y)*cos(z); // dx
            gradient[1] = -cos(x)*cos(y)*cos(z); // dy
            gradient[2] =  cos(x)*sin(y)*sin(z); // dz
        }
        if(istate==3) {
            // z-velocity
            gradient[0] = 0.0; // dx
            gradient[1] = 0.0; // dy
            gradient[2] = 0.0; // dz
        }
        if(istate==4) {
            // pressure
            gradient[0] = -(1.0/8.0)*sin(2.0*x)*(cos(2.0*z)+2.0); // dx
            gradient[1] = -(1.0/8.0)*sin(2.0*y)*(cos(2.0*z)+2.0); // dy
            gradient[2] = -(1.0/8.0)*(cos(2.0*x)+cos(2.0*y))*sin(2.0*z); // dz
        }
    }
    return gradient;
}
// done
template <int dim, typename real>
dealii::Tensor<1,dim,real> InitialConditionFunction_TaylorGreenVortex<dim,real>
::convert_primitive_to_conversative_gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient;
    if (dim == 3) {
        const real rho = primitive_value(point,0);
        const real u   = primitive_value(point,1);
        const real v   = primitive_value(point,2);
        const real w   = primitive_value(point,3);
        const real p   = primitive_value(point,4);
        const dealii::Tensor<1,dim,real> rho_grad = primitive_gradient(point,0);
        const dealii::Tensor<1,dim,real> u_grad   = primitive_gradient(point,1);
        const dealii::Tensor<1,dim,real> v_grad   = primitive_gradient(point,2);
        const dealii::Tensor<1,dim,real> w_grad   = primitive_gradient(point,3);
        const dealii::Tensor<1,dim,real> p_grad   = primitive_gradient(point,4);
        
        // convert to primitive to gradient of conservative variables using product rule
        if(istate==0) {
            // density
            for(int d=0; d<dim; d++) { 
                gradient[d] = rho_grad[d];
            }
        }
        if(istate==1) {
            // x-momentum
            for(int d=0; d<dim; d++) {
                gradient[d] = u*rho_grad[d] + rho*u_grad[d];
            }
        }
        if(istate==2) {
            // y-momentum
            for(int d=0; d<dim; d++) {
                gradient[d] = v*rho_grad[d] + rho*v_grad[d];
            }
        }
        if(istate==3) {
            // z-momentum
            for(int d=0; d<dim; d++) {
                gradient[d] = w*rho_grad[d] + rho*w_grad[d];
            }
        }
        if(istate==4) {
            // total energy
            for(int d=0; d<dim; d++) {
                gradient[d] = p_grad[d]/(1.4-1.0) + 0.5*rho_grad[d]*(u*u + v*v + w*w) + rho*(u*u_grad[d]+v*v_grad[d]+w*w_grad[d]);
            }
        }
    }
    return gradient;
}
// done
template <int dim, typename real>
inline dealii::Tensor<1,dim,real> InitialConditionFunction_TaylorGreenVortex<dim,real>
::gradient (const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    dealii::Tensor<1,dim,real> gradient = convert_primitive_to_conversative_gradient(point, istate);
    return gradient;
}
// ========================================================
// FLOW SOLVER -- Initial Condition Base Class + Factory
// ========================================================
// done
template <int dim, typename real>
InitialConditionFunction_FlowSolver<dim,real>
::InitialConditionFunction_FlowSolver (const unsigned int nstate)
    :
    dealii::Function<dim,real>(nstate,0.0) // 0.0 denotes initial time (t=0)
    , nstate(nstate)
{ 
    // Nothing to do here yet
}
// done
template <int dim, typename real>
std::shared_ptr< InitialConditionFunction_FlowSolver<dim,real> > 
InitialConditionFactory_FlowSolver<dim,real>::create_InitialConditionFunction_FlowSolver(
    Parameters::AllParameters const *const param, 
    int                                    nstate)
{
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    FlowCaseEnum flow_type = param->flow_solver_param.flow_case_type;

    return create_InitialConditionFunction_FlowSolver(flow_type, nstate);
}
// done
template <int dim, typename real>
std::shared_ptr< InitialConditionFunction_FlowSolver<dim,real> >
InitialConditionFactory_FlowSolver<dim,real>::create_InitialConditionFunction_FlowSolver(
    Parameters::FlowSolverParam::FlowCaseType flow_type,
    int                                       nstate)
{
    if(flow_type == FlowCaseEnum::inviscid_taylor_green_vortex){
        if constexpr((dim==3) /*&& (nstate==dim+2)*/) {
            return std::make_shared<InitialConditionFunction_TaylorGreenVortex<dim,real>>(nstate);
        }
    }else if(flow_type == FlowCaseEnum::viscous_taylor_green_vortex){
        if constexpr((dim==3) /*&& (nstate==dim+2)*/) {
            return std::make_shared<InitialConditionFunction_TaylorGreenVortex<dim,real>>(nstate);
        }
    }else{
        std::cout << "Invalid Flow Case Type." << std::endl;
    }

    return nullptr;
}
// ========================================================
// FLOW SOLVER TEST CASE -- What runs the test
// ========================================================
template <int dim, int nstate>
FlowSolver<dim, nstate>::FlowSolver(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{
    // Assign initial condition function from the InitialConditionFunction
    // initial_condition_function = InitialConditionFactory_FlowSolver<dim,double>::create_InitialConditionFunction_FlowSolver(parameters_input, nstate);
}

// template <int dim, int nstate>
// void FlowSolver<dim,nstate>::get_grid() const
// {
//     // -- 
//     /** Triangulation to store the grid.
//      *  In 1D, dealii::Triangulation<dim> is used.
//      *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
//      */
//     // For 2D and 3D, the MeshType == Triangulation, is
//     using Triangulation = dealii::parallel::distributed::Triangulation<dim>; // Triangulation == MeshType
//     // create 'grid' pointer
//     std::shared_ptr<Triangulation> grid = MeshFactory<Triangulation>::create_MeshType(this->mpi_communicator);
//     // Create the grid refinement study object: grs
//     // std::shared_ptr<TestBase> grs = std::make_shared < TestBase::GridRefinementStudy<dim,nstate,Triangulation> >(param);
//     GridRefinementStudy<dim,nstate,Triangulation> grs = GridRefinementStudy(param);
//     // Generate the grid based on parameter file
//     grs.get_grid(grid, grs_param);
// }

// template <int dim, int nstate>
// void FlowSolver<dim,nstate>::initialize_solution(PHiLiP::DGBase<dim,double> &dg, const PHiLiP::Physics::PhysicsBase<dim,nstate,double> &physics) const
// {
//     dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
//     solution_no_ghost.reinit(dg.locally_owned_dofs, MPI_COMM_WORLD);
//     dealii::VectorTools::interpolate(dg.dof_handler, *initial_condition_function, solution_no_ghost);
//     dg.solution = solution_no_ghost;
// }

template <int dim, int nstate>
int FlowSolver<dim,nstate>::run_test() const
{
    pcout << " Running Flow Solver. " << std::endl;
    // TO DO: Display the flow case string
    //----------------------------------------------------
    // Parameters
    //----------------------------------------------------
    const Parameters::AllParameters param                = *(TestsBase::all_parameters);
    const Parameters::GridRefinementStudyParam grs_param = param.grid_refinement_study_param;
    //----------------------------------------------------
    // Initialization
    //----------------------------------------------------
    const unsigned int poly_degree = grs_param.poly_degree;
    // const unsigned int poly_degree_max  = grs_param.poly_degree_max;
    // const unsigned int poly_degree_grid = grs_param.poly_degree_grid;
    // const unsigned int num_refinements = grs_param.num_refinements;
    //----------------------------------------------------
    // Physics
    //----------------------------------------------------
    // creating the physics object
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > physics_double = Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&param);
    //----------------------------------------------------
    // Grid -- (fixed for now)
    //----------------------------------------------------
    // TO DO: Move this to a member function
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    // For 2D and 3D, the MeshType == Triangulation, is
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>; // Triangulation == MeshType
    // create 'grid' pointer
    std::shared_ptr<Triangulation> grid = MeshFactory<Triangulation>::create_MeshType(this->mpi_communicator);
    // Create the grid refinement study object: grs
    // std::shared_ptr<TestBase> grs = std::make_shared < TestBase::GridRefinementStudy<dim,nstate,Triangulation> >(param);
    GridRefinementStudy<dim,nstate,Triangulation> grs = GridRefinementStudy(param);
    // Generate the grid based on parameter file
    grs.get_grid(grid, grs_param);
    //----------------------------------------------------
    // Discontinuous Galerkin
    //----------------------------------------------------
    // Create DG object using the factory
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, grid);
    dg->allocate_system ();
    //----------------------------------------------------
    // Initialize the solution
    //----------------------------------------------------
    // Create initial condition function from InitialConditionFactory_FlowSolver
    // TO DO: Drop the "_FlowSolver"
    std::shared_ptr< InitialConditionFunction_FlowSolver<dim,double> > initial_condition_function 
                = InitialConditionFactory_FlowSolver<dim,double>::create_InitialConditionFunction_FlowSolver(parameters_input, nstate);
    // TO DO: Move this to a member function
    std::cout << "Initializing solution with initial condition function." << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *initial_condition_function, solution_no_ghost);
    dg->solution = solution_no_ghost;
}

#if PHILIP_DIM==3
    // InitialConditionFunction
    template class InitialConditionFunction_FlowSolver <PHILIP_DIM,double>;
    
    template class InitialConditionFunction_TaylorGreenVortex <PHILIP_DIM,double>;
    
    template class FlowSolver <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

