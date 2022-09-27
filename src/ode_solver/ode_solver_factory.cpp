#include "ode_solver_factory.h"
#include "parameters/all_parameters.h"
#include "ode_solver_base.h"
//#include "runge_kutta_ode_solver.h"
#include "explicit_ode_solver.h"
#include "implicit_ode_solver.h"
#include "rrk_explicit_ode_solver.h"
#include "pod_galerkin_ode_solver.h"
#include "pod_petrov_galerkin_ode_solver.h"
#include <deal.II/distributed/solution_transfer.h>
#include "runge_kutta_methods/runge_kutta_methods.h"
#include "runge_kutta_methods/rk_tableau_base.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    pcout << "Creating ODE Solver..." << std::endl;
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    const ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if((ode_solver_type == ODEEnum::runge_kutta_solver)||(ode_solver_type == ODEEnum::rrk_explicit_solver))     
        return create_RungeKuttaODESolver(dg_input);
    if(ode_solver_type == ODEEnum::implicit_solver)         
        return std::make_shared<ImplicitODESolver<dim,real,MeshType>>(dg_input);
    else {
        display_error_ode_solver_factory(ode_solver_type, false);
        return nullptr;
    }
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    pcout << "Creating ODE Solver..." << std::endl;
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    const ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver)
        return std::make_shared<PODGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver) 
        return std::make_shared<PODPetrovGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    else {
        display_error_ode_solver_factory(ode_solver_type, true);
        return nullptr;
    }
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    pcout << "Creating ODE Solver..." << std::endl;
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    if((ode_solver_type == ODEEnum::runge_kutta_solver)||(ode_solver_type == ODEEnum::rrk_explicit_solver))     
        return create_RungeKuttaODESolver(dg_input);
    if(ode_solver_type == ODEEnum::implicit_solver)         
        return std::make_shared<ImplicitODESolver<dim,real,MeshType>>(dg_input);
    else {
        display_error_ode_solver_factory(ode_solver_type, false);
        return nullptr;
    }
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim,real,MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    pcout << "Creating ODE Solver..." << std::endl;
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver) 
        return std::make_shared<PODGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver) 
        return std::make_shared<PODPetrovGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    else {
        display_error_ode_solver_factory(ode_solver_type, true);
        return nullptr;
    }
}


template <int dim, typename real, typename MeshType>
void ODESolverFactory<dim,real,MeshType>::display_error_ode_solver_factory(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, bool reduced_order) 
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;

    std::string solver_string;    
    if (ode_solver_type == ODEEnum::runge_kutta_solver)            solver_string = "runge_kutta";
    if (ode_solver_type == ODEEnum::implicit_solver)               solver_string = "implicit";
    if (ode_solver_type == ODEEnum::rrk_explicit_solver)           solver_string = "rrk_explicit";
    if (ode_solver_type == ODEEnum::pod_galerkin_solver)           solver_string = "pod_galerkin";
    if (ode_solver_type == ODEEnum::pod_petrov_galerkin_solver)    solver_string = "pod_petrov_galerkin";
    else solver_string = "undefined";

    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    pcout << "********************************************************************" << std::endl;
    pcout << "Can't create ODE solver since solver type is not clear." << std::endl;
    pcout << "Solver type specified: " << solver_string << std::endl;
    pcout << "Solver type possible: " << std::endl;
    if(reduced_order){
        pcout <<  "pod_galerkin" << std::endl;
        pcout <<  "pod_petrov_galerkin" << std::endl;
    }
    else{
        pcout <<  "runge_kutta" << std::endl;
        pcout <<  "implicit" << std::endl;
        pcout <<  "rrk_explicit" << std::endl;
        pcout << "    With rrk_explicit only being valid for " <<std::endl;
        pcout << "    pde_type = burgers, use_collocated_nodes = true and dim = 1" <<std::endl;
    }
    pcout << "********************************************************************" << std::endl;
    std::abort();
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_RungeKuttaODESolver(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau = create_RKTableau(dg_input);
    const int n_rk_stages = dg_input->all_parameters->ode_solver_param.n_rk_stages;
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    const ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if (ode_solver_type == ODEEnum::runge_kutta_solver) {
        // Hard-coded templating of n_rk_stages because it is not known at compile time
        pcout << "Creating Runge Kutta ODE Solver with " 
              << n_rk_stages << " stage(s)..." << std::endl;
        if (n_rk_stages == 1){
            return std::make_shared<RungeKuttaODESolver<dim,real,1,MeshType>>(dg_input,rk_tableau);
        }
        if (n_rk_stages == 2){
            return std::make_shared<RungeKuttaODESolver<dim,real,2,MeshType>>(dg_input,rk_tableau);
        }
        if (n_rk_stages == 3){
            return std::make_shared<RungeKuttaODESolver<dim,real,3,MeshType>>(dg_input,rk_tableau);
        }
        if (n_rk_stages == 4){
            return std::make_shared<RungeKuttaODESolver<dim,real,4,MeshType>>(dg_input,rk_tableau);
        }
        else{
            pcout << "Error: invalid number of stages. Aborting..." << std::endl;
            std::abort();
            return nullptr;
        }
    }
    if (ode_solver_type == ODEEnum::rrk_explicit_solver){
        if constexpr(dim==1){
            const bool use_collocated_nodes = dg_input->all_parameters->use_collocated_nodes;
            using PDEEnum = Parameters::AllParameters::PartialDifferentialEquation;
            const PDEEnum pde_type = dg_input->all_parameters->pde_type;
            const bool use_inviscid_burgers = (pde_type == PDEEnum::burgers_inviscid);
            if (use_collocated_nodes && use_inviscid_burgers){
                pcout << "Creating Relaxation Runge Kutta ODE Solver with " 
                      << n_rk_stages << " stage(s)..." << std::endl;
                if (n_rk_stages == 1){
                    return std::make_shared<RRKExplicitODESolver<dim,real,1,MeshType>>(dg_input,rk_tableau);
                }
                if (n_rk_stages == 2){
                    return std::make_shared<RRKExplicitODESolver<dim,real,2,MeshType>>(dg_input,rk_tableau);
                }
                if (n_rk_stages == 3){
                    return std::make_shared<RRKExplicitODESolver<dim,real,3,MeshType>>(dg_input,rk_tableau);
                }
                if (n_rk_stages == 4){
                    return std::make_shared<RRKExplicitODESolver<dim,real,4,MeshType>>(dg_input,rk_tableau);
                }
                else{
                    pcout << "Error: invalid number of stages. Aborting..." << std::endl;
                    std::abort();
                    return nullptr;
                }
            } else {
                pcout << "Error: RRK has only been tested on collocated nodes with Burgers. Aborting..."<<std::endl;
                std::abort();
            }
        } else {
            pcout << "Error: RRK has only been tested on 1D calculations. Aborting..."<<std::endl;
            std::abort();
        }
    }
    else {
        display_error_ode_solver_factory(ode_solver_type, false);
        return nullptr;
    }
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<RKTableauBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_RKTableau(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    using RKMethodEnum = Parameters::ODESolverParam::RKMethodEnum;
    const RKMethodEnum rk_method = dg_input->all_parameters->ode_solver_param.runge_kutta_method;

    const int n_rk_stages = dg_input->all_parameters->ode_solver_param.n_rk_stages;

    if (rk_method == RKMethodEnum::ssprk3_ex)   return std::make_shared<SSPRK3Explicit<dim, real, MeshType>> (n_rk_stages, "3rd order SSP (explicit)");
    if (rk_method == RKMethodEnum::rk4_ex)      return std::make_shared<RK4Explicit<dim, real, MeshType>>    (n_rk_stages, "4th order classical RK (explicit)");
    if (rk_method == RKMethodEnum::euler_ex) {
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
        if (ode_solver_type == ODEEnum::rrk_explicit_solver) {
            //forward Euler is invalid for RRK: sum(b_i*a_ij) = 0 (see Lemma 2.1 in Ketcheson 2019)
            pcout << "Error: RRK is not valid for Forward Euler. Aborting..." << std::endl;
            std::abort();
            return nullptr;
        } else return std::make_shared<EulerExplicit<dim, real, MeshType>>  (n_rk_stages, "Forward Euler (explicit)");
    }
    if constexpr(dim==1){
        //Only tested in 1D with Burgers and advection
        using PDEEnum = Parameters::AllParameters::PartialDifferentialEquation;
        const PDEEnum pde_type = dg_input->all_parameters->pde_type;
        if ((pde_type==PDEEnum::burgers_inviscid) || (pde_type==PDEEnum::advection)){
            if (rk_method == RKMethodEnum::euler_im)    return std::make_shared<EulerImplicit<dim, real, MeshType>>  (n_rk_stages, "Implicit Euler (implicit)");
            if (rk_method == RKMethodEnum::dirk_2_im)   return std::make_shared<DIRK2Implicit<dim, real, MeshType>>  (n_rk_stages, "2nd order diagonally-implicit (implicit)");
            else {
                pcout << "Error: invalid RK method. Aborting..." << std::endl;
                std::abort();
                return nullptr;
            }
        } else {
            pcout << "Error: implicit RK only tested for 1D advection and Burgers'. Aborting..."  << std::endl;
            std::abort();
            return nullptr;
        }
    }
    else {
        pcout << "Error: invalid RK method, or attempted to use implicit RK with dim > 1. Aborting..." << std::endl;
        std::abort();
        return nullptr;
    }
}

template class ODESolverFactory<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ODESolverFactory<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
    template class ODESolverFactory<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
