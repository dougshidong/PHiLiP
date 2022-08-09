#include "ode_solver_factory.h"
#include "parameters/all_parameters.h"
#include "ode_solver_base.h"
#include "explicit_ode_solver.h"
#include "implicit_ode_solver.h"
#include "rrk_explicit_ode_solver.h"
#include "pod_galerkin_ode_solver.h"
#include "pod_petrov_galerkin_ode_solver.h"
#include <deal.II/distributed/solution_transfer.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
    std::cout << "Creating ODE Solver..." << std::endl;
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if(ode_solver_type == ODEEnum::explicit_solver)        return std::make_shared<ExplicitODESolver<dim,real,MeshType>>(dg_input);
    if(ode_solver_type == ODEEnum::implicit_solver)        return std::make_shared<ImplicitODESolver<dim,real,MeshType>>(dg_input);
    if constexpr(dim==1){
        //RRK is only implemented for Burgers on collocated nodes, 1D
        const bool use_collocated_nodes = dg_input->all_parameters->use_collocated_nodes;
        using PDEEnum = Parameters::AllParameters::PartialDifferentialEquation;
        const PDEEnum pde_type = dg_input->all_parameters->pde_type;
        const bool use_inviscid_burgers = (pde_type == PDEEnum::burgers_inviscid);
        if ((ode_solver_type == ODEEnum::rrk_explicit_solver) && 
                use_collocated_nodes && use_inviscid_burgers){
            return std::make_shared<RRKExplicitODESolver<dim,real,MeshType>>(dg_input);
        }
        else{
            display_error_ode_solver_factory(ode_solver_type, false);
            return nullptr;
        }
    }
    else {
        display_error_ode_solver_factory(ode_solver_type, false);
        return nullptr;
    }
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver) return std::make_shared<PODGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver) return std::make_shared<PODPetrovGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    else {
        display_error_ode_solver_factory(ode_solver_type, true);
        return nullptr;
    }
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    if(ode_solver_type == ODEEnum::explicit_solver)        return std::make_shared<ExplicitODESolver<dim,real,MeshType>>(dg_input);
    if(ode_solver_type == ODEEnum::implicit_solver)        return std::make_shared<ImplicitODESolver<dim,real,MeshType>>(dg_input);
    if constexpr(dim==1){
        //RRK is only implemented for Burgers on collocated nodes, 1D
        const bool use_collocated_nodes = dg_input->all_parameters->use_collocated_nodes;
        using PDEEnum = Parameters::AllParameters::PartialDifferentialEquation;
        const PDEEnum pde_type = dg_input->all_parameters->pde_type;
        const bool use_inviscid_burgers = (pde_type == PDEEnum::burgers_inviscid);
        if ((ode_solver_type == ODEEnum::rrk_explicit_solver) && 
                use_collocated_nodes && use_inviscid_burgers){
            return std::make_shared<RRKExplicitODESolver<dim,real,MeshType>>(dg_input);
        }
        else{
            display_error_ode_solver_factory(ode_solver_type, false);
            return nullptr;
        }
    }
    else {
        display_error_ode_solver_factory(ode_solver_type, false);
        return nullptr;
    }
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim,real,MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver) return std::make_shared<PODGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver) return std::make_shared<PODPetrovGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    else {
        display_error_ode_solver_factory(ode_solver_type, true);
        return nullptr;
    }
}


template <int dim, typename real, typename MeshType>
void ODESolverFactory<dim,real,MeshType>::display_error_ode_solver_factory(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, bool reduced_order) {
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;

    std::string solver_string;    
    if (ode_solver_type == ODEEnum::explicit_solver)               solver_string = "explicit";
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
        pcout <<  "explicit" << std::endl;
        pcout <<  "implicit" << std::endl;
        pcout <<  "rrk_explicit" << std::endl;
        pcout << "    With rrk_explicit only being valid for " <<std::endl;
        pcout << "    pde_type = burgers, use_collocated_nodes = true and dim = 1" <<std::endl;
    }
    pcout << "********************************************************************" << std::endl;
    std::abort();
}

template class ODESolverFactory<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ODESolverFactory<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
    template class ODESolverFactory<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
