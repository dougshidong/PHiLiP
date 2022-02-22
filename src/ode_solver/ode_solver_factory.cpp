#include "ode_solver_factory.h"
#include "parameters/all_parameters.h"
#include "ode_solver_base.h"
#include "explicit_ode_solver.h"
#include "implicit_ode_solver.h"
#include "pod_galerkin_ode_solver.h"
#include "pod_petrov_galerkin_ode_solver.h"
#include <deal.II/distributed/solution_transfer.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if(ode_solver_type == ODEEnum::explicit_solver) return std::make_shared<ExplicitODESolver<dim,real,MeshType>>(dg_input);
    if(ode_solver_type == ODEEnum::implicit_solver) return std::make_shared<ImplicitODESolver<dim,real,MeshType>>(dg_input);
    else {
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    pcout << "********************************************************************" << std::endl;
    pcout << "Can't create ODE solver since solver type is not clear." << std::endl;
    pcout << "Solver type specified: " << ode_solver_type << std::endl;
    pcout << "Solver type possible: " << std::endl;
    pcout <<  ODEEnum::explicit_solver << std::endl;
    pcout <<  ODEEnum::implicit_solver << std::endl;
    pcout <<  ODEEnum::pod_galerkin_solver << std::endl;
    pcout <<  ODEEnum::pod_petrov_galerkin_solver << std::endl;
    pcout << "********************************************************************" << std::endl;
    std::abort();
    return nullptr;
}
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver(std::shared_ptr< DGBase<dim,real,MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod)
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    ODEEnum ode_solver_type = dg_input->all_parameters->ode_solver_param.ode_solver_type;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver) return std::make_shared<PODGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver) return std::make_shared<PODPetrovGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    else {
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        pcout << "********************************************************************" << std::endl;
        pcout << "Can't create ODE solver since solver type is not clear." << std::endl;
        pcout << "Solver type specified: " << ode_solver_type << std::endl;
        pcout << "Solver type possible: " << std::endl;
        pcout <<  ODEEnum::explicit_solver << std::endl;
        pcout <<  ODEEnum::implicit_solver << std::endl;
        pcout <<  ODEEnum::pod_galerkin_solver << std::endl;
        pcout <<  ODEEnum::pod_petrov_galerkin_solver << std::endl;
        pcout << "********************************************************************" << std::endl;
        std::abort();
        return nullptr;
    }
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim,real,MeshType> > dg_input)
{
using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
if(ode_solver_type == ODEEnum::explicit_solver) return std::make_shared<ExplicitODESolver<dim,real,MeshType>>(dg_input);
if(ode_solver_type == ODEEnum::implicit_solver) return std::make_shared<ImplicitODESolver<dim,real,MeshType>>(dg_input);
else {
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    pcout << "********************************************************************" << std::endl;
    pcout << "Can't create ODE solver since solver type is not clear." << std::endl;
    pcout << "Solver type specified: " << ode_solver_type << std::endl;
    pcout << "Solver type possible: " << std::endl;
    pcout <<  ODEEnum::explicit_solver << std::endl;
    pcout <<  ODEEnum::implicit_solver << std::endl;
    pcout <<  ODEEnum::pod_galerkin_solver << std::endl;
    pcout <<  ODEEnum::pod_petrov_galerkin_solver << std::endl;
    pcout << "********************************************************************" << std::endl;
    std::abort();
    return nullptr;
}
}

template <int dim, typename real, typename MeshType>
std::shared_ptr<ODESolverBase<dim,real,MeshType>> ODESolverFactory<dim,real,MeshType>::create_ODESolver_manual(Parameters::ODESolverParam::ODESolverEnum ode_solver_type, std::shared_ptr< DGBase<dim,real,MeshType> > dg_input, std::shared_ptr<ProperOrthogonalDecomposition::POD<dim>> pod)
{
    using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
    if(ode_solver_type == ODEEnum::pod_galerkin_solver) return std::make_shared<PODGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    if(ode_solver_type == ODEEnum::pod_petrov_galerkin_solver) return std::make_shared<PODPetrovGalerkinODESolver<dim,real,MeshType>>(dg_input, pod);
    else {
        dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
        pcout << "********************************************************************" << std::endl;
        pcout << "Can't create ODE solver since solver type is not clear." << std::endl;
        pcout << "Solver type specified: " << ode_solver_type << std::endl;
        pcout << "Solver type possible: " << std::endl;
        pcout <<  ODEEnum::explicit_solver << std::endl;
        pcout <<  ODEEnum::implicit_solver << std::endl;
        pcout <<  ODEEnum::pod_galerkin_solver << std::endl;
        pcout <<  ODEEnum::pod_petrov_galerkin_solver << std::endl;
        pcout << "********************************************************************" << std::endl;
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
