#include "dg_factory.hpp"
#include "weak_dg.hpp"
#include "strong_dg.hpp"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
std::shared_ptr< DGBase<dim,real,MeshType> >
DGFactory<dim,real,MeshType>
::create_discontinuous_galerkin(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

    PDE_enum pde_type = parameters_input->pde_type;
    if (parameters_input->use_weak_form) {
        if (pde_type == PDE_enum::advection) {
            return std::make_shared< DGWeak<dim,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::advection_vector) {
            return std::make_shared< DGWeak<dim,2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::diffusion) {
            return std::make_shared< DGWeak<dim,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return std::make_shared< DGWeak<dim,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_inviscid) {
            return std::make_shared< DGWeak<dim,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_rewienski) {
            return std::make_shared< DGWeak<dim,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::euler) {
            return std::make_shared< DGWeak<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes) {
            return std::make_shared< DGWeak<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
    } else {
        if (pde_type == PDE_enum::advection) {
            return std::make_shared< DGStrong<dim,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::advection_vector) {
            return std::make_shared< DGStrong<dim,2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::diffusion) {
            return std::make_shared< DGStrong<dim,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return std::make_shared< DGStrong<dim,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_inviscid) {
            return std::make_shared< DGStrong<dim,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_rewienski) {
            return std::make_shared< DGStrong<dim,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::euler) {
            return std::make_shared< DGStrong<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes) {
            return std::make_shared< DGStrong<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
    }
    std::cout << "Can't create DGBase in create_discontinuous_galerkin(). Invalid PDE type: " << pde_type << std::endl;
    return nullptr;
}

template <int dim, typename real, typename MeshType>
std::shared_ptr< DGBase<dim,real,MeshType> >
DGFactory<dim,real,MeshType>
::create_discontinuous_galerkin(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
{
    return create_discontinuous_galerkin(parameters_input, degree, max_degree_input, degree+1, triangulation_input);
}

template <int dim, typename real, typename MeshType>
std::shared_ptr< DGBase<dim,real,MeshType> >
DGFactory<dim,real,MeshType>
::create_discontinuous_galerkin(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const std::shared_ptr<Triangulation> triangulation_input)
{
    return create_discontinuous_galerkin(parameters_input, degree, degree, triangulation_input);
}

template class DGFactory <PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGFactory <PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM!=1
template class DGFactory <PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
