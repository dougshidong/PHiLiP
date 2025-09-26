#include "dg_factory.hpp"
#include "weak_dg.hpp"
#include "strong_dg.hpp"

namespace PHiLiP {

template <int dim, int nspecies, typename real, typename MeshType>
std::shared_ptr< DGBase<dim,nspecies,real,MeshType> >
DGFactory<dim,nspecies,real,MeshType>
::create_discontinuous_galerkin(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
{
    using PDE_enum   = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = parameters_input->pde_type;
    using Model_enum = Parameters::AllParameters::ModelType;
    const Model_enum model_type = parameters_input->model_type;
    using RANSModel_enum = Parameters::PhysicsModelParam::ReynoldsAveragedNavierStokesModel;
    const RANSModel_enum rans_model_type = parameters_input->physics_model_param.RANS_model_type;

    if (parameters_input->use_weak_form) {
        if (pde_type == PDE_enum::advection && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::advection_vector && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::diffusion && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::convection_diffusion && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_inviscid && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_viscous && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_rewienski && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::euler && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::real_gas) {
            return std::make_shared< DGWeak<dim,nspecies,dim+2+(nspecies-1),real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::multi_species_calorically_perfect_euler) {
            return std::make_shared< DGWeak<dim,nspecies,dim+2+(nspecies-1),real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::reynolds_averaged_navier_stokes) && (rans_model_type == RANSModel_enum::SA_negative) && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,dim+3,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
#if PHILIP_DIM==3
        else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::large_eddy_simulation) && nspecies==1) {
            return std::make_shared< DGWeak<dim,nspecies,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
#endif
    } else {
        if (pde_type == PDE_enum::advection && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::advection_vector && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::diffusion && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::convection_diffusion && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,1,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_inviscid && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_viscous && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_rewienski && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::euler && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::real_gas) {
            return std::make_shared< DGStrong<dim,nspecies,dim+2+(nspecies-1),real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::multi_species_calorically_perfect_euler) {
            return std::make_shared< DGStrong<dim,nspecies,dim+2+(nspecies-1),real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);        
        } else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::reynolds_averaged_navier_stokes) && (rans_model_type == RANSModel_enum::SA_negative) && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,dim+3,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
#if PHILIP_DIM==3
        else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::large_eddy_simulation) && nspecies==1) {
            return std::make_shared< DGStrong<dim,nspecies,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
#endif
    }
    std::cout << "Can't create DGBase in create_discontinuous_galerkin(). Invalid PDE type: " << pde_type << std::endl;
    return nullptr;
}

template <int dim, int nspecies, typename real, typename MeshType>
std::shared_ptr< DGBase<dim,nspecies,real,MeshType> >
DGFactory<dim,nspecies,real,MeshType>
::create_discontinuous_galerkin(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
{
    return create_discontinuous_galerkin(parameters_input, degree, max_degree_input, degree+1, triangulation_input);
}

template <int dim, int nspecies, typename real, typename MeshType>
std::shared_ptr< DGBase<dim,nspecies,real,MeshType> >
DGFactory<dim,nspecies,real,MeshType>
::create_discontinuous_galerkin(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const std::shared_ptr<Triangulation> triangulation_input)
{
    return create_discontinuous_galerkin(parameters_input, degree, degree, triangulation_input);
}

template class DGFactory <PHILIP_DIM, PHILIP_SPECIES, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGFactory <PHILIP_DIM, PHILIP_SPECIES, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM!=1
template class DGFactory <PHILIP_DIM, PHILIP_SPECIES, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
