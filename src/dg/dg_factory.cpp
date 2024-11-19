#include "dg_factory.hpp"
#include "weak_dg.hpp"
#include "strong_dg.hpp"
#include "strong_dg_les.hpp"

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
    using PDE_enum   = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = parameters_input->pde_type;
    using Model_enum = Parameters::AllParameters::ModelType;
    const Model_enum model_type = parameters_input->model_type;
    using RANSModel_enum = Parameters::PhysicsModelParam::ReynoldsAveragedNavierStokesModel;
    const RANSModel_enum rans_model_type = parameters_input->physics_model_param.RANS_model_type;

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
        } else if (pde_type == PDE_enum::burgers_viscous) {
            return std::make_shared< DGWeak<dim,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_rewienski) {
            return std::make_shared< DGWeak<dim,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::euler) {
            return std::make_shared< DGWeak<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes) {
            return std::make_shared< DGWeak<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes_channel_flow_constant_source_term) {
            return std::make_shared< DGWeak<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::reynolds_averaged_navier_stokes) && (rans_model_type == RANSModel_enum::SA_negative)) {
            return std::make_shared< DGWeak<dim,dim+3,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
#if PHILIP_DIM==3
        else if ((pde_type == PDE_enum::physics_model || pde_type == PDE_enum::physics_model_filtered) && (model_type == Model_enum::large_eddy_simulation || model_type == Model_enum::navier_stokes_model)) {
            return std::make_shared< DGWeak<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
#endif
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
        } else if (pde_type == PDE_enum::burgers_viscous) {
            return std::make_shared< DGStrong<dim,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_rewienski) {
            return std::make_shared< DGStrong<dim,dim,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::euler) {
            return std::make_shared< DGStrong<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes) {
            return std::make_shared< DGStrong<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes_channel_flow_constant_source_term) {
            return std::make_shared< DGStrong<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::navier_stokes_channel_flow_constant_source_term_wall_model) {
            return std::make_shared< DGStrong<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if ((pde_type == PDE_enum::physics_model) && (model_type == Model_enum::reynolds_averaged_navier_stokes) && (rans_model_type == RANSModel_enum::SA_negative)) {
            return std::make_shared< DGStrong<dim,dim+3,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
#if PHILIP_DIM==3
        else if ((pde_type == PDE_enum::physics_model || pde_type == PDE_enum::physics_model_filtered) && (model_type == Model_enum::large_eddy_simulation || model_type == Model_enum::navier_stokes_model)) {
            using FlowCaseType_enum = Parameters::FlowSolverParam::FlowCaseType;
            const FlowCaseType_enum flow_case_type = parameters_input->flow_solver_param.flow_case_type;
            if(model_type == Model_enum::large_eddy_simulation) {
                using SGS_enum = Parameters::PhysicsModelParam::SubGridScaleModel;
                const SGS_enum SGS_model_type = parameters_input->physics_model_param.SGS_model_type;
                if(SGS_model_type == SGS_enum::shear_improved_smagorinsky) {
                    // this->do_compute_mean_strain_rate_tensor = true;
                    // return std::make_shared< DGStrongLES_ShearImproved<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
                    // TO DO: Create DGStrongLES_Base
                    // TO DO: Create DGStrongLES_ConstantModelCoefficient
                    // TO DO: Create DGStrongLES_ShearImproved
                    return std::make_shared< DGStrongLES_ShearImproved<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
                } else if(SGS_model_type == SGS_enum::dynamic_smagorinsky) {
                    return std::make_shared< DGStrongLES_DynamicSmagorinsky<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
                } else {
                    return std::make_shared< DGStrongLES<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
                }
            } else if (model_type == Model_enum::navier_stokes_model && flow_case_type == FlowCaseType_enum::channel_flow) {
                // TO DO: Update this so that LES works for the channel flow
                return std::make_shared< DGStrong_ChannelFlow<dim,dim+2,real,MeshType> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);    
            }
        }
#endif
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
