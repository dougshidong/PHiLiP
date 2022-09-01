#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/lac/vector.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "parameters/parameters_physics_model.h"
#include "numerical_flux/numerical_flux_factory.hpp"
#include "physics/model_factory.h"
#include "physics/model.h"
#include "physics/physics_factory.h"
#include "dg/artificial_dissipation_factory.h"

using PDEType      = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;
using ModelType    = PHiLiP::Parameters::AllParameters::ModelType;
using ConvType     = PHiLiP::Parameters::AllParameters::ConvectiveNumericalFlux;
using DissType     = PHiLiP::Parameters::AllParameters::DissipativeNumericalFlux;
using SGSModelType = PHiLiP::Parameters::PhysicsModelParam::SubGridScaleModel;

#define TOLERANCE 1E-12

template<int dim, int nstate>
void compare_array ( const std::array<double, nstate> &array1, const std::array<double, nstate> &array2, double scale2)
{
    for (int s=0; s<nstate; s++) {
        const double diff = std::abs(array1[s] - scale2*array2[s]);
        std::cout
            << "State " << (s+1) << " out of " << nstate
            << std::endl
            << "Array 1 = " << array1[s]
            << std::endl
            << "Array 2 = " << array2[s]
            << std::endl
            << "Difference = " << diff
            << std::endl;
        assert(diff < TOLERANCE);
    }
    std::cout << std::endl
              << std::endl
              << std::endl;
}

template<int dim, int nstate>
void initialize_model_variables(std::shared_ptr <PHiLiP::Physics::ModelBase<dim, nstate, double>> pde_model)
{
    /* Initializes the model variables defined in src/physics/model.h
     * since this is normally accomplished by DGBase (but in this case
     * we do not have a grid). Values arbitrarily chosen.
     */
    if(pde_model != nullptr) {
        // allocate (i.e. reinit) dealii vectors
        pde_model->cellwise_poly_degree.reinit(2);
        pde_model->cellwise_volume.reinit(2);
        // initialize
        pde_model->cellwise_poly_degree[0] = 2;
        pde_model->cellwise_poly_degree[1] = 3;
        pde_model->cellwise_volume[0] = 10.0;
        pde_model->cellwise_volume[1] = 20.0;
    }
}

template<int dim, int nstate>
int test_dissipative_numerical_flux_conservation (const PHiLiP::Parameters::AllParameters *const all_parameters)
{
    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    std::cout << std::scientific;


    using namespace PHiLiP;
    std::shared_ptr <Physics::ModelBase<dim, nstate, double>> pde_model = Physics::ModelFactory<dim, nstate, double>::create_Model(all_parameters);
    initialize_model_variables(pde_model);
    std::shared_ptr <Physics::PhysicsBase<dim, nstate, double>> pde_physics = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters,pde_model);
    std::shared_ptr <ArtificialDissipationBase<dim,nstate>> artificial_dissipation_pointer = ArtificialDissipationFactory<dim,nstate>::create_artificial_dissipation(all_parameters);
    
    dealii::Tensor<1,dim,double> normal_int;
    std::array<double, nstate> soln_int, soln_ext;
    std::array<dealii::Tensor<1,dim,double>, nstate> soln_grad_int, soln_grad_ext;
    dealii::Point<dim> point_1;
    dealii::Point<dim> point_2;
    for(int d=0; d<dim; d++) {
        point_1[d] = 0;
        point_2[d] = 1;
    }
    for(int s=0; s<nstate; s++) {
        soln_int[s] = pde_physics->manufactured_solution_function->value(point_1,s);
        soln_ext[s] = pde_physics->manufactured_solution_function->value(point_2,s);
        soln_grad_int[s] = pde_physics->manufactured_solution_function->gradient(point_1,s);
        soln_grad_ext[s] = pde_physics->manufactured_solution_function->gradient(point_2,s);
    }

    std::unique_ptr<NumericalFlux::NumericalFluxDissipative<dim, nstate, double>> diss_num_flux = 
        NumericalFlux::NumericalFluxFactory<dim, nstate, double>
        ::create_dissipative_numerical_flux (all_parameters->diss_num_flux_type, pde_physics, artificial_dissipation_pointer);
    std::array<double, nstate> diss_num_flux_dot_n_1 = diss_num_flux->evaluate_solution_flux(soln_int, soln_ext, normal_int);
    std::array<double, nstate> diss_num_flux_dot_n_2 = diss_num_flux->evaluate_solution_flux(soln_ext, soln_int, -normal_int);

    double penalty = 100;
    const double artificial_diss_int = 1.0, artificial_diss_ext = 2.0;
    const dealii::types::global_dof_index cell_index_int = 0;
    const dealii::types::global_dof_index cell_index_ext = 1;
    std::array<double, nstate> diss_auxi_num_flux_dot_n_1 = diss_num_flux->evaluate_auxiliary_flux(
                 cell_index_int, cell_index_ext,
                 artificial_diss_int, artificial_diss_ext,
                 soln_int, soln_ext,
                 soln_grad_int, soln_grad_ext,
                 normal_int, penalty);

    std::array<double, nstate> diss_auxi_num_flux_dot_n_2 = diss_num_flux->evaluate_auxiliary_flux(
                 cell_index_ext, cell_index_int,
                 artificial_diss_ext, artificial_diss_int,
                 soln_ext, soln_int,
                 soln_grad_ext, soln_grad_int,
                 -normal_int, penalty);
 

    std::cout << "Dissipative solution flux conservation (should be equal since not dotted with normal)" << std::endl;
    compare_array<dim,nstate> (diss_num_flux_dot_n_1, diss_num_flux_dot_n_2, 1.0);

    std::cout << "Dissipative auxiliary flux conservation (should be equal and opposite since it is dotted with normal)" << std::endl;
    compare_array<dim,nstate> (diss_auxi_num_flux_dot_n_1, diss_auxi_num_flux_dot_n_2, -1.0);

    return 0;
}

template<int dim, int nstate>
int test_dissipative_numerical_flux_consistency (const PHiLiP::Parameters::AllParameters *const all_parameters)
{
    using namespace PHiLiP;
    std::shared_ptr <Physics::ModelBase<dim, nstate, double>> pde_model = Physics::ModelFactory<dim, nstate, double>::create_Model(all_parameters);
    initialize_model_variables(pde_model);
    std::shared_ptr <Physics::PhysicsBase<dim, nstate, double>> pde_physics = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters, pde_model);
    std::shared_ptr <ArtificialDissipationBase<dim,nstate>> artificial_dissipation_pointer = ArtificialDissipationFactory<dim,nstate> ::create_artificial_dissipation(all_parameters);
    
    std::unique_ptr<NumericalFlux::NumericalFluxDissipative<dim, nstate, double>> diss_num_flux = 
        NumericalFlux::NumericalFluxFactory<dim, nstate, double>
        ::create_dissipative_numerical_flux (all_parameters->diss_num_flux_type, pde_physics, artificial_dissipation_pointer);

    dealii::Tensor<1,dim,double> normal_int;
    std::array<double, nstate> soln_int, soln_ext;
    std::array<dealii::Tensor<1,dim,double>, nstate> soln_grad_int, soln_grad_ext;
    //set_solution<dim,nstate> (soln_int, soln_ext, soln_grad_int, soln_grad_ext, normal_int);
    dealii::Point<dim> point_1;
    dealii::Point<dim> point_2;
    for(int d=0; d<dim; d++) {
        point_1[d] = 0.9;
        point_2[d] = 1;
    }
    for(int s=0; s<nstate; s++) {
        soln_int[s] = pde_physics->manufactured_solution_function->value(point_1,s);
        soln_ext[s] = soln_int[s];
        soln_grad_int[s] = pde_physics->manufactured_solution_function->gradient(point_1,s);
        soln_grad_ext[s] = soln_grad_int[s];
    }

    // Copy state to ext cell
    soln_int = soln_ext;
    soln_grad_int = soln_grad_ext;

    // Evaluate numerical fluxes
    const std::array<double, nstate> diss_soln_num_flux_dot_n = diss_num_flux->evaluate_solution_flux(soln_int, soln_ext, normal_int);
    double penalty = 100;
    const double artificial_diss_int = 1.0, artificial_diss_ext = 2.0;
    const dealii::types::global_dof_index cell_index_int = 0;
    const dealii::types::global_dof_index cell_index_ext = 1;
    const std::array<double, nstate> diss_auxi_num_flux_dot_n = diss_num_flux->evaluate_auxiliary_flux(
                 cell_index_int, cell_index_ext,
                 artificial_diss_int, artificial_diss_ext,
                 soln_int, soln_ext,
                 soln_grad_int, soln_grad_ext,
                 normal_int, penalty);


    std::array<dealii::Tensor<1,dim,double>, nstate> diss_phys_flux_int;
    pde_physics->dissipative_flux (soln_int, diss_phys_flux_int, cell_index_int);

    std::array<double, nstate> diss_phys_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        diss_phys_flux_dot_n[s] = diss_phys_flux_int[s]*normal_int;
    }

    std::cout << "Dissipative solution flux should be consistent (equal) with solution, when same values on each side of boundary" << std::endl;
    compare_array<dim,nstate> (diss_soln_num_flux_dot_n, soln_int, 1.0);

    std::cout << "Dissipative auxiliary flux should be consistent (equal) with gradient, when same values on each side of boundary" << std::endl;
    compare_array<dim,nstate> (diss_auxi_num_flux_dot_n, diss_phys_flux_dot_n, 1.0);

    return 0;
}

template<int dim, int nstate>
int test_convective_numerical_flux_conservation (const PHiLiP::Parameters::AllParameters *const all_parameters)
{
    using namespace PHiLiP;
    std::shared_ptr <Physics::ModelBase<dim, nstate, double>> pde_model = Physics::ModelFactory<dim, nstate, double>::create_Model(all_parameters);
    initialize_model_variables(pde_model);
    std::shared_ptr <Physics::PhysicsBase<dim, nstate, double>> pde_physics = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters,pde_model);
    
    std::unique_ptr<NumericalFlux::NumericalFluxConvective<dim, nstate, double>> conv_num_flux = 
        NumericalFlux::NumericalFluxFactory<dim, nstate, double>
        ::create_convective_numerical_flux (all_parameters->conv_num_flux_type, all_parameters->pde_type, all_parameters->model_type, pde_physics);

    dealii::Tensor<1,dim,double> normal_int;
    std::array<double, nstate> soln_int, soln_ext;
    std::array<dealii::Tensor<1,dim,double>, nstate> soln_grad_int, soln_grad_ext;
    dealii::Point<dim> point_1;
    dealii::Point<dim> point_2;
    for(int d=0; d<dim; d++) {
        point_1[d] = 0.9;
        point_2[d] = 1;
    }
    for(int s=0; s<nstate; s++) {
        soln_int[s] = pde_physics->manufactured_solution_function->value(point_1,s);
        soln_ext[s] = soln_int[s];
        soln_grad_int[s] = pde_physics->manufactured_solution_function->gradient(point_1,s);
        soln_grad_ext[s] = soln_grad_int[s];
    }

    std::array<double, nstate> conv_num_flux_dot_n_1 = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);
    std::array<double, nstate> conv_num_flux_dot_n_2 = conv_num_flux->evaluate_flux(soln_ext, soln_int, -normal_int);

    std::cout << "Convective numerical flux conservation (should be equal and opposite)" << std::endl;
    compare_array<dim,nstate> (conv_num_flux_dot_n_1, conv_num_flux_dot_n_2, -1.0);

    return 0;
}

template<int dim, int nstate>
int test_convective_numerical_flux_consistency (const PHiLiP::Parameters::AllParameters *const all_parameters)
{
    using namespace PHiLiP;
    std::shared_ptr <Physics::ModelBase<dim, nstate, double>> pde_model = Physics::ModelFactory<dim, nstate, double>::create_Model(all_parameters);
    initialize_model_variables(pde_model);
    std::shared_ptr <Physics::PhysicsBase<dim, nstate, double>> pde_physics = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(all_parameters,pde_model);

    std::unique_ptr<NumericalFlux::NumericalFluxConvective<dim, nstate, double>> conv_num_flux = 
        NumericalFlux::NumericalFluxFactory<dim, nstate, double>
        ::create_convective_numerical_flux (all_parameters->conv_num_flux_type, all_parameters->pde_type, all_parameters->model_type, pde_physics);

    dealii::Tensor<1,dim,double> normal_int;
    std::array<double, nstate> soln_int, soln_ext;
    std::array<dealii::Tensor<1,dim,double>, nstate> soln_grad_int, soln_grad_ext;
    dealii::Point<dim> point_1;
    dealii::Point<dim> point_2;
    for(int d=0; d<dim; d++) {
        point_1[d] = 0.9;
        point_2[d] = 1;
    }
    for(int s=0; s<nstate; s++) {
        soln_int[s] = pde_physics->manufactured_solution_function->value(point_1,s);
        soln_ext[s] = soln_int[s];
        soln_grad_int[s] = pde_physics->manufactured_solution_function->gradient(point_1,s);
        soln_grad_ext[s] = soln_grad_int[s];
    }

    // Consistent numerical flux should be equal to physical flux when both states are equal
    // Therefore, f1 - f2 = 0
    soln_int = soln_ext;
    std::array<double, nstate> conv_num_flux_dot_n = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);

    const std::array<dealii::Tensor<1,dim,double>, nstate> conv_phys_flux_int = pde_physics->convective_flux (soln_int);

    std::array<double, nstate> conv_phys_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        conv_phys_flux_dot_n[s] = conv_phys_flux_int[s]*normal_int;
    }

    std::cout << "Convective numerical flux should be consistent with physical flux (equal), when same values on each side of boundary" << std::endl;
    compare_array<dim,nstate> (conv_num_flux_dot_n, conv_phys_flux_dot_n, 1.0);

    return 0;
}

void print_model_type(const ModelType model)
{
    std::string model_string = "WARNING: invalid model";
    // assign model string
    if(model==ModelType::large_eddy_simulation) model_string = "large_eddy_simulation";
    
    // print model type
    std::cout << "----------------------------------------------------------------------------" << std::endl;
    std::cout << "-- Model Type: " << model_string << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;
}

void print_sub_grid_scale_model_type(const SGSModelType sgs_model)
{
    // sub-grid scale (SGS)
    std::string sgs_model_string = "WARNING: invalid SGS model";
    // assign SGS model string
    if     (sgs_model==SGSModelType::smagorinsky) sgs_model_string = "smagorinsky";
    else if(sgs_model==SGSModelType::wall_adaptive_local_eddy_viscosity) sgs_model_string = "wall_adaptive_local_eddy_viscosity";
    else if(sgs_model==SGSModelType::vreman) sgs_model_string = "vreman";
    
    // print SGS model type
    std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;
    std::cout << "-- SGS Model Type: " << sgs_model_string << std::endl;
    std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;
}

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const std::vector<PDEType> pde_type {
        PDEType::advection,
        PDEType::diffusion,
        PDEType::convection_diffusion,
        PDEType::advection_vector,
        PDEType::burgers_inviscid,
        PDEType::euler,
        PDEType::navier_stokes,
#if PHILIP_DIM==3
        PDEType::physics_model
#endif
    };

    const std::vector<ModelType> model_type {
#if PHILIP_DIM==3
        ModelType::large_eddy_simulation
#endif
    };

    const std::vector<SGSModelType> sgs_model_type {
        SGSModelType::smagorinsky,
        SGSModelType::wall_adaptive_local_eddy_viscosity,
        SGSModelType::vreman
    };

    std::vector<ConvType> conv_type {
        ConvType::lax_friedrichs,
        ConvType::roe,
        ConvType::l2roe,
        ConvType::central_flux,
        ConvType::two_point_flux,
        ConvType::two_point_flux_with_lax_friedrichs_dissipation,
        ConvType::two_point_flux_with_roe_dissipation,
        ConvType::two_point_flux_with_l2roe_dissipation
    };
    std::vector<DissType> diss_type {
        DissType::symm_internal_penalty
    };

    int success = 0;

    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);
    PHiLiP::Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);

    for (auto pde = pde_type.begin(); pde != pde_type.end() && success == 0; pde++) {

        all_parameters.pde_type = *pde;

        std::string pde_string;
        if(*pde==PDEType::advection)            pde_string = "advection";
        if(*pde==PDEType::diffusion)            pde_string = "diffusion";
        if(*pde==PDEType::convection_diffusion) pde_string = "convection_diffusion";
        if(*pde==PDEType::advection_vector)     pde_string = "advection_vector";
        if(*pde==PDEType::burgers_inviscid)     pde_string = "burgers_inviscid";
        if(*pde==PDEType::euler)                pde_string = "euler";
        if(*pde==PDEType::navier_stokes)        pde_string = "navier_stokes";
        if(*pde==PDEType::physics_model)        pde_string = "physics_model";


        if((*pde==PDEType::navier_stokes) || 
           ((*pde==PDEType::physics_model) && (all_parameters.model_type==ModelType::large_eddy_simulation))) {
            // We want a non-zero viscous (dissipative) flux for testing Navier-Stokes
            all_parameters.navier_stokes_param.reynolds_number_inf = 1.0; // default is 10000000.0 (i.e. inviscid Navier-Stokes)
        }
        
        for (auto conv = conv_type.begin(); conv != conv_type.end() && success == 0; conv++) {

            // Roe-type fluxes are defined only for the Euler and Navier-Stokes equations
            if(((*conv == ConvType::roe) || (*conv == ConvType::l2roe) ||
                (*conv == ConvType::two_point_flux_with_roe_dissipation) || 
                (*conv == ConvType::two_point_flux_with_l2roe_dissipation))
               && ((*pde!=PDEType::euler) && (*pde!=PDEType::navier_stokes) && (*pde!=PDEType::physics_model))) continue;
            all_parameters.conv_num_flux_type = *conv;

            std::string conv_string;
            if(*conv==ConvType::lax_friedrichs)    conv_string = "lax_friedrichs";
            if(*conv==ConvType::roe)               conv_string = "roe";
            if(*conv==ConvType::l2roe)             conv_string = "l2roe";
            if(*conv==ConvType::central_flux)      conv_string = "central_flux";
            if(*conv==ConvType::two_point_flux) conv_string = "two_point_flux";
            if(*conv==ConvType::two_point_flux_with_lax_friedrichs_dissipation) conv_string = "two_point_flux_with_lax_friedrichs_dissipation";
            if(*conv==ConvType::two_point_flux_with_roe_dissipation) conv_string = "two_point_flux_with_roe_dissipation";
            if(*conv==ConvType::two_point_flux_with_l2roe_dissipation) conv_string = "two_point_flux_with_l2roe_dissipation";

            std::cout << "============================================================================" << std::endl;
            std::cout << "PDE Type: " << pde_string << "\t Convective Flux Type: " << conv_string << std::endl;
            std::cout << "----------------------------------------------------------------------------" << std::endl;

            if(*pde==PDEType::advection) success = test_convective_numerical_flux_conservation<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::diffusion) success = test_convective_numerical_flux_conservation<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::convection_diffusion) success = test_convective_numerical_flux_conservation<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::advection_vector) success = test_convective_numerical_flux_conservation<PHILIP_DIM,2> (&all_parameters);
            if(*pde==PDEType::burgers_inviscid) success = test_convective_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM> (&all_parameters);
            if(*pde==PDEType::euler) success = test_convective_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
            if(*pde==PDEType::navier_stokes) success = test_convective_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
            if(*pde==PDEType::physics_model) {
                for (auto model = model_type.begin(); model != model_type.end() && success == 0; model++) {
                    all_parameters.model_type = *model;
                    print_model_type(*model);
                    if(all_parameters.model_type == ModelType::large_eddy_simulation) {
                        for (auto sgs_model = sgs_model_type.begin(); sgs_model != sgs_model_type.end() && success == 0; sgs_model++) {
                            print_sub_grid_scale_model_type(*sgs_model);
                            success = test_convective_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
                        }
                    }
                    else {
                        success = test_convective_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
                    }
                }
            }

            if(*pde==PDEType::advection) success = test_convective_numerical_flux_consistency<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::diffusion) success = test_convective_numerical_flux_consistency<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::convection_diffusion) success = test_convective_numerical_flux_consistency<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::advection_vector) success = test_convective_numerical_flux_consistency<PHILIP_DIM,2> (&all_parameters);
            if(*pde==PDEType::burgers_inviscid) success = test_convective_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM> (&all_parameters);
            if(*pde==PDEType::euler) success = test_convective_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
            if(*pde==PDEType::navier_stokes) success = test_convective_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
            if(*pde==PDEType::physics_model) {
                for (auto model = model_type.begin(); model != model_type.end() && success == 0; model++) {
                    all_parameters.model_type = *model;
                    print_model_type(*model);
                    if(all_parameters.model_type == ModelType::large_eddy_simulation) {
                        for (auto sgs_model = sgs_model_type.begin(); sgs_model != sgs_model_type.end() && success == 0; sgs_model++) {
                            print_sub_grid_scale_model_type(*sgs_model);
                            success = test_convective_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
                        }
                    }
                    else {
                        success = test_convective_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
                    }
                }
            }
        }
        for (auto diss = diss_type.begin(); diss != diss_type.end() && success == 0; diss++) {

            all_parameters.diss_num_flux_type = *diss;

            std::string diss_string;
            if(*diss==DissType::symm_internal_penalty) diss_string = "symm_internal_penalty";

            std::cout << "============================================================================" << std::endl;
            std::cout << "PDE Type: " << pde_string << "\t Dissipative Flux Type: " << diss_string << std::endl;
            std::cout << "----------------------------------------------------------------------------" << std::endl;

            if(*pde==PDEType::advection) success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::diffusion) success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::convection_diffusion) success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::advection_vector) success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,2> (&all_parameters);
            if(*pde==PDEType::burgers_inviscid) success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM> (&all_parameters);
            if(*pde==PDEType::euler) success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
            if(*pde==PDEType::navier_stokes) success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
            if(*pde==PDEType::physics_model) {
                for (auto model = model_type.begin(); model != model_type.end() && success == 0; model++) {
                    all_parameters.model_type = *model;
                    print_model_type(*model);
                    if(all_parameters.model_type == ModelType::large_eddy_simulation) {
                        for (auto sgs_model = sgs_model_type.begin(); sgs_model != sgs_model_type.end() && success == 0; sgs_model++) {
                            print_sub_grid_scale_model_type(*sgs_model);
                            success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
                        }
                    }
                    else {
                        success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
                    }
                }
            }


            if(*pde==PDEType::advection) success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::diffusion) success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::convection_diffusion) success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,1> (&all_parameters);
            if(*pde==PDEType::advection_vector) success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,2> (&all_parameters);
            if(*pde==PDEType::burgers_inviscid) success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM> (&all_parameters);
            if(*pde==PDEType::euler) success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
            if(*pde==PDEType::navier_stokes) success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
            if(*pde==PDEType::physics_model) {
                for (auto model = model_type.begin(); model != model_type.end() && success == 0; model++) {
                    all_parameters.model_type = *model;
                    print_model_type(*model);
                    if(all_parameters.model_type == ModelType::large_eddy_simulation) {
                        for (auto sgs_model = sgs_model_type.begin(); sgs_model != sgs_model_type.end() && success == 0; sgs_model++) {
                            print_sub_grid_scale_model_type(*sgs_model);
                            success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
                        }
                    }
                    else {
                        success = test_dissipative_numerical_flux_consistency<PHILIP_DIM,PHILIP_DIM+2> (&all_parameters);
                    }
                }
            }
        }
    }
    return success;
}

