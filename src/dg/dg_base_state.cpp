#include <boost/preprocessor/seq/for_each.hpp>


#include "dg_base_state.hpp"
#include "physics/model_factory.h"
#include "physics/physics_factory.h"
#include "ADTypes.hpp"
namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
DGBaseState<dim, nstate, real, MeshType>::DGBaseState(const Parameters::AllParameters *const parameters_input,
                                                      const unsigned int degree, const unsigned int max_degree_input,
                                                      const unsigned int grid_degree_input,
                                                      const std::shared_ptr<Triangulation> triangulation_input)
    : DGBase<dim, real, MeshType>::DGBase(nstate, parameters_input, degree, max_degree_input, grid_degree_input,
                                          triangulation_input)  // Use DGBase constructor
{
    artificial_dissip = ArtificialDissipationFactory<dim, nstate>::create_artificial_dissipation(parameters_input);

    pde_model_double = Physics::ModelFactory<dim, nstate, real>::create_Model(parameters_input);
    pde_physics_double = Physics::PhysicsFactory<dim, nstate, real>::create_Physics(parameters_input, pde_model_double);

    pde_model_fad = Physics::ModelFactory<dim, nstate, FadType>::create_Model(parameters_input);
    pde_physics_fad = Physics::PhysicsFactory<dim, nstate, FadType>::create_Physics(parameters_input, pde_model_fad);

    pde_model_rad = Physics::ModelFactory<dim, nstate, RadType>::create_Model(parameters_input);
    pde_physics_rad = Physics::PhysicsFactory<dim, nstate, RadType>::create_Physics(parameters_input, pde_model_rad);

    pde_model_fad_fad = Physics::ModelFactory<dim, nstate, FadFadType>::create_Model(parameters_input);
    pde_physics_fad_fad =
        Physics::PhysicsFactory<dim, nstate, FadFadType>::create_Physics(parameters_input, pde_model_fad_fad);

    pde_model_rad_fad = Physics::ModelFactory<dim, nstate, RadFadType>::create_Model(parameters_input);
    pde_physics_rad_fad =
        Physics::PhysicsFactory<dim, nstate, RadFadType>::create_Physics(parameters_input, pde_model_rad_fad);

    reset_numerical_fluxes();
}

template <int dim, int nstate, typename real, typename MeshType>
void DGBaseState<dim, nstate, real, MeshType>::reset_numerical_fluxes() {
    conv_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real>::create_convective_numerical_flux(
        all_parameters->conv_num_flux_type, all_parameters->pde_type, all_parameters->model_type, pde_physics_double);
    diss_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real>::create_dissipative_numerical_flux(
        all_parameters->diss_num_flux_type, pde_physics_double, artificial_dissip);

    conv_num_flux_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, FadType>::create_convective_numerical_flux(
        all_parameters->conv_num_flux_type, all_parameters->pde_type, all_parameters->model_type, pde_physics_fad);
    diss_num_flux_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, FadType>::create_dissipative_numerical_flux(
        all_parameters->diss_num_flux_type, pde_physics_fad, artificial_dissip);

    conv_num_flux_rad = NumericalFlux::NumericalFluxFactory<dim, nstate, RadType>::create_convective_numerical_flux(
        all_parameters->conv_num_flux_type, all_parameters->pde_type, all_parameters->model_type, pde_physics_rad);
    diss_num_flux_rad = NumericalFlux::NumericalFluxFactory<dim, nstate, RadType>::create_dissipative_numerical_flux(
        all_parameters->diss_num_flux_type, pde_physics_rad, artificial_dissip);

    conv_num_flux_fad_fad =
        NumericalFlux::NumericalFluxFactory<dim, nstate, FadFadType>::create_convective_numerical_flux(
            all_parameters->conv_num_flux_type, all_parameters->pde_type, all_parameters->model_type,
            pde_physics_fad_fad);
    diss_num_flux_fad_fad =
        NumericalFlux::NumericalFluxFactory<dim, nstate, FadFadType>::create_dissipative_numerical_flux(
            all_parameters->diss_num_flux_type, pde_physics_fad_fad, artificial_dissip);

    conv_num_flux_rad_fad =
        NumericalFlux::NumericalFluxFactory<dim, nstate, RadFadType>::create_convective_numerical_flux(
            all_parameters->conv_num_flux_type, all_parameters->pde_type, all_parameters->model_type,
            pde_physics_rad_fad);
    diss_num_flux_rad_fad =
        NumericalFlux::NumericalFluxFactory<dim, nstate, RadFadType>::create_dissipative_numerical_flux(
            all_parameters->diss_num_flux_type, pde_physics_rad_fad, artificial_dissip);
}

template <int dim, int nstate, typename real, typename MeshType>
void DGBaseState<dim, nstate, real, MeshType>::set_physics(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real> > pde_physics_double_input,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, FadType> > pde_physics_fad_input,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, RadType> > pde_physics_rad_input,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, FadFadType> > pde_physics_fad_fad_input,
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, RadFadType> > pde_physics_rad_fad_input) {
    pde_physics_double = pde_physics_double_input;
    pde_physics_fad = pde_physics_fad_input;
    pde_physics_rad = pde_physics_rad_input;
    pde_physics_fad_fad = pde_physics_fad_fad_input;
    pde_physics_rad_fad = pde_physics_rad_fad_input;

    reset_numerical_fluxes();
}

template <int dim, int nstate, typename real, typename MeshType>
void DGBaseState<dim, nstate, real, MeshType>::allocate_model_variables() {
    // allocate all model variables for each ModelBase object
    // -- double
    pde_model_double->cellwise_poly_degree.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    pde_model_double->cellwise_volume.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    // -- FadType
    pde_model_fad->cellwise_poly_degree.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    pde_model_fad->cellwise_volume.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    // -- RadType
    pde_model_rad->cellwise_poly_degree.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    pde_model_rad->cellwise_volume.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    // -- FadFadType
    pde_model_fad_fad->cellwise_poly_degree.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    pde_model_fad_fad->cellwise_volume.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    // -- RadFadType
    pde_model_rad_fad->cellwise_poly_degree.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    pde_model_rad_fad->cellwise_volume.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
}

template <int dim, int nstate, typename real, typename MeshType>
void DGBaseState<dim, nstate, real, MeshType>::update_model_variables() {
    // allocate/reinit the model variables
    allocate_model_variables();

    // get FEValues of volume
    const auto mapping = (*(this->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values;
    dealii::hp::FEValues<dim, dim> fe_values_collection_volume(mapping_collection, this->fe_collection,
                                                               this->volume_quadrature_collection, update_flags);

    // loop through all cells
    for (auto cell : this->dof_handler.active_cell_iterators()) {
        if (!(cell->is_locally_owned() || cell->is_ghost())) continue;

        // get FEValues of volume for current cell
        const int i_fele = cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;
        fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim, dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

        // get cell polynomial degree
        const dealii::FESystem<dim, dim> &fe_high = this->fe_collection[i_fele];
        const unsigned int cell_poly_degree = fe_high.tensor_degree();

        // get cell volume
        const dealii::Quadrature<dim> &quadrature = fe_values_volume.get_quadrature();
        const unsigned int n_quad_pts = quadrature.size();
        const std::vector<real> &JxW = fe_values_volume.get_JxW_values();
        real cell_volume_estimate = 0.0;
        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            cell_volume_estimate = cell_volume_estimate + JxW[iquad];
        }
        const real cell_volume = cell_volume_estimate;

        // get cell index for assignment
        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        // const dealii::types::global_dof_index cell_index = cell->global_active_cell_index(); //
        // https://www.dealii.org/current/doxygen/deal.II/classCellAccessor.html

        // assign values
        // -- double
        pde_model_double->cellwise_poly_degree[cell_index] = cell_poly_degree;
        pde_model_double->cellwise_volume[cell_index] = cell_volume;
        // -- FadType
        pde_model_fad->cellwise_poly_degree[cell_index] = cell_poly_degree;
        pde_model_fad->cellwise_volume[cell_index] = cell_volume;
        // -- RadType
        pde_model_rad->cellwise_poly_degree[cell_index] = cell_poly_degree;
        pde_model_rad->cellwise_volume[cell_index] = cell_volume;
        // -- FadFadType
        pde_model_fad_fad->cellwise_poly_degree[cell_index] = cell_poly_degree;
        pde_model_fad_fad->cellwise_volume[cell_index] = cell_volume;
        // -- RadRadType
        pde_model_rad_fad->cellwise_poly_degree[cell_index] = cell_poly_degree;
        pde_model_rad_fad->cellwise_volume[cell_index] = cell_volume;
    }
    pde_model_double->cellwise_poly_degree.update_ghost_values();
    pde_model_double->cellwise_volume.update_ghost_values();
    pde_model_fad->cellwise_poly_degree.update_ghost_values();
    pde_model_fad->cellwise_volume.update_ghost_values();
    pde_model_rad->cellwise_poly_degree.update_ghost_values();
    pde_model_rad->cellwise_volume.update_ghost_values();
    pde_model_fad_fad->cellwise_poly_degree.update_ghost_values();
    pde_model_fad_fad->cellwise_volume.update_ghost_values();
    pde_model_rad_fad->cellwise_poly_degree.update_ghost_values();
    pde_model_rad_fad->cellwise_volume.update_ghost_values();
}

template <int dim, int nstate, typename real, typename MeshType>
void DGBaseState<dim, nstate, real, MeshType>::set_use_auxiliary_eq() {
    this->use_auxiliary_eq = pde_physics_double->has_nonzero_diffusion;
}

template <int dim, int nstate, typename real, typename MeshType>
real DGBaseState<dim, nstate, real, MeshType>::evaluate_CFL(std::vector<std::array<real, nstate> > soln_at_q,
                                                            const real artificial_dissipation, const real cell_diameter,
                                                            const unsigned int cell_degree) {
    const unsigned int n_pts = soln_at_q.size();
    std::vector<real> convective_eigenvalues(n_pts);
    std::vector<real> viscosities(n_pts);
    for (unsigned int isol = 0; isol < n_pts; ++isol) {
        convective_eigenvalues[isol] = pde_physics_double->max_convective_eigenvalue(soln_at_q[isol]);
        viscosities[isol] = pde_physics_double->max_viscous_eigenvalue(soln_at_q[isol]);
    }
    const real max_eig = *(std::max_element(convective_eigenvalues.begin(), convective_eigenvalues.end()));
    const real max_diffusive = *(std::max_element(viscosities.begin(), viscosities.end()));

    // const real cfl_convective = cell_diameter / max_eig;
    // const real cfl_diffusive  = artificial_dissipation != 0.0 ? 0.5*cell_diameter*cell_diameter /
    // artificial_dissipation : 1e200; real min_cfl = std::min(cfl_convective, cfl_diffusive) / (2*cell_degree + 1.0);

    const unsigned int p = std::max((unsigned int)1, cell_degree);
    const real cfl_convective = (cell_diameter / max_eig) / (2 * p + 1);  //(p * p);
    const real cfl_diffusive = artificial_dissipation != 0.0
                                   ? (0.5 * cell_diameter * cell_diameter / artificial_dissipation) / (p * p * p * p)
                                   : ((this->all_parameters->ode_solver_param.ode_solver_type !=
                                       Parameters::ODESolverParam::ODESolverEnum::implicit_solver)
                                          ?  // if explicit use pseudotime stepping CFL
                                          (0.5 * cell_diameter * cell_diameter / max_diffusive) / (2 * p + 1)
                                          : 1e200);
    real min_cfl = std::min(cfl_convective, cfl_diffusive);

    if (min_cfl >= 1e190) min_cfl = cell_diameter / 1;

    return min_cfl;
}
    
template <int dim, int nstate, typename real, typename MeshType>
template<typename adtype>
Physics::PhysicsBase<dim, nstate, adtype> & DGBaseState<dim, nstate, real, MeshType>::get_physics() const
{
    if(std::is_same<adtype, double>::value)
    {
        return *pde_physics_double;
    }
    else if(std::is_same<adtype, FadType>::value)
    {
        return *pde_physics_fad;
    }
    else if(std::is_same<adtype, RadType>::value)
    {
        return *pde_physics_rad;
    }
    else if(std::is_same<adtype, FadFadType>::value)
    {
        return *pde_physics_fad_fad;
    }
    else if(std::is_same<adtype, RadFadType>::value)
    {
        return *pde_physics_rad_fad;
    }
    else 
    {
        std::cout<<"Unknown adtype to get physics object. Aborting..."<<std::endl;
        std::abort();
    }
    return *pde_physics_double;
}

// Define a sequence of indices representing the range [1, 5]
#define POSSIBLE_NSTATE (1)(2)(3)(4)(5)(6)

// Define a macro to instantiate MyTemplate for a specific index
#define INSTANTIATE_DISTRIBUTED(r, data, index) \
    template class DGBaseState <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_DISTRIBUTED, _, POSSIBLE_NSTATE)
#endif

#define INSTANTIATE_SHARED(r, data, index) \
    template class DGBaseState <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_SHARED, _, POSSIBLE_NSTATE)

#define INSTANTIATE_TRIA(r, data, index) \
    template class DGBaseState <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TRIA, _, POSSIBLE_NSTATE)

#define INSTANTIATE_GET_PHYSICS_DISTRIBUTED(r, data, index) {\
    template Physics::PhysicsBase<PHILIP_DIM, index, double> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> :: get_physics<double>() const; \
    template Physics::PhysicsBase<PHILIP_DIM, index, FadType> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> :: get_physics<FadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, RadType> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> :: get_physics<RadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, FadFadType> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> :: get_physics<FadFadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, RadFadType> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> :: get_physics<RadFadType>() const; \ 
}

#define INSTANTIATE_GET_PHYSICS_SHARED(r, data, index) {\
    template Physics::PhysicsBase<PHILIP_DIM, index, double> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>> :: get_physics<double>() const; \
    template Physics::PhysicsBase<PHILIP_DIM, index, FadType> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>> :: get_physics<FadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, RadType> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>> :: get_physics<RadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, FadFadType> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>> :: get_physics<FadFadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, RadFadType> & DGBaseState <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>> :: get_physics<RadFadType>() const; \ 
}

#define INSTANTIATE_GET_PHYSICS_TRIA(r, data, index) {\
    template Physics::PhysicsBase<PHILIP_DIM, index, double> & DGBaseState <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>> :: get_physics<double>() const; \
    template Physics::PhysicsBase<PHILIP_DIM, index, FadType> & DGBaseState <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>> :: get_physics<FadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, RadType> & DGBaseState <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>> :: get_physics<RadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, FadFadType> & DGBaseState <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>> :: get_physics<FadFadType>() const; \ 
    template Physics::PhysicsBase<PHILIP_DIM, index, RadFadType> & DGBaseState <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>> :: get_physics<RadFadType>() const; \ 
}

#if PHILIP_DIM!=1
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_GET_PHYSICS_DISTRIBUTED, _, POSSIBLE_NSTATE)
#endif
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_GET_PHYSICS_SHARED, _, POSSIBLE_NSTATE)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_GET_PHYSICS_TRIA, _, POSSIBLE_NSTATE)
}  // namespace PHiLiP
