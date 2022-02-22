#include<limits>
#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/base/qprojector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>

//#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/fe/mapping_q.h> // Might need mapping_q
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/mapping_manifold.h>
#include <deal.II/fe/mapping_fe_field.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/dofs/dof_renumbering.h>

#include "dg.h"
#include "physics/physics_factory.h"
#include "post_processor/physics_post_processor.h"

#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/distributed/grid_refinement.h>

#include <EpetraExt_Transpose_RowMatrix.h>


#include "global_counter.hpp"

unsigned int n_vmult;
unsigned int dRdW_form;
unsigned int dRdW_mult;
unsigned int dRdX_mult;
unsigned int d2R_mult;


namespace PHiLiP {

// Forward declaration.
template<int dim, typename real>
std::vector< real > project_function(
    const std::vector< real > &function_coeff,
    const dealii::FESystem<dim,dim> &fe_input,
    const dealii::FESystem<dim,dim> &fe_output,
    const dealii::QGauss<dim> &projection_quadrature);


template <int dim, typename real, typename MeshType>
DGBase<dim,real,MeshType>::DGBase(
    const int nstate_input,
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBase<dim,real,MeshType>(nstate_input, parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input, this->create_collection_tuple(max_degree_input, nstate_input, parameters_input))
{ }

template <int dim, typename real, typename MeshType>
DGBase<dim,real,MeshType>::DGBase(
    const int nstate_input,
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input,
    const MassiveCollectionTuple collection_tuple)
    : all_parameters(parameters_input)
    , nstate(nstate_input)
    , initial_degree(degree)
    , max_degree(max_degree_input)
    , triangulation(triangulation_input)
    , fe_collection(std::get<0>(collection_tuple))
    , volume_quadrature_collection(std::get<1>(collection_tuple))
    , face_quadrature_collection(std::get<2>(collection_tuple))
    , oned_quadrature_collection(std::get<3>(collection_tuple))
    , fe_collection_lagrange(std::get<4>(collection_tuple))
    , dof_handler(*triangulation, true)
    , high_order_grid(std::make_shared<HighOrderGrid<dim,real,MeshType>>(grid_degree_input, triangulation))
    , operators(all_parameters, nstate, max_degree, max_degree, grid_degree_input)
    , fe_q_artificial_dissipation(1)
    , dof_handler_artificial_dissipation(*triangulation, false)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
    , freeze_artificial_dissipation(false)
    , max_artificial_dissipation_coeff(0.0)
{

    dof_handler.initialize(*triangulation, fe_collection);
    dof_handler_artificial_dissipation.initialize(*triangulation, fe_q_artificial_dissipation);

    set_all_cells_fe_degree(degree);

}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::reinit()
{
    high_order_grid->reinit();

    dof_handler.initialize(*triangulation, fe_collection);
    set_all_cells_fe_degree(initial_degree);
}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::set_high_order_grid(std::shared_ptr<HighOrderGrid<dim,real,MeshType>> new_high_order_grid)
{
    high_order_grid = new_high_order_grid;
    triangulation = high_order_grid->triangulation;
    dof_handler.initialize(*triangulation, fe_collection);
    dof_handler_artificial_dissipation.initialize(*triangulation, fe_q_artificial_dissipation);
    set_all_cells_fe_degree(max_degree);
}

template <int dim, typename real, typename MeshType>
std::tuple<
        //dealii::hp::MappingCollection<dim>, // Mapping
        dealii::hp::FECollection<dim>, // Solution FE
        dealii::hp::QCollection<dim>,  // Volume quadrature
        dealii::hp::QCollection<dim-1>, // Face quadrature
        dealii::hp::QCollection<1>, // 1D quadrature for strong form
        dealii::hp::FECollection<dim> >   // Lagrange polynomials for strong form
DGBase<dim,real,MeshType>::create_collection_tuple(
    const unsigned int max_degree, 
    const int nstate, 
    const Parameters::AllParameters *const parameters_input) const
{
    dealii::hp::FECollection<dim>      fe_coll;
    dealii::hp::QCollection<dim>       volume_quad_coll;
    dealii::hp::QCollection<dim-1>     face_quad_coll;
    dealii::hp::QCollection<1>         oned_quad_coll;

    dealii::hp::FECollection<dim>      fe_coll_lagr;

    // for p=0, we use a p=1 FE for collocation, since there's no p=0 quadrature for Gauss Lobatto
    if (parameters_input->use_collocated_nodes==true)
    {
        int degree = 1;

        const dealii::FE_DGQ<dim> fe_dg(degree);
        const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
        fe_coll.push_back (fe_system);

        dealii::Quadrature<1>     oned_quad(degree+1);
        dealii::Quadrature<dim>   volume_quad(degree+1);
        dealii::Quadrature<dim-1> face_quad(degree+1); //removed const

        if (parameters_input->use_collocated_nodes) {

            dealii::QGaussLobatto<1> oned_quad_Gauss_Lobatto (degree+1);
            dealii::QGaussLobatto<dim> vol_quad_Gauss_Lobatto (degree+1);
            oned_quad = oned_quad_Gauss_Lobatto;
            volume_quad = vol_quad_Gauss_Lobatto;

            if(dim == 1) {
                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
                face_quad = face_quad_Gauss_Legendre;
            } else {
                dealii::QGaussLobatto<dim-1> face_quad_Gauss_Lobatto (degree+1);
                face_quad = face_quad_Gauss_Lobatto;
            }
        } else {
#if 0
            dealii::QGaussChebyshev<1> oned_quad_Gauss_Legendre (degree+1);
            dealii::QGaussChebyshev<dim> vol_quad_Gauss_Legendre (degree+1);
            if(dim == 1) {
                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
            face_quad = face_quad_Gauss_Legendre;
            } else {
                dealii::QGaussChebyshev<dim-1> face_quad_Gauss_Legendre (degree+1);
            face_quad = face_quad_Gauss_Legendre;
            }
#endif
//#if 0
            dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1);
            dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
//#endif
            oned_quad = oned_quad_Gauss_Legendre;
            volume_quad = vol_quad_Gauss_Legendre;
            face_quad = face_quad_Gauss_Legendre;
        }

        volume_quad_coll.push_back (volume_quad);
        face_quad_coll.push_back (face_quad);
        oned_quad_coll.push_back (oned_quad);

        dealii::FE_DGQArbitraryNodes<dim,dim> lagrange_poly(oned_quad);
        fe_coll_lagr.push_back (lagrange_poly);
    }

    int minimum_degree = (parameters_input->use_collocated_nodes==true) ?  1 :  0;
    for (unsigned int degree=minimum_degree; degree<=max_degree; ++degree) {

        // Solution FECollection
        const dealii::FE_DGQ<dim> fe_dg(degree);
        //const dealii::FE_DGQArbitraryNodes<dim,dim> fe_dg(dealii::QGauss<1>(degree+1));
        //std::cout << degree << " fe_dg.tensor_degree " << fe_dg.tensor_degree() << " fe_dg.n_dofs_per_cell " << fe_dg.n_dofs_per_cell() << std::endl;
        const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
        fe_coll.push_back (fe_system);

        dealii::Quadrature<1>     oned_quad(degree+1);
        dealii::Quadrature<dim>   volume_quad(degree+1);
        dealii::Quadrature<dim-1> face_quad(degree+1); //removed const

        if (parameters_input->use_collocated_nodes) {
            dealii::QGaussLobatto<1> oned_quad_Gauss_Lobatto (degree+1);
            dealii::QGaussLobatto<dim> vol_quad_Gauss_Lobatto (degree+1);
            oned_quad = oned_quad_Gauss_Lobatto;
            volume_quad = vol_quad_Gauss_Lobatto;

            if(dim == 1)
            {
                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
                face_quad = face_quad_Gauss_Legendre;
            }
            else
            {
                dealii::QGaussLobatto<dim-1> face_quad_Gauss_Lobatto (degree+1);
                face_quad = face_quad_Gauss_Lobatto;
            }
        } else {
            const unsigned int overintegration = parameters_input->overintegration;
#if 0
            dealii::QGaussChebyshev<1> oned_quad_Gauss_Legendre (degree+1+overintegration);
            dealii::QGaussChebyshev<dim> vol_quad_Gauss_Legendre (degree+1+overintegration);
            if(dim == 1) {
                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
            face_quad = face_quad_Gauss_Legendre;
            } else {
                dealii::QGaussChebyshev<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
            face_quad = face_quad_Gauss_Legendre;
            }
#endif
//#if 0
            dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1+overintegration);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1+overintegration);
            dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
//#endif
            oned_quad = oned_quad_Gauss_Legendre;
            volume_quad = vol_quad_Gauss_Legendre;
            face_quad = face_quad_Gauss_Legendre;
        }

        volume_quad_coll.push_back (volume_quad);
        face_quad_coll.push_back (face_quad);
        oned_quad_coll.push_back (oned_quad);

        dealii::FE_DGQArbitraryNodes<dim,dim> lagrange_poly(oned_quad);
        fe_coll_lagr.push_back (lagrange_poly);
    }
    return std::make_tuple(fe_coll, volume_quad_coll, face_quad_coll, oned_quad_coll, fe_coll_lagr);
}

template <int dim, int nstate, typename real, typename MeshType>
DGBaseState<dim,nstate,real,MeshType>::DGBaseState(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBase<dim,real,MeshType>::DGBase(nstate, parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input) // Use DGBase constructor
{
    pde_physics_double  = Physics::PhysicsFactory<dim,nstate,real>            ::create_Physics(parameters_input);
    pde_physics_fad     = Physics::PhysicsFactory<dim,nstate,FadType>         ::create_Physics(parameters_input);
    pde_physics_rad     = Physics::PhysicsFactory<dim,nstate,RadType>         ::create_Physics(parameters_input);
    pde_physics_fad_fad = Physics::PhysicsFactory<dim,nstate,FadFadType>      ::create_Physics(parameters_input);
    pde_physics_rad_fad = Physics::PhysicsFactory<dim,nstate,RadFadType>      ::create_Physics(parameters_input);
    artificial_dissip = ArtificialDissipationFactory<dim,nstate> ::create_artificial_dissipation(parameters_input);
    
    reset_numerical_fluxes();

}

template <int dim, int nstate, typename real, typename MeshType>
void DGBaseState<dim,nstate,real,MeshType>::reset_numerical_fluxes()
{
    conv_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_convective_numerical_flux (all_parameters->conv_num_flux_type, pde_physics_double);
    diss_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_dissipative_numerical_flux (all_parameters->diss_num_flux_type, pde_physics_double, artificial_dissip);

    conv_num_flux_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, FadType> ::create_convective_numerical_flux (all_parameters->conv_num_flux_type, pde_physics_fad);
    diss_num_flux_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, FadType> ::create_dissipative_numerical_flux (all_parameters->diss_num_flux_type, pde_physics_fad, artificial_dissip);

    conv_num_flux_rad = NumericalFlux::NumericalFluxFactory<dim, nstate, RadType> ::create_convective_numerical_flux (all_parameters->conv_num_flux_type, pde_physics_rad);
    diss_num_flux_rad = NumericalFlux::NumericalFluxFactory<dim, nstate, RadType> ::create_dissipative_numerical_flux (all_parameters->diss_num_flux_type, pde_physics_rad, artificial_dissip);

    conv_num_flux_fad_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, FadFadType> ::create_convective_numerical_flux (all_parameters->conv_num_flux_type, pde_physics_fad_fad);
    diss_num_flux_fad_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, FadFadType> ::create_dissipative_numerical_flux (all_parameters->diss_num_flux_type, pde_physics_fad_fad, artificial_dissip);

    conv_num_flux_rad_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, RadFadType> ::create_convective_numerical_flux (all_parameters->conv_num_flux_type, pde_physics_rad_fad);
    diss_num_flux_rad_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, RadFadType> ::create_dissipative_numerical_flux (all_parameters->diss_num_flux_type, pde_physics_rad_fad, artificial_dissip);
}

template <int dim, int nstate, typename real, typename MeshType>
void DGBaseState<dim,nstate,real,MeshType>::set_physics(
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, real       > > pde_physics_double_input,
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, FadType    > > pde_physics_fad_input,
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, RadType    > > pde_physics_rad_input,
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, FadFadType > > pde_physics_fad_fad_input,
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, RadFadType > > pde_physics_rad_fad_input)
{
    pde_physics_double = pde_physics_double_input;
    pde_physics_fad = pde_physics_fad_input;
    pde_physics_rad = pde_physics_rad_input;
    pde_physics_fad_fad = pde_physics_fad_fad_input;
    pde_physics_rad_fad = pde_physics_rad_fad_input;

    reset_numerical_fluxes();
}

template <int dim, int nstate, typename real, typename MeshType>
real DGBaseState<dim,nstate,real,MeshType>::evaluate_CFL (
    std::vector< std::array<real,nstate> > soln_at_q,
    const real artificial_dissipation,
    const real cell_diameter,
    const unsigned int cell_degree
    )
{
    const unsigned int n_pts = soln_at_q.size();
    std::vector< real > convective_eigenvalues(n_pts);
    for (unsigned int isol = 0; isol < n_pts; ++isol) {
        convective_eigenvalues[isol] = pde_physics_double->max_convective_eigenvalue (soln_at_q[isol]);
        //viscosities[isol] = pde_physics_double->compute_diffusion_coefficient (soln_at_q[isol]);
    }
    const real max_eig = *(std::max_element(convective_eigenvalues.begin(), convective_eigenvalues.end()));

    //const real cfl_convective = cell_diameter / max_eig;
    //const real cfl_diffusive  = artificial_dissipation != 0.0 ? 0.5*cell_diameter*cell_diameter / artificial_dissipation : 1e200;
    //real min_cfl = std::min(cfl_convective, cfl_diffusive) / (2*cell_degree + 1.0);

    const unsigned int p = std::max((unsigned int)1,cell_degree);
    const real cfl_convective = (cell_diameter / max_eig) / (2*p+1);//(p * p);
    const real cfl_diffusive  = artificial_dissipation != 0.0 ?
                                (0.5*cell_diameter*cell_diameter / artificial_dissipation) / (p*p*p*p)
                                : 1e200;
    real min_cfl = std::min(cfl_convective, cfl_diffusive);

    if (min_cfl >= 1e190) min_cfl = cell_diameter / 1;

    return min_cfl;
}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::time_scale_solution_update ( dealii::LinearAlgebra::distributed::Vector<double> &solution_update, const real CFL ) const
{
    std::vector<dealii::types::global_dof_index> dofs_indices;

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {

        if (!cell->is_locally_owned()) continue;


        const int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();

        const real dt = CFL * max_dt_cell[cell_index];
        for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
            const dealii::types::global_dof_index dof_index = dofs_indices[idof];
            solution_update[dof_index] *= dt;
        }
    }
}


template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::set_all_cells_fe_degree ( const unsigned int degree )
{
    triangulation->prepare_coarsening_and_refinement();
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
    {
        if (cell->is_locally_owned()) cell->set_future_fe_index (degree);
    }

    triangulation->execute_coarsening_and_refinement();
}

template <int dim, typename real, typename MeshType>
unsigned int DGBase<dim,real,MeshType>::get_max_fe_degree()
{
    unsigned int max_fe_degree = 0;

    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        if(cell->is_locally_owned() && cell->active_fe_index() > max_fe_degree)
            max_fe_degree = cell->active_fe_index();

    return dealii::Utilities::MPI::max(max_fe_degree, MPI_COMM_WORLD);
}

template <int dim, typename real, typename MeshType>
unsigned int DGBase<dim,real,MeshType>::get_min_fe_degree()
{
    unsigned int min_fe_degree = max_degree;

    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        if(cell->is_locally_owned() && cell->active_fe_index() < min_fe_degree)
            min_fe_degree = cell->active_fe_index();

    return dealii::Utilities::MPI::min(min_fe_degree, MPI_COMM_WORLD);
}

template <int dim, typename real, typename MeshType>
template<typename DoFCellAccessorType>
real DGBase<dim,real,MeshType>::evaluate_penalty_scaling (
    const DoFCellAccessorType &cell,
    const int iface,
    const dealii::hp::FECollection<dim> fe_collection) const
{

    const unsigned int fe_index = cell->active_fe_index();
    const unsigned int degree = fe_collection[fe_index].tensor_degree();
    const unsigned int degsq = (degree == 0) ? 1 : degree * (degree+1);

    const unsigned int normal_direction = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
    const real vol_div_facearea = cell->extent_in_direction(normal_direction);

    const real penalty = degsq / vol_div_facearea * this->all_parameters->sipg_penalty_factor;// * 20;

    return penalty;
}

template <int dim, typename real, typename MeshType>
template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
bool DGBase<dim,real,MeshType>::current_cell_should_do_the_work (
    const DoFCellAccessorType1 &current_cell, 
    const DoFCellAccessorType2 &neighbor_cell) const
{
    if (neighbor_cell->has_children()) {
    // Only happens in 1D where neither faces have children, but neighbor has some children
    // Can't do the computation now since we need to query the children's DoF
        AssertDimension(dim,1);
        return false;
    } else if (neighbor_cell->is_ghost()) {
    // In the case the neighbor is a ghost cell, we let the processor with the lower rank do the work on that face
    // We cannot use the cell->index() because the index is relative to the distributed triangulation
    // Therefore, the cell index of a ghost cell might be different to the physical cell index even if they refer to the same cell
        return (current_cell->subdomain_id() < neighbor_cell->subdomain_id());
        //return true;
    } else {
    // Locally owned neighbor cell
        Assert(neighbor_cell->is_locally_owned(), dealii::ExcMessage("If not ghost, neighbor should be locally owned."));

        if (current_cell->index() < neighbor_cell->index()) {
        // Cell with lower index does work
            return true;
        } else if (neighbor_cell->index() == current_cell->index()) {
        // If both cells have same index
        // See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
        // then cell at the lower level does the work
            return (current_cell->level() < neighbor_cell->level());
        }
        return false;
    }
    Assert(0==1, dealii::ExcMessage("Should not have reached here. Somehow another possible case has not been considered when two cells have the same coarseness."));
    return false;
}

template <int dim, typename real, typename MeshType>
template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
void DGBase<dim,real,MeshType>::assemble_cell_residual (
    const DoFCellAccessorType1 &current_cell,
    const DoFCellAccessorType2 &current_metric_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
    dealii::hp::FEValues<dim,dim>        &fe_values_collection_volume,
    dealii::hp::FEFaceValues<dim,dim>    &fe_values_collection_face_int,
    dealii::hp::FEFaceValues<dim,dim>    &fe_values_collection_face_ext,
    dealii::hp::FESubfaceValues<dim,dim> &fe_values_collection_subface,
    dealii::hp::FEValues<dim,dim>        &fe_values_collection_volume_lagrange,
    dealii::LinearAlgebra::distributed::Vector<double> &rhs)
{
    std::vector<dealii::types::global_dof_index> current_dofs_indices;
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices;

    // Current reference element related to this physical cell
    const int i_fele = current_cell->active_fe_index();
    const int i_quad = i_fele;
    const int i_mapp = 0;

    const dealii::FESystem<dim,dim> &current_fe_ref = fe_collection[i_fele];
    const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

    // Local vector contribution from each cell
    dealii::Vector<real> current_cell_rhs (n_dofs_curr_cell); // Defaults to 0.0 initialization

    // Obtain the mapping from local dof indices to global dof indices
    current_dofs_indices.resize(n_dofs_curr_cell);
    current_cell->get_dof_indices (current_dofs_indices);

    fe_values_collection_volume.reinit (current_cell, i_quad, i_mapp, i_fele);
    const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

    dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (current_cell);
    //if (!(all_parameters->use_weak_form)) fe_values_collection_volume_lagrange.reinit (current_cell, i_quad, i_mapp, i_fele);
    fe_values_collection_volume_lagrange.reinit (cell_iterator, i_quad, i_mapp, i_fele);
    const dealii::FEValues<dim,dim> &fe_values_lagrange = fe_values_collection_volume_lagrange.get_present_fe_values();

    const unsigned int n_metric_dofs_cell = high_order_grid->fe_system.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> current_metric_dofs_indices(n_metric_dofs_cell);
    std::vector<dealii::types::global_dof_index> neighbor_metric_dofs_indices(n_metric_dofs_cell);
    current_metric_cell->get_dof_indices (current_metric_dofs_indices);

    //if (all_parameters->add_artificial_dissipation) {
    //    const unsigned int n_soln_dofs = fe_values_volume.dofs_per_cell;
    //    const double cell_diameter = current_cell->diameter();
    //    std::vector< real > soln_coeff(n_soln_dofs);
    //    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
    //        soln_coeff[idof] = solution(current_dofs_indices[idof]);
    //    }
    //    const double artificial_diss_coeff = discontinuity_sensor(cell_diameter, soln_coeff, fe_values_volume.get_fe());
    //    artificial_dissipation_coeffs[current_cell->active_cell_index()] = artificial_diss_coeff;
    //}

    const dealii::types::global_dof_index current_cell_index = current_cell->active_cell_index();

    const unsigned int grid_degree = this->high_order_grid->fe_system.tensor_degree();
    const unsigned int poly_degree = i_fele;
  //  const unsigned int grid_degree = current_metric_cell->active_fe_index();

    if(!this->all_parameters->use_weak_form 
       && this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::explicit_solver )//only for strong form explicit
   {
        assemble_volume_term_explicit (
            current_cell,
            current_cell_index,
            fe_values_volume,
            current_dofs_indices,
            current_metric_dofs_indices,
            poly_degree, grid_degree,
            current_cell_rhs,
            fe_values_lagrange);
    }
    else {
        //Note the explicit is called first to set the max_dt_cell to a non-zero value.
        assemble_volume_term_explicit (
            current_cell,
            current_cell_index,
            fe_values_volume,
            current_dofs_indices,
            current_metric_dofs_indices,
            poly_degree, grid_degree,
            current_cell_rhs,
            fe_values_lagrange);
        //set current rhs to zero since the explicit call was just to set the max_dt_cell.
        current_cell_rhs*=0.0;

        assemble_volume_term_derivatives (
            current_cell,
            current_cell_index,
            fe_values_volume, current_fe_ref, volume_quadrature_collection[i_quad],
            current_metric_dofs_indices, current_dofs_indices,
            current_cell_rhs, fe_values_lagrange,
            compute_dRdW, compute_dRdX, compute_d2R);
 
   }


                    //// Add local contribution from current cell to global vector
                    //for (unsigned int i=0; i<n_dofs_curr_cell; ++i) {
                    //    rhs[current_dofs_indices[i]] += current_cell_rhs[i];
                    //}
                    //return;

    (void) fe_values_collection_face_int;
    (void) fe_values_collection_face_ext;
    (void) fe_values_collection_subface;
    for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {

        auto current_face = current_cell->face(iface);

        // CASE 1: FACE AT BOUNDARY
        if (current_face->at_boundary() && !current_cell->has_periodic_neighbor(iface) ) {

            fe_values_collection_face_int.reinit(current_cell, iface, i_quad, i_mapp, i_fele);
        //for 1D periodic

                if(current_face->at_boundary() && all_parameters->use_periodic_bc == true && dim == 1) //using periodic BCs (for 1d)
                {
                    int cell_index  = current_cell->index();
                    auto neighbor_cell = dof_handler.begin_active();
                    //int cell_index = current_cell->index();
                    #if 0
                    if (cell_index == 0 && iface == 0)
                    {
                        fe_values_collection_face_int.reinit(current_cell, iface, i_quad, i_mapp, i_fele);
                        neighbor_cell = dof_handler.begin_active();
                       for (unsigned int i = 0 ; i < triangulation->n_active_cells() - 1; ++i)
                       {
                           ++neighbor_cell;
                       }
                        neighbor_dofs_indices.resize(n_dofs_curr_cell);
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                         const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;

                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1,quad_index_neigh_cell,mapping_index_neigh_cell,fe_index_neigh_cell);

                    }
                    #endif
                   // else if (cell_index == (int) triangulation->n_active_cells() - 1 && iface == 1)
                    if (cell_index == (int) triangulation->n_active_cells() - 1 && iface == 1)
                    {
                     //   fe_values_collection_face_int.reinit(current_cell, iface, i_quad, i_mapp, i_fele);
                        neighbor_cell = dof_handler.begin_active();
                        neighbor_dofs_indices.resize(n_dofs_curr_cell);
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                        const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;
                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell); //not sure how changing the face number would work in dim!=1-dimensions.
                    }
                    else {
                        continue;
                    }

                    //std::cout << "cell " << current_cell->index() << "'s " << iface << "th face has neighbour: " << neighbor_cell->index() << std::endl;
                    const int neighbor_face_no = (iface ==1) ? 0:1;
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();

                    //check neighbour cell face on boundary
                    auto neigh_face_check = neighbor_cell->face(neighbor_face_no);
                    if(neigh_face_check->at_boundary()){
                        //do nothing
                    }
                    else{
                        pcout<<"FACE NOT ON BOUNDARY LOL"<<std::endl;
                    } 

                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    const dealii::FESystem<dim,dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
                   // const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                    const auto metric_neighbor_cell = high_order_grid->dof_handler_grid.begin_active();
                    metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);

                    const real penalty = evaluate_penalty_scaling (current_cell, iface, fe_collection);

              //      fe_values_collection_volume_neigh.reinit (neighbor_cell, i_quad, i_mapp, fe_index_neigh_cell);
               //     const dealii::FEValues<dim,dim> &fe_values_volume_neigh = fe_values_collection_volume_neigh.get_present_fe_values();

              //      if ( compute_dRdW ) {
              //         assemble_face_term_implicit (
              //                                     iface, neighbor_face_no, 
              //                                     fe_values_face_int, fe_values_face_ext,
              //                                     fe_values_volume, fe_values_volume_neigh,
              //                                     penalty,
              //                                     current_dofs_indices, neighbor_dofs_indices,
              //                                     current_cell_rhs, neighbor_cell_rhs);
              //     } else {
    const dealii::types::global_dof_index neighbor_cell_index = neighbor_cell->active_cell_index();
                if(!this->all_parameters->use_weak_form 
                    && this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::explicit_solver )//only for strong form explicit
                {
                    assemble_face_term_explicit (
                        iface, neighbor_face_no, 
                        current_cell,
                        current_cell_index,
                        neighbor_cell_index,
                        poly_degree, grid_degree,
                        fe_values_face_int, fe_values_face_ext,
                        penalty,
                        current_dofs_indices, neighbor_dofs_indices,
                        current_metric_dofs_indices, neighbor_metric_dofs_indices,
                        current_cell_rhs, neighbor_cell_rhs);

                }
                else {
                //implicit weak 1D per boundary
                }
                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                    rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
                }

                } else {//at boundary and not 1D periodic

        
            const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();

            const real penalty = evaluate_penalty_scaling (current_cell, iface, fe_collection);

            const unsigned int boundary_id = current_face->boundary_id();
                if(!this->all_parameters->use_weak_form 
                    && this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::explicit_solver )//only for strong form explicit
                {
                    assemble_boundary_term_explicit (
                        current_cell,
                        current_cell_index,
                        boundary_id, fe_values_face_int, penalty, current_dofs_indices, current_cell_rhs);
                }
                else {
                   const dealii::Quadrature<dim-1> face_quadrature = face_quadrature_collection[i_quad];
                   assemble_boundary_term_derivatives (
                       current_cell,
                       current_cell_index,
                       iface, boundary_id, fe_values_face_int, penalty,
                       current_fe_ref, face_quadrature,
                       current_metric_dofs_indices, current_dofs_indices, current_cell_rhs,
                       compute_dRdW, compute_dRdX, compute_d2R);
               }

            }

        //CASE 2: PERIODIC BOUNDARY CONDITIONS
        //note that periodicity is not adapted for hp adaptivity yet. this needs to be figured out in the future
        } else if (current_face->at_boundary() && current_cell->has_periodic_neighbor(iface)){

            const auto neighbor_cell = current_cell->periodic_neighbor(iface);
            //std::cout << "cell " << current_cell->index() << " at boundary" <<std::endl;
            //std::cout << "periodic neighbour on face " << iface << " is " << neighbor_cell->index() << std::endl;


            if (!current_cell->periodic_neighbor_is_coarser(iface) && current_cell_should_do_the_work(current_cell, neighbor_cell)) {

                Assert (current_cell->periodic_neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());

                const unsigned int n_dofs_neigh_cell = fe_collection[neighbor_cell->active_fe_index()].n_dofs_per_cell();
                dealii::Vector<real> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                fe_values_collection_face_int.reinit (current_cell, iface, i_quad, i_mapp, i_fele);
                const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();

                // Corresponding face of the neighbor.
                const unsigned int neighbor_iface = current_cell->periodic_neighbor_of_periodic_neighbor(iface);

                const int i_fele_n = neighbor_cell->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;
                fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_iface, i_quad_n, i_mapp_n, i_fele_n);
                const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                const real penalty1 = evaluate_penalty_scaling (current_cell, iface, fe_collection);
                const real penalty2 = evaluate_penalty_scaling (neighbor_cell, neighbor_iface, fe_collection);
                const real penalty = 0.5 * (penalty1 + penalty2);

                const dealii::types::global_dof_index neighbor_cell_index = neighbor_cell->active_cell_index();
                    const auto metric_neighbor_cell = current_metric_cell->periodic_neighbor(iface);
                    metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);

                    const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad_n]; // or i_quad

                    std::pair<unsigned int, int> face_subface_int = std::make_pair(iface, -1);
                    std::pair<unsigned int, int> face_subface_ext = std::make_pair(neighbor_iface, -1);
                    const auto face_data_set_int = dealii::QProjector<dim>::DataSetDescriptor::face (
                                                                                                  dealii::ReferenceCell::get_hypercube(dim),
                                                                                                  iface,
                                                                                                  current_cell->face_orientation(iface),
                                                                                                  current_cell->face_flip(iface),
                                                                                                  current_cell->face_rotation(iface),
                                                                                                  used_face_quadrature.size());
                    const auto face_data_set_ext = dealii::QProjector<dim>::DataSetDescriptor::face (
                                                                                                  dealii::ReferenceCell::get_hypercube(dim),
                                                                                                  neighbor_iface,
                                                                                                  neighbor_cell->face_orientation(neighbor_iface),
                                                                                                  neighbor_cell->face_flip(neighbor_iface),
                                                                                                  neighbor_cell->face_rotation(neighbor_iface),
                                                                                                  used_face_quadrature.size());
                if(!this->all_parameters->use_weak_form 
                    && this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::explicit_solver )//only for strong form explicit
                {
                    assemble_face_term_explicit (
                        iface, neighbor_iface, 
                        current_cell,
                        current_cell_index,
                        neighbor_cell_index,
                        poly_degree, grid_degree,
                        fe_values_face_int, fe_values_face_ext,
                        penalty,
                        current_dofs_indices, neighbor_dofs_indices,
                        current_metric_dofs_indices, neighbor_metric_dofs_indices,
                        current_cell_rhs, neighbor_cell_rhs);
                }
                else {
                    assemble_face_term_derivatives (
                        current_cell,
                        current_cell_index,
                        neighbor_cell_index,
                        face_subface_int, face_subface_ext,
                        face_data_set_int,
                        face_data_set_ext,
                        fe_values_face_int, fe_values_face_ext,
                        penalty,
                        fe_collection[i_fele], fe_collection[i_fele_n],
                        used_face_quadrature,
                        current_metric_dofs_indices, neighbor_metric_dofs_indices,
                        current_dofs_indices, neighbor_dofs_indices,
                        current_cell_rhs, neighbor_cell_rhs,
                        compute_dRdW, compute_dRdX, compute_d2R);
                }

                // Add local contribution from neighbor cell to global vector
                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                    rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
                }
            } else {
                //do nothing
            }


        // CASE 3: NEIGHBOUR IS FINER
        // Occurs if the face has children
        // Do nothing.
        // The face contribution from the current cell will appear then the finer neighbor cell is assembled.
        } else if (current_cell->face(iface)->has_children()) {

        // CASE 4: NEIGHBOR IS COARSER
        // Assemble face residual.
        } else if (current_cell->neighbor(iface)->face(current_cell->neighbor_face_no(iface))->has_children()) {

            Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
            Assert (!(current_cell->neighbor(iface)->has_children()), dealii::ExcInternalError());

            // Obtain cell neighbour
            const auto neighbor_cell = current_cell->neighbor(iface);
            const unsigned int neighbor_iface = current_cell->neighbor_face_no(iface);

            // Find corresponding subface
            unsigned int neighbor_i_subface = 0;
            unsigned int n_subface = dealii::GeometryInfo<dim>::n_subfaces(neighbor_cell->subface_case(neighbor_iface));

            for (; neighbor_i_subface < n_subface; ++neighbor_i_subface) {
                if (neighbor_cell->neighbor_child_on_subface (neighbor_iface, neighbor_i_subface) == current_cell) {
                    break;
                }
            }
            Assert(neighbor_i_subface != n_subface, dealii::ExcInternalError());

            const int i_fele_n = neighbor_cell->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;

            const unsigned int n_dofs_neigh_cell = fe_collection[i_fele_n].n_dofs_per_cell();
            dealii::Vector<real> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

            // Obtain the mapping from local dof indices to global dof indices for neighbor cell
            neighbor_dofs_indices.resize(n_dofs_neigh_cell);
            neighbor_cell->get_dof_indices (neighbor_dofs_indices);

            fe_values_collection_face_int.reinit (current_cell, iface, i_quad, i_mapp, i_fele);
            const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();

            fe_values_collection_subface.reinit (neighbor_cell, neighbor_iface, neighbor_i_subface, i_quad_n, i_mapp_n, i_fele_n);
            const dealii::FESubfaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_subface.get_present_fe_values();

            const real penalty1 = evaluate_penalty_scaling (current_cell, iface, fe_collection);
            const real penalty2 = evaluate_penalty_scaling (neighbor_cell, neighbor_iface, fe_collection);
            const real penalty = 0.5 * (penalty1 + penalty2);

            const dealii::types::global_dof_index neighbor_cell_index = neighbor_cell->active_cell_index();
                const auto metric_neighbor_cell = current_metric_cell->neighbor(iface);
                metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);

                const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad_n]; // or i_quad
                std::pair<unsigned int, int> face_subface_int = std::make_pair(iface, -1);
                std::pair<unsigned int, int> face_subface_ext = std::make_pair(neighbor_iface, (int)neighbor_i_subface);

                const auto face_data_set_int = dealii::QProjector<dim>::DataSetDescriptor::face( 
                                                                                                 dealii::ReferenceCell::get_hypercube(dim),
                                                                                                 iface,
                                                                                                 current_cell->face_orientation(iface),
                                                                                                 current_cell->face_flip(iface),
                                                                                                 current_cell->face_rotation(iface),
                                                                                                 used_face_quadrature.size());
                const auto face_data_set_ext = dealii::QProjector<dim>::DataSetDescriptor::subface (
                                                                                                    dealii::ReferenceCell::get_hypercube(dim),
                                                                                                    neighbor_iface,
                                                                                                    neighbor_i_subface,
                                                                                                    neighbor_cell->face_orientation(neighbor_iface),
                                                                                                    neighbor_cell->face_flip(neighbor_iface),
                                                                                                    neighbor_cell->face_rotation(neighbor_iface),
                                                                                                    used_face_quadrature.size(),
                                                                                                    neighbor_cell->subface_case(neighbor_iface));
            if(!this->all_parameters->use_weak_form 
                && this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::explicit_solver )//only for strong form explicit
            {
                assemble_face_term_explicit (
                    iface, neighbor_iface, 
                    current_cell,
                    current_cell_index,
                    neighbor_cell_index,
                    poly_degree, grid_degree,
                    fe_values_face_int, fe_values_face_ext,
                    penalty,
                    current_dofs_indices, neighbor_dofs_indices,
                    current_metric_dofs_indices, neighbor_metric_dofs_indices,
                    current_cell_rhs, neighbor_cell_rhs);
            }
            else {
                assemble_face_term_derivatives (
                    current_cell,
                    current_cell_index,
                    neighbor_cell_index,
                    face_subface_int, face_subface_ext,
                    face_data_set_int,
                    face_data_set_ext,
                    fe_values_face_int, fe_values_face_ext,
                    penalty,
                    fe_collection[i_fele], fe_collection[i_fele_n],
                    used_face_quadrature,
                    current_metric_dofs_indices, neighbor_metric_dofs_indices,
                    current_dofs_indices, neighbor_dofs_indices,
                    current_cell_rhs, neighbor_cell_rhs,
                    compute_dRdW, compute_dRdX, compute_d2R);
            }
            // Add local contribution from neighbor cell to global vector
            for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
            }
        // CASE 5: NEIGHBOR CELL HAS SAME COARSENESS
        // Therefore, we need to choose one of them to do the work
        } else if ( current_cell_should_do_the_work(current_cell, current_cell->neighbor(iface)) ) {
            Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());

            const auto neighbor_cell = current_cell->neighbor_or_periodic_neighbor(iface);
            // Corresponding face of the neighbor.
            // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
            const unsigned int neighbor_iface = current_cell->neighbor_of_neighbor(iface);

            // Get information about neighbor cell
            const unsigned int n_dofs_neigh_cell = fe_collection[neighbor_cell->active_fe_index()].n_dofs_per_cell();

            // Local rhs contribution from neighbor
            dealii::Vector<real> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

            // Obtain the mapping from local dof indices to global dof indices for neighbor cell
            neighbor_dofs_indices.resize(n_dofs_neigh_cell);
            neighbor_cell->get_dof_indices (neighbor_dofs_indices);

            fe_values_collection_face_int.reinit (current_cell, iface, i_quad, i_mapp, i_fele);
            const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();

            const int i_fele_n = neighbor_cell->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;
            fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_iface, i_quad_n, i_mapp_n, i_fele_n);
            const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

            const real penalty1 = evaluate_penalty_scaling (current_cell, iface, fe_collection);
            const real penalty2 = evaluate_penalty_scaling (neighbor_cell, neighbor_iface, fe_collection);
            const real penalty = 0.5 * (penalty1 + penalty2);

            const dealii::types::global_dof_index neighbor_cell_index = neighbor_cell->active_cell_index();
                const auto metric_neighbor_cell = current_metric_cell->neighbor_or_periodic_neighbor(iface);
                metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);

                const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad_n]; // or i_quad
                std::pair<unsigned int, int> face_subface_int = std::make_pair(iface, -1);
                std::pair<unsigned int, int> face_subface_ext = std::make_pair(neighbor_iface, -1);
                const auto face_data_set_int = dealii::QProjector<dim>::DataSetDescriptor::face (
                                                                                              dealii::ReferenceCell::get_hypercube(dim),
                                                                                              iface,
                                                                                              current_cell->face_orientation(iface),
                                                                                              current_cell->face_flip(iface),
                                                                                              current_cell->face_rotation(iface),
                                                                                              used_face_quadrature.size());
                const auto face_data_set_ext = dealii::QProjector<dim>::DataSetDescriptor::face (
                                                                                              dealii::ReferenceCell::get_hypercube(dim),
                                                                                              neighbor_iface,
                                                                                              neighbor_cell->face_orientation(neighbor_iface),
                                                                                              neighbor_cell->face_flip(neighbor_iface),
                                                                                              neighbor_cell->face_rotation(neighbor_iface),
                                                                                              used_face_quadrature.size());
            if(!this->all_parameters->use_weak_form 
                && this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::explicit_solver )//only for strong form explicit
            {
                assemble_face_term_explicit (
                    iface, neighbor_iface, 
                    current_cell,
                    current_cell_index,
                    neighbor_cell_index,
                    poly_degree, grid_degree,
                    fe_values_face_int, fe_values_face_ext,
                    penalty,
                    current_dofs_indices, neighbor_dofs_indices,
                    current_metric_dofs_indices, neighbor_metric_dofs_indices,
                    current_cell_rhs, neighbor_cell_rhs);
            }
           else {
              assemble_face_term_derivatives (
                  current_cell,
                  current_cell_index,
                  neighbor_cell_index,
                  face_subface_int, face_subface_ext,
                  face_data_set_int,
                  face_data_set_ext,
                  fe_values_face_int, fe_values_face_ext,
                  penalty,
                  fe_collection[i_fele], fe_collection[i_fele_n],
                  used_face_quadrature,
                  current_metric_dofs_indices, neighbor_metric_dofs_indices,
                  current_dofs_indices, neighbor_dofs_indices,
                  current_cell_rhs, neighbor_cell_rhs,
                  compute_dRdW, compute_dRdX, compute_d2R);
          }

            // Add local contribution from neighbor cell to global vector
            for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
            }
        } else {
            // Should be faces where the neighbor cell has the same coarseness
            // but will be evaluated when we visit the other cell.
        }


    } // end of face loop

    // Add local contribution from current cell to global vector
    for (unsigned int i=0; i<n_dofs_curr_cell; ++i) {
        rhs[current_dofs_indices[i]] += current_cell_rhs[i];
    }
}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::set_dual(const dealii::LinearAlgebra::distributed::Vector<real> &dual_input)
{
    dual = dual_input;
}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::update_artificial_dissipation_discontinuity_sensor()
{
    const auto mapping = (*(high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values;
    dealii::hp::FEValues<dim,dim> fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, update_flags); ///< FEValues of volume.

    std::vector< double > soln_coeff_high;
    std::vector<dealii::types::global_dof_index> dof_indices;

    const unsigned int n_dofs_arti_diss = fe_q_artificial_dissipation.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);

    if (freeze_artificial_dissipation) return;
    artificial_dissipation_c0 *= 0.0;
    for (auto cell : dof_handler.active_cell_iterators()) {
        if (!(cell->is_locally_owned() || cell->is_ghost())) continue;

        dealii::types::global_dof_index cell_index = cell->active_cell_index();
        artificial_dissipation_coeffs[cell_index] = 0.0;
        artificial_dissipation_se[cell_index] = 0.0;
        //artificial_dissipation_coeffs[cell_index] = 1e-2;
        //artificial_dissipation_se[cell_index] = 0.0;
        //continue;

        const int i_fele = cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;

        const dealii::FESystem<dim,dim> &fe_high = fe_collection[i_fele];
        const unsigned int degree = fe_high.tensor_degree();

        if (degree == 0) continue;

        const unsigned int nstate = fe_high.components;
        const unsigned int n_dofs_high = fe_high.dofs_per_cell;

        fe_values_collection_volume.reinit (cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

        dof_indices.resize(n_dofs_high);
        cell->get_dof_indices (dof_indices);

        soln_coeff_high.resize(n_dofs_high);
        for (unsigned int idof=0; idof<n_dofs_high; ++idof) {
            soln_coeff_high[idof] = solution[dof_indices[idof]];
        }

        // Lower degree basis.
        const unsigned int lower_degree = degree-1;
        const dealii::FE_DGQLegendre<dim> fe_dgq_lower(lower_degree);
        const dealii::FESystem<dim,dim> fe_lower(fe_dgq_lower, nstate);

        // Projection quadrature.
        const dealii::QGauss<dim> projection_quadrature(degree+5);
        std::vector< double > soln_coeff_lower = project_function<dim,double>( soln_coeff_high, fe_high, fe_lower, projection_quadrature);

        // Quadrature used for solution difference.
        const dealii::Quadrature<dim> &quadrature = fe_values_volume.get_quadrature();
        const std::vector<dealii::Point<dim,double>> &unit_quad_pts = quadrature.get_points();

        const unsigned int n_quad_pts = quadrature.size();
        const unsigned int n_dofs_lower = fe_lower.dofs_per_cell;

        double element_volume = 0.0;
        double error = 0.0;
        double soln_norm = 0.0;
        std::vector<double> soln_high(nstate);
        std::vector<double> soln_lower(nstate);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (unsigned int s=0; s<nstate; ++s) {
                soln_high[s] = 0.0;
                soln_lower[s] = 0.0;
            }
            // Interpolate solution
            for (unsigned int idof=0; idof<n_dofs_high; ++idof) {
                  const unsigned int istate = fe_high.system_to_component_index(idof).first;
                  soln_high[istate] += soln_coeff_high[idof] * fe_high.shape_value_component(idof,unit_quad_pts[iquad],istate);
            }
            // Interpolate low order solution
            for (unsigned int idof=0; idof<n_dofs_lower; ++idof) {
                  const unsigned int istate = fe_lower.system_to_component_index(idof).first;
                  soln_lower[istate] += soln_coeff_lower[idof] * fe_lower.shape_value_component(idof,unit_quad_pts[iquad],istate);
            }
            // Quadrature
            element_volume += fe_values_volume.JxW(iquad);
            // Only integrate over the first state variable.
            for (unsigned int s=0; s<1/*nstate*/; ++s) 
            {
                error += (soln_high[s] - soln_lower[s]) * (soln_high[s] - soln_lower[s]) * fe_values_volume.JxW(iquad);
                soln_norm += soln_high[s] * soln_high[s] * fe_values_volume.JxW(iquad);
            }
        }

        //std::cout << " error: " << error
        //          << " soln_norm: " << soln_norm << std::endl;
        //if (error < 1e-12) continue;
        if (soln_norm < 1e-12) 
        {
            continue;
        }

        double S_e, s_e;
        S_e = sqrt(error / soln_norm);
        s_e = log10(S_e);

        //const double mu_scale = 1.0;
        //const double s_0 = log10(0.1) - 4.25*log10(degree);
        //const double s_0 = -0.5 - 4.25*log10(degree);
        //const double kappa = 1.0;

        const double mu_scale = all_parameters->artificial_dissipation_param.mu_artificial_dissipation; //1.0
        //const double s_0 = - 4.25*log10(degree);
        const double s_0 = -0.00 - 4.00*log10(degree);
        const double kappa = all_parameters->artificial_dissipation_param.kappa_artificial_dissipation; //1.0
        const double low = s_0 - kappa;
        const double upp = s_0 + kappa;

        const double diameter = std::pow(element_volume, 1.0/dim);
        const double eps_0 = mu_scale * diameter / (double)degree;
    
        //std::cout << " lower < s_e < upp " << low << " < " << s_e << " < " << upp << " ? " << std::endl;

        if ( s_e < low) continue;

        if ( s_e > upp) {
            artificial_dissipation_coeffs[cell_index] += eps_0;
            if(eps_0 > max_artificial_dissipation_coeff)
            {
                max_artificial_dissipation_coeff = eps_0;
            }
            continue;
        }

        const double PI = 4*atan(1);
        double eps = 1.0 + sin(PI * (s_e - s_0) * 0.5 / kappa);
        eps *= eps_0 * 0.5;
    
        if(eps > max_artificial_dissipation_coeff)
        {
            max_artificial_dissipation_coeff = eps;
        }


        artificial_dissipation_coeffs[cell_index] += eps;
        artificial_dissipation_se[cell_index] = s_e;

        typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
            triangulation.get(), cell->level(), cell->index(), &dof_handler_artificial_dissipation);

        dof_indices_artificial_dissipation.resize(n_dofs_arti_diss);
        artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);
        for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
            const unsigned int index = dof_indices_artificial_dissipation[idof];
            artificial_dissipation_c0[index] = std::max(artificial_dissipation_c0[index], eps);
        }

        //const unsigned int dofs_per_face = fe_q_artificial_dissipation.n_dofs_per_face();
        //for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
        //    const auto face = cell->face(iface);
        //    if (face->at_boundary()) {
        //        for (unsigned int idof_face=0; idof_face<dofs_per_face; ++idof_face) {
        //            unsigned int idof_cell = fe_q_artificial_dissipation.face_to_cell_index(idof_face, iface);
        //            const dealii::types::global_dof_index index = dof_indices_artificial_dissipation[idof_cell];
        //            artificial_dissipation_c0[index] = 0.0;
        //        }
        //    }
        //}
    }
    dealii::IndexSet boundary_dofs(dof_handler_artificial_dissipation.n_dofs());
    dealii::DoFTools::extract_boundary_dofs(dof_handler_artificial_dissipation,
                                dealii::ComponentMask(),
                                boundary_dofs);
    for (unsigned int i = 0; i < dof_handler_artificial_dissipation.n_dofs(); ++i) {
        if (boundary_dofs.is_element(i)) {
            artificial_dissipation_c0[i] = 0.0;
        }
    }
    // artificial_dissipation_c0 *= 0.0;
    // artificial_dissipation_c0.add(1e-1);
    artificial_dissipation_c0.update_ghost_values();
}


template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::assemble_residual (const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R, const double CFL_mass)
{
    dealii::deal_II_exceptions::disable_abort_on_exception(); // Allows us to catch negative Jacobians.
    Assert( !(compute_dRdW && compute_dRdX)
        &&  !(compute_dRdW && compute_d2R)
        &&  !(compute_dRdX && compute_d2R)
            , dealii::ExcMessage("Can only do one at a time compute_dRdW or compute_dRdX or compute_d2R"));

    max_artificial_dissipation_coeff = 0.0;
    //pcout << "Assembling DG residual...";
    if (compute_dRdW) {
        pcout << " with dRdW...";

        auto diff_sol = solution;
        diff_sol -= solution_dRdW;
        const double l2_norm_sol = diff_sol.l2_norm();

        if (l2_norm_sol == 0.0) {

            auto diff_node = high_order_grid->volume_nodes;
            diff_node -= volume_nodes_dRdW;
            const double l2_norm_node = diff_node.l2_norm();

            if (l2_norm_node == 0.0) {
                if (CFL_mass_dRdW == CFL_mass) {
                    pcout << " which is already assembled..." << std::endl;
                    return;
                }
            }
        }
        {
            int n_stencil = 1 + std::pow(2,dim);
            int n_dofs_cell = nstate*std::pow(max_degree+1,dim);
            n_vmult += n_stencil*n_dofs_cell;
            dRdW_form += 1;
        }
        solution_dRdW = solution;
        volume_nodes_dRdW = high_order_grid->volume_nodes;
        CFL_mass_dRdW = CFL_mass;

        system_matrix = 0;
    }
    if (compute_dRdX) {
        pcout << " with dRdX...";

        auto diff_sol = solution;
        diff_sol -= solution_dRdX;
        const double l2_norm_sol = diff_sol.l2_norm();

        if (l2_norm_sol == 0.0) {

            auto diff_node = high_order_grid->volume_nodes;
            diff_node -= volume_nodes_dRdX;
            const double l2_norm_node = diff_node.l2_norm();

            if (l2_norm_node == 0.0) {
                pcout << " which is already assembled..." << std::endl;
                return;
            }
        }
        solution_dRdX = solution;
        volume_nodes_dRdX = high_order_grid->volume_nodes;

        if (   dRdXv.m() != solution.size() || dRdXv.n() != high_order_grid->volume_nodes.size()) {

            allocate_dRdX();
        }
        dRdXv = 0;
    }
    if (compute_d2R) {
        pcout << " with d2RdWdW, d2RdWdX, d2RdXdX...";
        auto diff_sol = solution;
        diff_sol -= solution_d2R;
        const double l2_norm_sol = diff_sol.l2_norm();

        if (l2_norm_sol == 0.0) {

            auto diff_node = high_order_grid->volume_nodes;
            diff_node -= volume_nodes_d2R;
            const double l2_norm_node = diff_node.l2_norm();

            if (l2_norm_node == 0.0) {

                auto diff_dual = dual;
                diff_dual -= dual_d2R;
                const double l2_norm_dual = diff_dual.l2_norm();
                if (l2_norm_dual == 0.0) {
                    pcout << " which is already assembled..." << std::endl;
                    return;
                }
            }
        }
        solution_d2R = solution;
        volume_nodes_d2R = high_order_grid->volume_nodes;
        dual_d2R = dual;

        if (   d2RdWdW.m() != solution.size()
            || d2RdWdX.m() != solution.size()
            || d2RdWdX.n() != high_order_grid->volume_nodes.size()
            || d2RdXdX.m() != high_order_grid->volume_nodes.size()) {

            allocate_second_derivatives();
        }
        d2RdWdW = 0;
        d2RdWdX = 0;
        d2RdXdX = 0;
    }
    right_hand_side = 0;

    //pcout << std::endl;

    //const dealii::MappingManifold<dim,dim> mapping;
    //const dealii::MappingQ<dim,dim> mapping(10);//;max_degree+1);
    //const dealii::MappingQ<dim,dim> mapping(high_order_grid->max_degree);
    //const dealii::MappingQGeneric<dim,dim> mapping(high_order_grid->max_degree);
    const auto mapping = (*(high_order_grid->mapping_fe_field));

    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, fe_collection, face_quadrature_collection, this->face_update_flags); ///< FEValues of interior face.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, fe_collection, face_quadrature_collection, this->neighbor_face_update_flags); ///< FEValues of exterior face.
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface (mapping_collection, fe_collection, face_quadrature_collection, this->face_update_flags); ///< FEValues of subface.

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_lagrange (mapping_collection, fe_collection_lagrange, volume_quadrature_collection, this->volume_update_flags);

    solution.update_ghost_values();

    int assembly_error = 0;
    try {

        update_artificial_dissipation_discontinuity_sensor();
        
        //assembles and solves for auxiliary variable if necessary.
        assemble_auxiliary_residual();

//for(unsigned int i=0; i<auxiliary_solution[0].size(); i++){
//auxiliary_solution[0].zero_out_ghosts();
//pcout<<" aux "<<auxiliary_solution[0][i]<<std::endl;
//}

        auto metric_cell = high_order_grid->dof_handler_grid.begin_active();
        for (auto soln_cell = dof_handler.begin_active(); soln_cell != dof_handler.end(); ++soln_cell, ++metric_cell) {
        //for (auto cell = triangulation->begin_active(); cell != triangulation->end(); ++cell) {
            if (!soln_cell->is_locally_owned()) continue;

            //const int tria_level = cell->level();
            //const int tria_index = cell->index();
            //dealii::DoFCellAccessor<dim,dim,false> soln_cell(triangulation.get(), tria_level, tria_index, &dof_handler);
            //dealii::DoFCellAccessor<dim,dim,false> metric_cell(triangulation.get(), tria_level, tria_index, &high_order_grid->dof_handler_grid);

            //dealii::TriaActiveIterator< dealii::DoFCellAccessor<dim,dim,false> >

            //DoFCellAccessor<dim,dim,false> soln_cell(triangulation.get(), tria_level, tria_index, &dof_handler);
            //dealii::DoFCellAccessor<dim,dim,false> metric_cell(triangulation.get(), tria_level, tria_index, &high_order_grid->dof_handler_grid);


            // Add right-hand side contributions this cell can compute
            assemble_cell_residual (
                soln_cell,
                metric_cell,
                compute_dRdW, compute_dRdX, compute_d2R,
                fe_values_collection_volume,
                fe_values_collection_face_int,
                fe_values_collection_face_ext,
                fe_values_collection_subface,
                fe_values_collection_volume_lagrange,
                right_hand_side);
        } // end of cell loop
    } catch(...) {
        assembly_error = 1;
    }
    const int mpi_assembly_error = dealii::Utilities::MPI::sum(assembly_error, mpi_communicator);

    if (mpi_assembly_error != 0) {
        std::cout << "Invalid residual assembly encountered..."
                  << " Filling up RHS with 1s. " << std::endl;
        right_hand_side *= 0.0;
        right_hand_side.add(1.0);
        if (compute_dRdW) {
            std::cout << " Filling up Jacobian with mass matrix. " << std::endl;
            const bool do_inverse_mass_matrix = false;
            evaluate_mass_matrices (do_inverse_mass_matrix);
            system_matrix.copy_from(global_mass_matrix);
        }
        //if (compute_dRdX) {
        //    dRdXv.trilinos_matrix().
        //}
        //if (compute_d2R) {
        //    d2RdWdW = 0;
        //    d2RdWdX = 0;
        //    d2RdXdX = 0;
        //}
    }

    right_hand_side.compress(dealii::VectorOperation::add);
    right_hand_side.update_ghost_values();
    if ( compute_dRdW ) {
        system_matrix.compress(dealii::VectorOperation::add);

        if (global_mass_matrix.m() != system_matrix.m()) {
            const bool do_inverse_mass_matrix = false;
            evaluate_mass_matrices (do_inverse_mass_matrix);
        }
        if (CFL_mass != 0.0) {
            time_scaled_mass_matrices(CFL_mass);
            add_time_scaled_mass_matrices();
        }

        Epetra_CrsMatrix *input_matrix  = const_cast<Epetra_CrsMatrix *>(&(system_matrix.trilinos_matrix()));
        Epetra_CrsMatrix *output_matrix;
        epetra_rowmatrixtransposer_dRdW = std::make_unique<Epetra_RowMatrixTransposer> ( input_matrix );
        const bool make_data_contiguous = true;
        int error_transpose = epetra_rowmatrixtransposer_dRdW->CreateTranspose( make_data_contiguous, output_matrix);
        if (error_transpose) {
            std::cout << "Failed to create dRdW transpose... Aborting" << std::endl;
            //std::abort();
        }
        bool copy_values = true;
        system_matrix_transpose.reinit(*output_matrix, copy_values);
        delete(output_matrix);

        //Epetra_CrsMatrix *input_matrix  = const_cast<Epetra_CrsMatrix *>(&(system_matrix.trilinos_matrix()));
        //std::shared_ptr<Epetra_CrsMatrix> output_matrix = std::make_shared<Epetra_CrsMatrix> ();
        //epetra_rowmatrixtransposer_dRdW = std::make_unique<Epetra_RowMatrixTransposer> ( input_matrix );
        //const bool make_data_contiguous = true;
        //epetra_rowmatrixtransposer_dRdW->CreateTranspose( make_data_contiguous, output_matrix.get());
        //system_matrix_transpose.reinit(*output_matrix);
        
        //EpetraExt::RowMatrix_Transpose transposer;
        //Epetra_CrsMatrix* input_matrix  = const_cast<Epetra_CrsMatrix *>(&(system_matrix.trilinos_matrix()));
        //Epetra_CrsMatrix* output_matrix = dynamic_cast<Epetra_CrsMatrix*>(&transposer(*input_matrix));
        //system_matrix_transpose.reinit(*output_matrix);
        //std::cout << output_matrix << std::endl;
        //delete(output_matrix);


        //double condition_estimate;
        //dRdW_preconditioner_builder.ConstructPreconditioner(condition_estimate);
    }
    if ( compute_dRdX ) dRdXv.compress(dealii::VectorOperation::add);
    if ( compute_d2R ) {
        d2RdWdW.compress(dealii::VectorOperation::add);
        d2RdXdX.compress(dealii::VectorOperation::add);
        d2RdWdX.compress(dealii::VectorOperation::add);
    }
    //if ( compute_dRdW ) system_matrix.compress(dealii::VectorOperation::insert);
    //system_matrix.print(std::cout);

} // end of assemble_system_explicit ()

template <int dim, typename real, typename MeshType>
double DGBase<dim,real,MeshType>::get_residual_linfnorm () const
{
    pcout << "Evaluating residual Linf-norm..." << std::endl;
    const auto mapping = (*(high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    double residual_linf_norm = 0.0;
    std::vector<dealii::types::global_dof_index> dofs_indices;
    const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values;
    dealii::hp::FEValues<dim,dim> fe_values_collection_volume (mapping_collection,
                                                               fe_collection,
                                                               volume_quadrature_collection,
                                                               update_flags);

    // Obtain the mapping from local dof indices to global dof indices
    for (const auto cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;

        const int i_fele = cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;

        fe_values_collection_volume.reinit (cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_vol = fe_values_collection_volume.get_present_fe_values();

        const dealii::FESystem<dim,dim> &fe_ref = fe_collection[i_fele];
        const unsigned int n_dofs = fe_ref.n_dofs_per_cell();
        const unsigned int n_quad = fe_values_vol.n_quadrature_points;

        dofs_indices.resize(n_dofs);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad = 0; iquad < n_quad; ++iquad) {
            double residual_val = 0.0;
            for (unsigned int idof = 0; idof < n_dofs; ++idof) {
                const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
                residual_val += right_hand_side[dofs_indices[idof]] * fe_values_vol.shape_value_component(idof, iquad, istate);
            }
            residual_linf_norm = std::max(std::abs(residual_val), residual_val);
        }

    }
    const double mpi_residual_linf_norm = dealii::Utilities::MPI::max(residual_linf_norm, mpi_communicator);
    return mpi_residual_linf_norm;
}


template <int dim, typename real, typename MeshType>
double DGBase<dim,real,MeshType>::get_residual_l2norm () const
{
    //return get_residual_linfnorm ();
    //return right_hand_side.l2_norm();
    //return right_hand_side.l2_norm() / right_hand_side.size();
    //auto scaled_residual = right_hand_side;
    //global_mass_matrix.vmult(scaled_residual, right_hand_side);
    //return scaled_residual.l2_norm();
    //pcout << "Evaluating residual L2-norm..." << std::endl;
    const auto mapping = (*(high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    double residual_l2_norm = 0.0;
    double domain_volume = 0.0;
    std::vector<dealii::types::global_dof_index> dofs_indices;
    const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values;
    dealii::hp::FEValues<dim,dim> fe_values_collection_volume (mapping_collection,
                                                               fe_collection,
                                                               volume_quadrature_collection,
                                                               update_flags);

    // Obtain the mapping from local dof indices to global dof indices
    for (const auto cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;

        const int i_fele = cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;

        fe_values_collection_volume.reinit (cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_vol = fe_values_collection_volume.get_present_fe_values();

        const dealii::FESystem<dim,dim> &fe_ref = fe_collection[i_fele];
        const unsigned int n_dofs = fe_ref.n_dofs_per_cell();
        const unsigned int n_quad = fe_values_vol.n_quadrature_points;

        dofs_indices.resize(n_dofs);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad = 0; iquad < n_quad; ++iquad) {
            double residual_val = 0.0;
            for (unsigned int idof = 0; idof < n_dofs; ++idof) {
                const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
                residual_val += right_hand_side[dofs_indices[idof]] * fe_values_vol.shape_value_component(idof, iquad, istate);
            }
            residual_l2_norm += residual_val*residual_val * fe_values_vol.JxW(iquad);
            domain_volume += fe_values_vol.JxW(iquad);
        }

    }
    const double mpi_residual_l2_norm = dealii::Utilities::MPI::sum(residual_l2_norm, mpi_communicator);
    const double mpi_domain_volume    = dealii::Utilities::MPI::sum(domain_volume, mpi_communicator);
    return std::sqrt(mpi_residual_l2_norm) / mpi_domain_volume;
}
template <int dim, typename real, typename MeshType>
unsigned int DGBase<dim,real,MeshType>::n_dofs () const
{
    return dof_handler.n_dofs();
}


#if PHILIP_DIM > 1
template <int dim, typename DoFHandlerType = dealii::DoFHandler<dim>>
class DataOutEulerFaces : public dealii::DataOutFaces<dim, DoFHandlerType>
{
    static const unsigned int dimension = DoFHandlerType::dimension;
    static const unsigned int space_dimension = DoFHandlerType::space_dimension;
    using cell_iterator = typename dealii::DataOut_DoFData<DoFHandlerType, dimension - 1, dimension>::cell_iterator;

    using FaceDescriptor = typename std::pair<cell_iterator, unsigned int>;
    /**
     * Return the first face which we want output for. The default
     * implementation returns the first face of a (locally owned) active cell
     * or, if the @p surface_only option was set in the destructor (as is the
     * default), the first such face that is located on the boundary.
     *
     * If you want to use a different logic to determine which faces should
     * contribute to the creation of graphical output, you can overload this
     * function in a derived class.
     */
    virtual FaceDescriptor first_face() override;
 
    /**
     * Return the next face after which we want output for. If there are no more
     * faces, <tt>dofs->end()</tt> is returned as the first component of the
     * return value.
     *
     * The default implementation returns the next face of a (locally owned)
     * active cell, or the next such on the boundary (depending on whether the
     * @p surface_only option was provided to the constructor).
     *
     * This function traverses the mesh active cell by active cell (restricted to
     * locally owned cells), and then through all faces of the cell. As a result,
     * interior faces are output twice, a feature that is useful for
     * discontinuous Galerkin methods or if a DataPostprocessor is used that
     * might produce results that are discontinuous between cells).
     *
     * This function can be overloaded in a derived class to select a
     * different set of faces. Note that the default implementation assumes that
     * the given @p face is active, which is guaranteed as long as first_face()
     * is also used from the default implementation. Overloading only one of the
     * two functions should be done with care.
     */
    virtual FaceDescriptor next_face(const FaceDescriptor &face) override;

};

template <int dim, typename DoFHandlerType>
typename DataOutEulerFaces<dim, DoFHandlerType>::FaceDescriptor
DataOutEulerFaces<dim, DoFHandlerType>::first_face()
{
    // simply find first active cell with a face on the boundary
    typename dealii::Triangulation<dimension, space_dimension>::active_cell_iterator
        cell = this->triangulation->begin_active();
    for (; cell != this->triangulation->end(); ++cell)
        if (cell->is_locally_owned())
            for (const unsigned int f : dealii::GeometryInfo<dimension>::face_indices())
                if (cell->face(f)->at_boundary())
                    if (cell->face(f)->boundary_id() == 1001)
                        return FaceDescriptor(cell, f);
  
    // just return an invalid descriptor if we haven't found a locally
    // owned face. this can happen in parallel where all boundary
    // faces are owned by other processors
    return FaceDescriptor();
}
    
template <int dim, typename DoFHandlerType>
typename DataOutEulerFaces<dim, DoFHandlerType>::FaceDescriptor
DataOutEulerFaces<dim, DoFHandlerType>::next_face(const FaceDescriptor &old_face)
{
    FaceDescriptor face = old_face;
  
    // first check whether the present cell has more faces on the boundary. since
    // we started with this face, its cell must clearly be locally owned
    Assert(face.first->is_locally_owned(), dealii::ExcInternalError());
    for (unsigned int f = face.second + 1; f < dealii::GeometryInfo<dimension>::faces_per_cell; ++f)
        if (face.first->face(f)->at_boundary()) 
            if (face.first->face(f)->boundary_id() == 1001) {
                // yup, that is so, so return it
                face.second = f;
                return face;
            }
  
    // otherwise find the next active cell that has a face on the boundary
  
    // convert the iterator to an active_iterator and advance this to the next
    // active cell
    typename dealii::Triangulation<dimension, space_dimension>::active_cell_iterator
      active_cell = face.first;
  
    // increase face pointer by one
    ++active_cell;
  
    // while there are active cells
    while (active_cell != this->triangulation->end()) {
        // check all the faces of this active cell. but skip it altogether
        // if it isn't locally owned
        if (active_cell->is_locally_owned())
            for (const unsigned int f : dealii::GeometryInfo<dimension>::face_indices())
                if (active_cell->face(f)->at_boundary())
                    if (active_cell->face(f)->boundary_id() == 1001) {
                        face.first  = active_cell;
                        face.second = f;
                        return face;
                    }
            
        // the present cell had no faces on the boundary (or was not locally
        // owned), so check next cell
        ++active_cell;
    }   
      
    // we fell off the edge, so return with invalid pointer
    face.first  = this->triangulation->end();
    face.second = 0;
    return face;
} 

template <int dim>
class NormalPostprocessor : public dealii::DataPostprocessorVector<dim>
{
public:
    NormalPostprocessor ()
    : dealii::DataPostprocessorVector<dim> ("normal", dealii::update_normal_vectors)
    {}
    virtual void
    evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &input_data, std::vector<dealii::Vector<double>> &computed_quantities) const override
    {
        // ensure that there really are as many output slots
        // as there are points at which DataOut provides the
        // gradients:
        AssertDimension (input_data.normals.size(), computed_quantities.size());
        // then loop over all of these inputs:
        for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p) {
            // ensure that each output slot has exactly 'dim'
            // components (as should be expected, given that we
            // want to create vector-valued outputs), and copy the
            // gradients of the solution at the evaluation points
            // into the output slots:
            AssertDimension (computed_quantities[p].size(), dim);
            for (unsigned int d=0; d<dim; ++d)
              computed_quantities[p][d] = input_data.normals[p][d];
        }
    }
    virtual void
    evaluate_scalar_field (const dealii::DataPostprocessorInputs::Scalar<dim> &input_data, std::vector<dealii::Vector<double> > &computed_quantities) const override
    {
        // ensure that there really are as many output slots
        // as there are points at which DataOut provides the
        // gradients:
        AssertDimension (input_data.normals.size(), computed_quantities.size());
        // then loop over all of these inputs:
        for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p) {
            // ensure that each output slot has exactly 'dim'
            // components (as should be expected, given that we
            // want to create vector-valued outputs), and copy the
            // gradients of the solution at the evaluation points
            // into the output slots:
            AssertDimension (computed_quantities[p].size(), dim);
            for (unsigned int d=0; d<dim; ++d)
              computed_quantities[p][d] = input_data.normals[p][d];
        }
    }
};



template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::output_face_results_vtk (const unsigned int cycle)// const
{

    DataOutEulerFaces<dim, dealii::DoFHandler<dim>> data_out;

    data_out.attach_dof_handler (dof_handler);

    std::vector<std::string> position_names;
    for(int d=0;d<dim;++d) {
        if (d==0) position_names.push_back("x");
        if (d==1) position_names.push_back("y");
        if (d==2) position_names.push_back("z");
    }
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, dealii::DataComponentInterpretation::component_is_scalar);
    data_out.add_data_vector (high_order_grid->dof_handler_grid, high_order_grid->volume_nodes, position_names, data_component_interpretation);

    dealii::Vector<float> subdomain(triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) {
        subdomain(i) = triangulation->locally_owned_subdomain();
    }
    const std::string name = "subdomain";
    data_out.add_data_vector(subdomain, name, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);

    //if (all_parameters->add_artificial_dissipation) {
        data_out.add_data_vector(artificial_dissipation_coeffs, std::string("artificial_dissipation_coeffs"), dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);
        data_out.add_data_vector(artificial_dissipation_se, std::string("artificial_dissipation_se"), dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);
    //}
    data_out.add_data_vector(dof_handler_artificial_dissipation, artificial_dissipation_c0, std::string("artificial_dissipation"));

    data_out.add_data_vector(max_dt_cell, std::string("max_dt_cell"), dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);

    data_out.add_data_vector(cell_volume, std::string("cell_volume"), dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);


    // Let the physics post-processor determine what to output.
    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = Postprocess::PostprocessorFactory<dim>::create_Postprocessor(all_parameters);
    data_out.add_data_vector (solution, *post_processor);

    NormalPostprocessor<dim> normals_post_processor;
    data_out.add_data_vector (solution, normals_post_processor);

    // Output the polynomial degree in each cell
    std::vector<unsigned int> active_fe_indices;
    dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    dealii::Vector<double> cell_poly_degree = active_fe_indices_dealiivector;

    data_out.add_data_vector (active_fe_indices_dealiivector, "PolynomialDegree", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);

    // Output absolute value of the residual so that we can visualize it on a logscale.
    std::vector<std::string> residual_names;
    for(int s=0;s<nstate;++s) {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }
    auto residual = right_hand_side;
    for (auto &&rhs_value : residual) {
        if (std::signbit(rhs_value)) rhs_value = -rhs_value;
        if (rhs_value == 0.0) rhs_value = std::numeric_limits<double>::min();
    }
    residual.update_ghost_values();
    data_out.add_data_vector (residual, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_dof_data);

    //for(int s=0;s<nstate;++s) {
    //    residual_names[s] = "scaled_" + residual_names[s];
    //}
    //global_mass_matrix.vmult(residual, right_hand_side);
    //for (auto &&rhs_value : residual) {
    //    if (std::signbit(rhs_value)) rhs_value = -rhs_value;
    //    if (rhs_value == 0.0) rhs_value = std::numeric_limits<double>::min();
    //}
    //residual.update_ghost_values();
    //data_out.add_data_vector (residual, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_dof_data);


    //typename dealii::DataOut<dim,dealii::DoFHandler<dim>>::CurvedCellRegion curved = dealii::DataOut<dim,dealii::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells;
    //typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::curved_boundary;
    //typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::no_curved_cells;

    const dealii::Mapping<dim> &mapping = (*(high_order_grid->mapping_fe_field));
    const int grid_degree = high_order_grid->max_degree;
    //const int n_subdivisions = max_degree+1;//+30; // if write_higher_order_cells, n_subdivisions represents the order of the cell
    //const int n_subdivisions = 1;//+30; // if write_higher_order_cells, n_subdivisions represents the order of the cell
    const int n_subdivisions = grid_degree;
    data_out.build_patches(mapping, n_subdivisions);
    //const bool write_higher_order_cells = (dim>1 && max_degree > 1) ? true : false;
    const bool write_higher_order_cells = false;//(dim>1 && grid_degree > 1) ? true : false;
    dealii::DataOutBase::VtkFlags vtkflags(0.0,cycle,true,dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::best_compression,write_higher_order_cells);
    data_out.set_flags(vtkflags);

    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    std::string filename = "surface_solution-" + dealii::Utilities::int_to_string(dim, 1) +"D_maxpoly"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
    //std::cout << "Writing out file: " << filename << std::endl;

    if (iproc == 0) {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) {
            std::string fn = "surface_solution-" + dealii::Utilities::int_to_string(dim, 1) +"D_maxpoly"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + ".";
            fn += dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        std::string master_fn = "surface_solution-" + dealii::Utilities::int_to_string(dim, 1) +"D_maxpoly"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
        master_fn += dealii::Utilities::int_to_string(cycle, 4) + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }

}
#endif

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::output_results_vtk (const unsigned int cycle)// const
{
#if PHILIP_DIM>1
    output_face_results_vtk (cycle);
#endif

    dealii::DataOut<dim, dealii::DoFHandler<dim>> data_out;

    data_out.attach_dof_handler (dof_handler);

    std::vector<std::string> position_names;
    for(int d=0;d<dim;++d) {
        if (d==0) position_names.push_back("x");
        if (d==1) position_names.push_back("y");
        if (d==2) position_names.push_back("z");
    }
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, dealii::DataComponentInterpretation::component_is_scalar);
    data_out.add_data_vector (high_order_grid->dof_handler_grid, high_order_grid->volume_nodes, position_names, data_component_interpretation);

    dealii::Vector<float> subdomain(triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) {
        subdomain(i) = triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    //if (all_parameters->add_artificial_dissipation) {
        data_out.add_data_vector(artificial_dissipation_coeffs, "artificial_dissipation_coeffs", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
        data_out.add_data_vector(artificial_dissipation_se, "artificial_dissipation_se", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    //}
    data_out.add_data_vector(dof_handler_artificial_dissipation, artificial_dissipation_c0, "artificial_dissipation");

    data_out.add_data_vector(max_dt_cell, "max_dt_cell", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    data_out.add_data_vector(cell_volume, "cell_volume", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);


    // Let the physics post-processor determine what to output.
    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = Postprocess::PostprocessorFactory<dim>::create_Postprocessor(all_parameters);
    data_out.add_data_vector (solution, *post_processor);

    // Output the polynomial degree in each cell
    std::vector<unsigned int> active_fe_indices;
    dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    dealii::Vector<double> cell_poly_degree = active_fe_indices_dealiivector;

    data_out.add_data_vector (active_fe_indices_dealiivector, "PolynomialDegree", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // Output absolute value of the residual so that we can visualize it on a logscale.
    std::vector<std::string> residual_names;
    for(int s=0;s<nstate;++s) {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }
    auto residual = right_hand_side;
    for (auto &&rhs_value : residual) {
        if (std::signbit(rhs_value)) rhs_value = -rhs_value;
        if (rhs_value == 0.0) rhs_value = std::numeric_limits<double>::min();
    }
    residual.update_ghost_values();
    data_out.add_data_vector (residual, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    //for(int s=0;s<nstate;++s) {
    //    residual_names[s] = "scaled_" + residual_names[s];
    //}
    //global_mass_matrix.vmult(residual, right_hand_side);
    //for (auto &&rhs_value : residual) {
    //    if (std::signbit(rhs_value)) rhs_value = -rhs_value;
    //    if (rhs_value == 0.0) rhs_value = std::numeric_limits<double>::min();
    //}
    //residual.update_ghost_values();
    //data_out.add_data_vector (residual, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);


    typename dealii::DataOut<dim,dealii::DoFHandler<dim>>::CurvedCellRegion curved = dealii::DataOut<dim,dealii::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells;
    //typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::curved_boundary;
    //typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::no_curved_cells;

    const dealii::Mapping<dim> &mapping = (*(high_order_grid->mapping_fe_field));
    const int grid_degree = high_order_grid->max_degree;
    //const int n_subdivisions = max_degree+1;//+30; // if write_higher_order_cells, n_subdivisions represents the order of the cell
    //const int n_subdivisions = 1;//+30; // if write_higher_order_cells, n_subdivisions represents the order of the cell
    const int n_subdivisions = grid_degree;
    data_out.build_patches(mapping, n_subdivisions, curved);
    //const bool write_higher_order_cells = (dim>1 && max_degree > 1) ? true : false;
    const bool write_higher_order_cells = (dim>1 && grid_degree > 1) ? true : false;
    dealii::DataOutBase::VtkFlags vtkflags(0.0,cycle,true,dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::best_compression,write_higher_order_cells);
    data_out.set_flags(vtkflags);

    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    std::string filename = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D_maxpoly"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
    //std::cout << "Writing out file: " << filename << std::endl;

    if (iproc == 0) {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) {
            std::string fn = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D_maxpoly"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + ".";
            fn += dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        std::string master_fn = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D_maxpoly"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
        master_fn += dealii::Utilities::int_to_string(cycle, 4) + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }

}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::allocate_system ()
{
    pcout << "Allocating DG system and initializing FEValues" << std::endl;
    // This function allocates all the necessary memory to the
    // system matrices and vectors.

    dof_handler.distribute_dofs(fe_collection);
    dealii::DoFRenumbering::Cuthill_McKee(dof_handler,true);
    //const bool reversed_numbering = true;
    //dealii::DoFRenumbering::Cuthill_McKee(dof_handler, reversed_numbering);
    //const bool reversed_numbering = false;
    //const bool use_constraints = false;
    //dealii::DoFRenumbering::boost::minimum_degree(dof_handler, reversed_numbering, use_constraints);
    //dealii::DoFRenumbering::boost::king_ordering(dof_handler, reversed_numbering, use_constraints);

    //dealii::MappingFEField<dim,dim,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> mapping = high_order_grid->get_MappingFEField();
    //dealii::MappingFEField<dim,dim,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> mapping = *(high_order_grid->mapping_fe_field);

    //int minimum_degree = (all_parameters->use_collocated_nodes==true) ?  1 :  0;
    //int current_fe_index = 0;
    //for (unsigned int degree=minimum_degree; degree<=max_degree; ++degree) {
    //    //mapping_collection.push_back(mapping);
    //    if(current_fe_index <= mapping_collection.size()) {
    //        mapping_collection.push_back(mapping);
    //    } else {
    //        mapping_collection[current_fe_index] = std::shared_ptr<const dealii::Mapping<dim, dim>>(mapping.clone());
    //    }
    //    current_fe_index++;
    //}

    // Solution and RHS
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, ghost_dofs);
    locally_relevant_dofs = ghost_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);
    //dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    dof_handler_artificial_dissipation.distribute_dofs(fe_q_artificial_dissipation);
    const dealii::IndexSet locally_owned_dofs_artificial_dissipation = dof_handler_artificial_dissipation.locally_owned_dofs();

    dealii::IndexSet ghost_dofs_artificial_dissipation;
    dealii::IndexSet locally_relevant_dofs_artificial_dissipation;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_artificial_dissipation, ghost_dofs_artificial_dissipation);
    locally_relevant_dofs_artificial_dissipation = ghost_dofs_artificial_dissipation;
    ghost_dofs_artificial_dissipation.subtract_set(locally_owned_dofs_artificial_dissipation);

    artificial_dissipation_c0.reinit(locally_owned_dofs_artificial_dissipation, ghost_dofs_artificial_dissipation, mpi_communicator);
    artificial_dissipation_c0.update_ghost_values();

    artificial_dissipation_coeffs.reinit(triangulation->n_active_cells());
    artificial_dissipation_se.reinit(triangulation->n_active_cells());
    max_dt_cell.reinit(triangulation->n_active_cells());
    cell_volume.reinit(triangulation->n_active_cells());

    solution.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
    solution *= 0.0;
    solution.add(std::numeric_limits<real>::lowest());
    //right_hand_side.reinit(locally_owned_dofs, mpi_communicator);
    right_hand_side.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
    right_hand_side.add(1.0); // Avoid 0 initial residual for output and logarithmic visualization.
    dual.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);

    //Allocate for auxiliary equation only.
    allocate_auxiliary_equation ();

    // System matrix allocation
    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(locally_owned_dofs, sparsity_pattern, mpi_communicator);

    // system_matrix_transpose.reinit(system_matrix);
    // Epetra_CrsMatrix *input_matrix  = const_cast<Epetra_CrsMatrix *>(&(system_matrix.trilinos_matrix()));
    // Epetra_CrsMatrix *output_matrix;
    // epetra_rowmatrixtransposer_dRdW = std::make_unique<Epetra_RowMatrixTransposer> ( input_matrix );
    // const bool make_data_contiguous = true;
    // epetra_rowmatrixtransposer_dRdW->CreateTranspose( make_data_contiguous, output_matrix);
    // system_matrix_transpose.reinit(*output_matrix);
    // delete(output_matrix);

    // {
    //     dRdW_preconditioner_builder.SetUserMatrix(const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix()));
    //     dRdW_preconditioner_builder.SetAztecOption(AZ_precond, AZ_dom_decomp);
    //     dRdW_preconditioner_builder.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
    //     dRdW_preconditioner_builder.SetAztecOption(AZ_overlap, 0);
    //     dRdW_preconditioner_builder.SetAztecOption(AZ_reorder, 1); // RCM re-ordering

    //     const Parameters::LinearSolverParam &linear_parameters = all_parameters->linear_solver_param;
    //     const double ilut_drop = linear_parameters.ilut_drop;
    //     const double ilut_rtol = linear_parameters.ilut_rtol;
    //     const double ilut_atol = linear_parameters.ilut_atol;
    //     const int ilut_fill = linear_parameters.ilut_fill;

    //     dRdW_preconditioner_builder.SetAztecParam(AZ_drop, ilut_drop);
    //     dRdW_preconditioner_builder.SetAztecParam(AZ_ilut_fill, ilut_fill);
    //     dRdW_preconditioner_builder.SetAztecParam(AZ_athresh, ilut_atol);
    //     dRdW_preconditioner_builder.SetAztecParam(AZ_rthresh, ilut_rtol);

    // }

    // dRdXv matrix allocation
    // dealii::SparsityPattern dRdXv_sparsity_pattern = get_dRdX_sparsity_pattern ();
    // const dealii::IndexSet &row_parallel_partitioning = locally_owned_dofs;
    // const dealii::IndexSet &col_parallel_partitioning = high_order_grid->locally_owned_dofs_grid;
    // //const dealii::IndexSet &col_parallel_partitioning = high_order_grid->locally_relevant_dofs_grid;
    // dRdXv.reinit(row_parallel_partitioning, col_parallel_partitioning, dRdXv_sparsity_pattern, MPI_COMM_WORLD);

    // Make sure that derivatives are cleared when reallocating DG objects.
    // The call to assemble the derivatives will reallocate those derivatives
    // if they are ever needed.
    system_matrix_transpose.clear();
    dRdXv.clear();
    d2RdWdX.clear();
    d2RdWdW.clear();
    d2RdXdX.clear();

    solution_dRdW.reinit(solution);
    solution_dRdW *= 0.0;
    volume_nodes_dRdW.reinit(high_order_grid->volume_nodes);
    volume_nodes_dRdW *= 0.0;

    CFL_mass_dRdW = 0.0;

    solution_dRdX.reinit(solution);
    solution_dRdX *= 0.0;
    volume_nodes_dRdX.reinit(high_order_grid->volume_nodes);
    volume_nodes_dRdX *= 0.0;

    solution_d2R.reinit(solution);
    solution_d2R *= 0.0;
    volume_nodes_d2R.reinit(high_order_grid->volume_nodes);
    volume_nodes_d2R *= 0.0;
    dual_d2R.reinit(dual);
    dual_d2R *= 0.0;
}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::allocate_second_derivatives ()
{
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    {
        dealii::SparsityPattern sparsity_pattern_d2RdWdX = get_d2RdWdX_sparsity_pattern ();
        const dealii::IndexSet &row_parallel_partitioning_d2RdWdX = locally_owned_dofs;
        const dealii::IndexSet &col_parallel_partitioning_d2RdWdX = high_order_grid->locally_owned_dofs_grid;
        d2RdWdX.reinit(row_parallel_partitioning_d2RdWdX, col_parallel_partitioning_d2RdWdX, sparsity_pattern_d2RdWdX, mpi_communicator);
    }

    {
        dealii::SparsityPattern sparsity_pattern_d2RdWdW = get_d2RdWdW_sparsity_pattern ();
        const dealii::IndexSet &row_parallel_partitioning_d2RdWdW = locally_owned_dofs;
        const dealii::IndexSet &col_parallel_partitioning_d2RdWdW = locally_owned_dofs;
        d2RdWdW.reinit(row_parallel_partitioning_d2RdWdW, col_parallel_partitioning_d2RdWdW, sparsity_pattern_d2RdWdW, mpi_communicator);
    }

    {
        dealii::SparsityPattern sparsity_pattern_d2RdXdX = get_d2RdXdX_sparsity_pattern ();
        const dealii::IndexSet &row_parallel_partitioning_d2RdXdX = high_order_grid->locally_owned_dofs_grid;
        const dealii::IndexSet &col_parallel_partitioning_d2RdXdX = high_order_grid->locally_owned_dofs_grid;
        d2RdXdX.reinit(row_parallel_partitioning_d2RdXdX, col_parallel_partitioning_d2RdXdX, sparsity_pattern_d2RdXdX, mpi_communicator);
    }
}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::allocate_dRdX ()
{
    // dRdXv matrix allocation
    dealii::SparsityPattern dRdXv_sparsity_pattern = get_dRdX_sparsity_pattern ();
    const dealii::IndexSet &row_parallel_partitioning = locally_owned_dofs;
    const dealii::IndexSet &col_parallel_partitioning = high_order_grid->locally_owned_dofs_grid;
    dRdXv.reinit(row_parallel_partitioning, col_parallel_partitioning, dRdXv_sparsity_pattern, MPI_COMM_WORLD);
}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::evaluate_mass_matrices (bool do_inverse_mass_matrix)
{

    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const bool use_auxiliary_eq = (all_parameters->pde_type == PDE_enum::convection_diffusion || all_parameters->pde_type == PDE_enum::diffusion) ? true : false;//bool to simplify aux check

    // Mass matrix sparsity pattern
    //dealii::SparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs(), dof_handler.get_fe_collection().max_dofs_per_cell());
    //dealii::SparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs(), dof_handler.get_fe_collection().max_dofs_per_cell());
    //dealii::DynamicSparsityPattern dsp(dof_handler.n_locally_owned_dofs(), dof_handler.n_locally_owned_dofs(), dof_handler.get_fe_collection().max_dofs_per_cell());
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    std::vector<dealii::types::global_dof_index> dofs_indices;
    for (auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++cell) {

        if (!cell->is_locally_owned()) continue;

        const unsigned int fe_index_curr_cell = cell->active_fe_index();

        // Current reference element related to this physical cell
        const dealii::FESystem<dim,dim> &current_fe_ref = fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_cell = current_fe_ref.n_dofs_per_cell();

        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                dsp.add(dofs_indices[itest], dofs_indices[itrial]);
            }
        }
    }
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_owned_dofs);
    mass_sparsity_pattern.copy_from(dsp);
    if (do_inverse_mass_matrix == true) {
        global_inverse_mass_matrix.reinit(locally_owned_dofs, mass_sparsity_pattern);
        if (use_auxiliary_eq){
            for(int idim=0; idim<dim; idim++){
                global_inverse_mass_matrix_auxiliary[idim].reinit(locally_owned_dofs, mass_sparsity_pattern);
            }
        }
        if (this->all_parameters->use_energy == true){//for split form get energy
            global_mass_matrix.reinit(locally_owned_dofs, mass_sparsity_pattern);
            if (use_auxiliary_eq){
                for(int idim=0; idim<dim; idim++){
                    global_mass_matrix_auxiliary[idim].reinit(locally_owned_dofs, mass_sparsity_pattern);
                }
            }
        }
    } else {
        global_mass_matrix.reinit(locally_owned_dofs, mass_sparsity_pattern);
    }

    auto metric_cell = high_order_grid->dof_handler_grid.begin_active();
    for (auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++cell, ++metric_cell) {

        if (!cell->is_locally_owned()) continue;

        const unsigned int fe_index_curr_cell = cell->active_fe_index();

        // Current reference element related to this physical cell
        const dealii::FESystem<dim,dim> &current_fe_ref = fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_cell = current_fe_ref.n_dofs_per_cell();
        const unsigned int n_quad_pts = volume_quadrature_collection[fe_index_curr_cell].size();

        dealii::FullMatrix<real> local_mass_matrix(n_dofs_cell);

        //quadrature weights
        const std::vector<real> &quad_weights = operators.volume_quadrature_collection[fe_index_curr_cell].get_weights();
        //setup metric cell
        const dealii::FESystem<dim> &fe_metric = high_order_grid->fe_system;
        const unsigned int n_metric_dofs = high_order_grid->fe_system.dofs_per_cell;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        const unsigned int grid_degree = high_order_grid->fe_system.tensor_degree();
        //get mapping_support points
        std::vector<std::vector<real>> mapping_support_points(dim);
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_metric_dofs/dim);
        }
        dealii::QGaussLobatto<dim> vol_GLL(grid_degree +1);
        for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
            for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
                const real val = (high_order_grid->volume_nodes[metric_dof_indices[idof]]);
                const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
                mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate); 
            }
        }

        //get determinant of Jacobian
        std::vector<real> determinant_Jacobian(n_quad_pts);
        operators.build_local_vol_determinant_Jac(grid_degree, fe_index_curr_cell, n_quad_pts, n_metric_dofs/dim, mapping_support_points, determinant_Jacobian);

 
        if(this->all_parameters->use_weight_adjusted_mass == false){
           // const std::vector<real> &JxW = fe_values_volume.get_JxW_values();
            std::vector<real> JxW(n_quad_pts);
         //   const std::vector<real> &JxW_dealii = fe_values_volume.get_JxW_values();
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                JxW[iquad] = quad_weights[iquad] * determinant_Jacobian[iquad];
            }
            operators.build_local_Mass_Matrix(JxW, n_dofs_cell, n_quad_pts, fe_index_curr_cell, local_mass_matrix);

            std::vector<dealii::FullMatrix<real>> local_mass_matrix_aux(dim);
            for(int idim=0; idim<dim; idim++){
                local_mass_matrix_aux[idim].reinit(n_dofs_cell, n_dofs_cell);
                if(use_auxiliary_eq){
                    local_mass_matrix_aux[idim].add(1.0,local_mass_matrix);
                }
            }

            if(this->all_parameters->flux_reconstruction_type != Parameters::AllParameters::Flux_Reconstruction::cDG){
                //For flux reconstruction
                dealii::FullMatrix<real> K_operator(n_dofs_cell);
                const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
                operators.build_local_K_operator(local_mass_matrix, n_dofs_cell, curr_cell_degree, K_operator);
                for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                    for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                        local_mass_matrix[itest][itrial] = local_mass_matrix[itest][itrial] + K_operator[itest][itrial];
                    }
                }
            }
            if(use_auxiliary_eq && this->all_parameters->flux_reconstruction_aux_type != Parameters::AllParameters::Flux_Reconstruction_Aux::kDG){
                std::vector<dealii::FullMatrix<real>> K_operator_aux(dim);
                for(int idim=0; idim<dim; idim++){
                    K_operator_aux[idim].reinit(n_dofs_cell, n_dofs_cell);
                }
                const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
                operators.build_local_K_operator_AUX(local_mass_matrix, n_dofs_cell, curr_cell_degree, K_operator_aux);
                for(int idim=0; idim<dim; idim++){
                    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                        for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                            local_mass_matrix_aux[idim][itest][itrial] = local_mass_matrix[itest][itrial] + K_operator_aux[idim][itest][itrial];
                        }
                    }
                }
            }

            dofs_indices.resize(n_dofs_cell);
            cell->get_dof_indices (dofs_indices);
            if (do_inverse_mass_matrix == true) {
                dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
                local_inverse_mass_matrix.invert(local_mass_matrix);
                global_inverse_mass_matrix.set (dofs_indices, local_inverse_mass_matrix);

                if(use_auxiliary_eq){
                    dealii::FullMatrix<real> local_inverse_mass_matrix_aux(n_dofs_cell);
                    for(int idim=0; idim<dim; idim++){
                        local_inverse_mass_matrix_aux.invert(local_mass_matrix_aux[idim]);
                        global_inverse_mass_matrix_auxiliary[idim].set (dofs_indices, local_inverse_mass_matrix_aux);
                    }                 
                }
                if (this->all_parameters->use_energy == true){//for split form energy
                    global_mass_matrix.set (dofs_indices, local_mass_matrix);
                    if(use_auxiliary_eq){
                        for(int idim=0; idim<dim; idim++){
                            global_mass_matrix_auxiliary[idim].set (dofs_indices, local_mass_matrix_aux[idim]);
                        }                 
                    }
                }
            } else {
                global_mass_matrix.set (dofs_indices, local_mass_matrix);
            }
        } else {//use weight adjusted Mass Matrix (it's based off the inverse)
            std::vector<real> W_J_inv(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                W_J_inv[iquad] = quad_weights[iquad] / determinant_Jacobian[iquad]; 
            }
            const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
            operators.build_local_Mass_Matrix(W_J_inv, n_dofs_cell, n_quad_pts, curr_cell_degree, local_mass_matrix);
            if(this->all_parameters->flux_reconstruction_type != Parameters::AllParameters::Flux_Reconstruction::cDG){
                //For flux reconstruction
                dealii::FullMatrix<real> K_operator(n_dofs_cell);
                operators.build_local_K_operator(local_mass_matrix, n_dofs_cell, curr_cell_degree, K_operator);
                for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                    for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                        local_mass_matrix[itest][itrial] = local_mass_matrix[itest][itrial] + K_operator[itest][itrial];
                    }
                }
            }
            dofs_indices.resize(n_dofs_cell);
            cell->get_dof_indices (dofs_indices);
            if (do_inverse_mass_matrix == true) {
                dealii::FullMatrix<real> temp(n_dofs_cell);
                operators.FR_mass_inv[curr_cell_degree].mmult(temp, local_mass_matrix);
                dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
                temp.mmult(local_inverse_mass_matrix, operators.FR_mass_inv[curr_cell_degree]); 
                global_inverse_mass_matrix.set (dofs_indices, local_inverse_mass_matrix);
            
                if (this->all_parameters->use_energy == true){//for split form energy
                    dealii::FullMatrix<real> inverse_of_weighted_mass_inverse(n_dofs_cell);
                    inverse_of_weighted_mass_inverse.invert(local_inverse_mass_matrix);
                    global_mass_matrix.set (dofs_indices, inverse_of_weighted_mass_inverse);
                }
            } else {
                dealii::FullMatrix<real> inv_weighted_mass_inv(n_dofs_cell);
                inv_weighted_mass_inv.invert(local_mass_matrix);
                dealii::FullMatrix<real> M_K(n_dofs_cell);
                M_K.add(1.0, operators.local_mass[curr_cell_degree], 1.0, operators.local_K_operator[curr_cell_degree]);
                dealii::FullMatrix<real> temp(n_dofs_cell);
                M_K.mmult(temp, inv_weighted_mass_inv);
                dealii::FullMatrix<real> inverse_of_weighted_mass_inverse(n_dofs_cell);
                temp.mmult(inverse_of_weighted_mass_inverse, inv_weighted_mass_inv);
                global_mass_matrix.set (dofs_indices, inverse_of_weighted_mass_inverse);
            }

        }//end of weight-adjusted mass matrix condition
    }//end of cell loop

    if (do_inverse_mass_matrix == true) {
        global_inverse_mass_matrix.compress(dealii::VectorOperation::insert);
        if (use_auxiliary_eq){
            for(int idim=0; idim<dim; idim++){
                global_inverse_mass_matrix_auxiliary[idim].compress(dealii::VectorOperation::insert);
            }
        }
        if (this->all_parameters->use_energy == true){//for split form energy
            global_mass_matrix.compress(dealii::VectorOperation::insert);
        }
    } else {
        global_mass_matrix.compress(dealii::VectorOperation::insert);
        //std::cout << " global_mass_matrix "  << std::endl;
        //std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
        //global_mass_matrix.print(std::cout);
        ////std::abort();
    }

    return;
}

template<int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::add_mass_matrices(const real scale)
{
    system_matrix.add(scale, global_mass_matrix);
}
template<int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::add_time_scaled_mass_matrices()
{
    system_matrix.add(1.0, time_scaled_global_mass_matrix);
}
template<int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::time_scaled_mass_matrices(const real dt_scale)
{
    time_scaled_global_mass_matrix.reinit(system_matrix);
    time_scaled_global_mass_matrix = 0.0;
    std::vector<dealii::types::global_dof_index> dofs_indices;
    for (auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++cell) {

        if (!cell->is_locally_owned()) continue;

        const unsigned int fe_index_curr_cell = cell->active_fe_index();

        // Current reference element related to this physical cell
        const dealii::FESystem<dim,dim> &current_fe_ref = fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_cell = current_fe_ref.n_dofs_per_cell();

        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);

        const double max_dt = max_dt_cell[cell->active_cell_index()];

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = current_fe_ref.system_to_component_index(itest).first;
            for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {
                const unsigned int istate_trial = current_fe_ref.system_to_component_index(itrial).first;

                if(istate_test==istate_trial) {
                    const unsigned int row = dofs_indices[itest];
                    const unsigned int col = dofs_indices[itrial];
                    const double value = global_mass_matrix.el(row, col);
                    const double new_val = value / (dt_scale * max_dt);
                    AssertIsFinite(new_val);
                    time_scaled_global_mass_matrix.set(row, col, new_val);
                    if (row!=col) time_scaled_global_mass_matrix.set(col, row, new_val);
                }
            }
        }
    }
    time_scaled_global_mass_matrix.compress(dealii::VectorOperation::insert);
}

template<int dim, typename real>
std::vector< real > project_function(
    const std::vector< real > &function_coeff,
    const dealii::FESystem<dim,dim> &fe_input,
    const dealii::FESystem<dim,dim> &fe_output,
    const dealii::QGauss<dim> &projection_quadrature)
{
    const unsigned int nstate = fe_input.n_components();
    const unsigned int n_vector_dofs_in = fe_input.dofs_per_cell;
    const unsigned int n_vector_dofs_out = fe_output.dofs_per_cell;
    const unsigned int n_dofs_in = n_vector_dofs_in / nstate;
    const unsigned int n_dofs_out = n_vector_dofs_out / nstate;

    assert(n_vector_dofs_in == function_coeff.size());
    assert(nstate == fe_output.n_components());

    const unsigned int n_quad_pts = projection_quadrature.size();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = projection_quadrature.get_points();

    std::vector< real > function_coeff_out(n_vector_dofs_out); // output function coefficients.
    for (unsigned istate = 0; istate < nstate; ++istate) {

        std::vector< real > function_at_quad(n_quad_pts);

        // Output interpolation_operator is V^T in the notes.
        dealii::FullMatrix<double> interpolation_operator(n_dofs_out,n_quad_pts);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            function_at_quad[iquad] = 0.0;
            for (unsigned int idof=0; idof<n_dofs_in; ++idof) {
                const unsigned int idof_vector = fe_input.component_to_system_index(istate,idof);
                function_at_quad[iquad] += function_coeff[idof_vector] * fe_input.shape_value_component(idof_vector,unit_quad_pts[iquad],istate);
            }
            function_at_quad[iquad] *= projection_quadrature.weight(iquad);

            for (unsigned int idof=0; idof<n_dofs_out; ++idof) {
                const unsigned int idof_vector = fe_output.component_to_system_index(istate,idof);
                interpolation_operator[idof][iquad] = fe_output.shape_value_component(idof_vector,unit_quad_pts[iquad],istate);
            }
        }

        std::vector< real > rhs(n_dofs_out);
        for (unsigned int idof=0; idof<n_dofs_out; ++idof) {
            rhs[idof] = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                rhs[idof] += interpolation_operator[idof][iquad] * function_at_quad[iquad];
            }
        }

        dealii::FullMatrix<double> mass(n_dofs_out, n_dofs_out);
        for(unsigned int row=0; row<n_dofs_out; ++row) {
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                mass[row][col] = 0;
            }
        }
        for(unsigned int row=0; row<n_dofs_out; ++row) {
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    mass[row][col] += interpolation_operator[row][iquad] * interpolation_operator[col][iquad] * projection_quadrature.weight(iquad);
                }
            }
        }
        dealii::FullMatrix<double> inverse_mass(n_dofs_out, n_dofs_out);
        inverse_mass.invert(mass);

        for(unsigned int row=0; row<n_dofs_out; ++row) {
            const unsigned int idof_vector = fe_output.component_to_system_index(istate,row);
            function_coeff_out[idof_vector] = 0.0;
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                function_coeff_out[idof_vector] += inverse_mass[row][col] * rhs[col];
            }
        }
    }

    return function_coeff_out;
    //
    //int mpi_rank;
    //(void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    //if (mpi_rank==0) mass.print(std::cout);

}

template <int dim, typename real,typename MeshType>
template <typename real2>
real2 DGBase<dim,real,MeshType>::discontinuity_sensor(
    const dealii::Quadrature<dim> &volume_quadrature,
    const std::vector< real2 > &soln_coeff_high,
    const dealii::FiniteElement<dim,dim> &fe_high,
    const std::vector<real2> &jac_det)
{
    const unsigned int degree = fe_high.tensor_degree();

    if (degree == 0) return 0;

    const unsigned int nstate = fe_high.components;
    const unsigned int n_dofs_high = fe_high.dofs_per_cell;

    // Lower degree basis.
    const unsigned int lower_degree = degree-1;
    const dealii::FE_DGQLegendre<dim> fe_dgq_lower(lower_degree);
    const dealii::FESystem<dim,dim> fe_lower(fe_dgq_lower, nstate);

    // Projection quadrature.
    const dealii::QGauss<dim> projection_quadrature(degree+5);
    std::vector< real2 > soln_coeff_lower = project_function<dim,real2>( soln_coeff_high, fe_high, fe_lower, projection_quadrature);

    // Quadrature used for solution difference.
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = volume_quadrature.get_points();

    const unsigned int n_quad_pts = volume_quadrature.size();
    const unsigned int n_dofs_lower = fe_lower.dofs_per_cell;

    real2 element_volume = 0.0;
    real2 error = 0.0;
    real2 soln_norm = 0.0;
    std::vector<real2> soln_high(nstate);
    std::vector<real2> soln_lower(nstate);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int s=0; s<nstate; ++s) {
            soln_high[s] = 0.0;
            soln_lower[s] = 0.0;
        }
        // Interpolate solution
        for (unsigned int idof=0; idof<n_dofs_high; ++idof) {
              const unsigned int istate = fe_high.system_to_component_index(idof).first;
              soln_high[istate] += soln_coeff_high[idof] * fe_high.shape_value_component(idof,unit_quad_pts[iquad],istate);
        }
        // Interpolate low order solution
        for (unsigned int idof=0; idof<n_dofs_lower; ++idof) {
              const unsigned int istate = fe_lower.system_to_component_index(idof).first;
              soln_lower[istate] += soln_coeff_lower[idof] * fe_lower.shape_value_component(idof,unit_quad_pts[iquad],istate);
        }
        // Quadrature
        const real2 JxW = jac_det[iquad] * volume_quadrature.weight(iquad);
        element_volume += JxW;
        // Only integrate over the first state variable.
        // Persson and Peraire only did density.
        for (unsigned int s=0; s<1/*nstate*/; ++s) 
        {
            error += (soln_high[s] - soln_lower[s]) * (soln_high[s] - soln_lower[s]) * JxW;
            soln_norm += soln_high[s] * soln_high[s] * JxW;
        }
    }

    if (soln_norm < 1e-15) return 0;

    const real2 S_e = sqrt(error / soln_norm);
    const real2 s_e = log10(S_e);

    const double mu_scale = all_parameters->artificial_dissipation_param.mu_artificial_dissipation;
    const double s_0 = -0.00 - 4.00*log10(degree);
    const double kappa = all_parameters->artificial_dissipation_param.kappa_artificial_dissipation;
    const double low = s_0 - kappa;
    const double upp = s_0 + kappa;

    const real2 diameter = pow(element_volume, 1.0/dim);
    const real2 eps_0 = mu_scale * diameter / (double)degree;

    if ( s_e < low) return 0.0;

    if ( s_e > upp) 
    {
        return eps_0;
    }

    const double PI = 4*atan(1);
    real2 eps = 1.0 + sin(PI * (s_e - s_0) * 0.5 / kappa);
    eps *= eps_0 * 0.5;
    return eps;
}

//template<int dim, typename real, typename MeshType>
//void DGBase<dim,real,MeshType>::refine_residual_based()
//{
//    dealii::Vector<float> gradient_indicator(high_order_grid->triangulation->n_active_cells());
//
//    const auto mapping = (*(high_order_grid->mapping_fe_field));
//    dealii::DerivativeApproximation::approximate_gradient(mapping,
//                                                  dof_handler,
//                                                  solution,
//                                                  gradient_indicator);
//
//    for (const auto &cell : high_order_grid->triangulation->active_cell_iterators()) {
//        gradient_indicator[cell->active_cell_index()] *= std::pow(cell->diameter(), 1 + 1.0 * dim / 2);
//    }
//    std::vector<dealii::types::global_dof_index> dofs_indices;
//    for (const auto &cell : dof_handler.active_cell_iterators()) {
//        if (!cell->is_locally_owned()) continue;
//
//        const int i_fele = cell->active_fe_index();
//        const dealii::FESystem<dim,dim> &fe_ref = fe_collection[i_fele];
//        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();
//        dofs_indices.resize(n_dofs_cell);
//        cell->get_dof_indices (dofs_indices);
//        double max_residual = 0;
//        for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
//        //for (const auto &idof : dofs_indices) {
//            const unsigned int index = dofs_indices[idof];
//            const unsigned int istate = fe_ref.system_to_component_index(idof).first;
//            if (istate == dim+2-1) {
//                const double res = std::abs(right_hand_side[index]);
//                if (res > max_residual) max_residual = res;
//            }
//        }
//        gradient_indicator[cell->active_cell_index()] = max_residual;
//    }
//
//    dealii::LinearAlgebra::distributed::Vector<double> old_solution(solution);
//    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dof_handler);
//    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
//    high_order_grid->prepare_for_coarsening_and_refinement();
//
//    //high_order_grid->triangulation->refine_global (1);
//    if constexpr(dim == 1 ||
//        !std::is_same<MeshType, dealii::parallel::distributed::Triangulation<dim>>::value) {
//        dealii::GridRefinement::refine_and_coarsen_fixed_number(*high_order_grid->triangulation,
//                                                        gradient_indicator,
//                                                        0.05,
//                                                        0.025);
//    } else {
//        dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(*(high_order_grid->triangulation),
//                                                        gradient_indicator,
//                                                        0.05,
//                                                        0.01);
//    }
//    high_order_grid->triangulation->execute_coarsening_and_refinement();
//    high_order_grid->execute_coarsening_and_refinement();
//    allocate_system ();
//    solution.zero_out_ghosts();
//    solution_transfer.interpolate(solution);
//    solution.update_ghost_values();
//
//    assemble_residual ();
//
//}

template <int dim, typename real, typename MeshType>
void DGBase<dim,real,MeshType>::set_current_time(const real time)
{
    this->current_time = time;
}
// No support for anisotropic mesh refinement with parallel::distributed::Triangulation
// template<int dim, typename real>
// void DGBase<dim,real,MeshType>::set_anisotropic_flags()
// {
//     dealii::UpdateFlags face_update_flags = dealii::UpdateFlags(dealii::update_values | dealii::update_JxW_values);
//
//     const auto mapping = (*(high_order_grid->mapping_fe_field));
//
//     dealii::hp::MappingCollection<dim>   mapping_collection(mapping);
//     dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, fe_collection, face_quadrature_collection, face_update_flags);
//     dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, fe_collection, face_quadrature_collection, face_update_flags);
//     dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface  (mapping_collection, fe_collection, face_quadrature_collection, face_update_flags);
//
//     for (const auto &cell : dof_handler.active_cell_iterators()) {
//         if (!cell->is_locally_owned()) continue;
//         if (cell->refine_flag_set()) {
//             dealii::Point<dim> jump;
//             dealii::Point<dim> area;
//
//             // Current reference element related to this physical cell
//             const int i_fele = cell->active_fe_index();
//             const int i_quad = i_fele;
//             const int i_mapp = 0;
//
//             for (const auto face_no : cell->face_indices()) {
//
//                 const auto face = cell->face(face_no);
//
//                 if (!face->at_boundary()) {
//
//                     Assert(cell->neighbor(face_no).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
//                     const auto neighbor = cell->neighbor(face_no);
//
//                     if (face->has_children()) {
//
//                         unsigned int neighbor_iface = cell->neighbor_face_no(face_no);
//                         for (unsigned int subface_no = 0; subface_no < face->number_of_children(); ++subface_no) {
//
//                             const auto neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);
//                             const int i_fele_n = neighbor_child->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;
//                             Assert(!neighbor_child->has_children(), dealii::ExcInternalError());
//
//                             fe_values_collection_subface.reinit(cell, face_no, subface_no, i_fele, i_quad, i_mapp);
//                             fe_values_collection_face_ext.reinit(neighbor_child, neighbor_iface, i_fele_n, i_quad_n, i_mapp_n);
//
//                             const dealii::FESubfaceValues<dim,dim> &fe_values_face_int = fe_values_collection_subface.get_present_fe_values();
//                             const dealii::FEFaceValues<dim,dim>    &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();
//
//                             std::vector<dealii::Vector<double>> u(fe_values_face_int.n_quadrature_points);
//                             std::vector<dealii::Vector<double>> u_neighbor(fe_values_face_int.n_quadrature_points);
//                             std::fill(u.begin(), u.end(), dealii::Vector<double>(nstate));
//                             std::fill(u_neighbor.begin(), u_neighbor.end(), dealii::Vector<double>(nstate));
//
//                             fe_values_face_int.get_function_values(solution, u);
//                             fe_values_face_ext.get_function_values(solution, u_neighbor);
//
//                             const std::vector<double> &JxW = fe_values_face_int.get_JxW_values();
//
//                             for (unsigned int iquad = 0; iquad < fe_values_face_int.n_quadrature_points; ++iquad) {
//                                 u[iquad].add(-1.0, u_neighbor[iquad]);
//                                 const double diff_u = (u[iquad]).l2_norm();
//                                 jump[face_no / 2] += std::abs(diff_u) * JxW[iquad];
//                                 area[face_no / 2] += JxW[iquad];
//                             }
//                         }
//
//                     } else if (!cell->neighbor_is_coarser(face_no)) {
//                         unsigned int neighbor_iface = cell->neighbor_of_neighbor(face_no);
//                         const int i_fele_n = neighbor->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;
//
//                         fe_values_collection_face_int.reinit(cell, face_no, i_fele, i_quad, i_mapp);
//                         fe_values_collection_face_ext.reinit(neighbor, neighbor_iface, i_fele_n, i_quad_n, i_mapp_n);
//
//                         const dealii::FEFaceValues<dim,dim>    &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
//                         const dealii::FEFaceValues<dim,dim>    &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();
//
//                         std::vector<dealii::Vector<double>> u(fe_values_face_int.n_quadrature_points);
//                         std::vector<dealii::Vector<double>> u_neighbor(fe_values_face_int.n_quadrature_points);
//                         std::fill(u.begin(), u.end(), dealii::Vector<double>(nstate));
//                         std::fill(u_neighbor.begin(), u_neighbor.end(), dealii::Vector<double>(nstate));
//
//                         fe_values_face_int.get_function_values(solution, u);
//                         fe_values_face_ext.get_function_values(solution, u_neighbor);
//
//                         const std::vector<double> &JxW = fe_values_face_int.get_JxW_values();
//
//                         for (unsigned int iquad = 0; iquad < fe_values_face_int.n_quadrature_points; ++iquad) {
//                             u[iquad].add(-1.0, u_neighbor[iquad]);
//                             const double diff_u = (u[iquad]).l2_norm();
//                             jump[face_no / 2] += std::abs(diff_u) * JxW[iquad];
//                             area[face_no / 2] += JxW[iquad];
//                         }
//                     } else { // i.e. neighbor is coarser than cell
//                         std::pair<unsigned int, unsigned int> neighbor_face_subface = cell->neighbor_of_coarser_neighbor(face_no);
//                         Assert(neighbor_face_subface.first < cell->n_faces(), dealii::ExcInternalError());
//                         Assert(neighbor_face_subface.second < neighbor->face(neighbor_face_subface.first) ->number_of_children(), dealii::ExcInternalError());
//                         Assert(neighbor->neighbor_child_on_subface( neighbor_face_subface.first, neighbor_face_subface.second) == cell, dealii::ExcInternalError());
//
//                         const int i_fele_n = neighbor->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;
//
//                         fe_values_collection_face_int.reinit(cell, face_no, i_fele, i_quad, i_mapp);
//                         fe_values_collection_subface.reinit(neighbor, neighbor_face_subface.first, neighbor_face_subface.second, i_fele_n, i_quad_n, i_mapp_n);
//
//                         const dealii::FEFaceValues<dim,dim>       &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
//                         const dealii::FESubfaceValues<dim,dim>    &fe_values_face_ext = fe_values_collection_subface.get_present_fe_values();
//
//                         std::vector<dealii::Vector<double>> u(fe_values_face_int.n_quadrature_points);
//                         std::vector<dealii::Vector<double>> u_neighbor(fe_values_face_int.n_quadrature_points);
//                         std::fill(u.begin(), u.end(), dealii::Vector<double>(nstate));
//                         std::fill(u_neighbor.begin(), u_neighbor.end(), dealii::Vector<double>(nstate));
//
//                         fe_values_face_int.get_function_values(solution, u);
//                         fe_values_face_ext.get_function_values(solution, u_neighbor);
//
//                         const std::vector<double> &JxW = fe_values_face_int.get_JxW_values();
//
//                         for (unsigned int iquad = 0; iquad < fe_values_face_int.n_quadrature_points; ++iquad) {
//                             u[iquad].add(-1.0, u_neighbor[iquad]);
//                             const double diff_u = (u[iquad]).l2_norm();
//                             jump[face_no / 2] += std::abs(diff_u) * JxW[iquad];
//                             area[face_no / 2] += JxW[iquad];
//                         }
//                     }
//                 }
//             }
//             std::array<double, dim> average_jumps;
//             double                  sum_of_average_jumps = 0.;
//             for (unsigned int i = 0; i < dim; ++i) {
//                 average_jumps[i] = jump(i) / area(i);
//                 sum_of_average_jumps += average_jumps[i];
//             }
//
//             const double anisotropic_threshold_ratio = 3.0;
//             for (unsigned int i = 0; i < dim; ++i) {
//                 if (average_jumps[i] > anisotropic_threshold_ratio * (sum_of_average_jumps - average_jumps[i])) {
//                     cell->set_refine_flag(dealii::RefinementCase<dim>::cut_axis(i));
//                 }
//             }
//         }
//     }
// }

template class DGBase <PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGBase <PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM!=1
template class DGBase <PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGBaseState <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

template double DGBase<PHILIP_DIM,double,dealii::Triangulation<PHILIP_DIM>>::discontinuity_sensor<double>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< double > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<double>  &jac_det);
template FadType DGBase<PHILIP_DIM,double,dealii::Triangulation<PHILIP_DIM>>::discontinuity_sensor<FadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< FadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<FadType>  &jac_det);
template RadType DGBase<PHILIP_DIM,double,dealii::Triangulation<PHILIP_DIM>>::discontinuity_sensor<RadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< RadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<RadType>  &jac_det);
template FadFadType DGBase<PHILIP_DIM,double,dealii::Triangulation<PHILIP_DIM>>::discontinuity_sensor<FadFadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< FadFadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<FadFadType>  &jac_det);
template RadFadType DGBase<PHILIP_DIM,double,dealii::Triangulation<PHILIP_DIM>>::discontinuity_sensor<RadFadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< RadFadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<RadFadType>  &jac_det);


template double DGBase<PHILIP_DIM,double,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::discontinuity_sensor<double>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< double > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<double>  &jac_det);
template FadType DGBase<PHILIP_DIM,double,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::discontinuity_sensor<FadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< FadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<FadType>  &jac_det);
template RadType DGBase<PHILIP_DIM,double,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::discontinuity_sensor<RadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< RadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<RadType>  &jac_det);
template FadFadType DGBase<PHILIP_DIM,double,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::discontinuity_sensor<FadFadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< FadFadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<FadFadType>  &jac_det);
template RadFadType DGBase<PHILIP_DIM,double,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::discontinuity_sensor<RadFadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< RadFadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<RadFadType>  &jac_det);


template double DGBase<PHILIP_DIM,double,dealii::parallel::shared::Triangulation<PHILIP_DIM>>::discontinuity_sensor<double>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< double > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<double>  &jac_det);
template FadType DGBase<PHILIP_DIM,double,dealii::parallel::shared::Triangulation<PHILIP_DIM>>::discontinuity_sensor<FadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< FadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<FadType>  &jac_det);
template RadType DGBase<PHILIP_DIM,double,dealii::parallel::shared::Triangulation<PHILIP_DIM>>::discontinuity_sensor<RadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< RadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<RadType>  &jac_det);
template FadFadType DGBase<PHILIP_DIM,double,dealii::parallel::shared::Triangulation<PHILIP_DIM>>::discontinuity_sensor<FadFadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< FadFadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<FadFadType>  &jac_det);
template RadFadType DGBase<PHILIP_DIM,double,dealii::parallel::shared::Triangulation<PHILIP_DIM>>::discontinuity_sensor<RadFadType>(const dealii::Quadrature<PHILIP_DIM> &volume_quadrature, const std::vector< RadFadType > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high, const std::vector<RadFadType>  &jac_det);

} // PHiLiP namespace
