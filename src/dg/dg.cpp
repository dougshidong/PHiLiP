#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/base/qprojector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

//#include <deal.II/lac/solver_control.h>
//#include <deal.II/lac/trilinos_precondition.h>
//#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/fe/fe_dgq.h>

//#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/fe/mapping_q.h> // Might need mapping_q
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/mapping_manifold.h> 
#include <deal.II/fe/mapping_fe_field.h> 

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/dofs/dof_renumbering.h>


#include "dg.h"
#include "post_processor/physics_post_processor.h"

//template class dealii::MappingFEField<PHILIP_DIM,PHILIP_DIM,dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<PHILIP_DIM> >;
namespace PHiLiP {
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    template <int dim> using Triangulation = dealii::Triangulation<dim>;
#else
    template <int dim> using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif

// DGFactory ***********************************************************************
template <int dim, typename real>
std::shared_ptr< DGBase<dim,real> >
DGFactory<dim,real>
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
            return std::make_shared< DGWeak<dim,1,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::advection_vector) {
            return std::make_shared< DGWeak<dim,2,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::diffusion) {
            return std::make_shared< DGWeak<dim,1,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return std::make_shared< DGWeak<dim,1,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_inviscid) {
            return std::make_shared< DGWeak<dim,dim,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::euler) {
            return std::make_shared< DGWeak<dim,dim+2,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
    } else {
        if (pde_type == PDE_enum::advection) {
            return std::make_shared< DGStrong<dim,1,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::advection_vector) {
            return std::make_shared< DGStrong<dim,2,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::diffusion) {
            return std::make_shared< DGStrong<dim,1,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return std::make_shared< DGStrong<dim,1,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_inviscid) {
            return std::make_shared< DGStrong<dim,dim,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        } else if (pde_type == PDE_enum::euler) {
            return std::make_shared< DGStrong<dim,dim+2,real> >(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input);
        }
    }
    std::cout << "Can't create DGBase in create_discontinuous_galerkin(). Invalid PDE type: " << pde_type << std::endl;
    return nullptr;
}

template <int dim, typename real>
std::shared_ptr< DGBase<dim,real> >
DGFactory<dim,real>
::create_discontinuous_galerkin(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
{
    return create_discontinuous_galerkin(parameters_input, degree, max_degree_input, degree+1, triangulation_input);
}

template <int dim, typename real>
std::shared_ptr< DGBase<dim,real> >
DGFactory<dim,real>
::create_discontinuous_galerkin(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const std::shared_ptr<Triangulation> triangulation_input)
{
    return create_discontinuous_galerkin(parameters_input, degree, degree, triangulation_input);
}

// DGBase ***************************************************************************
template <int dim, typename real>
DGBase<dim,real>::DGBase(
    const int nstate_input,
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBase<dim,real>(nstate_input, parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input, this->create_collection_tuple(max_degree_input, nstate_input, parameters_input))
{ }

template <int dim, typename real>
DGBase<dim,real>::DGBase( 
    const int nstate_input,
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input,
    const MassiveCollectionTuple collection_tuple)
    : all_parameters(parameters_input)
    , nstate(nstate_input)
    , max_degree(max_degree_input)
    , triangulation(triangulation_input)
    , fe_collection(std::get<0>(collection_tuple))
    , volume_quadrature_collection(std::get<1>(collection_tuple))
    , face_quadrature_collection(std::get<2>(collection_tuple))
    , oned_quadrature_collection(std::get<3>(collection_tuple))
    , fe_collection_lagrange(std::get<4>(collection_tuple))
    , dof_handler(*triangulation)
    , high_order_grid(grid_degree_input, triangulation)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{ 

    dof_handler.initialize(*triangulation, fe_collection);

    set_all_cells_fe_degree(degree); 

}

template <int dim, typename real> 
std::tuple<
        //dealii::hp::MappingCollection<dim>, // Mapping
        dealii::hp::FECollection<dim>, // Solution FE
        dealii::hp::QCollection<dim>,  // Volume quadrature
        dealii::hp::QCollection<dim-1>, // Face quadrature
        dealii::hp::QCollection<1>, // 1D quadrature for strong form
        dealii::hp::FECollection<dim> >   // Lagrange polynomials for strong form
DGBase<dim,real>::create_collection_tuple(const unsigned int max_degree, const int nstate, const Parameters::AllParameters *const parameters_input) const
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

		//

		dealii::Quadrature<1>     oned_quad(degree+1);
		dealii::Quadrature<dim>   volume_quad(degree+1);
		dealii::Quadrature<dim-1> face_quad(degree+1); //removed const

		if (parameters_input->use_collocated_nodes)
			{
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


			}
			else
			{
				dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1);
				dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1);
				dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
				oned_quad = oned_quad_Gauss_Legendre;
				volume_quad = vol_quad_Gauss_Legendre;
				face_quad = face_quad_Gauss_Legendre;
			}
		//


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
            const unsigned int overintegration = 0;
            dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1+overintegration);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1+overintegration);
            dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
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


template <int dim, typename real>
void DGBase<dim,real>::set_all_cells_fe_degree ( const unsigned int degree )
{
    triangulation->prepare_coarsening_and_refinement();
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
    {
        if (cell->is_locally_owned()) cell->set_future_fe_index (degree);
    }

    triangulation->execute_coarsening_and_refinement();
}



// Destructor
template <int dim, typename real>
DGBase<dim,real>::~DGBase () 
{ 
    dof_handler.clear ();
}

// template <int dim, typename real>
// void DGBase<dim,real>::set_triangulation(Triangulation *triangulation_input)
// { 
//     // dof_handler.clear();
// 
//     // triangulation = triangulation_input;
// 
//     // dof_handler.initialize(*triangulation, fe_collection);
// 
//     //set_all_cells_fe_degree (fe_collection.size()-1); // Always sets it to the maximum degree
// 
//     //set_all_cells_fe_degree ( max_degree-min_degree);
// }

template <int dim, typename real>
template<typename DoFCellAccessorType>
real DGBase<dim,real>::evaluate_penalty_scaling (
    const DoFCellAccessorType &cell,
    const int iface,
    const dealii::hp::FECollection<dim> fe_collection) const
{

    const unsigned int fe_index = cell->active_fe_index();
    const unsigned int degree = fe_collection[fe_index].tensor_degree();
    const unsigned int degsq = (degree == 0) ? 1 : degree * (degree+1);

    const unsigned int normal_direction = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
    const real vol_div_facearea = cell->extent_in_direction(normal_direction);

    const real penalty = degsq / vol_div_facearea;

    return penalty;
}

template <int dim, typename real>
template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
bool DGBase<dim,real>::current_cell_should_do_the_work (const DoFCellAccessorType1 &current_cell, const DoFCellAccessorType2 &neighbor_cell) const
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

template <int dim, typename real>
template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
void DGBase<dim,real>::assemble_cell_residual (
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

    const unsigned int n_metric_dofs_cell = high_order_grid.fe_system.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> current_metric_dofs_indices(n_metric_dofs_cell);
    std::vector<dealii::types::global_dof_index> neighbor_metric_dofs_indices(n_metric_dofs_cell);
    current_metric_cell->get_dof_indices (current_metric_dofs_indices);

    if (all_parameters->add_artificial_dissipation) {
        const unsigned int n_soln_dofs = fe_values_volume.dofs_per_cell;
        const double cell_diameter = current_cell->diameter();
        std::vector< real > soln_coeff(n_soln_dofs);
        for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
            soln_coeff[idof] = solution(current_dofs_indices[idof]);
        }
        const double artificial_diss_coeff = discontinuity_sensor(cell_diameter, soln_coeff, fe_values_volume.get_fe());
        artificial_dissipation_coeffs[current_cell->active_cell_index()] = artificial_diss_coeff;
    }

    if ( compute_dRdW || compute_dRdX || compute_d2R ) {
        assemble_volume_terms_derivatives (
            fe_values_volume, current_fe_ref, volume_quadrature_collection[i_quad],
            current_metric_dofs_indices, current_dofs_indices,
            current_cell_rhs, fe_values_lagrange,
            compute_dRdW, compute_dRdX, compute_d2R);
    } else {
        assemble_volume_terms_explicit (fe_values_volume, current_dofs_indices, current_cell_rhs, fe_values_lagrange);
    }

    for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {

        auto current_face = current_cell->face(iface);

        // Case 1: Face at boundary
        if (current_face->at_boundary() && !current_cell->has_periodic_neighbor(iface) ) {

            fe_values_collection_face_int.reinit(current_cell, iface, i_quad, i_mapp, i_fele);

            // Case 1.1: 1D Periodic boundary condition
            if(current_face->at_boundary() && all_parameters->use_periodic_bc == true && dim == 1) {

                const int neighbor_iface = (iface == 1) ? 0 : 1;

                int cell_index = current_cell->index();
                auto neighbor_cell = dof_handler.begin_active();
                if (cell_index == 0 && iface == 0) {
                // First cell of the domain, neighbor is the last.
                    for (unsigned int i = 0 ; i < triangulation->n_active_cells() - 1; ++i) {
                        ++neighbor_cell;
                    }
                } else if (cell_index == (int) triangulation->n_active_cells() - 1 && iface == 1) {
                // Last cell of the domain, neighbor is the first.
                }

                const int i_fele_n = neighbor_cell->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;
                const unsigned int n_dofs_neigh_cell = fe_collection[i_fele_n].n_dofs_per_cell();
                neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                fe_values_collection_face_ext.reinit(neighbor_cell, neighbor_iface, i_quad_n, i_mapp_n, i_fele_n);

                const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                dealii::Vector<real> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                const real penalty1 = evaluate_penalty_scaling (current_cell, iface, fe_collection);
                const real penalty2 = evaluate_penalty_scaling (neighbor_cell, neighbor_iface, fe_collection);
                const real penalty = 0.5 * (penalty1 + penalty2);

                if ( compute_dRdW || compute_dRdX || compute_d2R ) {
                    auto metric_neighbor_cell = high_order_grid.dof_handler_grid.begin_active();
                    if (cell_index == 0 && iface == 0) {
                    // First cell of the domain, neighbor is the last.
                        for (unsigned int i = 0 ; i < triangulation->n_active_cells() - 1; ++i) {
                            ++neighbor_cell;
                        }
                    } else if (cell_index == (int) triangulation->n_active_cells() - 1 && iface == 1) {
                    // Last cell of the domain, neighbor is the first.
                    }
                    metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);
                    const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad_n]; // or i_quad
                    const dealii::Quadrature<dim> quadrature_int =
                        dealii::QProjector<dim>::project_to_face(used_face_quadrature,iface);
                    const dealii::Quadrature<dim> quadrature_ext =
                        dealii::QProjector<dim>::project_to_face(used_face_quadrature,neighbor_iface);
                    assemble_face_term_derivatives (   iface, neighbor_iface,
                                                fe_values_face_int, fe_values_face_ext,
                                                penalty,
                                                fe_collection[i_fele], fe_collection[i_fele_n],
                                                quadrature_int, quadrature_ext,
                                                current_metric_dofs_indices, neighbor_metric_dofs_indices,
                                                current_dofs_indices, neighbor_dofs_indices,
                                                current_cell_rhs, neighbor_cell_rhs,
                                                compute_dRdW, compute_dRdX, compute_d2R);
                } else {
                    assemble_face_term_explicit (
                                                fe_values_face_int, fe_values_face_ext,
                                                penalty,
                                                current_dofs_indices, neighbor_dofs_indices,
                                                current_cell_rhs, neighbor_cell_rhs);
                }

            } else {
            // Actual boundary term
                const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();

                const real penalty = evaluate_penalty_scaling (current_cell, iface, fe_collection);

                const unsigned int boundary_id = current_face->boundary_id();
                if (compute_dRdW || compute_dRdX || compute_d2R) {
                    const dealii::Quadrature<dim-1> face_quadrature = face_quadrature_collection[i_quad];
                    assemble_boundary_term_derivatives (
                        iface, boundary_id, fe_values_face_int, penalty,
                        current_fe_ref, face_quadrature,
                        current_metric_dofs_indices, current_dofs_indices, current_cell_rhs,
                        compute_dRdW, compute_dRdX, compute_d2R);
    
                } else {
                    assemble_boundary_term_explicit (boundary_id, fe_values_face_int, penalty, current_dofs_indices, current_cell_rhs);
                }
            }

        //CASE 1.5: periodic boundary conditions
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

                if ( compute_d2R ) {
                    const auto metric_neighbor_cell = current_metric_cell->periodic_neighbor(iface);
                    metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);
                    const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad_n]; // or i_quad
                    const dealii::Quadrature<dim> quadrature_int =
                        dealii::QProjector<dim>::project_to_face(used_face_quadrature,iface);
                    const dealii::Quadrature<dim> quadrature_ext =
                        dealii::QProjector<dim>::project_to_face(used_face_quadrature,neighbor_iface);
                    assemble_face_term_derivatives (   iface, neighbor_iface,
                                                fe_values_face_int, fe_values_face_ext,
                                                penalty,
                                                fe_collection[i_fele], fe_collection[i_fele_n],
                                                quadrature_int, quadrature_ext,
                                                current_metric_dofs_indices, neighbor_metric_dofs_indices,
                                                current_dofs_indices, neighbor_dofs_indices,
                                                current_cell_rhs, neighbor_cell_rhs,
                                                compute_dRdW, compute_dRdX, compute_d2R);
                } else {
                    assemble_face_term_explicit (
                            fe_values_face_int, fe_values_face_ext,
                            penalty,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);
                }

                // Add local contribution from neighbor cell to global vector
                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                    rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
                }
            } else {
                //do nothing
            }


        // Case 2:
        // Neighbour is finer occurs if the face has children
        // In this case, we loop over the current large face's subfaces and visit multiple neighbors
        } else if (current_cell->face(iface)->has_children()) {
        //} else if ( (current_cell->level() > current_cell->neighbor(iface)->level()) ) {

//            Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
//            Assert (current_cell->neighbor(iface)->has_children(), dealii::ExcInternalError());
//
//            // Obtain cell neighbour
//            const unsigned int neighbor_iface = current_cell->neighbor_face_no(iface);
//
//            for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {
//
//                // Get neighbor on ith subface
//                auto neighbor_cell = current_cell->neighbor_child_on_subface (iface, subface_no);
//                // Since the neighbor cell is finer than the current cell, it should not have more children
//                Assert (!neighbor_cell->has_children(), dealii::ExcInternalError());
//                Assert (neighbor_cell->neighbor(neighbor_iface) == current_cell, dealii::ExcInternalError());
//
//                const int i_fele_n = neighbor_cell->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;
//
//                const unsigned int n_dofs_neigh_cell = fe_collection[i_fele_n].n_dofs_per_cell();
//                dealii::Vector<real> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization
//
//                // Obtain the mapping from local dof indices to global dof indices for neighbor cell
//                neighbor_dofs_indices.resize(n_dofs_neigh_cell);
//                neighbor_cell->get_dof_indices (neighbor_dofs_indices);
//
//                fe_values_collection_subface.reinit (current_cell, iface, subface_no, i_quad, i_mapp, i_fele);
//                const dealii::FESubfaceValues<dim,dim> &fe_values_face_int = fe_values_collection_subface.get_present_fe_values();
//
//                fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_iface, i_quad_n, i_mapp_n, i_fele_n);
//                const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();
//
//                const real penalty1 = evaluate_penalty_scaling (current_cell, iface, fe_collection);
//                const real penalty2 = evaluate_penalty_scaling (neighbor_cell, neighbor_iface, fe_collection);
//                const real penalty = 0.5 * (penalty1 + penalty2);
//
//                if ( compute_dRdW ) {
//                    assemble_face_term_implicit (
//                            fe_values_face_int, fe_values_face_ext,
//                            penalty,
//                            current_dofs_indices, neighbor_dofs_indices,
//                            current_cell_rhs, neighbor_cell_rhs);
//                }
//                if ( compute_dRdX ) {
//                    const auto metric_neighbor_cell = metric_cell->neighbor_child_on_subface (iface, subface_no);
//                    metric_neighbor_cell.get_dof_indices(neighbor_metric_dofs_indices);
//                    const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad_n]; // or i_quad
//                    const dealii::Quadrature<dim> quadrature_int =
//                        dealii::QProjector<dim>::project_to_face(used_face_quadrature,iface);
//                    const dealii::Quadrature<dim> quadrature_ext =
//                        dealii::QProjector<dim>::project_to_face(used_face_quadrature,neighbor_iface);
//                    assemble_face_term_dRdX (   iface, neighbor_iface,
//                                                fe_values_face_int, fe_values_face_ext,
//                                                penalty,
//                                                fe_collection[i_fele], fe_collection[i_fele_n],
//                                                face_quadrature_collection[i_quad_n],
//                                                current_metric_dofs_indices, neighbor_metric_dofs_indices,
//                                                current_dofs_indices, neighbor_dofs_indices,
//                                                current_cell_rhs, neighbor_cell_rhs);
//                }
//                if ( !compute_dRdX && !compute_dRdW ) {
//                    assemble_face_term_explicit (
//                        fe_values_face_int, fe_values_face_ext,
//                        penalty,
//                        current_dofs_indices, neighbor_dofs_indices,
//                        current_cell_rhs, neighbor_cell_rhs);
//                }
//                // Add local contribution from neighbor cell to global vector
//                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
//                    rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
//                }
//            }

        //} else if (current_cell->neighbor(iface)->level() < current_cell->level()) {
        } else if (current_cell->neighbor(iface)->face(current_cell->neighbor_face_no(iface))->has_children()) {
        //} else if (current_cell->neighbor(iface)->level() < current_cell->level()) {
            // Case 4: Neighbor is coarser
            // Do nothing.
            // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces

            Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
            Assert (!(current_cell->neighbor(iface)->has_children()), dealii::ExcInternalError());

            // Obtain cell neighbour
            const auto neighbor_cell = current_cell->neighbor(iface);
            const unsigned int neighbor_iface = current_cell->neighbor_face_no(iface);

            // Find corresponding subface
            unsigned int i_subface = 0;
            unsigned int n_subface = dealii::GeometryInfo<dim>::n_subfaces(neighbor_cell->subface_case(neighbor_iface));

            for (; i_subface < n_subface; ++i_subface) {
                if (neighbor_cell->neighbor_child_on_subface (neighbor_iface, i_subface) == current_cell) {
                    break;
                }
            }
            Assert(i_subface != n_subface, dealii::ExcInternalError());

            const int i_fele_n = neighbor_cell->active_fe_index(), i_quad_n = i_fele_n, i_mapp_n = 0;

            const unsigned int n_dofs_neigh_cell = fe_collection[i_fele_n].n_dofs_per_cell();
            dealii::Vector<real> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

            // Obtain the mapping from local dof indices to global dof indices for neighbor cell
            neighbor_dofs_indices.resize(n_dofs_neigh_cell);
            neighbor_cell->get_dof_indices (neighbor_dofs_indices);

            fe_values_collection_face_int.reinit (current_cell, iface, i_quad, i_mapp, i_fele);
            const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();

            fe_values_collection_subface.reinit (neighbor_cell, neighbor_iface, i_subface, i_quad_n, i_mapp_n, i_fele_n);
            const dealii::FESubfaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_subface.get_present_fe_values();

            const real penalty1 = evaluate_penalty_scaling (current_cell, iface, fe_collection);
            const real penalty2 = evaluate_penalty_scaling (neighbor_cell, neighbor_iface, fe_collection);
            const real penalty = 0.5 * (penalty1 + penalty2);

            if ( compute_dRdW || compute_dRdX || compute_d2R ) {
                const auto metric_neighbor_cell = current_metric_cell->neighbor(iface);
                metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);

                const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad_n]; // or i_quad
                const dealii::Quadrature<dim> quadrature_int =
                    dealii::QProjector<dim>::project_to_face(used_face_quadrature,iface);
                const dealii::Quadrature<dim> quadrature_ext =
                    dealii::QProjector<dim>::project_to_subface(used_face_quadrature,neighbor_iface,i_subface, dealii::RefinementCase<dim-1>::isotropic_refinement);
                assemble_face_term_derivatives (   iface, neighbor_iface,
                                            fe_values_face_int, fe_values_face_ext,
                                            penalty,
                                            fe_collection[i_fele], fe_collection[i_fele_n],
                                            quadrature_int, quadrature_ext,
                                            current_metric_dofs_indices, neighbor_metric_dofs_indices,
                                            current_dofs_indices, neighbor_dofs_indices,
                                            current_cell_rhs, neighbor_cell_rhs,
                                            compute_dRdW, compute_dRdX, compute_d2R);
            } else {
                assemble_face_term_explicit (
                    fe_values_face_int, fe_values_face_ext,
                    penalty,
                    current_dofs_indices, neighbor_dofs_indices,
                    current_cell_rhs, neighbor_cell_rhs);
            }
            // Add local contribution from neighbor cell to global vector
            for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
            }
        // Case 3:
        // Neighbor cell is NOT coarser
        // Therefore, they have the same coarseness, and we need to choose one of them to do the work
        //} else if ( !(current_cell->neighbor_is_coarser(iface)) && current_cell_should_do_the_work(current_cell, current_cell->neighbor(iface)) ) {
        } else if ( current_cell_should_do_the_work(current_cell, current_cell->neighbor(iface)) ) {
        //} else if ( (current_cell->level() == current_cell->neighbor(iface)->level()) && current_cell_should_do_the_work(current_cell, current_cell->neighbor(iface)) ) {
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

            if ( compute_dRdW || compute_dRdX || compute_d2R ) {
                const auto metric_neighbor_cell = current_metric_cell->neighbor_or_periodic_neighbor(iface);
                metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);
                const dealii::Quadrature<dim-1> &used_face_quadrature = face_quadrature_collection[i_quad_n]; // or i_quad
                const dealii::Quadrature<dim> quadrature_int =
                    dealii::QProjector<dim>::project_to_face(used_face_quadrature,iface);
                const dealii::Quadrature<dim> quadrature_ext =
                    dealii::QProjector<dim>::project_to_face(used_face_quadrature,neighbor_iface);
                assemble_face_term_derivatives (   iface, neighbor_iface,
                                            fe_values_face_int, fe_values_face_ext,
                                            penalty,
                                            fe_collection[i_fele], fe_collection[i_fele_n],
                                            quadrature_int, quadrature_ext,
                                            current_metric_dofs_indices, neighbor_metric_dofs_indices,
                                            current_dofs_indices, neighbor_dofs_indices,
                                            current_cell_rhs, neighbor_cell_rhs,
                                            compute_dRdW, compute_dRdX, compute_d2R);
            } else {
                assemble_face_term_explicit (
                        fe_values_face_int, fe_values_face_ext,
                        penalty,
                        current_dofs_indices, neighbor_dofs_indices,
                        current_cell_rhs, neighbor_cell_rhs);
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

template <int dim, typename real>
void DGBase<dim,real>::set_dual(const dealii::LinearAlgebra::distributed::Vector<real> &dual_input)
{
    dual = dual_input;
}


template <int dim, typename real>
void DGBase<dim,real>::assemble_residual (const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    dealii::deal_II_exceptions::disable_abort_on_exception(); // Allows us to catch negative Jacobians.
    Assert( !(compute_dRdW && compute_dRdX)
        &&  !(compute_dRdW && compute_d2R)
        &&  !(compute_dRdX && compute_d2R)
            , dealii::ExcMessage("Can only do one at a time compute_dRdW or compute_dRdX or compute_d2R"));
    if (compute_d2R) {
        //dual.reinit(locally_owned_dofs,mpi_communicator);

        {
            dealii::SparsityPattern sparsity_pattern_d2RdWdX = get_d2RdWdX_sparsity_pattern ();
            const dealii::IndexSet &row_parallel_partitioning_d2RdWdX = locally_owned_dofs;
            const dealii::IndexSet &col_parallel_partitioning_d2RdWdX = high_order_grid.locally_owned_dofs_grid;
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
            const dealii::IndexSet &row_parallel_partitioning_d2RdXdX = high_order_grid.locally_owned_dofs_grid;
            const dealii::IndexSet &col_parallel_partitioning_d2RdXdX = high_order_grid.locally_owned_dofs_grid;
            d2RdXdX.reinit(row_parallel_partitioning_d2RdXdX, col_parallel_partitioning_d2RdXdX, sparsity_pattern_d2RdXdX, mpi_communicator);
        }

        AssertDimension(dual.size(), right_hand_side.size());
    }

    right_hand_side = 0;

    pcout << "Assembling DG residual...";
    if (compute_dRdW) {
        pcout << " with dRdW...";
        system_matrix = 0;
    }
    if (compute_dRdX) {
        pcout << " with dRdX...";
        dRdXv = 0;
    }
    if (compute_d2R) {
        pcout << " with d2RdWdW, d2RdWdX, d2RdXdX...";
        d2RdWdW = 0;
        d2RdWdX = 0;
        d2RdXdX = 0;
    }
    pcout << std::endl;

    //const dealii::MappingManifold<dim,dim> mapping;
    //const dealii::MappingQ<dim,dim> mapping(10);//;max_degree+1);
    //const dealii::MappingQ<dim,dim> mapping(high_order_grid.max_degree);
    //const dealii::MappingQGeneric<dim,dim> mapping(high_order_grid.max_degree);
    const auto mapping = (*(high_order_grid.mapping_fe_field));

    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, fe_collection, face_quadrature_collection, this->face_update_flags); ///< FEValues of interior face.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, fe_collection, face_quadrature_collection, this->neighbor_face_update_flags); ///< FEValues of exterior face.
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface (mapping_collection, fe_collection, face_quadrature_collection, this->face_update_flags); ///< FEValues of subface.

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_lagrange (mapping_collection, fe_collection_lagrange, volume_quadrature_collection, this->volume_update_flags);

    solution.update_ghost_values();

    int assembly_error = 0;
    try {
        auto current_metric_cell = high_order_grid.dof_handler_grid.begin_active();
        for (auto current_cell = dof_handler.begin_active(); current_cell != dof_handler.end(); ++current_cell, ++current_metric_cell) {
            if (!current_cell->is_locally_owned()) continue;

            // Add right-hand side contributions this cell can compute
            assemble_cell_residual (
                current_cell, 
                current_metric_cell, 
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
    if ( compute_dRdW ) system_matrix.compress(dealii::VectorOperation::add);
    if ( compute_dRdX ) dRdXv.compress(dealii::VectorOperation::add);
    if ( compute_d2R ) {
        d2RdWdW.compress(dealii::VectorOperation::add);
        d2RdXdX.compress(dealii::VectorOperation::add);
        d2RdWdX.compress(dealii::VectorOperation::add);
    }
    //if ( compute_dRdW ) system_matrix.compress(dealii::VectorOperation::insert);
    //system_matrix.print(std::cout);

} // end of assemble_system_explicit ()


template <int dim, typename real>
double DGBase<dim,real>::get_residual_l2norm () const
{
    return right_hand_side.l2_norm();
}
template <int dim, typename real>
unsigned int DGBase<dim,real>::n_dofs () const
{
    return dof_handler.n_dofs();
}

template <int dim, typename real>
void DGBase<dim,real>::output_results_vtk (const unsigned int cycle)// const
{

    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> data_out;
    data_out.attach_dof_handler (dof_handler);

    //std::vector<std::string> solution_names;
    //for(int s=0;s<nstate;++s) {
    //    std::string varname = "u" + dealii::Utilities::int_to_string(s,1);
    //    solution_names.push_back(varname);
    //}
    //std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(nstate, dealii::DataComponentInterpretation::component_is_scalar);
    //data_out.add_data_vector (solution, solution_names, dealii::DataOut<dim>::type_dof_data, data_component_interpretation);

    dealii::Vector<float> subdomain(triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) {
        subdomain(i) = triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    if (all_parameters->add_artificial_dissipation) {
        data_out.add_data_vector(artificial_dissipation_coeffs, "artificial_dissipation_coeffs", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    }

    data_out.add_data_vector(max_dt_cell, "max_dt_cell", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);


    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = Postprocess::PostprocessorFactory<dim>::create_Postprocessor(all_parameters);
    data_out.add_data_vector (solution, *post_processor);

    // Output the polynomial degree in each cell
    std::vector<unsigned int> active_fe_indices;
    dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    dealii::Vector<double> cell_poly_degree = active_fe_indices_dealiivector;

//    int index = 0;
//    for (auto current_cell_poly = cell_poly_degree.begin(); current_cell_poly != cell_poly_degree.end(); ++current_cell_poly) {
//        current_cell_poly[index] = fe_collection[active_fe_indices_dealiivector[index]].tensor_degree();
//        index++;
//    }
//    //using DVTenum = dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType;
//    data_out.add_data_vector (cell_poly_degree, "PolynomialDegree", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    data_out.add_data_vector (active_fe_indices_dealiivector, "PolynomialDegree", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);


    //assemble_residual (false);
    std::vector<std::string> residual_names;
    for(int s=0;s<nstate;++s) {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }
    //std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(nstate, dealii::DataComponentInterpretation::component_is_scalar);
    //data_out.add_data_vector (right_hand_side, residual_names, dealii::DataOut<dim, dealii::hp::DoFHandler<dim>>::type_dof_data, data_component_interpretation);
    data_out.add_data_vector (right_hand_side, residual_names, dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);


    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    // //data_out.build_patches (mapping_collection[mapping_collection.size()-1]);
    // data_out.build_patches(*(high_order_grid.mapping_fe_field), max_degree, dealii::DataOut<dim, dealii::hp::DoFHandler<dim>>::CurvedCellRegion::no_curved_cells);
    // //data_out.build_patches(*(high_order_grid.mapping_fe_field), fe_collection.size(), dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells);

    typename dealii::DataOut<dim,dealii::hp::DoFHandler<dim>>::CurvedCellRegion curved = dealii::DataOut<dim,dealii::hp::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells;
    //typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::curved_boundary;
    //typename dealii::DataOut<dim>::CurvedCellRegion curved = dealii::DataOut<dim>::CurvedCellRegion::no_curved_cells;

    const dealii::Mapping<dim> &mapping = (*(high_order_grid.mapping_fe_field));
    //const int n_subdivisions = max_degree;;//+30; // if write_higher_order_cells, n_subdivisions represents the order of the cell
    const int n_subdivisions = 0;//+30; // if write_higher_order_cells, n_subdivisions represents the order of the cell
    data_out.build_patches(mapping, n_subdivisions, curved);
    const bool write_higher_order_cells = (dim>1 && max_degree > 1) ? true : false; 
    dealii::DataOutBase::VtkFlags vtkflags(0.0,cycle,true,dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::best_compression,write_higher_order_cells);
    data_out.set_flags(vtkflags);


    std::string filename = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D_maxpoly"+dealii::Utilities::int_to_string(max_degree, 2)+"-";
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
	std::cout << "Writing out file: " << filename << std::endl;

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

template <int dim, typename real>
void DGBase<dim,real>::allocate_system ()
{
    pcout << "Allocating DG system and initializing FEValues" << std::endl;
    // This function allocates all the necessary memory to the 
    // system matrices and vectors.

    dof_handler.distribute_dofs(fe_collection);
    dealii::DoFRenumbering::Cuthill_McKee(dof_handler);
	

    //dealii::MappingFEField<dim,dim,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> mapping = high_order_grid.get_MappingFEField();
    //dealii::MappingFEField<dim,dim,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> mapping = *(high_order_grid.mapping_fe_field);

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

    artificial_dissipation_coeffs.reinit(triangulation->n_active_cells());
    max_dt_cell.reinit(triangulation->n_active_cells());

    solution.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
    //right_hand_side.reinit(locally_owned_dofs, mpi_communicator);
    right_hand_side.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
    dual.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);

    // System matrix allocation
    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(locally_owned_dofs, sparsity_pattern, mpi_communicator);

    // dRdXv matrix allocation
    dealii::SparsityPattern dRdXv_sparsity_pattern = get_dRdX_sparsity_pattern ();
    const dealii::IndexSet &row_parallel_partitioning = locally_owned_dofs;
    const dealii::IndexSet &col_parallel_partitioning = high_order_grid.locally_owned_dofs_grid;
    //const dealii::IndexSet &col_parallel_partitioning = high_order_grid.locally_relevant_dofs_grid;
    dRdXv.reinit(row_parallel_partitioning, col_parallel_partitioning, dRdXv_sparsity_pattern, MPI_COMM_WORLD);
}

template <int dim, typename real>
void DGBase<dim,real>::evaluate_mass_matrices (bool do_inverse_mass_matrix)
{
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
    } else {
        global_mass_matrix.reinit(locally_owned_dofs, mass_sparsity_pattern);
    }

    //dealii::TrilinosWrappers::SparseMatrix 
    //    matrix_with_correct_size(locally_owned_dofs,
    //            mpi_communicator,
    //            dof_handler.get_fe_collection().max_dofs_per_cell());
    //pcout << "Before compress" << std::endl;
    //matrix_with_correct_size.compress(dealii::VectorOperation::unknown);
    //if (do_inverse_mass_matrix == true) {
    //    global_inverse_mass_matrix.reinit(matrix_with_correct_size);
    //} else {
    //    global_mass_matrix.reinit(matrix_with_correct_size);
    //}
    //pcout << "AFter reinit" << std::endl;

    //const dealii::MappingManifold<dim,dim> mapping;
    //const dealii::MappingQ<dim,dim> mapping(max_degree+1);
    //const dealii::MappingQ<dim,dim> mapping(high_order_grid.max_degree);
    // std::cout << "Grid degree: " << high_order_grid.max_degree << std::endl;
    //const dealii::MappingQGeneric<dim,dim> mapping(high_order_grid.max_degree);
    const auto mapping = (*(high_order_grid.mapping_fe_field));

    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim> fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
    for (auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++cell) {

        if (!cell->is_locally_owned()) continue;

        const unsigned int mapping_index = 0;
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const unsigned int quad_index = fe_index_curr_cell;

        // Current reference element related to this physical cell
        const dealii::FESystem<dim,dim> &current_fe_ref = fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_cell = current_fe_ref.n_dofs_per_cell();
        const unsigned int n_quad_pts = volume_quadrature_collection[fe_index_curr_cell].size();

        dealii::FullMatrix<real> local_mass_matrix(n_dofs_cell);

        fe_values_collection_volume.reinit (cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_volume.get_fe().system_to_component_index(itest).first;
            for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {
                const unsigned int istate_trial = fe_values_volume.get_fe().system_to_component_index(itrial).first;
                real value = 0.0;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    value +=
                        fe_values_volume.shape_value_component(itest,iquad,istate_test)
                        * fe_values_volume.shape_value_component(itrial,iquad,istate_trial)
                        * fe_values_volume.JxW(iquad);
                }
                local_mass_matrix[itrial][itest] = 0.0;
                local_mass_matrix[itest][itrial] = 0.0;
                if(istate_test==istate_trial) { 
                    local_mass_matrix[itrial][itest] = value;
                    local_mass_matrix[itest][itrial] = value;
                }
            }
        }

        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);
        if (do_inverse_mass_matrix == true) {
            dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
            local_inverse_mass_matrix.invert(local_mass_matrix);
            global_inverse_mass_matrix.set (dofs_indices, local_inverse_mass_matrix);
        } else {
            global_mass_matrix.set (dofs_indices, local_mass_matrix);
        }
    }

    if (do_inverse_mass_matrix == true) {
        global_inverse_mass_matrix.compress(dealii::VectorOperation::insert);
    } else {
        global_mass_matrix.compress(dealii::VectorOperation::insert);
        time_scaled_global_mass_matrix.reinit(global_mass_matrix);
    }

    return;
}
template<int dim, typename real>
void DGBase<dim,real>::add_mass_matrices(const real scale)
{
    system_matrix.add(scale, global_mass_matrix);
}
template<int dim, typename real>
void DGBase<dim,real>::add_time_scaled_mass_matrices()
{
    system_matrix.add(1.0, time_scaled_global_mass_matrix);
}
template<int dim, typename real>
void DGBase<dim,real>::time_scaled_mass_matrices(const real dt_scale)
{
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
                    const double value = global_mass_matrix.el(dofs_indices[itest],dofs_indices[itrial]);
                    const double new_val = value / (dt_scale * max_dt);
                    time_scaled_global_mass_matrix.set(dofs_indices[itest],dofs_indices[itrial],new_val);
                    time_scaled_global_mass_matrix.set(dofs_indices[itrial],dofs_indices[itest],new_val);
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
        mass.gauss_jordan();

        for(unsigned int row=0; row<n_dofs_out; ++row) {
            const unsigned int idof_vector = fe_output.component_to_system_index(istate,row);
            function_coeff_out[idof_vector] = 0.0;
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                function_coeff_out[idof_vector] += mass[row][col] * rhs[col];
            }
        }
    }

    return function_coeff_out;
    //
    //int mpi_rank;
    //(void) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    //if (mpi_rank==0) mass.print(std::cout);

}


template <int dim, typename real>
template <typename real2>
real2 DGBase<dim,real>::discontinuity_sensor(
    const double diameter,
    const std::vector< real2 > &soln_coeff_high,
    const dealii::FiniteElement<dim,dim> &fe_high)
{
    //return 0;
    //return 0.01;
    const unsigned int degree = fe_high.tensor_degree();
    const unsigned int nstate = fe_high.components;
    if (degree == 0) return 0;
    const unsigned int lower_degree = degree-1;//degree-1;
    const dealii::FE_DGQLegendre<dim> fe_dgq_lower(lower_degree);
    const dealii::FESystem<dim,dim> fe_lower(fe_dgq_lower, nstate);

    const unsigned int n_dofs_high = fe_high.dofs_per_cell;
    const unsigned int n_dofs_lower = fe_lower.dofs_per_cell;
    //dealii::FullMatrix<double> projection_matrix(n_dofs_lower,n_dofs_high);
    //dealii::FETools::get_projection_matrix(fe_high, fe_lower, projection_matrix);

    //std::vector< real2 > soln_coeff_lower(n_dofs_lower);
    //for(unsigned int row=0; row<n_dofs_lower; ++row) {
    //    soln_coeff_lower[row] = 0.0;
    //    for(unsigned int col=0; col<n_dofs_high; ++col) {
    //        soln_coeff_lower[row] += projection_matrix[row][col] * soln_coeff_high[col];
    //    }
    //}

    const dealii::QGauss<dim> quadrature(degree+5);
    const unsigned int n_quad_pts = quadrature.size();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = quadrature.get_points();

    std::vector< real2 > soln_coeff_lower = project_function<dim,real2>( soln_coeff_high, fe_high, fe_lower, quadrature);

    real2 error = 0.0;
    real2 soln_norm = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        real2 soln_high = 0.0;
        real2 soln_lower = 0.0;
        for (unsigned int idof=0; idof<n_dofs_high; ++idof) {
              soln_high += soln_coeff_high[idof] * fe_high.shape_value(idof,unit_quad_pts[iquad]);
        }
        for (unsigned int idof=0; idof<n_dofs_lower; ++idof) {
              soln_lower += soln_coeff_lower[idof] * fe_lower.shape_value(idof,unit_quad_pts[iquad]);
        }
        // Need JxW not just W
        // However, this happens at the cell faces, and therefore can't query the 
        // the volume Jacobians
        //std::cout << "soln_high" << std::endl;
        //std::cout << soln_high << std::endl;
        //std::cout << "soln_lower" << std::endl;
        //std::cout << soln_lower << std::endl;
        //error += std::pow(soln_high - soln_lower, 2) * quadrature.weight(iquad);
        //soln_norm += std::pow(soln_high, 2) * quadrature.weight(iquad);
        error += (soln_high - soln_lower) * (soln_high - soln_lower) * quadrature.weight(iquad);
        soln_norm += soln_high * soln_high * quadrature.weight(iquad);
    }

    if (error < 1e-12) return 0.0;
    if (soln_norm < 1e-12) return 0.0;

    real2 S_e, s_e;
    S_e = error / soln_norm;
    s_e = std::log10(S_e);

    //double S_0, s_0;
    //S_0 = 1.0 / std::pow(degree,4);
    //s_0 = std::log10(S_0);
    //const double kappa = 0.1 * std::abs(s_0);

    const double skappa = -2.3;
    const double s_0 = skappa-4.25*std::log10(degree);
    const double kappa = 10.2; // 1.2
    const double mu_scale = 1.0;

    const double low = s_0 - kappa;
    const double upp = s_0 + kappa;
    const real2 eps_0 = mu_scale * diameter / degree;

    if ( s_e < low) return 0.0;
    //std::cout << "error: " << error << " norm: " << soln_norm << " s_e " << s_e << " low: " << low << " upp: " << upp << std::endl;
    if ( s_e > upp) return eps_0;

    const double PI = 4*atan(1);
    real2 eps = 1.0 + std::sin(PI * (s_e - s_0) * 0.5 / kappa);
    eps *= eps_0 * 0.5;

    return eps;

}


template class DGBase <PHILIP_DIM, double>;
template class DGFactory <PHILIP_DIM, double>;

template double
DGBase<PHILIP_DIM,double>::discontinuity_sensor(const double diameter, const std::vector< double > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high);
template Sacado::Fad::DFad<double>
DGBase<PHILIP_DIM,double>::discontinuity_sensor(const double diameter, const std::vector< Sacado::Fad::DFad<double> > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high);
template Sacado::Fad::DFad<Sacado::Fad::DFad<double>>
DGBase<PHILIP_DIM,double>::discontinuity_sensor(const double diameter, const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<double>> > &soln_coeff_high, const dealii::FiniteElement<PHILIP_DIM,PHILIP_DIM> &fe_high);

} // PHiLiP namespace
