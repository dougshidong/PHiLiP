#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

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

//#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/fe/mapping_q.h> // Might need mapping_q
#include <deal.II/fe/mapping_manifold.h> 
#include <deal.II/fe/mapping_fe_field.h> 

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>


#include "dg.h"
#include "post_processor/physics_post_processor.h"
#include <string>

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
    Triangulation *const triangulation_input)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

    PDE_enum pde_type = parameters_input->pde_type;
    if (parameters_input->use_weak_form) {
        if (pde_type == PDE_enum::advection) {
            return std::make_shared< DGWeak<dim,1,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::advection_vector) {
            return std::make_shared< DGWeak<dim,2,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::diffusion) {
            return std::make_shared< DGWeak<dim,1,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return std::make_shared< DGWeak<dim,1,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_inviscid) {
            return std::make_shared< DGWeak<dim,dim,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::euler) {
            return std::make_shared< DGWeak<dim,dim+2,real> >(parameters_input, degree, triangulation_input);
        }
    } else {
        if (pde_type == PDE_enum::advection) {
            return std::make_shared< DGStrong<dim,1,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::advection_vector) {
            return std::make_shared< DGStrong<dim,2,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::diffusion) {
            return std::make_shared< DGStrong<dim,1,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return std::make_shared< DGStrong<dim,1,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::burgers_inviscid) {
            return std::make_shared< DGStrong<dim,dim,real> >(parameters_input, degree, triangulation_input);
        } else if (pde_type == PDE_enum::euler) {
            return std::make_shared< DGStrong<dim,dim+2,real> >(parameters_input, degree, triangulation_input);
        }
    }
    std::cout << "Can't create DGBase in create_discontinuous_galerkin(). Invalid PDE type: " << pde_type << std::endl;
    return nullptr;
}

// DGBase ***************************************************************************
template <int dim, typename real>
DGBase<dim,real>::DGBase(
    const int nstate_input,
    const Parameters::AllParameters *const parameters_input,
    const unsigned int max_degree_input,
    Triangulation *const triangulation_input)
    : DGBase<dim,real>(nstate_input, parameters_input, max_degree_input, triangulation_input, this->create_collection_tuple(max_degree_input, nstate_input, parameters_input))
{ }

template <int dim, typename real>
DGBase<dim,real>::DGBase( 
    const int nstate_input,
    const Parameters::AllParameters *const parameters_input,
    const unsigned int max_degree_input,
    Triangulation *const triangulation_input,
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

//for flux points
    , fe_collection_flux(std::get<5>(collection_tuple))
    , volume_quadrature_collection_flux(std::get<6>(collection_tuple))
    , face_quadrature_collection_flux(std::get<7>(collection_tuple))
    , oned_quadrature_collection_flux(std::get<8>(collection_tuple))
//for jac at soln points
    , volume_quadrature_collection_jac_sol(std::get<9>(collection_tuple))

    , dof_handler(*triangulation)
    , high_order_grid(all_parameters, max_degree_input+1, triangulation)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{ 

    dof_handler.initialize(*triangulation, fe_collection);

    set_all_cells_fe_degree (fe_collection.size()-1); // Always sets it to the maximum degree

}

template <int dim, typename real> 
std::tuple<
        //dealii::hp::MappingCollection<dim>, // Mapping
        dealii::hp::FECollection<dim>, // Solution FE
        dealii::hp::QCollection<dim>,  // Volume quadrature
        dealii::hp::QCollection<dim-1>, // Face quadrature
        dealii::hp::QCollection<1>, // 1D quadrature for strong form
        dealii::hp::FECollection<dim>,   // Lagrange polynomials for strong form
        dealii::hp::FECollection<dim>, //Flux FE
        dealii::hp::QCollection<dim>, //Flux Volume Quadrature
        dealii::hp::QCollection<dim-1>, //Flux Face Quadrature
        dealii::hp::QCollection<1>, //Flux 1D Quadrature For Strong Form
        dealii::hp::QCollection<dim> > //Jac soln points quadrature
DGBase<dim,real>::create_collection_tuple(const unsigned int max_degree, const int nstate, const Parameters::AllParameters *const parameters_input) const
{
    //dealii::hp::MappingCollection<dim> mapping_coll;
    dealii::hp::FECollection<dim>      fe_coll;
    dealii::hp::QCollection<dim>       volume_quad_coll;
    dealii::hp::QCollection<dim-1>     face_quad_coll;
    dealii::hp::QCollection<1>         oned_quad_coll;

    dealii::hp::FECollection<dim>      fe_coll_lagr;

    //for flux points
    dealii::hp::FECollection<dim>      fe_coll_flux;
    dealii::hp::QCollection<dim>       volume_quad_coll_flux;
    dealii::hp::QCollection<dim-1>     face_quad_coll_flux;
    dealii::hp::QCollection<1>         oned_quad_coll_flux;
//for Jac soln points
    dealii::hp::QCollection<dim>       volume_quad_coll_jac_sol;

    // for p=0, we use a p=1 FE for collocation, since there's no p=0 quadrature for Gauss Lobatto
    if (parameters_input->use_collocated_nodes==true)
    {
    	int degree = 1;
		//const dealii::MappingQ<dim,dim> mapping(degree, true);
		//const dealii::MappingQ<dim,dim> mapping(degree+1, true);
		//const dealii::MappingManifold<dim,dim> mapping;
		//mapping_coll.push_back(mapping);

		const dealii::FE_DGQ<dim> fe_dg(degree);
                //const dealii::FE_DGQLegendre<dim> fe_dg(degree);
		const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
		fe_coll.push_back (fe_system);

                //proj flux
                const dealii::FE_DGQ<dim> fe_dg_flux(degree);
                const dealii::FESystem<dim,dim> fe_system_flux(fe_dg_flux, nstate);
                fe_coll_flux.push_back (fe_system_flux);
                //for flux points
                dealii::Quadrature<1>     oned_quad_flux(degree+1);
                dealii::Quadrature<dim>   volume_quad_flux(degree+1);
                dealii::Quadrature<dim-1> face_quad_flux(degree+1); //removed const
		//end proj flux

		dealii::Quadrature<1>     oned_quad(degree+1);
		dealii::Quadrature<dim>   volume_quad(degree+1);
		dealii::Quadrature<dim-1> face_quad(degree+1); //removed const

		if (parameters_input->use_collocated_nodes)
			{
				dealii::QGaussLobatto<1> oned_quad_Gauss_Lobatto (degree+1);
				dealii::QGaussLobatto<dim> vol_quad_Gauss_Lobatto (degree+1);
				oned_quad = oned_quad_Gauss_Lobatto;
				volume_quad = vol_quad_Gauss_Lobatto;

                                //flux point
                                oned_quad_flux = oned_quad_Gauss_Lobatto;
                                volume_quad_flux = vol_quad_Gauss_Lobatto;
                                //flux point end

				if(dim == 1)
				{
					dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
					face_quad = face_quad_Gauss_Legendre;
                                        //flux point
                                        face_quad_flux = face_quad_Gauss_Legendre;
                                        //end flux point
				}
				else
				{
					dealii::QGaussLobatto<dim-1> face_quad_Gauss_Lobatto (degree+1);
					face_quad = face_quad_Gauss_Lobatto;
                                        //flux point
                                        face_quad_flux = face_quad_Gauss_Lobatto;
                                        //end flux point
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
                                //for flux points
                                dealii::QGauss<1> oned_quad_Gauss_Legendre_flux (degree+1);
                                dealii::QGauss<dim> vol_quad_Gauss_Legendre_flux (degree+1);
                                dealii::QGauss<dim-1> face_quad_Gauss_Legendre_flux (degree+1);
                                oned_quad_flux = oned_quad_Gauss_Legendre_flux;
                                volume_quad_flux = vol_quad_Gauss_Legendre_flux;
                                face_quad_flux = face_quad_Gauss_Legendre_flux;
                                //end flux point
			}
		//


		volume_quad_coll.push_back (volume_quad);
		face_quad_coll.push_back (face_quad);
		oned_quad_coll.push_back (oned_quad);

		dealii::FE_DGQArbitraryNodes<dim,dim> lagrange_poly(oned_quad);
		fe_coll_lagr.push_back (lagrange_poly);

        //for flux points
        volume_quad_coll_flux.push_back (volume_quad_flux);
        face_quad_coll_flux.push_back (face_quad_flux);
        oned_quad_coll_flux.push_back (oned_quad_flux);
    }

    int minimum_degree = (parameters_input->use_collocated_nodes==true) ?  1 :  0;
    for (unsigned int degree=minimum_degree; degree<=max_degree; ++degree) {
        //const dealii::MappingQ<dim,dim> mapping(degree, true);
        //const dealii::MappingQ<dim,dim> mapping(degree+1, true);
        //const dealii::MappingManifold<dim,dim> mapping;
        //mapping_coll.push_back(mapping);

        // Solution FECollection
        const dealii::FE_DGQ<dim> fe_dg(degree);
        //const dealii::FE_DGQLegendre<dim> fe_dg(degree);
        //const dealii::FE_DGP<dim> fe_dg(degree);
        const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
        fe_coll.push_back (fe_system);

        dealii::Quadrature<1>     oned_quad(degree+1);
        dealii::Quadrature<dim>   volume_quad(degree+1);
        dealii::Quadrature<dim-1> face_quad(degree+1); //removed const

//for jac soln points
        dealii::Quadrature<dim>   volume_quad_jac(degree+1);

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
            dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1);
            dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
            oned_quad = oned_quad_Gauss_Legendre;
            volume_quad = vol_quad_Gauss_Legendre;
            face_quad = face_quad_Gauss_Legendre;
            if(parameters_input->use_jac_sol_points == true){
                unsigned int degree_GLL;
                if(degree == 0)
                    degree_GLL = degree + 1;
                else
                    degree_GLL = degree;
                dealii::QGaussLobatto<dim> vol_quad_Gauss_Lobatto (degree_GLL+1);
                volume_quad_jac = vol_quad_Gauss_Lobatto;
            }
            else{
                volume_quad_jac = vol_quad_Gauss_Legendre;
            }
        }

        volume_quad_coll.push_back (volume_quad);
        face_quad_coll.push_back (face_quad);
        oned_quad_coll.push_back (oned_quad);
        //for jac soln points
        volume_quad_coll_jac_sol.push_back(volume_quad_jac);

        dealii::FE_DGQArbitraryNodes<dim,dim> lagrange_poly(oned_quad);
        fe_coll_lagr.push_back (lagrange_poly);
    }

    minimum_degree = 0;
    unsigned int maximum_degree;
    if(parameters_input->use_projected_flux == true){
        minimum_degree++;
        maximum_degree = max_degree + 1;
    }
    else
        maximum_degree = max_degree;

   // for (unsigned int degree=(minimum_degree+1); degree<=(max_degree +1); ++degree) {
   // for (unsigned int degree=(minimum_degree); degree<=(max_degree ); ++degree) {
    if(parameters_input->use_collocated_nodes==true){
        minimum_degree = minimum_degree + 1;
    }
    for (unsigned int degree=(minimum_degree); degree<=(maximum_degree); ++degree) {
        //const dealii::MappingQ<dim,dim> mapping(degree, true);
        //const dealii::MappingQ<dim,dim> mapping(degree+1, true);
       // const dealii::MappingManifold<dim,dim> mapping;
       // mapping_coll_flux.push_back(mapping);

        const dealii::FE_DGQ<dim> fe_dg_flux(degree);
        //const dealii::FE_DGQLegendre<dim> fe_dg_flux(degree);
        //const dealii::FE_DGP<dim> fe_dg_flux(degree);
        const dealii::FESystem<dim,dim> fe_system_flux(fe_dg_flux, nstate);
        fe_coll_flux.push_back (fe_system_flux);

        //

//for flux points
        dealii::Quadrature<1>     oned_quad_flux(degree+1);
        dealii::Quadrature<dim>   volume_quad_flux(degree+1);
        dealii::Quadrature<dim-1> face_quad_flux(degree+1); //removed const


        if (parameters_input->use_collocated_nodes)
            {
                dealii::QGaussLobatto<1> oned_quad_Gauss_Lobatto (degree+1);
                dealii::QGaussLobatto<dim> vol_quad_Gauss_Lobatto (degree+1);
                oned_quad_flux = oned_quad_Gauss_Lobatto;
                volume_quad_flux = vol_quad_Gauss_Lobatto;

                if(dim == 1)
                {
                    dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
                    face_quad_flux = face_quad_Gauss_Legendre;
                }
                else
                {
                    dealii::QGaussLobatto<dim-1> face_quad_Gauss_Lobatto (degree+1);
                    face_quad_flux = face_quad_Gauss_Lobatto;
                }


            }
            else
            {
                //for flux points
                dealii::QGauss<1> oned_quad_Gauss_Legendre_flux (degree+1);
                dealii::QGauss<dim> vol_quad_Gauss_Legendre_flux (degree+1);
                dealii::QGauss<dim-1> face_quad_Gauss_Legendre_flux (degree+1);
                oned_quad_flux = oned_quad_Gauss_Legendre_flux;
                volume_quad_flux = vol_quad_Gauss_Legendre_flux;
                face_quad_flux = face_quad_Gauss_Legendre_flux;

            }
        //


        //for flux points
        volume_quad_coll_flux.push_back (volume_quad_flux);
        face_quad_coll_flux.push_back (face_quad_flux);
        oned_quad_coll_flux.push_back (oned_quad_flux);

      //  dealii::FE_DGQArbitraryNodes<dim,dim> lagrange_poly(oned_quad_flux);
      //  fe_coll_lagr_flux.push_back (lagrange_poly);

    }
    return std::make_tuple(fe_coll, volume_quad_coll, face_quad_coll, oned_quad_coll, fe_coll_lagr, fe_coll_flux, volume_quad_coll_flux, face_quad_coll_flux, oned_quad_coll_flux, volume_quad_coll_jac_sol);
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
void DGBase<dim,real>::assemble_residual (const bool compute_dRdW)
{
    right_hand_side = 0;

    if (compute_dRdW) system_matrix = 0;

    // For now assume same polynomial degree across domain
    const unsigned int max_dofs_per_cell = dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);

    //dealii::hp::MappingCollection<dim> mapping_collection(*(high_order_grid.mapping_fe_field));
    //const dealii::MappingManifold<dim,dim> mapping;
    //const dealii::MappingQ<dim,dim> mapping(max_degree+1);
    const auto mapping = (*(high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, fe_collection, face_quadrature_collection, this->face_update_flags); ///< FEValues of interior face.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, fe_collection, face_quadrature_collection, this->neighbor_face_update_flags); ///< FEValues of exterior face.
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface (mapping_collection, fe_collection, face_quadrature_collection, this->face_update_flags); ///< FEValues of subface.

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_lagrange (mapping_collection, fe_collection_lagrange, volume_quadrature_collection, this->volume_update_flags);

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_flux (mapping_collection, DGBase<dim,real>::fe_collection_flux, DGBase<dim,real>::volume_quadrature_collection_flux, this->volume_update_flags); ///< FEValues of volume.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int_flux (mapping_collection, DGBase<dim,real>::fe_collection_flux, DGBase<dim,real>::face_quadrature_collection_flux, this->face_update_flags); ///< FEValues of interior face.
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext_flux (mapping_collection, DGBase<dim,real>::fe_collection_flux, DGBase<dim,real>::face_quadrature_collection_flux, this->neighbor_face_update_flags); ///< FEValues of exterior face.
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface_flux (mapping_collection, DGBase<dim,real>::fe_collection_flux, DGBase<dim,real>::face_quadrature_collection_flux, this->face_update_flags); ///< FEValues of subface.

//solution basis functions evaluated at flux volume nodes
    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_soln_flux (mapping_collection, DGBase<dim,real>::fe_collection, DGBase<dim,real>::volume_quadrature_collection_flux, this->volume_update_flags); ///< FEValues of volume.

    unsigned int n_cell_visited = 0;
    unsigned int n_face_visited = 0;

    solution.update_ghost_values();
    for (auto current_cell = dof_handler.begin_active(); current_cell != dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
        n_cell_visited++;

        // Current reference element related to this physical cell
        const unsigned int mapping_index = 0;
        const unsigned int fe_index_curr_cell = current_cell->active_fe_index();
        const unsigned int quad_index = fe_index_curr_cell;
        const dealii::FESystem<dim,dim> &current_fe_ref = fe_collection[fe_index_curr_cell];
        const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Local vector contribution from each cell
        dealii::Vector<double> current_cell_rhs (n_dofs_curr_cell); // Defaults to 0.0 initialization

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        current_cell->get_dof_indices (current_dofs_indices);

        // fe_values_collection.reinit(current_cell, quad_collection_index, mapping_collection_index, fe_collection_index)
        fe_values_collection_volume.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();


        dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (current_cell);
        //if (!(all_parameters->use_weak_form)) fe_values_collection_volume_lagrange.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
        fe_values_collection_volume_lagrange.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);

        fe_values_collection_volume_flux.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);

        const dealii::FEValues<dim,dim> &fe_values_volume_flux = fe_values_collection_volume_flux.get_present_fe_values();

        fe_values_collection_volume_soln_flux.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);

        const dealii::FEValues<dim,dim> &fe_values_volume_soln_flux = fe_values_collection_volume_soln_flux.get_present_fe_values();

        const dealii::FEValues<dim,dim> &fe_values_lagrange = fe_values_collection_volume_lagrange.get_present_fe_values();
        if ( compute_dRdW ) {
            assemble_volume_terms_implicit (fe_values_volume, fe_values_volume_flux, fe_values_volume_soln_flux, current_dofs_indices, current_cell_rhs, fe_values_lagrange);
        } else {
            assemble_volume_terms_explicit (fe_values_volume, fe_values_volume_flux, fe_values_volume_soln_flux, current_dofs_indices, current_cell_rhs, fe_values_lagrange);
        }

        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {

            auto current_face = current_cell->face(iface);
            auto neighbor_cell = current_cell->neighbor(iface);

            // See tutorial step-30 for breakdown of 4 face cases

            // Case 1:
            // Face at boundary
            if (current_face->at_boundary() && !current_cell->has_periodic_neighbor(iface) ) {

                n_face_visited++;

                fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);

                if(current_face->at_boundary() && all_parameters->use_periodic_bc == true && dim == 1) //using periodic BCs (for 1d)
                {
                    int cell_index  = current_cell->index();
                    //int cell_index = current_cell->index();
                    if (cell_index == 0 && iface == 0)
                    {
                        fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                        neighbor_cell = dof_handler.begin_active();
                        for (unsigned int i = 0 ; i < triangulation->n_active_cells() - 1; ++i)
                        {
                            ++neighbor_cell;
                        }
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                         const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;

                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1,quad_index_neigh_cell,mapping_index_neigh_cell,fe_index_neigh_cell);

                    }
                    else if (cell_index == (int) triangulation->n_active_cells() - 1 && iface == 1)
                    {
                        fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                        neighbor_cell = dof_handler.begin_active();
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                        const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                        const unsigned int mapping_index_neigh_cell = 0;
                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell); //not sure how changing the face number would work in dim!=1-dimensions.
                    }

                    //std::cout << "cell " << current_cell->index() << "'s " << iface << "th face has neighbour: " << neighbor_cell->index() << std::endl;
                    const int neighbor_face_no = (iface ==1) ? 0:1;
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();

                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    const dealii::FESystem<dim,dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization


                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);

                    //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                    const real penalty1 = deg1sq / vol_div_facearea1;
                    const real penalty2 = deg2sq / vol_div_facearea2;

                    real penalty = 0.5 * ( penalty1 + penalty2 );

                    if ( compute_dRdW ) {
                        assemble_face_term_implicit (
                                                    fe_values_face_int, fe_values_face_ext,
                                                    penalty,
                                                    current_dofs_indices, neighbor_dofs_indices,
                                                    current_cell_rhs, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_explicit (
                                                    fe_values_face_int, fe_values_face_ext,
                                                    penalty,
                                                    current_dofs_indices, neighbor_dofs_indices,
                                                    current_cell_rhs, neighbor_cell_rhs);
                    }

                } else {
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                    const unsigned int normal_direction = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction);

                    real penalty = deg1sq / vol_div_facearea1;

                    const unsigned int boundary_id = current_face->boundary_id();
                    // Need to somehow get boundary type from the mesh
                    if ( compute_dRdW ) {
                        assemble_boundary_term_implicit (boundary_id, fe_values_face_int, penalty, current_dofs_indices, current_cell_rhs);
                    } else {
                        assemble_boundary_term_explicit (boundary_id, fe_values_face_int, penalty, current_dofs_indices, current_cell_rhs);
                    }
                }

                //CASE 1.5: periodic boundary conditions
                //note that periodicity is not adapted for hp adaptivity yet. this needs to be figured out in the future
            } else if (current_face->at_boundary() && current_cell->has_periodic_neighbor(iface)){

                neighbor_cell = current_cell->periodic_neighbor(iface);
                //std::cout << "cell " << current_cell->index() << " at boundary" <<std::endl;
                //std::cout << "periodic neighbour on face " << iface << " is " << neighbor_cell->index() << std::endl;


                if (!current_cell->periodic_neighbor_is_coarser(iface) &&
                    (neighbor_cell->index() > current_cell->index() ||
                     (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
                    )
                   )
                {
                     n_face_visited++;
                    Assert (current_cell->periodic_neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());


                    // Corresponding face of the neighbor.
                    // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
                    const unsigned int neighbor_face_no = current_cell->periodic_neighbor_of_periodic_neighbor(iface);

                    // Get information about neighbor cell
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                    const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                    const unsigned int mapping_index_neigh_cell = 0;
                    const dealii::FESystem<dim,dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    // Local rhs contribution from neighbor
                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                    // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                    neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                    fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                    fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);

                    //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                    const real penalty1 = deg1sq / vol_div_facearea1;
                    const real penalty2 = deg2sq / vol_div_facearea2;

                    real penalty = 0.5 * ( penalty1 + penalty2 );
                    //penalty = 1;//99;

                    if ( compute_dRdW ) {
                        assemble_face_term_implicit (
                                fe_values_face_int, fe_values_face_ext,
                                penalty,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_explicit (
                                fe_values_face_int, fe_values_face_ext,
                                penalty,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs, neighbor_cell_rhs);
                    }

                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                        right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
                    }
                }
                else
                {
                    //do nothing
                }


            // Case 2:
            // Neighbour is finer occurs if the face has children
            // In this case, we loop over the current large face's subfaces and visit multiple neighbors
            } else if (current_face->has_children()) {

                Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());

                // Obtain cell neighbour
                const unsigned int neighbor_face_no = current_cell->neighbor_face_no(iface);

                for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {

                    n_face_visited++;

                    // Get neighbor on ith subface
                    auto neighbor_cell = current_cell->neighbor_child_on_subface (iface, subface_no);
                    // Since the neighbor cell is finer than the current cell, it should not have more children
                    Assert (!neighbor_cell->has_children(), dealii::ExcInternalError());

                    // Get information about neighbor cell
                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                    const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                    const unsigned int mapping_index_neigh_cell = 0;
                    const dealii::FESystem<dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                    // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                    neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                    fe_values_collection_subface.reinit (current_cell, iface, subface_no, quad_index, mapping_index, fe_index_curr_cell);
                    const dealii::FESubfaceValues<dim,dim> &fe_values_face_int = fe_values_collection_subface.get_present_fe_values();

                    fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);

                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                    const real penalty1 = deg1sq / vol_div_facearea1;
                    const real penalty2 = deg2sq / vol_div_facearea2;
                    
                    real penalty = 0.5 * ( penalty1 + penalty2 );

                    if ( compute_dRdW ) {
                        assemble_face_term_implicit (
                                fe_values_face_int, fe_values_face_ext,
                                penalty,
                                current_dofs_indices, neighbor_dofs_indices,
                                current_cell_rhs, neighbor_cell_rhs);
                    } else {
                        assemble_face_term_explicit (
                            fe_values_face_int, fe_values_face_ext,
                            penalty,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);
                    }
                    // Add local contribution from neighbor cell to global vector
                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                        right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
                    }
                }

            // Case 3:
            // Neighbor cell is NOT coarser
            // Therefore, they have the same coarseness, and we need to choose one of them to do the work
            } else if (
                (   !(current_cell->neighbor_is_coarser(iface))
                    // In the case the neighbor is a ghost cell, we let the processor with the lower rank do the work on that face
                    // We cannot use the cell->index() because the index is relative to the distributed triangulation
                    // Therefore, the cell index of a ghost cell might be different to the physical cell index even if they refer to the same cell
                 && neighbor_cell->is_ghost()
                 && current_cell->subdomain_id() < neighbor_cell->subdomain_id()
                )
                ||
                (   !(current_cell->neighbor_is_coarser(iface))
                    // In the case the neighbor is a local cell, we let the cell with the lower index do the work on that face
                 && neighbor_cell->is_locally_owned()
                 &&
                    (  // Cell with lower index does work
                       current_cell->index() < neighbor_cell->index()
                     ||
                       // If both cells have same index
                       // See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
                       // then cell at the lower level does the work
                       (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
                    )
                )
            )
            {
                n_face_visited++;
                Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());

                auto neighbor_cell = current_cell->neighbor_or_periodic_neighbor(iface);
                // Corresponding face of the neighbor.
                // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
                const unsigned int neighbor_face_no = current_cell->neighbor_of_neighbor(iface);

                // Get information about neighbor cell
                const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
                const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
                const unsigned int mapping_index_neigh_cell = 0;
                const dealii::FESystem<dim,dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
                const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();

                // Local rhs contribution from neighbor
                dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization

                // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                neighbor_cell->get_dof_indices (neighbor_dofs_indices);

                fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
                fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
                const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();

                const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
                const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
                const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
                const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);

                //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
                const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
                const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);

                const real penalty1 = deg1sq / vol_div_facearea1;
                const real penalty2 = deg2sq / vol_div_facearea2;
                
                real penalty = 0.5 * ( penalty1 + penalty2 );
                //penalty = 1;//99;

                if ( compute_dRdW ) {
                    assemble_face_term_implicit (
                            fe_values_face_int, fe_values_face_ext,
                            penalty,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);
                } else {
                    assemble_face_term_explicit (
                            fe_values_face_int, fe_values_face_ext,
                            penalty,
                            current_dofs_indices, neighbor_dofs_indices,
                            current_cell_rhs, neighbor_cell_rhs);
                }

                // Add local contribution from neighbor cell to global vector
                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                    right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
                }
            } else {
                // Case 4: Neighbor is coarser
                // Do nothing.
                // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces
            }

        } // end of face loop

        for (unsigned int i=0; i<n_dofs_curr_cell; ++i) {
            right_hand_side(current_dofs_indices[i]) += current_cell_rhs(i);
        }

    } // end of cell loop
    right_hand_side.compress(dealii::VectorOperation::add);
    if ( compute_dRdW ) system_matrix.compress(dealii::VectorOperation::add);

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

    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = Postprocess::PostprocessorFactory<dim>::create_Postprocessor(all_parameters);
    data_out.add_data_vector (solution, *post_processor);

    dealii::Vector<float> subdomain(triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) {
        subdomain(i) = triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

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
    //data_out.build_patches (mapping_collection[mapping_collection.size()-1]);
    data_out.build_patches(*(high_order_grid.mapping_fe_field), max_degree, dealii::DataOut<dim, dealii::hp::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells);
    //data_out.build_patches(*(high_order_grid.mapping_fe_field), fe_collection.size(), dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells);
    std::string filename = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D-";
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    if (iproc == 0) {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) {
            std::string fn = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + ".";
            fn += dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        std::string master_fn = "solution-" + dealii::Utilities::int_to_string(dim, 1) +"D-";
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

    solution.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
    //right_hand_side.reinit(locally_owned_dofs, mpi_communicator);
    right_hand_side.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);

    // System matrix allocation
    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.compute_n_locally_owned_dofs_per_processor(), mpi_communicator, locally_relevant_dofs);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(locally_owned_dofs, sparsity_pattern, mpi_communicator);
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
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.compute_n_locally_owned_dofs_per_processor(), mpi_communicator, locally_owned_dofs);
    mass_sparsity_pattern.copy_from(dsp);
    if (do_inverse_mass_matrix == true) {
        global_inverse_mass_matrix.reinit(locally_owned_dofs, mass_sparsity_pattern);

        global_inverse_mass_correction_matrix.resize(dim);
        for(int idim=0; idim<dim; idim++){
            global_inverse_mass_correction_matrix[idim].reinit(locally_owned_dofs, mass_sparsity_pattern);
        }
        if (this->all_parameters->use_energy == true){//for split form get energy
            global_mass_matrix.reinit(locally_owned_dofs, mass_sparsity_pattern);
        }
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

    //dealii::hp::MappingCollection<dim> mapping_collection(*(high_order_grid.mapping_fe_field));
    //const dealii::MappingManifold<dim,dim> mapping;
    //const dealii::MappingQ<dim,dim> mapping(max_degree+1);
    const auto mapping = (*(high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim> fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
//#if 0
//for jac soln points
    dealii::hp::FEValues<dim,dim> fe_values_collection_volume_jac (mapping_collection, fe_collection, volume_quadrature_collection_jac_sol, this->volume_update_flags); ///< FEValues of volume.
//#endif

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

        std::vector<dealii::FullMatrix<real>> local_mass_correction_matrix(dim);
        for(int idim=0; idim<dim; idim++){
            local_mass_correction_matrix[idim].reinit(n_dofs_cell, n_dofs_cell);
        }

        fe_values_collection_volume.reinit (cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
//#if 0
        //for jac soln points
        fe_values_collection_volume_jac.reinit (cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume_jac = fe_values_collection_volume_jac.get_present_fe_values();
        const std::vector<real> &quad_weights = volume_quadrature_collection[fe_index_curr_cell].get_weights ();
        const std::vector<real> &quad_weights_jac = volume_quadrature_collection_jac_sol[fe_index_curr_cell].get_weights ();
//#endif


//#if 0
        dealii::FullMatrix<real> Chi_operator(n_quad_pts, n_dofs_cell);
        dealii::FullMatrix<real> Chi_operator_soln(n_quad_pts, n_dofs_cell);
        dealii::FullMatrix<real> Quad(n_quad_pts);
        dealii::FullMatrix<real> Jac(n_quad_pts);
        for(int istate=0; istate<nstate; istate++){
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                Chi_operator[iquad][itest] = fe_values_volume.shape_value_component(itest,iquad,istate);
                Chi_operator_soln[iquad][itest] = fe_values_volume_jac.shape_value_component(itest,iquad,istate);
                if(itest == iquad){
                    Quad[iquad][iquad] = quad_weights[iquad];
                    Jac[iquad][iquad] = fe_values_volume_jac.JxW(iquad) / quad_weights_jac[iquad];
                }
                else{
                    Quad[iquad][itest] = 0.0;
                    Jac[iquad][itest] = 0.0;
                }
            }
        }
        }
        dealii::FullMatrix<real> Chi_operator_soln_inv(n_quad_pts, n_dofs_cell);
        Chi_operator_soln_inv.invert(Chi_operator_soln);
        dealii::FullMatrix<real> Chi_Quad(n_dofs_cell, n_quad_pts);
        Chi_operator.Tmmult(Chi_Quad, Quad);
        dealii::FullMatrix<real> interp_flux_sol(n_quad_pts, n_quad_pts); 
        Chi_operator.mmult(interp_flux_sol, Chi_operator_soln_inv);
        dealii::FullMatrix<real> interp_Jac(n_quad_pts, n_quad_pts);
        interp_flux_sol.mmult(interp_Jac, Jac);
        dealii::FullMatrix<real> Chi_quad_Jac(n_dofs_cell, n_quad_pts);
        Chi_Quad.mmult(Chi_quad_Jac, interp_Jac);
        Chi_quad_Jac.mmult(local_mass_matrix, Chi_operator); 

//#endif


#if 0
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_volume.get_fe().system_to_component_index(itest).first;
            for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {
                const unsigned int istate_trial = fe_values_volume.get_fe().system_to_component_index(itrial).first;
                real value = 0.0;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
               // #if 0
                    value +=
                        fe_values_volume.shape_value_component(itest,iquad,istate_test)
                        * fe_values_volume.shape_value_component(itrial,iquad,istate_trial)
                        * fe_values_volume.JxW(iquad);
               // #endif
                #if 0
                    value +=
                        fe_values_volume.shape_value_component(itest,iquad,istate_test)
                        * fe_values_volume.shape_value_component(itrial,iquad,istate_trial)
                        * quad_weights[iquad] * fe_values_volume_jac.JxW(iquad) / quad_weights_jac[iquad];
                #endif
                }
                local_mass_matrix[itrial][itest] = 0.0;
                local_mass_matrix[itest][itrial] = 0.0;
                if(istate_test==istate_trial) { 
                    local_mass_matrix[itrial][itest] = value;
                    local_mass_matrix[itest][itrial] = value;
                }
            }
        }
#endif


        //For flux reconstruction
        dealii::FullMatrix<real> K_operator(n_dofs_cell);
        std::vector<dealii::FullMatrix<real>> K_operator_aux(dim);
        for(int idim=0; idim<dim; idim++){
            K_operator_aux[idim].reinit(n_dofs_cell, n_dofs_cell);
        }
        const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
        get_K_operator_FR(fe_collection, fe_index_curr_cell, fe_values_volume, n_quad_pts, n_dofs_cell, curr_cell_degree, K_operator, K_operator_aux);

        for(int idim=0; idim<dim; idim++){
            for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                    local_mass_correction_matrix[idim][itest][itrial] = local_mass_matrix[itest][itrial] + K_operator_aux[idim][itest][itrial];
                }
            }
        }

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                local_mass_matrix[itest][itrial] = local_mass_matrix[itest][itrial] + K_operator[itest][itrial];
            }
        }


        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);
        if (do_inverse_mass_matrix == true) {
            dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
            local_inverse_mass_matrix.invert(local_mass_matrix);
            global_inverse_mass_matrix.set (dofs_indices, local_inverse_mass_matrix);

            for(int idim=0; idim<dim; idim++){
            dealii::FullMatrix<real> local_inverse_mass_correction_matrix(n_dofs_cell);
            local_inverse_mass_correction_matrix.invert(local_mass_correction_matrix[idim]);
            global_inverse_mass_correction_matrix[idim].set (dofs_indices, local_inverse_mass_correction_matrix);
            }
            if (this->all_parameters->use_energy == true){//for split form energy
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                local_mass_matrix[itest][itrial] = local_mass_matrix[itest][itrial] - K_operator[itest][itrial];
            }
        }
                global_mass_matrix.set (dofs_indices, local_mass_matrix);
            }
        } else {
            global_mass_matrix.set (dofs_indices, local_mass_matrix);
        }
    }

    if (do_inverse_mass_matrix == true) {
        global_inverse_mass_matrix.compress(dealii::VectorOperation::insert);

        for(int idim=0; idim<dim; idim++){
        global_inverse_mass_correction_matrix[idim].compress(dealii::VectorOperation::insert);
        }
        if (this->all_parameters->use_energy == true){//for split form energy
            global_mass_matrix.compress(dealii::VectorOperation::insert);
        }
    } else {
        global_mass_matrix.compress(dealii::VectorOperation::insert);
    }

    return;
}
template<int dim, typename real>
void DGBase<dim,real>::add_mass_matrices(const real scale)
{
    system_matrix.add(scale, global_mass_matrix);
}

template <int dim, typename real>
std::vector<real> DGBase<dim,real>::evaluate_time_steps (const bool exact_time_stepping)
{
    // TO BE DONE
    std::vector<real> time_steps(10);
    if(exact_time_stepping) return time_steps;
    return time_steps;
}

template <int dim, typename real>
void DGBase<dim,real>::build_global_projection_operator ()
{
    const auto mapping = (*(DGBase<dim,real>::high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_flux (mapping_collection, fe_collection_flux, volume_quadrature_collection_flux, this->volume_update_flags); ///< FEValues of volume.
    
    unsigned int n_dif_cells = 0;
   // std::array<std::array<unsigned int, 2>, 100> dif_cells;
   // dealii::Vector<std::array<unsigned int, 2>> dif_cells(this->dof_handler.n_dofs());
    std::vector<std::array<unsigned int, 2>> dif_cells(this->dof_handler.n_dofs());
    unsigned int sum_n_dof = 0;
    unsigned int sum_n_quad = 0;

    for (auto current_cell = dof_handler.begin_active(); current_cell != dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;

        // Current reference element related to this physical cell
        const unsigned int mapping_index = 0;
        const unsigned int fe_index_curr_cell = current_cell->active_fe_index();
        const unsigned int quad_index = fe_index_curr_cell;

        fe_values_collection_volume.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
        dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (current_cell);
        fe_values_collection_volume_flux.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume_flux = fe_values_collection_volume_flux.get_present_fe_values();

        const unsigned int n_quad_pts      = fe_values_volume.n_quadrature_points;
        const unsigned int n_dofs_cell     = fe_values_volume.dofs_per_cell;
        const unsigned int n_quad_pts_flux      = fe_values_volume_flux.n_quadrature_points;
        const unsigned int n_dofs_cell_flux     = fe_values_volume_flux.dofs_per_cell;
        if ( n_dif_cells == 0){
            dif_cells[0][0] = n_dofs_cell;    
            dif_cells[0][1] = n_quad_pts;    
            n_dif_cells++;
            sum_n_dof += n_dofs_cell_flux;
            sum_n_quad += n_quad_pts_flux;
        }
            unsigned int this_cell_dif = 0;
        for (unsigned int icell_dif=0; icell_dif<n_dif_cells; icell_dif++){
            if(n_dofs_cell != dif_cells[icell_dif][0] && n_quad_pts != dif_cells[icell_dif][1]){
                this_cell_dif++; 
            }
        }
        if(this_cell_dif == n_dif_cells){
            dif_cells[this_cell_dif][0] = n_dofs_cell;
            dif_cells[this_cell_dif][1] = n_quad_pts;
            n_dif_cells++;
            sum_n_dof += n_dofs_cell_flux;
            sum_n_quad += n_quad_pts_flux;
        } 
    }

    dif_order_cells.resize(n_dif_cells);
   // global_projection_operator.resize(n_dif_cells);
   // global_projection_operator(sum_n_dof, sum_n_quad);
    global_projection_operator.resize(sum_n_dof);
    for(unsigned int idof =0; idof<sum_n_dof; idof++)
        global_projection_operator[idof].resize(sum_n_quad);

    n_dif_cells = 0;
    sum_n_dof = 0;
    sum_n_quad = 0;

    for (auto current_cell = dof_handler.begin_active(); current_cell != dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;

        // Current reference element related to this physical cell
        const unsigned int mapping_index = 0;
        const unsigned int fe_index_curr_cell = current_cell->active_fe_index();
        const unsigned int quad_index = fe_index_curr_cell;

        fe_values_collection_volume.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
        dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (current_cell);
        fe_values_collection_volume_flux.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume_flux = fe_values_collection_volume_flux.get_present_fe_values();

        const unsigned int n_quad_pts      = fe_values_volume.n_quadrature_points;
        const unsigned int n_dofs_cell     = fe_values_volume.dofs_per_cell;
        const unsigned int n_quad_pts_flux      = fe_values_volume_flux.n_quadrature_points;
        const unsigned int n_dofs_cell_flux     = fe_values_volume_flux.dofs_per_cell;

        if ( n_dif_cells == 0){
            dif_order_cells[0][0] = n_dofs_cell;    
            dif_order_cells[0][1] = n_quad_pts;    
            dif_order_cells[0][2] = sum_n_dof;    
            dif_order_cells[0][3] = sum_n_quad;    
            n_dif_cells++;
            dealii::FullMatrix<real> local_projection_oper(n_dofs_cell_flux, n_quad_pts_flux);
            get_projection_operator(fe_values_volume_flux, n_quad_pts_flux, n_dofs_cell_flux, local_projection_oper);
            for(unsigned int idof = sum_n_dof; idof<(sum_n_dof + n_dofs_cell_flux); idof++){
                for(unsigned int iquad = sum_n_quad; iquad<(sum_n_quad + n_quad_pts_flux); iquad++){
                    global_projection_operator[idof][iquad] = local_projection_oper[idof - sum_n_dof][iquad - sum_n_quad];
                }
            }
            sum_n_dof += n_dofs_cell_flux;
            sum_n_quad += n_quad_pts_flux;
        }
            unsigned int this_cell_dif = 0;
        for (unsigned int icell_dif=0; icell_dif<n_dif_cells; icell_dif++){
            if(n_dofs_cell != dif_order_cells[icell_dif][0] && n_quad_pts != dif_order_cells[icell_dif][1]){
                this_cell_dif++; 
            }
        }
        if(this_cell_dif == n_dif_cells){
            dif_order_cells[this_cell_dif][0] = n_dofs_cell;
            dif_order_cells[this_cell_dif][1] = n_quad_pts;
            dif_order_cells[0][2] = sum_n_dof;    
            dif_order_cells[0][3] = sum_n_quad;    
            n_dif_cells++;
            dealii::FullMatrix<real> local_projection_oper(n_dofs_cell_flux, n_quad_pts_flux);
            get_projection_operator(fe_values_volume_flux, n_quad_pts_flux, n_dofs_cell_flux, local_projection_oper);
            for(unsigned int idof = sum_n_dof; idof<(sum_n_dof + n_dofs_cell_flux); idof++){
                for(unsigned int iquad = sum_n_quad; iquad<(sum_n_quad + n_quad_pts_flux); iquad++){
                    global_projection_operator[idof][iquad] = local_projection_oper[idof - sum_n_dof][iquad - sum_n_quad];
                }
            }
            sum_n_dof += n_dofs_cell_flux;
            sum_n_quad += n_quad_pts_flux;
        } 
    }

}
template <int dim, typename real>
void DGBase<dim,real>::get_projection_operator(
                const dealii::FEValues<dim,dim> &fe_values_volume, unsigned int n_quad_pts,
                 unsigned int n_dofs_cell, dealii::FullMatrix<real> &projection_matrix)
{


        dealii::FullMatrix<real> local_mass_matrix(n_dofs_cell);
        dealii::FullMatrix<real> local_inverse_mass_matrix(n_dofs_cell);
        dealii::FullMatrix<real> local_vandermonde(n_quad_pts, n_dofs_cell);
        dealii::FullMatrix<real> local_weights(n_quad_pts);
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
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            const unsigned int istate_test = fe_values_volume.get_fe().system_to_component_index(itest).first;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                        local_vandermonde[iquad][itest] = fe_values_volume.shape_value_component(itest,iquad,istate_test);
                }
        }
        for( unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for( unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                local_weights[iquad][iquad2] = 0.0;
                if(iquad==iquad2)
                    local_weights[iquad][iquad2] = fe_values_volume.JxW(iquad);
            }
        }

        local_inverse_mass_matrix.invert(local_mass_matrix);
        dealii::FullMatrix<real> V_trans_W(n_dofs_cell, n_quad_pts);

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            for( unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                V_trans_W[itest][iquad] = local_vandermonde[iquad][itest] * fe_values_volume.JxW(iquad);
            }
        }


    for (unsigned int idof=0; idof<n_dofs_cell; idof++){
        for (unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            projection_matrix[idof][iquad] = 0.0;
            for (unsigned int idof2=0; idof2<n_dofs_cell; idof2++){
                projection_matrix[idof][iquad] += local_inverse_mass_matrix[idof][idof2] * V_trans_W[idof2][iquad];
            }
        }
    } 


}
template <int dim, typename real>
void DGBase<dim,real>::get_K_operator_FR(
                 const dealii::hp::FECollection<dim> fe_collection, const unsigned int fe_index_curr_cell,
                 const dealii::FEValues<dim,dim> &fe_values_vol, unsigned int n_quad_pts,
                 unsigned int n_dofs_cell, const unsigned int curr_cell_degree,
                 dealii::FullMatrix<real> &K_operator,
                 std::vector<dealii::FullMatrix<real>> &K_operator_aux/*, std::string correction*/)
{
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    using FR_Aux_enum = Parameters::AllParameters::Flux_Reconstruction_Aux;

    std::vector<dealii::FullMatrix<real>> local_derivative_operator(dim);
    for(int idim=0; idim<dim; idim++){
       local_derivative_operator[idim].reinit(n_quad_pts, n_dofs_cell);
    }

    dealii::FullMatrix<real> Chi_operator(n_quad_pts, n_dofs_cell);
    dealii::FullMatrix<real> Chi_operator_with_Jac(n_quad_pts, n_dofs_cell);
    dealii::FullMatrix<real> Chi_operator_with_Quad(n_quad_pts, n_dofs_cell);
    dealii::FullMatrix<real> Chi_inv_operator(n_quad_pts, n_dofs_cell);
    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();
    const std::vector<real> &quad_weights = volume_quadrature_collection[fe_index_curr_cell].get_weights ();
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const dealii::Point<dim> qpoint  = volume_quadrature_collection[fe_index_curr_cell].point(iquad);
            Chi_operator[iquad][itest] = fe_collection[fe_index_curr_cell].shape_value_component(itest,qpoint,istate);
            Chi_operator_with_Jac[iquad][itest] = fe_collection[fe_index_curr_cell].shape_value_component(itest,qpoint,istate) * JxW[iquad] / quad_weights[iquad];
            Chi_operator_with_Quad[iquad][itest] = fe_collection[fe_index_curr_cell].shape_value_component(itest,qpoint,istate) * quad_weights[iquad];
        }
    }
    }

    Chi_inv_operator.invert(Chi_operator);
    dealii::FullMatrix<real> Jacobian_physical(n_dofs_cell);
    dealii::FullMatrix<real> local_Mass_Matrix_no_Jac(n_dofs_cell);
    Chi_inv_operator.mmult(Jacobian_physical, Chi_operator_with_Jac);//Chi^{-1}*Jm*Chi
    Chi_operator.Tmmult(local_Mass_Matrix_no_Jac, Chi_operator_with_Quad);//M=Chi^T*W*Chi

#if 0
printf("Chi operator Volume\n");
fflush(stdout);
for (unsigned int idof=0; idof<n_dofs_cell; idof++){
for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
    printf(" %g ",Chi_operator[idof][iquad]);
    fflush(stdout);
}
printf("\n");
fflush(stdout);
}
#endif
#if 0
printf("Jac physical\n");
fflush(stdout);
for (unsigned int idof=0; idof<n_dofs_cell; idof++){
for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
    printf(" %g ",Jacobian_physical[idof][iquad]);
    fflush(stdout);
}
printf("\n");
fflush(stdout);
}
#endif


    for(int istate=0; istate<nstate; istate++){
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            dealii::Tensor<1,dim,real> derivative;
            const dealii::Point<dim> qpoint  = volume_quadrature_collection[fe_index_curr_cell].point(iquad);
            derivative = fe_collection[fe_index_curr_cell].shape_grad_component(idof, qpoint, istate);
            for (int idim=0; idim<dim; idim++){
                local_derivative_operator[idim][iquad][idof] = derivative[idim];//store dChi/dXi
            }
        }
    }
    }

    //turn derivative of basis function to derivative operator D=Chi^{-1}*dChi/dXi
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<real> derivative_temp(n_quad_pts, n_dofs_cell);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                derivative_temp[iquad][idof] = local_derivative_operator[idim][iquad][idof];
            }
        }
        Chi_inv_operator.mmult(local_derivative_operator[idim],derivative_temp);
    }

    real c = 0.0;
    real k = 0.0;
    FR_enum c_input = this->all_parameters->flux_reconstruction_type; 
    FR_Aux_enum k_input = this->all_parameters->flux_reconstruction_aux_type; 
    if(c_input == FR_enum::cHU){ 
        const double pfact = factorial_DG(curr_cell_degree);
        const double pfact2 = factorial_DG(2.0 * curr_cell_degree);
       // double cp = 1.0/(pow(2.0,curr_cell_degree)) * pfact2/(pow(pfact,2));
        double cp = pfact2/(pow(pfact,2));//since ref element [0,1]
        c = 2.0 * (curr_cell_degree+1)/( (2.0*curr_cell_degree+1.0)*curr_cell_degree*(pow(pfact*cp,2)));  
        c/=2.0;//since orthonormal
    }
    else if(c_input == FR_enum::cSD){ 
        const double pfact = factorial_DG(curr_cell_degree);
        const double pfact2 = factorial_DG(2.0 * curr_cell_degree);
       // double cp = 1.0/(pow(2.0,curr_cell_degree)) * pfact2/(pow(pfact,2));
        double cp = pfact2/(pow(pfact,2));
        c = 2 * (curr_cell_degree)/( (2.0*curr_cell_degree+1.0)*(curr_cell_degree+1.0)*(pow(pfact*cp,2)));  
        c/=2.0;//since orthonormal
    }
    else if(c_input == FR_enum::cNegative){ 
        const double pfact = factorial_DG(curr_cell_degree);
        const double pfact2 = factorial_DG(2.0 * curr_cell_degree);
       // double cp = 1.0/(pow(2,curr_cell_degree)) * pfact2/(pow(pfact,2));
        double cp = pfact2/(pow(pfact,2));
        c = - 2.0 / ( (2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2)));  
        c/=2.0;//since orthonormal
    }
    else if(c_input == FR_enum::cNegative2){ 
        const double pfact = factorial_DG(curr_cell_degree);
        const double pfact2 = factorial_DG(2.0 * curr_cell_degree);
       // double cp = 1.0/(pow(2,curr_cell_degree)) * pfact2/(pow(pfact,2));
        double cp = pfact2/(pow(pfact,2));
        c = - 2.0 / ( (2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2)));  
        c/=2.0;//since orthonormal
        c/=2.0;
    }
    else if(c_input == FR_enum::cDG){ 
        c = 0.0;
    }
    else if(c_input == FR_enum::cPlus){ 
        if(curr_cell_degree == 2){
            c = 0.186;
    //        c = 0.173;//RK33
        }
        if(curr_cell_degree == 3)
            c = 3.67e-3;
        if(curr_cell_degree == 4){
            c = 4.79e-5;
     //       c = 4.92e-5;//RK33
        }
        if(curr_cell_degree == 5)
            c = 4.24e-7;
        c/=2.0;//since orthonormal
        c/=pow(pow(2.0,curr_cell_degree),2);//since ref elem [0,1]
    }
    if(k_input == FR_Aux_enum::kHU){ 
        const double pfact = factorial_DG(curr_cell_degree);
        const double pfact2 = factorial_DG(2.0 * curr_cell_degree);
       // double cp = 1.0/(pow(2.0,curr_cell_degree)) * pfact2/(pow(pfact,2));
        double cp = pfact2/(pow(pfact,2));
        k = 2.0 * (curr_cell_degree+1.0)/( (2.0*curr_cell_degree+1.0)*curr_cell_degree*(pow(pfact*cp,2)));  
        k/=2.0;//since orthonormal
    }
    else if(k_input == FR_Aux_enum::kSD){ 
        const double pfact = factorial_DG(curr_cell_degree);
        const double pfact2 = factorial_DG(2.0 * curr_cell_degree);
       // double cp = 1.0/(pow(2.0,curr_cell_degree)) * pfact2/(pow(pfact,2));
        double cp = pfact2/(pow(pfact,2));
        k = 2.0 * (curr_cell_degree)/( (2.0*curr_cell_degree+1.0)*(curr_cell_degree+1.0)*(pow(pfact*cp,2)));  
        k/=2.0;//since orthonormal
    }
    else if(k_input == FR_Aux_enum::kNegative){ 
        const double pfact = factorial_DG(curr_cell_degree);
        const double pfact2 = factorial_DG(2.0 * curr_cell_degree);
       // double cp = 1.0/(pow(2.0,curr_cell_degree)) * pfact2/(pow(pfact,2));
        double cp = pfact2/(pow(pfact,2));
        k = - 2.0 / ( (2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2)));  
        k/=2.0;//since orthonormal
    }
    else if(k_input == FR_Aux_enum::kNegative2){ 
        const double pfact = factorial_DG(curr_cell_degree);
        const double pfact2 = factorial_DG(2.0 * curr_cell_degree);
       // double cp = 1.0/(pow(2.0,curr_cell_degree)) * pfact2/(pow(pfact,2));
        double cp = pfact2/(pow(pfact,2));
        k = - 2.0 / ( (2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2)));  
        k/=2.0;//since orthonormal
        k/=2.0;
    }
    else if(k_input == FR_Aux_enum::kDG){ 
        k = 0.0;
    }
    else if(k_input == FR_Aux_enum::kPlus){ 
        if(curr_cell_degree == 2)
        {
            k = 0.186;
        //    k = 0.173;//RK33
        }
        if(curr_cell_degree == 3)
        {
            k = 3.67e-3;
        }
        if(curr_cell_degree == 4){
            k = 4.79e-5;
        //    k = 4.92e-5;//RK33
        }
        if(curr_cell_degree == 5)
            k = 4.24e-7;
        k/=2.0;//since orthonormal
        k/=pow(pow(2.0,curr_cell_degree),2);//since ref elem [0,1]
    }
//c=10000.0;

  //  c = 10.0;
#if 0
printf("\n\n\n");
fflush(stdout);
    printf("Hessian first\n");
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<dim> qpoint  = volume_quadrature_collection[fe_index_curr_cell].point(iquad);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            unsigned int istate =0;
            dealii::Tensor<2,dim,real> derivative;
            derivative = fe_collection[fe_index_curr_cell].shape_grad_grad_component(idof, qpoint, istate);
                printf("%g ",derivative[0][0]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
    printf("Hessian second\n");
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<dim> qpoint  = volume_quadrature_collection[fe_index_curr_cell].point(iquad);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            unsigned int istate =0;
            dealii::Tensor<2,dim,real> derivative;
            derivative = fe_collection[fe_index_curr_cell].shape_grad_grad_component(idof, qpoint, istate);
                printf("%g ",derivative[0][1]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
    printf("Hessian third\n");
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<dim> qpoint  = volume_quadrature_collection[fe_index_curr_cell].point(iquad);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            unsigned int istate =0;
            dealii::Tensor<2,dim,real> derivative;
            derivative = fe_collection[fe_index_curr_cell].shape_grad_grad_component(idof, qpoint, istate);
                printf("%g ",derivative[1][0]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
    printf("Hessian fourth\n");
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<dim> qpoint  = volume_quadrature_collection[fe_index_curr_cell].point(iquad);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            unsigned int istate =0;
            dealii::Tensor<2,dim,real> derivative;
            derivative = fe_collection[fe_index_curr_cell].shape_grad_grad_component(idof, qpoint, istate);
                printf("%g ",derivative[1][1]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
#endif
    
    if(dim == 1){
        dealii::FullMatrix<real> K_operator_no_Jac(n_dofs_cell);
        dealii::FullMatrix<real> K_operator_no_Jac_aux(n_dofs_cell);

        dealii::FullMatrix<real> derivative_p(n_quad_pts, n_dofs_cell);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                if(idof == iquad){
                    derivative_p[idof][iquad] = 1.0;//set it equal to identity
                }
            }
        }

        //(Chi^{-1}*dChi/dXi)^p
        for(unsigned int idegree=0; idegree< (curr_cell_degree); idegree++){
            dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
            derivative_p_temp.add(1, derivative_p);
            local_derivative_operator[0].mmult(derivative_p, derivative_p_temp);
        }

        //c*(Chi^{-1}*dChi/dXi)^p
        dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                derivative_p_temp[iquad][idof] = c * derivative_p[iquad][idof];
            }
        }

        dealii::FullMatrix<real> K_operator_temp(n_dofs_cell);
        derivative_p_temp.Tmmult(K_operator_temp, local_Mass_Matrix_no_Jac);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M
        K_operator_temp.mmult(K_operator_no_Jac, derivative_p);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)

#if 0
//BUILD WEAK K OPERATOR
        dealii::FullMatrix<real> derivative_weak(n_dofs_cell);
    std::vector<dealii::FullMatrix<real>> derivative_weak_temp(dim);
    for(int idim=0; idim<dim; idim++){
       derivative_weak_temp[idim].reinit(n_quad_pts, n_dofs_cell);
    }
        dealii::FullMatrix<real> Mass_inv(n_dofs_cell);
       // dealii::FullMatrix<real> derivative_weak_temp(n_dofs_cell);
        dealii::FullMatrix<real> derivative_weak_trans(n_dofs_cell);
        Mass_inv.invert(local_Mass_Matrix_no_Jac);
    for(int istate=0; istate<nstate; istate++){
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            dealii::Tensor<1,dim,real> derivative;
            const dealii::Point<dim> qpoint  = volume_quadrature_collection[fe_index_curr_cell].point(iquad);
            derivative = fe_collection[fe_index_curr_cell].shape_grad_component(idof, qpoint, istate);
            for (int idim=0; idim<dim; idim++){
                derivative_weak_temp[idim][iquad][idof] = derivative[idim];//store dChi/dXi
            }
        }
    }
    }
        derivative_weak_temp[0].Tmmult(derivative_weak_trans, Chi_operator_with_Quad);
        Mass_inv.mmult(derivative_weak, derivative_weak_trans);
        dealii::FullMatrix<real> derivative_p_weak(n_quad_pts, n_dofs_cell);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                if(idof == iquad){
                    derivative_p_weak[idof][iquad] = 1.0;//set it equal to identity
                }
            }
        }
        for(unsigned int idegree=0; idegree< (curr_cell_degree); idegree++){
            dealii::FullMatrix<real> derivative_p_temp_weak(n_quad_pts, n_dofs_cell);
            derivative_p_temp_weak.add(1, derivative_p_weak);
            derivative_weak.mmult(derivative_p_weak, derivative_p_temp_weak);
        }

        //c*(Chi^{-1}*dChi/dXi)^p
        dealii::FullMatrix<real> derivative_p_temp_weak(n_quad_pts, n_dofs_cell);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                derivative_p_temp_weak[iquad][idof] = c * derivative_p_weak[iquad][idof];
            }
        }

        derivative_p_temp_weak.Tmmult(K_operator_temp, local_Mass_Matrix_no_Jac);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M
        K_operator_temp.mmult(K_operator_no_Jac, derivative_p_weak);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)
//END BUILD WEAK K
        
#endif


        //repeat for auxiliary equation
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                derivative_p_temp[iquad][idof] = k * derivative_p[iquad][idof];
            }
        }

        derivative_p_temp.Tmmult(K_operator_temp, local_Mass_Matrix_no_Jac);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M
        K_operator_temp.mmult(K_operator_no_Jac_aux, derivative_p);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)

        //include Jacobian dependence
        K_operator_no_Jac.mmult(K_operator,Jacobian_physical);//K*Chi^{-1}*Jm*Chi
        K_operator_no_Jac_aux.mmult(K_operator_aux[0],Jacobian_physical);

    }
    else if(dim == 2){
        //build K matrix without Jac dependence
        dealii::FullMatrix<real> K_operator_no_Jac(n_dofs_cell);
        std::vector<dealii::FullMatrix<real>> K_operator_no_Jac_aux(dim);
        for(int idim=0; idim<dim; idim++){
            K_operator_no_Jac_aux[idim].reinit(n_dofs_cell, n_dofs_cell);
        }
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int idof2=0; idof2<n_dofs_cell; idof2++){
                K_operator_no_Jac[idof][idof2] = 0.0;
                for(int idim=0; idim<dim; idim++){
                    K_operator_no_Jac_aux[idim][idof][idof2] = 0.0;
                }
            }
        }

        for(unsigned int v_deg=0; v_deg<=curr_cell_degree; v_deg++){
 
#if 0
            if(v_deg != 0){
                v_deg=curr_cell_degree;
            }
#endif
            dealii::FullMatrix<real> derivative_p(n_quad_pts, n_dofs_cell);
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    if(idof == iquad){
                        derivative_p[idof][iquad] = 1.0;//set it equal to identity
                    }
                    else{
                        derivative_p[idof][iquad] = 0.0;//set it equal to identity
                    }
                }
            }
        
            for(unsigned int idegree=0; idegree< (curr_cell_degree-v_deg); idegree++){
                dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
                derivative_p_temp.add(1, derivative_p);
                local_derivative_operator[0].mmult(derivative_p, derivative_p_temp);
            }
            for(unsigned int idegree=0; idegree< v_deg; idegree++){
                dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
                derivative_p_temp.add(1, derivative_p);
                local_derivative_operator[1].mmult(derivative_p, derivative_p_temp);
            }

#if 0
//testing for verification
            dealii::FullMatrix<real> temp_deriv_p(n_quad_pts, n_dofs_cell);
            Chi_operator.mmult(temp_deriv_p,derivative_p);
    printf("\n\n\n");
    fflush(stdout);
    printf("from Dp v is %d\n",v_deg);
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
               // printf("%g ",temp_deriv_p[iquad][idof]);
               if(fabs(derivative_p[iquad][idof])<1e-9)
                    derivative_p[iquad][idof]=0.0;
                printf("%.12g ",derivative_p[iquad][idof]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
    printf("from Dp ACTUAL vdeg %d\n",v_deg);
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
               // printf("%g ",temp_deriv_p[iquad][idof]);
                printf("%g ",temp_deriv_p[iquad][idof]);
                //printf("%g ",derivative_p[iquad][idof]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
#if 0
    printf("CHI INV\n");
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                printf("%g ",Chi_inv_operator[iquad][idof]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
#endif
#endif

            const double pfact = factorial_DG(curr_cell_degree);
            const double vfact = factorial_DG(v_deg);
            const double p_minus_v_fact = factorial_DG(curr_cell_degree - v_deg);
            double c_v = c * pfact / (vfact * p_minus_v_fact); 
            double k_v = k * pfact / (vfact * p_minus_v_fact); 

            dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    derivative_p_temp[iquad][idof] = c_v * derivative_p[iquad][idof];
                }
            }

            dealii::FullMatrix<real> K_operator_temp(n_dofs_cell);
            derivative_p_temp.Tmmult(K_operator_temp, local_Mass_Matrix_no_Jac);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M
            K_operator_temp.mmult(K_operator_no_Jac, derivative_p, true);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)
     //       dealii::FullMatrix<real> K_operator_temp2(n_dofs_cell);
      //      K_operator_temp.mmult(K_operator_temp2, derivative_p);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)

#if 0
    printf("\n\n\n");
    fflush(stdout);
    printf("K oper temp 2\n");
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                printf("%.12g ",K_operator_temp2[iquad][idof]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
#endif
#if 0
    printf("Mass\n");
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
               if(local_Mass_Matrix_no_Jac[iquad][idof]<1e-9)
                    local_Mass_Matrix_no_Jac[iquad][idof]=0.0;
                printf("%g ",local_Mass_Matrix_no_Jac[iquad][idof]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
    printf("K oper no Jac vdeg %d\n",v_deg);
    fflush(stdout);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
               if(K_operator_no_Jac[iquad][idof]<1e-9)
                    K_operator_no_Jac[iquad][idof]=0.0;
                printf("%g ",K_operator_no_Jac[iquad][idof]);
        fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    } 
#endif
           // derivative_p_temp.Tmmult(K_operator_no_Jac, derivative_p, true);
           
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    derivative_p_temp[iquad][idof] = k_v * derivative_p[iquad][idof];
                }
            }

//#if 0
            derivative_p_temp.Tmmult(K_operator_temp, local_Mass_Matrix_no_Jac, false);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M
            if (v_deg==0){
                K_operator_temp.mmult(K_operator_no_Jac_aux[0], derivative_p, false);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)
            }
            if(v_deg==curr_cell_degree){
                K_operator_temp.mmult(K_operator_no_Jac_aux[1], derivative_p, false);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)
            }


//#endif
#if 0
           // derivative_p_temp.Tmmult(K_operator_no_Jac_aux, derivative_p, true);
                dealii::FullMatrix<real> temp_mat(n_dofs_cell);
                dealii::FullMatrix<real> temp_mat2(n_dofs_cell);
                derivative_p_temp.TmTmult(temp_mat, Chi_operator);            
                temp_mat.mmult(temp_mat2, Chi_operator);            
                temp_mat2.mmult(K_operator_no_Jac_aux[0], derivative_p, true);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)


#endif
#if 0
printf("\n\n\n");
fflush(stdout);
        printf("K oper after\n");
fflush(stdout);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int idof2=0; idof2<n_dofs_cell; idof2++){
                if(fabs(K_operator_no_Jac_aux[0][idof][idof2])<1e-9)
                    K_operator_no_Jac_aux[1][idof][idof2]=0.0;
                printf("%g ",K_operator_no_Jac_aux[1][idof][idof2]);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
        }

#endif
        }
        //Include Jac dependence
        K_operator_no_Jac.mmult(K_operator,Jacobian_physical);
        for(int idim=0; idim<dim; idim++){
            K_operator_no_Jac_aux[idim].mmult(K_operator_aux[idim],Jacobian_physical);
        }
    }
    else{
        //build K matrix without Jac dependence
        dealii::FullMatrix<real> K_operator_no_Jac(n_dofs_cell);
        dealii::FullMatrix<real> K_operator_no_Jac_aux(n_dofs_cell);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            for(unsigned int idof2=0; idof2<n_dofs_cell; idof2++){
                K_operator_no_Jac[idof][idof2] = 0.0;
                K_operator_no_Jac_aux[idof][idof2] = 0.0;
            }
        }

        for(unsigned int v_deg=0; v_deg<=curr_cell_degree; v_deg++){
            for(unsigned int w_deg=0; w_deg<=v_deg; w_deg++){
    
                dealii::FullMatrix<real> derivative_p(n_quad_pts, n_dofs_cell);
                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        if(idof == iquad){
                            derivative_p[idof][iquad] = 1.0;//set it equal to identity
                        }
                    }
                }
        
                for(unsigned int idegree=0; idegree< (curr_cell_degree-v_deg); idegree++){
                    dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
                    derivative_p_temp.add(1, derivative_p);
                    local_derivative_operator[0].mmult(derivative_p, derivative_p_temp);
                }
                for(unsigned int idegree=0; idegree< (v_deg-w_deg); idegree++){
                    dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
                    derivative_p_temp.add(1, derivative_p);
                    local_derivative_operator[1].mmult(derivative_p, derivative_p_temp);
                }
                for(unsigned int idegree=0; idegree< w_deg; idegree++){
                    dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
                    derivative_p_temp.add(1, derivative_p);
                    local_derivative_operator[2].mmult(derivative_p, derivative_p_temp);
                }

                const double pfact = factorial_DG(curr_cell_degree);
                const double vfact = factorial_DG(v_deg);
                const double wfact = factorial_DG(w_deg);
                const double p_minus_v_fact = factorial_DG(curr_cell_degree - v_deg);
                const double v_minus_w_fact = factorial_DG(v_deg - w_deg);
                double c_vw = c * pfact / (vfact * p_minus_v_fact) * vfact / (wfact * v_minus_w_fact);//binomial coeff 
                double k_vw = k * pfact / (vfact * p_minus_v_fact) * vfact / (wfact * v_minus_w_fact);//binomial coeff 

                dealii::FullMatrix<real> derivative_p_temp(n_quad_pts, n_dofs_cell);
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                        derivative_p_temp[iquad][idof] = c_vw * derivative_p[iquad][idof];
                    }
                }

                dealii::FullMatrix<real> K_operator_temp(n_dofs_cell);
                derivative_p_temp.Tmmult(K_operator_temp, local_Mass_Matrix_no_Jac);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M
                K_operator_temp.mmult(K_operator_no_Jac, derivative_p, true);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)
               // derivative_p_temp.Tmmult(K_operator_no_Jac, derivative_p, true);

                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                        derivative_p_temp[iquad][idof] = k_vw * derivative_p[iquad][idof];
                    }
                }

                derivative_p_temp.Tmmult(K_operator_temp, local_Mass_Matrix_no_Jac);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M
                K_operator_temp.mmult(K_operator_no_Jac_aux, derivative_p, true);//(c*(Chi^{-1}*dChi/dXi)^p)^T*M*(Chi^{-1}*dChi/dXi)^p)
    
               // derivative_p_temp.Tmmult(K_operator_no_Jac_aux, derivative_p, true);
            }
        }
        //Include Jac dependence
        K_operator_no_Jac.mmult(K_operator,Jacobian_physical);
        K_operator_no_Jac_aux.mmult(K_operator_aux[0],Jacobian_physical);
    }

}
template <int dim, typename real>
double DGBase<dim,real>::factorial_DG(double n)
{
    if ((n==0)||(n==1))
      return 1;
   else
      return n*factorial_DG(n-1);
}

template <int dim, typename real>
void DGBase<dim,real>::set_current_time(const real time)
{
    this->current_time = time;
}



template class DGBase <PHILIP_DIM, double>;
template class DGFactory <PHILIP_DIM, double>;

} // PHiLiP namespace
