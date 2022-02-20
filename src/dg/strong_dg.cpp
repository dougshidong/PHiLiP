#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>

#include "ADTypes.hpp"

#include <deal.II/fe/fe_dgq.h> // Used for flux interpolation

#include "strong_dg.hpp"

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
DGStrong<dim,nstate,real,MeshType>::DGStrong(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBaseState<dim,nstate,real,MeshType>::DGBaseState(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
{ }
// Destructor
template <int dim, int nstate, typename real, typename MeshType>
DGStrong<dim,nstate,real,MeshType>::~DGStrong ()
{
    pcout << "Destructing DGStrong..." << std::endl;
}

/*******************************************************************
 *
 *
 *              AUXILIARY EQUATIONS
 *
 *
 *******************************************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::allocate_auxiliary_equation ()
{
    {
        DGBase<dim,real,MeshType>::auxiliary_RHS.resize(dim);
        DGBase<dim,real,MeshType>::auxiliary_solution.resize(dim);
        for (int idim=0; idim<dim; idim++){
            DGBase<dim,real,MeshType>::auxiliary_RHS[idim].reinit(DGBase<dim,real,MeshType>::locally_owned_dofs, DGBase<dim,real,MeshType>::ghost_dofs, DGBase<dim,real,MeshType>::mpi_communicator);
            this->auxiliary_RHS[idim].add(1.0);

            DGBase<dim,real,MeshType>::auxiliary_solution[idim].reinit(DGBase<dim,real,MeshType>::locally_owned_dofs, DGBase<dim,real,MeshType>::ghost_dofs, DGBase<dim,real,MeshType>::mpi_communicator);
            this->auxiliary_solution[idim] *= 0.0;
        }

    }
}
template <int dim, int nstate, typename real, typename MeshType>
template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
void DGStrong<dim,nstate,real,MeshType>::assemble_cell_auxiliary_residual (
    const DoFCellAccessorType1 &current_cell,
    const DoFCellAccessorType2 &current_metric_cell,
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rhs)
{
    std::vector<dealii::types::global_dof_index> current_dofs_indices;
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices;

    // Current reference element related to this physical cell
    const int i_fele = current_cell->active_fe_index();
   
    const dealii::FESystem<dim,dim> &current_fe_ref = this->operators.fe_collection_basis[i_fele];
    const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
   
    // Local vector contribution from each cell
    std::vector<dealii::Tensor<1,dim,double>> current_cell_rhs_aux (n_dofs_curr_cell);// Defaults to 0.0 initialization
   
    // Obtain the mapping from local dof indices to global dof indices
    current_dofs_indices.resize(n_dofs_curr_cell);
    current_cell->get_dof_indices (current_dofs_indices);
   
//    dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (current_cell);
   
    const unsigned int n_metric_dofs_cell = this->high_order_grid->fe_system.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> current_metric_dofs_indices(n_metric_dofs_cell);
    std::vector<dealii::types::global_dof_index> neighbor_metric_dofs_indices(n_metric_dofs_cell);
    current_metric_cell->get_dof_indices (current_metric_dofs_indices);
   
    const unsigned int grid_degree = this->high_order_grid->fe_system.tensor_degree();
    const unsigned int poly_degree = i_fele;
   
    assemble_volume_term_auxiliary_equation (
        current_dofs_indices,
        current_metric_dofs_indices,
        poly_degree, grid_degree,
        current_cell_rhs_aux);

    for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {

        auto current_face = current_cell->face(iface);

        // CASE 1: FACE AT BOUNDARY
        if (current_face->at_boundary() && !current_cell->has_periodic_neighbor(iface) ) {
        //for 1D periodic
                if(current_face->at_boundary() && all_parameters->use_periodic_bc == true && dim == 1) //using periodic BCs (for 1d)
                {
                    int cell_index  = current_cell->index();
                    auto neighbor_cell = this->dof_handler.begin_active();
                    if (cell_index == (int) DGBase<dim,real,MeshType>::triangulation->n_active_cells() - 1 && iface == 1)
                    {
                        neighbor_cell = this->dof_handler.begin_active();
                        neighbor_dofs_indices.resize(n_dofs_curr_cell);
                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
                    }
                    else {
                        continue;
                    }
        
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
        
                    const dealii::FESystem<dim,dim> &neigh_fe_ref = this->operators.fe_collection_basis[fe_index_neigh_cell];
                   // const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();
        
                    std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs_aux (n_dofs_neigh_cell ); // Defaults to 0.0 initialization
        
                    const auto metric_neighbor_cell = this->high_order_grid->dof_handler_grid.begin_active();
                    metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);
        
                    assemble_face_term_auxiliary (
                        iface, neighbor_face_no, 
                        poly_degree, grid_degree,
                        current_dofs_indices, neighbor_dofs_indices,
                        current_metric_dofs_indices, neighbor_metric_dofs_indices,
                        current_cell_rhs_aux, neighbor_cell_rhs_aux);

                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                        for(int idim=0; idim<dim; idim++){
                            rhs[idim][neighbor_dofs_indices[i]] += neighbor_cell_rhs_aux[i][idim];
                        }
                    }
        
                } else {//at boundary and not 1D periodic
        
                    const unsigned int boundary_id = current_face->boundary_id();
        
                    assemble_boundary_term_auxiliary_equation (
                        poly_degree, grid_degree, iface,
                        boundary_id, current_dofs_indices, 
                        current_metric_dofs_indices, current_cell_rhs_aux);
        
            }
        
        //CASE 2: PERIODIC BOUNDARY CONDITIONS
        //note that periodicity is not adapted for hp adaptivity yet. this needs to be figured out in the future
        } else if (current_face->at_boundary() && current_cell->has_periodic_neighbor(iface)){

            const auto neighbor_cell = current_cell->periodic_neighbor(iface);

            if (!current_cell->periodic_neighbor_is_coarser(iface) && this->current_cell_should_do_the_work(current_cell, neighbor_cell)) {
                Assert (current_cell->periodic_neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
//pcout<<"ON PERIODIC"<<std::endl;

                const unsigned int n_dofs_neigh_cell = this->operators.fe_collection_basis[neighbor_cell->active_fe_index()].n_dofs_per_cell();
                std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs_aux (n_dofs_neigh_cell ); // Defaults to 0.0 initialization

                // Obtain the mapping from local dof indices to global dof indices for neighbor cell
                neighbor_dofs_indices.resize(n_dofs_neigh_cell);
                neighbor_cell->get_dof_indices (neighbor_dofs_indices);
                
                // Corresponding face of the neighbor.
                const unsigned int neighbor_iface = current_cell->periodic_neighbor_of_periodic_neighbor(iface);
                
                const auto metric_neighbor_cell = current_metric_cell->periodic_neighbor(iface);
                metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);
                
                assemble_face_term_auxiliary (
                    iface, neighbor_iface, 
                    poly_degree, grid_degree,
                    current_dofs_indices, neighbor_dofs_indices,
                    current_metric_dofs_indices, neighbor_metric_dofs_indices,
                    current_cell_rhs_aux, neighbor_cell_rhs_aux);

                // Add local contribution from neighbor cell to global vector
                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                    for(int idim=0; idim<dim; idim++){
                        rhs[idim][neighbor_dofs_indices[i]] += neighbor_cell_rhs_aux[i][idim];
                    }
                }
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
        
            const int i_fele_n = neighbor_cell->active_fe_index();
            const unsigned int n_dofs_neigh_cell = this->operators.fe_collection_basis[i_fele_n].n_dofs_per_cell();
            std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs_aux (n_dofs_neigh_cell ); // Defaults to 0.0 initialization
        
            // Obtain the mapping from local dof indices to global dof indices for neighbor cell
            neighbor_dofs_indices.resize(n_dofs_neigh_cell);
            neighbor_cell->get_dof_indices (neighbor_dofs_indices);
        
            const auto metric_neighbor_cell = current_metric_cell->neighbor(iface);
            metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);
        
            assemble_face_term_auxiliary (
                iface, neighbor_iface, 
                poly_degree, grid_degree,
                current_dofs_indices, neighbor_dofs_indices,
                current_metric_dofs_indices, neighbor_metric_dofs_indices,
                current_cell_rhs_aux, neighbor_cell_rhs_aux);

            // Add local contribution from neighbor cell to global vector
            for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                for(int idim=0; idim<dim; idim++){
                    rhs[idim][neighbor_dofs_indices[i]] += neighbor_cell_rhs_aux[i][idim];
                }
            }
        // CASE 5: NEIGHBOR CELL HAS SAME COARSENESS
        // Therefore, we need to choose one of them to do the work
        } else if ( this->current_cell_should_do_the_work(current_cell, current_cell->neighbor(iface)) ) {
            Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
        
//pcout<<"ON normal face"<<std::endl;
            const auto neighbor_cell = current_cell->neighbor_or_periodic_neighbor(iface);
            // Corresponding face of the neighbor.
            // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
            const unsigned int neighbor_iface = current_cell->neighbor_of_neighbor(iface);
        
            // Get information about neighbor cell
            const unsigned int n_dofs_neigh_cell = this->operators.fe_collection_basis[neighbor_cell->active_fe_index()].n_dofs_per_cell();
        
            // Local rhs contribution from neighbor
            std::vector<dealii::Tensor<1,dim,double>> neighbor_cell_rhs_aux (n_dofs_neigh_cell ); // Defaults to 0.0 initialization
        
            // Obtain the mapping from local dof indices to global dof indices for neighbor cell
            neighbor_dofs_indices.resize(n_dofs_neigh_cell);
            neighbor_cell->get_dof_indices (neighbor_dofs_indices);
        
            const auto metric_neighbor_cell = current_metric_cell->neighbor_or_periodic_neighbor(iface);
            metric_neighbor_cell->get_dof_indices(neighbor_metric_dofs_indices);
        
            assemble_face_term_auxiliary (
                iface, neighbor_iface, 
                poly_degree, grid_degree,
                current_dofs_indices, neighbor_dofs_indices,
                current_metric_dofs_indices, neighbor_metric_dofs_indices,
                current_cell_rhs_aux, neighbor_cell_rhs_aux);
        
            // Add local contribution from neighbor cell to global vector
            for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
                for(int idim=0; idim<dim; idim++){
                    rhs[idim][neighbor_dofs_indices[i]] += neighbor_cell_rhs_aux[i][idim];
                }
            }
        } else {
            // Should be faces where the neighbor cell has the same coarseness
            // but will be evaluated when we visit the other cell.
        }


    } // end of face loop

    // Add local contribution from current cell to global vector
    for (unsigned int i=0; i<n_dofs_curr_cell; ++i) {
        for(int idim=0; idim<dim; idim++){
            rhs[idim][current_dofs_indices[i]] += current_cell_rhs_aux[i][idim];
        }
    }

}
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_auxiliary_residual ()
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    using ODE_enum = Parameters::ODESolverParam::ODESolverEnum;
    if ( (this->all_parameters->pde_type == PDE_enum::convection_diffusion || this->all_parameters->pde_type == PDE_enum::diffusion)
        && this->all_parameters->ode_solver_param.ode_solver_type == ODE_enum::explicit_solver )//auxiliary only works explicit for now
    {
        //set auxiliary rhs to 0
        for(int idim=0; idim<dim; idim++){
            this->auxiliary_RHS[idim] = 0;
        }
        //loop over cells solving for auxiliary rhs
        auto metric_cell = this->high_order_grid->dof_handler_grid.begin_active();
        for (auto soln_cell = DGBase<dim,real,MeshType>::dof_handler.begin_active(); soln_cell != DGBase<dim,real,MeshType>::dof_handler.end(); ++soln_cell, ++metric_cell) {
            if (!soln_cell->is_locally_owned()) continue;

            assemble_cell_auxiliary_residual (
                soln_cell,
                metric_cell,
                this->auxiliary_RHS);
        } // end of cell loop

        for(int idim=0; idim<dim; idim++){
            //compress auxiliary rhs for solution transfer across mpi ranks
            this->auxiliary_RHS[idim].compress(dealii::VectorOperation::add);
            //update ghost values
            this->auxiliary_RHS[idim].update_ghost_values();

            //solve for auxiliary solution for each dimension
            this->global_inverse_mass_matrix_auxiliary[idim].vmult(this->auxiliary_solution[idim], this->auxiliary_RHS[idim]);
            //update ghost values of auxiliary solution
            this->auxiliary_solution[idim].update_ghost_values();

        }
    }//end of if statement for diffusive
}

/**************************************************
 *
 *      AUXILIARY RESIDUAL FUNCTIONS
 *
 *      *******************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_auxiliary_equation(
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const unsigned int poly_degree,
        const unsigned int grid_degree,
        std::vector<dealii::Tensor<1,dim,double>> &local_auxiliary_RHS)
{

    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,ADtype>;

    const unsigned int n_quad_pts = this->operators.volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;

    AssertDimension (n_dofs_cell, current_dofs_indices.size());

    //Get the metric terms
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    //get local cofactor matrix
    std::vector<std::vector<real>> mapping_support_points(dim);
    for(int idim=0; idim<dim; idim++){
        mapping_support_points[idim].resize(n_metric_dofs/dim);
    }
    dealii::QGaussLobatto<dim> vol_GLL(grid_degree + 1);//Mapping supp points ALWYAS GLL nodes of grid degree + 1
    //Reindex the mapping support points to work with the operators finite elements.
    for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate); 
        }
    }
    std::vector<dealii::FullMatrix<real>> metric_cofactor(n_quad_pts);
    std::vector<real> determinant_Jacobian(n_quad_pts);
    for(unsigned int iquad=0;iquad<n_quad_pts; iquad++){
        metric_cofactor[iquad].reinit(dim, dim);
    }
    this->operators.build_local_vol_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, n_quad_pts, n_metric_dofs/dim, mapping_support_points, determinant_Jacobian, metric_cofactor);

    //build physical gradient operator based on skew-symmetric form
    //get physical split grdient in covariant basis
    std::vector<std::vector<dealii::FullMatrix<real>>> physical_gradient(nstate);
    for(unsigned int istate=0; istate<nstate; istate++){
        physical_gradient[istate].resize(dim);
        for(int idim=0; idim<dim; idim++){
            physical_gradient[istate][idim].reinit(n_quad_pts, n_quad_pts);    
        }
    }
    this->operators.get_Jacobian_scaled_physical_gradient(false, this->operators.gradient_flux_basis[poly_degree], metric_cofactor, n_quad_pts, nstate, physical_gradient); 

    // AD variable
    std::vector< ADtype > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real,MeshType>::solution(current_dofs_indices[idof]);
    }
    std::vector< ADArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_at_q[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_at_q[iquad][istate] += soln_coeff[idof] * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
        }
    }
    for(int istate=0; istate<nstate; istate++){
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            for(int idim=0; idim<dim; idim++){
                soln_grad_at_q[iquad][istate][idim] = 0.0;
                for(unsigned int iflux=0; iflux<n_quad_pts; iflux++){
                    soln_grad_at_q[iquad][istate][idim] += soln_at_q[iflux][istate] * physical_gradient[istate][idim][iquad][iflux];
                }
            }
        }
    }


    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        dealii::Tensor<1,dim,ADtype> rhs;
        for (int idim=0; idim<dim; idim++){
            rhs[idim] = 0.0;
        }

        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            rhs = rhs + this->operators.vol_integral_basis[poly_degree][iquad][itest] * soln_grad_at_q[iquad][istate];
        }

        for (int idim=0; idim<dim; idim++)
        {
            local_auxiliary_RHS[itest][idim] += rhs[idim];
        }

    }
}
/**************************************************************************************/
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_auxiliary_equation(
        const unsigned int poly_degree, const unsigned int grid_degree,
        const unsigned int iface,
        const unsigned int boundary_id,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS)
{
   // using ADtype = Sacado::Fad::DFad<real>;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,ADtype>;

    const unsigned int n_face_quad_pts = this->operators.face_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;

    AssertDimension (n_dofs_cell, current_dofs_indices.size());

    //Get metric terms
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    //Compute metric terms (cofactor and normals)

    //get local cofactor matrix
    std::vector<std::vector<real>> mapping_support_points(dim);//mapping support points of interior cell
    for(int idim=0; idim<dim; idim++){
        mapping_support_points[idim].resize(n_metric_dofs/dim);//there are n_metric_dofs/dim shape functions
    }
    dealii::QGaussLobatto<dim> vol_GLL(grid_degree + 1);//Note that the mapping supp points are always GLL quad nodes
    //get the mapping support points int and ext

    for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate); 
        }
    }

    std::vector<dealii::FullMatrix<real>> metric_cofactor_face(n_face_quad_pts);
    std::vector<real> determinant_Jacobian_face(n_face_quad_pts);
    for(unsigned int iquad=0;iquad<n_face_quad_pts; iquad++){
        metric_cofactor_face[iquad].reinit(dim, dim);
    }
    //surface metric cofactor
    this->operators.build_local_face_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, iface,
                                                                        n_face_quad_pts, n_metric_dofs / dim, mapping_support_points, 
                                                                        determinant_Jacobian_face, metric_cofactor_face);

    //get physical normal scaled by jac
    const dealii::Tensor<1,dim, real> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];

    std::vector<dealii::Tensor<1,dim, real> > normal_phys_int(n_face_quad_pts);
    std::vector<real> face_jac(n_face_quad_pts);
    std::vector<real> JxW_int(n_face_quad_pts);
    for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
        this->operators.compute_reference_to_physical(unit_normal_int, metric_cofactor_face[iquad], normal_phys_int[iquad]); 
        face_jac[iquad] = normal_phys_int[iquad].norm();
        normal_phys_int[iquad] /= face_jac[iquad];//normalize it
    }

    std::vector<std::vector<dealii::FullMatrix<real>>> physical_gradient(nstate);
    for(unsigned int istate=0; istate<nstate; istate++){
        physical_gradient[istate].resize(dim);
        for(int idim=0; idim<dim; idim++){
            physical_gradient[istate][idim].reinit(n_face_quad_pts, n_dofs_cell);    
        }
    }
    //hard coding here the physical gradient of the basis functions evaluated at the facet cubature nodes.
    const dealii::Quadrature<dim> face_quad = dealii::QProjector<dim>::project_to_face(dealii::ReferenceCell::get_hypercube(dim),
                                                                                                this->operators.face_quadrature_collection[poly_degree],
                                                                                                iface);
    for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            for(unsigned int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    physical_gradient[istate][idim][iquad][idof] += this->operators.fe_collection_basis[poly_degree].shape_grad_component(idof,face_quad.point(iquad),istate)[jdim] * metric_cofactor_face[iquad][idim][jdim] ;
                }
                physical_gradient[istate][idim][iquad][idof] /= face_jac[iquad];
            }
        }
    }


    std::vector<ADArray> soln_int(n_face_quad_pts);
    std::vector<ADArray> soln_ext(n_face_quad_pts);

    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);

    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_num_flux_difference(n_face_quad_pts);

    // AD variable
    std::vector< ADtype > soln_coeff_int(n_dofs_cell);
    //const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(current_dofs_indices[idof]);
    //    soln_coeff_int[idof].diff(idof, n_total_indep);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normal_phys_int[iquad];

        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * this->operators.basis_at_facet_cubature[poly_degree][iface][iquad][idof];
            for(int idim=0; idim<dim; idim++){
                soln_grad_int[iquad][istate][idim] += soln_coeff_int[idof] * physical_gradient[istate][idim][iquad][idof];
            }
        }

        dealii::Point<dim> quad_point;
        for(unsigned int imetric_dof=0; imetric_dof<n_metric_dofs/dim; imetric_dof++){
            for(int idim=0; idim<dim; idim++){
                quad_point[idim] += this->operators.mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][iquad][imetric_dof]
                                                    * mapping_support_points[idim][imetric_dof];
            }
        }
        const dealii::Point<dim, real> x_quad = quad_point;
       // pde_physics->boundary_face_values (boundary_id, x_quad, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);
        this->pde_physics_double->boundary_face_values (boundary_id, x_quad, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        diss_soln_num_flux[iquad] = this->diss_num_flux_double->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        for (int s=0; s<nstate; s++) {
            diss_num_flux_difference[iquad][s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;//(u*-u)*n
        }

    }

    // Boundary integral
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADTensor rhs;
        for (int idim=0; idim<dim; idim++){
            rhs[idim] = 0.0;
        }
        
        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            rhs = rhs + this->operators.face_integral_basis[poly_degree][iface][iquad][itest] * diss_num_flux_difference[iquad][istate] * face_jac[iquad];
        }

        for (int idim=0; idim<dim; idim ++)
        {
            local_auxiliary_RHS[itest][idim] += rhs[idim];   
        }
    }
}
/*********************************************************************************/
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_auxiliary(
        const unsigned int iface, const unsigned int neighbor_iface,
        const unsigned int poly_degree, const unsigned int grid_degree,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &/*metric_dof_indices_ext*/,
        std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS_int,
        std::vector<dealii::Tensor<1,dim,real>> &local_auxiliary_RHS_ext)
{
   // using ADtype = Sacado::Fad::DFad<real>;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;
    using ADTensor = dealii::Tensor<1,dim,ADtype>;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = this->operators.face_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_int = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;
    const unsigned int n_dofs_ext = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;

    AssertDimension (n_dofs_int, current_dofs_indices.size());
    AssertDimension (n_dofs_ext, neighbor_dofs_indices.size());

    //Get metric terms.
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    //Compute metric terms (cofactor and normals)

    //get local cofactor matrix
    std::vector<std::vector<real>> mapping_support_points_int(dim);//mapping support points of interior cell
    for(int idim=0; idim<dim; idim++){
        mapping_support_points_int[idim].resize(n_metric_dofs/dim);//there are n_metric_dofs/dim shape functions
    }
    dealii::QGaussLobatto<dim> vol_GLL(grid_degree + 1);//Note that the mapping supp points are always GLL quad nodes
    for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val_int = (this->high_order_grid->volume_nodes[metric_dof_indices_int[idof]]);
            const unsigned int istate_int = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points_int[istate_int][igrid_node] += val_int * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate_int); 
        }
    }

    std::vector<dealii::FullMatrix<real>> metric_cofactor_face(n_face_quad_pts);
    std::vector<real> determinant_Jacobian_face(n_face_quad_pts);
    for(unsigned int iquad=0;iquad<n_face_quad_pts; iquad++){
        metric_cofactor_face[iquad].reinit(dim, dim);
    }
    //surface metric cofactor
    this->operators.build_local_face_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, iface,
                                                                        n_face_quad_pts, n_metric_dofs / dim, mapping_support_points_int, 
                                                                        determinant_Jacobian_face, metric_cofactor_face);

    //get physical normal scaled by jac
    const dealii::Tensor<1,dim, real> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];

    std::vector<dealii::Tensor<1,dim, real> > normal_phys_int(n_face_quad_pts);
    std::vector<real> face_jac(n_face_quad_pts);
    for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
        this->operators.compute_reference_to_physical(unit_normal_int, metric_cofactor_face[iquad], normal_phys_int[iquad]); 
        face_jac[iquad] = normal_phys_int[iquad].norm();
        normal_phys_int[iquad] /= face_jac[iquad];//normalize it
    }


    // AD variable
    std::vector<ADtype> soln_coeff_int(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext(n_dofs_ext);

    // Interpolate solution to the face quadrature points
    std::vector< ADArray > soln_int_at_q(n_face_quad_pts);
    std::vector< ADArray > soln_ext_at_q(n_face_quad_pts);

    std::vector<ADArray> soln_num_flux(n_face_quad_pts); 
    std::vector<ADArrayTensor1> soln_num_flux_dot_n(n_face_quad_pts); 

    // AD variable
   // const unsigned int n_total_indep = n_dofs_int + n_dofs_ext;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(current_dofs_indices[idof]);
//if(isnan(soln_coeff_int[idof])){
//    pcout<<"THE SOLUTION INT  WAS NAN FIRST AHAHAHAHAHAH"<<std::endl;
//}
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext[idof] = DGBase<dim,real,MeshType>::solution(neighbor_dofs_indices[idof]);
//if(isnan(soln_coeff_ext[idof])){
//    pcout<<"THE SOLUTION EXT  WAS NAN FIRST AHAHAHAHAHAH"<<std::endl;
//}
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int_at_q[iquad][istate]      = 0;
            soln_ext_at_q[iquad][istate]      = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normal_phys_int[iquad];
       // const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_int_at_q[iquad][istate] += soln_coeff_int[idof] * this->operators.basis_at_facet_cubature[poly_degree][iface][iquad][idof];
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_ext_at_q[iquad][istate] += soln_coeff_ext[idof] * this->operators.basis_at_facet_cubature[poly_degree][neighbor_iface][iquad][idof];
        }

        soln_num_flux[iquad] = this->diss_num_flux_double->evaluate_solution_flux(soln_int_at_q[iquad], soln_ext_at_q[iquad], normal_int);

        for (int s=0; s<nstate; s++) {
            for(int idim=0; idim<dim; idim++){
                soln_num_flux_dot_n[iquad][s][idim] = soln_num_flux[iquad][s] * normal_int[idim];
            }
        }
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        ADTensor rhs;
        for (int idim=0; idim<dim; idim++){
            rhs[idim] = 0.0;
        }
        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            const ADTensor soln_surf_difference = face_jac[iquad] * soln_num_flux_dot_n[iquad][istate] - face_jac[iquad] * soln_int_at_q[iquad][istate] * normal_phys_int[iquad]; 
            rhs = rhs + this->operators.face_integral_basis[poly_degree][iface][iquad][itest_int] * soln_surf_difference;
        }

        for (int idim = 0; idim<dim; idim++)
        {
            local_auxiliary_RHS_int[itest_int][idim] += rhs[idim];
        }
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        ADTensor rhs;
        for (int idim=0; idim<dim; idim++){
            rhs[idim] = 0.0;
        }
        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            const ADTensor soln_surf_difference = face_jac[iquad] * (-soln_num_flux_dot_n[iquad][istate]) - face_jac[iquad] * soln_ext_at_q[iquad][istate] * (-normal_phys_int[iquad]); 
            rhs = rhs + this->operators.face_integral_basis[poly_degree][neighbor_iface][iquad][itest_ext] * soln_surf_difference;
        }

        for (int idim = 0; idim<dim; idim++)
        {
            local_auxiliary_RHS_ext[itest_ext][idim] += rhs[idim];
        }
    }
}

/*******************************************************************
 *
 *
 *              PRIMARY EQUATIONS
 *
 *              NOTE: the implicit functions have not been modified.
 *
 *******************************************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int ,//face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &,//fe,
    const dealii::Quadrature<dim-1> &,//quadrature,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const bool compute_dRdW,
    const bool compute_dRdX,
    const bool compute_d2R)
{ 
    (void) current_cell_index;
    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    using ADArray = std::array<FadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;
 
    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;
 
    AssertDimension (n_dofs_cell, soln_dof_indices.size());
 
    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();
 
    std::vector<real> residual_derivatives(n_dofs_cell);
 
    std::vector<ADArray> soln_int(n_face_quad_pts);
    std::vector<ADArray> soln_ext(n_face_quad_pts);
 
    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);
 
    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*
 
    std::vector<ADArrayTensor1> conv_phys_flux(n_face_quad_pts);
 
    // AD variable
    std::vector< FadType > soln_coeff_int(n_dofs_cell);
    const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices[idof]);
        soln_coeff_int[idof].diff(idof, n_total_indep);
    }
 
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
 
        const dealii::Tensor<1,dim,FadType> normal_int = normals[iquad];
        const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;
 
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }
 
        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
        dealii::Point<dim,FadType> ad_point;
        for (int d=0;d<dim;++d) { ad_point[d] = real_quad_point[d]; }
        this->pde_physics_fad->boundary_face_values (boundary_id, ad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);
 
        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);
 
        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
 
        // Used for strong form
        // Which physical convective flux to use?
        conv_phys_flux[iquad] = this->pde_physics_fad->convective_flux (soln_int[iquad]);
 
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);
 
        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
   for (int d=0; d<dim; d++) {
    diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
   }
        }
        diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
 
        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_auxiliary_flux(
            0.0, 0.0,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }
 
    // Boundary integral
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
 
        FadType rhs = 0.0;
 
        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;
 
        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
 
            // Convection
            const FadType flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux[iquad][istate]*normals[iquad];
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * flux_diff * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
            rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
        }
        // *******************
 
        local_rhs_int_cell(itest) += rhs.val();
 
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                //residual_derivatives[idof] = rhs.fastAccessDx(idof);
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
        }
    }
}
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &,//fe,
    const dealii::Quadrature<dim> &,//quadrature,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &fe_values_lagrange,
    const bool compute_dRdW,
    const bool compute_dRdX,
    const bool compute_d2R)
{
    (void) current_cell_index;
    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    using ADArray = std::array<FadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;

    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector< ADArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< ADArray > source_at_q(n_quad_pts);

    // AD variable
    std::vector< FadType > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real,MeshType>::solution(cell_dofs_indices[idof]);
        soln_coeff[idof].diff(idof, n_dofs_cell);
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the volume quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
        }
        //std::cout << "Density " << soln_at_q[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum " << soln_at_q[iquad][1] << std::endl;
        //std::cout << "Energy " << soln_at_q[iquad][nstate-1] << std::endl;
        // Evaluate physical convective flux and source term
        conv_phys_flux_at_q[iquad] = this->pde_physics_fad->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = this->pde_physics_fad->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);

        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            const dealii::Point<dim,real> real_quad_point = fe_values_vol.quadrature_point(iquad);
            dealii::Point<dim,FadType> ad_point;
            for (int d=0;d<dim;++d) { ad_point[d] = real_quad_point[d]; }
            source_at_q[iquad] = this->pde_physics_fad->source_term (ad_point, soln_at_q[iquad], DGBase<dim,real,MeshType>::current_time);
        }
    }


    // Evaluate flux divergence by interpolating the flux
    // Since we have nodal values of the flux, we use the Lagrange polynomials to obtain the gradients at the quadrature points.
    //const dealii::FEValues<dim,dim> &fe_values_lagrange = this->fe_values_collection_volume_lagrange.get_present_fe_values();
    std::vector<ADArray> flux_divergence(n_quad_pts);

    std::array<std::array<std::vector<FadType>,nstate>,dim> f;
    std::array<std::array<std::vector<FadType>,nstate>,dim> g;

    for (int istate = 0; istate<nstate; ++istate) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            flux_divergence[iquad][istate] = 0.0;
            for ( unsigned int flux_basis = 0; flux_basis < n_quad_pts; ++flux_basis ) {
                flux_divergence[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate] * fe_values_lagrange.shape_grad(flux_basis,iquad);
            }

        }
    }

    // Strong form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        FadType rhs = 0;


        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Convective
            // Now minus such 2 integrations by parts
            assert(JxW[iquad] - fe_values_lagrange.JxW(iquad) < 1e-14);

            rhs = rhs - fe_values_vol.shape_value_component(itest,iquad,istate) * flux_divergence[iquad][istate] * JxW[iquad];

            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW[iquad];
            // Source

            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
                rhs = rhs + fe_values_vol.shape_value_component(itest,iquad,istate) * source_at_q[iquad][istate] * JxW[iquad];
            }
        }

        local_rhs_int_cell(itest) += rhs.val();

        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(cell_dofs_indices[itest], cell_dofs_indices, residual_derivatives);
        }
    }
}
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const std::pair<unsigned int, int> /*face_subface_int*/,
    const std::pair<unsigned int, int> /*face_subface_ext*/,
    const typename dealii::QProjector<dim>::DataSetDescriptor /*face_data_set_int*/,
    const typename dealii::QProjector<dim>::DataSetDescriptor /*face_data_set_ext*/,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &,//fe_int,
    const dealii::FESystem<dim,dim> &,//fe_ext,
    const dealii::Quadrature<dim-1> &,//face_quadrature_int,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell,
    const bool compute_dRdW,
    const bool compute_dRdX,
    const bool compute_d2R)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;
    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    using ADArray = std::array<FadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_dofs_ext, soln_dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // even for non-conforming meshes?
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<FadType> soln_coeff_int_ad(n_dofs_int);
    std::vector<FadType> soln_coeff_ext_ad(n_dofs_ext);


    // Jacobian blocks
    std::vector<real> dR1_dW1(n_dofs_int);
    std::vector<real> dR1_dW2(n_dofs_ext);
    std::vector<real> dR2_dW1(n_dofs_int);
    std::vector<real> dR2_dW2(n_dofs_ext);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_ext(n_face_quad_pts);

    // Interpolate solution to the face quadrature points
    std::vector< ADArray > soln_int(n_face_quad_pts);
    std::vector< ADArray > soln_ext(n_face_quad_pts);

    std::vector< ADArrayTensor1 > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
    std::vector< ADArrayTensor1 > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros

    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_ext
    // AD variable
    const unsigned int n_total_indep = n_dofs_int + n_dofs_ext;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices_int[idof]);
        soln_coeff_int_ad[idof].diff(idof, n_total_indep);
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices_ext[idof]);
        soln_coeff_ext_ad[idof].diff(idof+n_dofs_int, n_total_indep);
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
            soln_ext[iquad][istate]      = 0;
            soln_grad_ext[iquad][istate] = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,FadType> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int_ad[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
            soln_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
            soln_grad_ext[iquad][istate] += soln_coeff_ext_ad[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);
        }
        //std::cout << "Density int" << soln_int[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum int" << soln_int[iquad][1] << std::endl;
        //std::cout << "Energy int" << soln_int[iquad][nstate-1] << std::endl;
        //std::cout << "Density ext" << soln_ext[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum ext" << soln_ext[iquad][1] << std::endl;
        //std::cout << "Energy ext" << soln_ext[iquad][nstate-1] << std::endl;

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        conv_phys_flux_int[iquad] = this->pde_physics_fad->convective_flux (soln_int[iquad]);
        conv_phys_flux_ext[iquad] = this->pde_physics_fad->convective_flux (soln_ext[iquad]);

        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
   for (int d=0; d<dim; d++) {
    diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
    diss_soln_jump_ext[s][d] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext[d];
   }
        }
        diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = this->pde_physics_fad->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_auxiliary_flux(
            0.0, 0.0,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        FadType rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            const FadType flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux_int[iquad][istate]*normals_int[iquad];
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * flux_diff * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
            rhs = rhs + fe_values_int.shape_grad_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
        }

        local_rhs_int_cell(itest_int) += rhs.val();
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
                dR1_dW1[idof] = rhs.fastAccessDx(idof);
            }
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                dR1_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
            }
            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_int, dR1_dW1);
            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_ext, dR1_dW2);
        }
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        FadType rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            const FadType flux_diff = (-conv_num_flux_dot_n[iquad][istate]) - conv_phys_flux_ext[iquad][istate]*(-normals_int[iquad]);
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * flux_diff * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-diss_auxi_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
            rhs = rhs + fe_values_ext.shape_grad_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
        }

        local_rhs_ext_cell(itest_ext) += rhs.val();
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
                dR2_dW1[idof] = rhs.fastAccessDx(idof);
            }
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                dR2_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
            }
            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_int, dR2_dW1);
            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_ext, dR2_dW2);
        }
    }
}

/*******************************************************
 *
 *              EXPLICIT
 *
 *              **********************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_explicit(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &/*fe_values_vol*/,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const unsigned int poly_degree,
    const unsigned int grid_degree,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    (void) current_cell_index;
    //std::cout << "assembling cell terms" << std::endl;
    using realtype = real;
    using realArray = std::array<realtype,nstate>;
    using realArrayTensor1 = std::array< dealii::Tensor<1,dim,realtype>, nstate >;

//    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
//    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;

    const unsigned int n_quad_pts = this->operators.volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector< realArray > soln_at_q(n_quad_pts);
    std::vector< realArrayTensor1 > aux_soln_at_q(n_quad_pts); //auxiliary sol at flux nodes

    std::vector< realArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
    std::vector< realArrayTensor1 > diffusive_phys_flux_at_q(n_quad_pts);
    std::vector< realArray > source_at_q(n_quad_pts);



    //get local cofactor matrix
    std::vector<std::vector<real>> mapping_support_points(dim);
    for(int idim=0; idim<dim; idim++){
        mapping_support_points[idim].resize(n_metric_dofs/dim);
    }
    dealii::QGaussLobatto<dim> vol_GLL(grid_degree + 1);//Mapping supp points ALWYAS GLL nodes of grid degree + 1
    for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate); 
        }
    }
    std::vector<dealii::FullMatrix<real>> metric_cofactor(n_quad_pts);
    std::vector<real> determinant_Jacobian(n_quad_pts);
    for(unsigned int iquad=0;iquad<n_quad_pts; iquad++){
        metric_cofactor[iquad].reinit(dim, dim);
    }
    this->operators.build_local_vol_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, n_quad_pts, n_metric_dofs/dim, mapping_support_points, determinant_Jacobian, metric_cofactor);

    //build physical gradient operator based on skew-symmetric form
    //get physical split grdient in covariant basis
    std::vector<std::vector<dealii::FullMatrix<real>>> physical_gradient(nstate);
    for(unsigned int istate=0; istate<nstate; istate++){
        physical_gradient[istate].resize(dim);
        for(int idim=0; idim<dim; idim++){
            physical_gradient[istate][idim].reinit(n_quad_pts, n_quad_pts);    
        }
    }
    //Note that this does curvilinear metric splitting built in.
    this->operators.get_Jacobian_scaled_physical_gradient(true, this->operators.gradient_flux_basis[poly_degree], metric_cofactor, n_quad_pts, nstate, physical_gradient); 



    // AD variable
    std::vector< realtype > soln_coeff(n_dofs_cell);
    std::vector< dealii::Tensor<1,dim,real>> aux_soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real,MeshType>::solution(cell_dofs_indices[idof]);
        for(int idim=0; idim<dim; idim++){
            aux_soln_coeff[idof][idim] = DGBase<dim,real,MeshType>::auxiliary_solution[idim](cell_dofs_indices[idof]);
        }
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_at_q[iquad][istate]      = 0;
            aux_soln_at_q[iquad][istate]  = 0;
        }
    }
    // Interpolate solution to the volume quadrature points
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_at_q[iquad][istate]     += soln_coeff[idof]     * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
            aux_soln_at_q[iquad][istate] += aux_soln_coeff[idof] * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
        }

        // Evaluate physical convective flux
        conv_phys_flux_at_q[iquad] = this->pde_physics_double->convective_flux (soln_at_q[iquad]);

        //Diffusion
        diffusive_phys_flux_at_q[iquad] = this->pde_physics_double->dissipative_flux(soln_at_q[iquad], aux_soln_at_q[iquad]);

        //Source
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            dealii::Point<dim> quad_point;
            for(int idim=0; idim<dim; idim++){
                quad_point[idim] = 0.0;
                for(unsigned int imetric_dof=0; imetric_dof<n_metric_dofs/dim; imetric_dof++){
                    quad_point[idim] += this->operators.mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][iquad][imetric_dof]
                                                        * mapping_support_points[idim][imetric_dof];
                }
            }
            source_at_q[iquad] = this->pde_physics_double->source_term (quad_point, soln_at_q[iquad], this->current_time);
        }
    }


    // Evaluate flux divergence by applying the Jacobian scaled physical divergence operator (physical_gradient) on the fluxes.
    // NOTE: the curvilinear split form is user defined incorporated in the assembly of the divergence operator.

    std::vector<realArray> flux_divergence(n_quad_pts);
    std::vector<realArray> divergence_diffusive_flux(n_quad_pts);
    for (int istate = 0; istate<nstate; ++istate) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            flux_divergence[iquad][istate] = 0.0;
            divergence_diffusive_flux[iquad][istate] = 0.0;
            for (unsigned int flux_basis=0; flux_basis<n_quad_pts; ++flux_basis) {
                if (this->all_parameters->use_split_form == true)
                {
                    for(int idim=0; idim<dim; idim++){
                        flux_divergence[iquad][istate] += 2* DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->convective_numerical_split_flux(soln_at_q[iquad],soln_at_q[flux_basis])[istate][idim] 
                                                        *  physical_gradient[istate][idim][iquad][flux_basis];
                        divergence_diffusive_flux[iquad][istate] += diffusive_phys_flux_at_q[flux_basis][istate][idim] 
                                                                  * physical_gradient[istate][idim][iquad][flux_basis];
                    }
                }
                else
                {
                    for(int idim=0; idim<dim; idim++){
                        flux_divergence[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate][idim] 
                                                        * physical_gradient[istate][idim][iquad][flux_basis];
                        divergence_diffusive_flux[iquad][istate] += diffusive_phys_flux_at_q[flux_basis][istate][idim] 
                                                                  * physical_gradient[istate][idim][iquad][flux_basis];
                    }
                }
            }
        }
    }



    // Strong form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        realtype rhs = 0;

        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Convective
            rhs = rhs - this->operators.vol_integral_basis[poly_degree][iquad][itest]  * flux_divergence[iquad][istate];

            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics. Since we used the auxiliary
            //// variable, put a negative here.
            rhs = rhs - this->operators.vol_integral_basis[poly_degree][iquad][itest]  * divergence_diffusive_flux[iquad][istate];

            // Source
            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
                rhs = rhs + this->operators.vol_integral_basis[poly_degree][iquad][itest] * source_at_q[iquad][istate] * determinant_Jacobian[iquad];
            }
        }

        local_rhs_int_cell(itest) += rhs;
    }
}


template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_explicit(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    (void) current_cell_index;
    using ADArray = std::array<FadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;

    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_dofs_cell, dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector<ADArray> soln_int(n_face_quad_pts);
    std::vector<ADArray> soln_ext(n_face_quad_pts);

    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<ADArrayTensor1> conv_phys_flux(n_face_quad_pts);

    // AD variable
    std::vector< FadType > soln_coeff_int(n_dofs_cell);
    const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(dof_indices_int[idof]);
        soln_coeff_int[idof].diff(idof, n_total_indep);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,FadType> normal_int = normals[iquad];
        const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;

        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }

        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
        dealii::Point<dim,FadType> ad_point;
        for (int d=0;d<dim;++d) { ad_point[d] = real_quad_point[d]; }
        this->pde_physics_fad->boundary_face_values (boundary_id, ad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        // Used for strong form
        // Which physical convective flux to use?
        conv_phys_flux[iquad] = this->pde_physics_fad->convective_flux (soln_int[iquad]);

        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
   for (int d=0; d<dim; d++) {
    diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
   }
        }
        diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int);

        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_auxiliary_flux(
            0.0, 0.0,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }

    // Boundary integral
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        FadType rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            const FadType flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux[iquad][istate]*normals[iquad];
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * flux_diff * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
            rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
        }
        // *******************

        local_rhs_int_cell(itest) += rhs.val();

        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                //residual_derivatives[idof] = rhs.fastAccessDx(idof);
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(dof_indices_int[itest], dof_indices_int, residual_derivatives);
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_explicit(
    const unsigned int iface, const unsigned int neighbor_iface, 
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int poly_degree, const unsigned int grid_degree,
    const dealii::FEFaceValuesBase<dim,dim>     &/*fe_values_int*/,
    const dealii::FEFaceValuesBase<dim,dim>     &/*fe_values_ext*/,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;
    //std::cout << "assembling face terms" << std::endl;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    // Currently assuming both interior and exterior cells have same poly order on face
    // In future will pass poly_degree_int and poly_degree_ext
    const unsigned int n_face_quad_pts = this->operators.face_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_vol = this->operators.volume_quadrature_collection[poly_degree].size();

    const unsigned int n_dofs_int = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;
    const unsigned int n_dofs_ext = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    // Question: Jacobian and normal should always be consistent between two elements
    // even for non-conforming meshes?
    // Answer: Yes. Please see surface conforming unit tests.


    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;


    //Compute metric terms (cofactor and normals)

    //get local cofactor matrix
    std::vector<std::vector<real>> mapping_support_points_int(dim);//mapping support points of interior cell
    std::vector<std::vector<real>> mapping_support_points_ext(dim);//mapping support points of exterior cell
    for(int idim=0; idim<dim; idim++){
        mapping_support_points_int[idim].resize(n_metric_dofs/dim);//there are n_metric_dofs/dim shape functions
        mapping_support_points_ext[idim].resize(n_metric_dofs/dim);
    }
    dealii::QGaussLobatto<dim> vol_GLL(grid_degree + 1);//Note that the mapping supp points are always GLL quad nodes
    //get the mapping support points int and ext
    for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val_int = (this->high_order_grid->volume_nodes[metric_dof_indices_int[idof]]);
            const unsigned int istate_int = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points_int[istate_int][igrid_node] += val_int * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate_int); 

            const real val_ext = (this->high_order_grid->volume_nodes[metric_dof_indices_ext[idof]]);
            const unsigned int istate_ext = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points_ext[istate_ext][igrid_node] += val_ext * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate_ext); 
        }
    }

    std::vector<dealii::FullMatrix<real>> metric_cofactor_face(n_face_quad_pts);//surface metric cofactor
    std::vector<real> determinant_Jacobian_face(n_face_quad_pts);//detemrinant metric Jacobian evaluated on face. NOTE: not the same as the surface JAcobian since that comes from normalizing the physical normal vector.
    for(unsigned int iquad=0;iquad<n_face_quad_pts; iquad++){
        metric_cofactor_face[iquad].reinit(dim, dim);
    }
    //surface metric cofactor
    this->operators.build_local_face_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, iface,
                                                                        n_face_quad_pts, n_metric_dofs / dim, mapping_support_points_int, 
                                                                        determinant_Jacobian_face, metric_cofactor_face);

    //get physical normal scaled by jac
    const dealii::Tensor<1,dim, real> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];

    std::vector<dealii::Tensor<1,dim, real> > normal_phys_int(n_face_quad_pts);
    std::vector<real> face_jac(n_face_quad_pts);//Surface determinant of metric Jacobian by definition: \hat{n}_m = \frac{1}{J^\Gamma} n_m.
    for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
        this->operators.compute_reference_to_physical(unit_normal_int, metric_cofactor_face[iquad], normal_phys_int[iquad]); 
        face_jac[iquad] = normal_phys_int[iquad].norm();
        normal_phys_int[iquad] /= face_jac[iquad];//normalize it to have physical unit normal.
    }

    std::vector<dealii::FullMatrix<real>> metric_cofactor_int(n_quad_pts_vol);//interior volume metric cofactor
    std::vector<dealii::FullMatrix<real>> metric_cofactor_ext(n_quad_pts_vol);//exterior volume metric cofactor
    std::vector<real> determinant_Jacobian(n_quad_pts_vol);//note we dont actually need this so use same for int and ext
    for(unsigned int iquad=0;iquad<n_quad_pts_vol; iquad++){
        metric_cofactor_int[iquad].reinit(dim, dim);
        metric_cofactor_ext[iquad].reinit(dim, dim);
    }
    //interior and exterior volume cofactors
    this->operators.build_local_vol_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, n_quad_pts_vol, n_metric_dofs/dim, mapping_support_points_int, determinant_Jacobian, metric_cofactor_int);
    this->operators.build_local_vol_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, n_quad_pts_vol, n_metric_dofs/dim, mapping_support_points_ext, determinant_Jacobian, metric_cofactor_ext);


    //Initialize the solution etc

    // Solution coefficients
    std::vector<ADtype> soln_coeff_int(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext(n_dofs_ext);
    std::vector< dealii::Tensor<1,dim,real>> aux_soln_coeff_int(n_dofs_int);
    std::vector< dealii::Tensor<1,dim,real>> aux_soln_coeff_ext(n_dofs_ext);

    std::vector<ADArrayTensor1> conv_ref_flux_int_on_face(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_ref_flux_ext_on_face(n_face_quad_pts);

    // AD variable
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(dof_indices_int[idof]);
        for(int idim=0; idim<dim; idim++){
            aux_soln_coeff_int[idof][idim] = DGBase<dim,real,MeshType>::auxiliary_solution[idim](dof_indices_int[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext[idof] = DGBase<dim,real,MeshType>::solution(dof_indices_ext[idof]);
        for(int idim=0; idim<dim; idim++){
            aux_soln_coeff_ext[idof][idim] = DGBase<dim,real,MeshType>::auxiliary_solution[idim](dof_indices_ext[idof]);
        }
    }


    std::vector< ADArray > soln_at_q_int(n_quad_pts_vol);
    std::vector< ADArrayTensor1 > aux_soln_at_q_int(n_quad_pts_vol); //auxiliary sol at flux nodes
    std::vector< ADArray > soln_at_q_ext(n_quad_pts_vol);
    std::vector< ADArrayTensor1 > aux_soln_at_q_ext(n_quad_pts_vol); //auxiliary sol at flux nodes
    for (unsigned int iquad=0; iquad<n_quad_pts_vol; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_at_q_int[iquad][istate]      = 0;
            soln_at_q_ext[iquad][istate]      = 0;
            aux_soln_at_q_int[iquad][istate]  = 0;
            aux_soln_at_q_ext[iquad][istate]  = 0;
        }
    }

    std::vector< ADArrayTensor1 > conv_ref_flux_vol_int(n_quad_pts_vol);
    std::vector< ADArrayTensor1 > conv_ref_flux_vol_ext(n_quad_pts_vol);
    std::vector< ADArrayTensor1 > ref_diff_flux_vol_int(n_quad_pts_vol);
    std::vector< ADArrayTensor1 > ref_diff_flux_vol_ext(n_quad_pts_vol);
    //Obtain the volume REFERENCE Fluxes.
    //First for the interior volume reference fluxes.
    for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_at_q_int[iquad][istate]      += soln_coeff_int[idof]     * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
            aux_soln_at_q_int[iquad][istate]  += aux_soln_coeff_int[idof] * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
        }
        //get phys flux in vol quad
        ADArrayTensor1 phys_flux_conv      = this->pde_physics_double->convective_flux  (soln_at_q_int[iquad]);
        ADArrayTensor1 phys_flux_diffusive = this->pde_physics_double->dissipative_flux (soln_at_q_int[iquad], aux_soln_at_q_int[iquad]);
        //transform to a reference flux
        for(int istate=0; istate<nstate; istate++){
            this->operators.compute_physical_to_reference(phys_flux_conv[istate],      metric_cofactor_int[iquad], conv_ref_flux_vol_int[iquad][istate]);
            this->operators.compute_physical_to_reference(phys_flux_diffusive[istate], metric_cofactor_int[iquad], ref_diff_flux_vol_int[iquad][istate]);
        }
    }
    //Next for the exterior volume reference fluxes.
    for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_at_q_ext[iquad][istate]      += soln_coeff_ext[idof]     * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
            aux_soln_at_q_ext[iquad][istate]  += aux_soln_coeff_ext[idof] * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
        }
        //get phys flux in vol quad
        ADArrayTensor1 phys_flux_conv      = this->pde_physics_double->convective_flux  (soln_at_q_ext[iquad]);
        ADArrayTensor1 phys_flux_diffusive = this->pde_physics_double->dissipative_flux (soln_at_q_ext[iquad], aux_soln_at_q_ext[iquad]);
        //transform to a reference flux
        for(int istate=0; istate<nstate; istate++){
            this->operators.compute_physical_to_reference(phys_flux_conv[istate],      metric_cofactor_ext[iquad], conv_ref_flux_vol_ext[iquad][istate]);
            this->operators.compute_physical_to_reference(phys_flux_diffusive[istate], metric_cofactor_ext[iquad], ref_diff_flux_vol_ext[iquad][istate]);
        }
    }

    std::vector<ADArrayTensor1> conv_ref_flux_interp_to_face_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_ref_flux_interp_to_face_ext(n_face_quad_pts);
    std::vector<ADArrayTensor1> diffusive_ref_flux_interp_to_face_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> diffusive_ref_flux_interp_to_face_ext(n_face_quad_pts);
    //Interpolate the volume REFERENCE fluxes to the facet. 
    //NOTE: this is not the same as the flux being evaluated on the facet for the nonlinear case!
    //i.e. whether the flux itself is nonlinear, OR the grid is nonlinear.
    for(int istate=0; istate<nstate; istate++){
        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            conv_ref_flux_interp_to_face_int[iquad][istate]      = 0;
            conv_ref_flux_interp_to_face_ext[iquad][istate]      = 0;
            diffusive_ref_flux_interp_to_face_int[iquad][istate] = 0;
            diffusive_ref_flux_interp_to_face_ext[iquad][istate] = 0;
            for (unsigned int iflux=0; iflux<n_quad_pts_vol; ++iflux) {
                conv_ref_flux_interp_to_face_int[iquad][istate] += conv_ref_flux_vol_int[iflux][istate] 
                                                                 * this->operators.flux_basis_at_facet_cubature[poly_degree][istate][iface][iquad][iflux];
                conv_ref_flux_interp_to_face_ext[iquad][istate] += conv_ref_flux_vol_ext[iflux][istate] 
                                                                 * this->operators.flux_basis_at_facet_cubature[poly_degree][istate][neighbor_iface][iquad][iflux];
                diffusive_ref_flux_interp_to_face_int[iquad][istate] += ref_diff_flux_vol_int[iflux][istate] 
                                                                      * this->operators.flux_basis_at_facet_cubature[poly_degree][istate][iface][iquad][iflux];
                diffusive_ref_flux_interp_to_face_ext[iquad][istate] += ref_diff_flux_vol_ext[iflux][istate] 
                                                                      * this->operators.flux_basis_at_facet_cubature[poly_degree][istate][neighbor_iface][iquad][iflux];
            }
        }
    }


    // Interpolate solution to the facet quadrature points
    std::vector< ADArray > soln_int(n_face_quad_pts);
    std::vector< ADArrayTensor1> aux_soln_int(n_face_quad_pts);
    std::vector< ADArray > soln_ext(n_face_quad_pts);
    std::vector< ADArrayTensor1> aux_soln_ext(n_face_quad_pts);

    //convective surface numerical flux PHYSICAL
    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    //diffusive surface numerical flux PHYSICAL
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts);


    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]     = 0;
            aux_soln_int[iquad][istate] = 0;
            soln_ext[iquad][istate]     = 0;
            aux_soln_ext[iquad][istate] = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normal_phys_int[iquad];//PHYSICAL UNIT Normal

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof]     * this->operators.basis_at_facet_cubature[poly_degree][iface][iquad][idof];
            aux_soln_int[iquad][istate]  += aux_soln_coeff_int[idof] * this->operators.basis_at_facet_cubature[poly_degree][iface][iquad][idof];
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_ext[iquad][istate]      += soln_coeff_ext[idof]     * this->operators.basis_at_facet_cubature[poly_degree][neighbor_iface][iquad][idof];
            aux_soln_ext[iquad][istate]  += aux_soln_coeff_ext[idof] * this->operators.basis_at_facet_cubature[poly_degree][neighbor_iface][iquad][idof];
        }

        //Evaluate the convective and diffusive numerical fluxes, along with any surface flux splitting.
        //Convective numerical flux. 
        conv_num_flux_dot_n[iquad] = this->conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);


        //Series of checks because I don't have curvilinear Euler uncollocated surface split working yet.
        if (this->all_parameters->use_split_form == true && this->all_parameters->use_curvilinear_split_form == false){
            ADArrayTensor1 conv_surface_ref_flux;//surface reference flux.
            ADArrayTensor1 phys_flux = this->pde_physics_double->convective_flux (soln_int[iquad]);
            //get Surface reference flux
            for(int istate=0; istate<nstate; istate++){
                this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_face[iquad], conv_surface_ref_flux[istate]);
            }
            //get surface splitting
            conv_ref_flux_int_on_face[iquad] = this->pde_physics_double->convective_surface_numerical_split_flux(conv_surface_ref_flux, conv_ref_flux_interp_to_face_int[iquad]); 
            phys_flux = this->pde_physics_double->convective_flux (soln_ext[iquad]);
            for(int istate=0; istate<nstate; istate++){
                this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_face[iquad], conv_surface_ref_flux[istate]);
            }
            conv_ref_flux_ext_on_face[iquad] = this->pde_physics_double->convective_surface_numerical_split_flux(conv_surface_ref_flux, conv_ref_flux_interp_to_face_ext[iquad]); 
        } else if(this->all_parameters->use_split_form == false && this->all_parameters->use_curvilinear_split_form == true){
            ADArrayTensor1 conv_surface_ref_flux;
            ADArrayTensor1 phys_flux = this->pde_physics_double->convective_flux (soln_int[iquad]);
            //get surface reference flux anf do curvilinear surface splitting simultaneously
            for(int istate=0; istate<nstate; istate++){
                this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_face[iquad], conv_surface_ref_flux[istate]);
                conv_ref_flux_int_on_face[iquad][istate] = 0.5 * (conv_surface_ref_flux[istate] + conv_ref_flux_interp_to_face_int[iquad][istate]);
            }

            phys_flux = this->pde_physics_double->convective_flux (soln_ext[iquad]);
            for(int istate=0; istate<nstate; istate++){
                this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_face[iquad], conv_surface_ref_flux[istate]);
                conv_ref_flux_ext_on_face[iquad][istate] = 0.5 * (conv_surface_ref_flux[istate] + conv_ref_flux_interp_to_face_ext[iquad][istate]);
            }

        } else {
            //Standard DG interp VOLUME REFERENCE flux to the surface. 
            //NOTE: This is NOT the same as a surface flux in the nonlinear case. 
            //It needs to be the VOLUME REFERENCE flux interpolated to the surface for the 
            //correct orders of convergence.
            conv_ref_flux_int_on_face[iquad] = conv_ref_flux_interp_to_face_int[iquad];
            conv_ref_flux_ext_on_face[iquad] = conv_ref_flux_interp_to_face_ext[iquad];
        }

        diss_auxi_num_flux_dot_n[iquad] = this->diss_num_flux_double->evaluate_auxiliary_flux(
            0.0, 0.0,
            soln_int[iquad], soln_ext[iquad],
            aux_soln_int[iquad], aux_soln_ext[iquad],
            normal_int, penalty);
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        ADtype rhs = 0.0;
        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            const ADtype flux_diff = face_jac[iquad]*conv_num_flux_dot_n[iquad][istate] - conv_ref_flux_int_on_face[iquad][istate]*unit_normal_int;
            rhs = rhs - this->operators.face_integral_basis[poly_degree][iface][iquad][itest_int] * flux_diff;


            // Diffusive
            const ADtype diffusive_diff = face_jac[iquad]*diss_auxi_num_flux_dot_n[iquad][istate] - diffusive_ref_flux_interp_to_face_int[iquad][istate]*unit_normal_int;
            rhs = rhs - this->operators.face_integral_basis[poly_degree][iface][iquad][itest_int] * diffusive_diff;
        }

        local_rhs_int_cell(itest_int) += rhs;
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        ADtype rhs = 0.0;
        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            const ADtype flux_diff = face_jac[iquad]*(-conv_num_flux_dot_n[iquad][istate]) - conv_ref_flux_ext_on_face[iquad][istate]*(-unit_normal_int);
            rhs = rhs - this->operators.face_integral_basis[poly_degree][neighbor_iface][iquad][itest_ext] * flux_diff;

            // Diffusive
            const ADtype diffusive_diff = face_jac[iquad]*(-diss_auxi_num_flux_dot_n[iquad][istate]) - diffusive_ref_flux_interp_to_face_ext[iquad][istate]*(-unit_normal_int);
            rhs = rhs - this->operators.face_integral_basis[poly_degree][neighbor_iface][iquad][itest_ext] * diffusive_diff;
        }

        local_rhs_ext_cell(itest_ext) += rhs;
    }
}

// using default MeshType = Triangulation
// 1D: dealii::Triangulation<dim>;
// OW: dealii::parallel::distributed::Triangulation<dim>;
template class DGStrong <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class DGStrong <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class DGStrong <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace

