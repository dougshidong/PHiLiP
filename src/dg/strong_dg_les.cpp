#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>

#include "ADTypes.hpp"

#include <deal.II/fe/fe_dgq.h> // Used for flux interpolation

// TO DO: review the above includes

#include "strong_dg_les.hpp"

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
DGStrongLES<dim,nstate,real,MeshType>::DGStrongLES(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGStrong<dim,nstate,real,MeshType>::DGStrong(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
    // , do_compute_filtered_solution(this->all_parameters->physics_model_param.do_compute_filtered_solution)
    // , apply_modal_high_pass_filter_on_filtered_solution(this->all_parameters->physics_model_param.apply_modal_high_pass_filter_on_filtered_solution)
    // , poly_degree_max_large_scales(this->all_parameters->physics_model_param.poly_degree_max_large_scales)
{ 
#if PHILIP_DIM==3
    // // TO DO: move this if statement logic to the DGFactory
    // if(((pde_type==PDE_enum::physics_model || pde_type==PDE_enum::physics_model_filtered) && 
    //     (model_type==Model_enum::large_eddy_simulation || model_type==Model_enum::navier_stokes_model))) 
    // {
        if constexpr (dim+2==nstate) {
            this->pde_model_les_double = std::dynamic_pointer_cast<Physics::LargeEddySimulationBase<dim,dim+2,real>>(this->pde_model_double);
        }
    // }
    // else if((pde_type==PDE_enum::physics_model  || pde_type==PDE_enum::physics_model_filtered) && 
    //          (model_type!=Model_enum::large_eddy_simulation && model_type!=Model_enum::navier_stokes_model)) 
    // {
    //     std::cout << "Invalid convective numerical flux for physics_model and/or corresponding baseline_physics_type" << std::endl;
    //     if(nstate!=(dim+2)) std::cout << "Error: Cannot create_euler_based_convective_numerical_flux() for nstate_baseline_physics != nstate." << std::endl;
    //     std::abort();
    // }
#endif

    // // TO DO: move this to the factory
    // // Determine if the mean strain rate tensor must be computed
    // using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    // const PDE_enum pde_type = this->all_param.pde_type;
    // if(pde_type == PDE_enum::physics_model  || pde_type == PDE_enum::physics_model_filtered) {
        
    // }
}

// Destructor
template <int dim, int nstate, typename real, typename MeshType>
DGStrongLES<dim,nstate,real,MeshType>::~DGStrongLES()
{
    pcout << "Destructing DGStrongLES..." << std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrongLES<dim,nstate,real,MeshType>::allocate_model_variables()
{
    // allocate all model variables for each ModelBase object
    // -- double
    this->pde_model_double->cellwise_poly_degree.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    this->pde_model_double->cellwise_volume.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrongLES<dim,nstate,real,MeshType>::update_model_variables()
{
    // allocate/reinit the model variables
    allocate_model_variables();

    // TO DO: should only call this once if no hp-adaptation
    update_cellwise_volume_and_poly_degree();

    // update the cellwise mean quantities
    update_cellwise_mean_quantities();
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrongLES<dim,nstate,real,MeshType>::update_cellwise_volume_and_poly_degree()
{
    // get FEValues of volume
    const auto mapping = (*(this->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values;
    dealii::hp::FEValues<dim,dim> fe_values_collection_volume (mapping_collection, 
                                                               this->fe_collection, 
                                                               this->volume_quadrature_collection, 
                                                               update_flags);

    // loop through all cells
    for (auto cell : this->dof_handler.active_cell_iterators()) {
        if (!(cell->is_locally_owned() || cell->is_ghost())) continue;

        // get FEValues of volume for current cell
        const int i_fele = cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;
        fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

        // get cell polynomial degree
        const dealii::FESystem<dim,dim> &fe_high = this->fe_collection[i_fele];
        const unsigned int cell_poly_degree = fe_high.tensor_degree();

        // get cell volume
        const dealii::Quadrature<dim> &quadrature = fe_values_volume.get_quadrature();
        const unsigned int n_quad_pts = quadrature.size();
        const std::vector<real> &JxW = fe_values_volume.get_JxW_values();
        real cell_volume_estimate = 0.0;
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            cell_volume_estimate = cell_volume_estimate + JxW[iquad];
        }
        const real cell_volume = cell_volume_estimate;
        
        // get cell index for assignment
        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        // const dealii::types::global_dof_index cell_index = cell->global_active_cell_index(); // https://www.dealii.org/current/doxygen/deal.II/classCellAccessor.html

        // assign values
        // -- double
        this->pde_model_double->cellwise_poly_degree[cell_index] = cell_poly_degree;
        this->pde_model_double->cellwise_volume[cell_index] = cell_volume;
    }
    this->pde_model_double->cellwise_poly_degree.update_ghost_values();
    this->pde_model_double->cellwise_volume.update_ghost_values();
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrongLES<dim,nstate,real,MeshType>::update_cellwise_mean_quantities()
{ 
    // do nothing
}


template <int dim, int nstate, typename real, typename MeshType>
DGStrongLES_ShearImproved<dim,nstate,real,MeshType>::DGStrongLES_ShearImproved(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGStrongLES<dim,nstate,real,MeshType>::DGStrongLES(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
    // , do_compute_filtered_solution(this->all_parameters->physics_model_param.do_compute_filtered_solution)
    // , apply_modal_high_pass_filter_on_filtered_solution(this->all_parameters->physics_model_param.apply_modal_high_pass_filter_on_filtered_solution)
    // , poly_degree_max_large_scales(this->all_parameters->physics_model_param.poly_degree_max_large_scales)
{ 
    // do nothing
}

// Destructor
template <int dim, int nstate, typename real, typename MeshType>
DGStrongLES_ShearImproved<dim,nstate,real,MeshType>::~DGStrongLES_ShearImproved()
{
    pcout << "Destructing DGStrongLES_ShearImproved..." << std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrongLES_ShearImproved<dim,nstate,real,MeshType>::allocate_model_variables()
{
    // allocate all model variables for each ModelBase object
    // -- double
    this->pde_model_double->cellwise_poly_degree.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    this->pde_model_double->cellwise_volume.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    // allocate the cellwise mean strain rate tensor magnitude distributed vector
    // -- double
    this->pde_model_double->cellwise_mean_strain_rate_tensor_magnitude.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrongLES_ShearImproved<dim,nstate,real,MeshType>::update_cellwise_mean_quantities()
{
    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10; // set to zero to reduce computational cost; currently set to 10 for peace of mind

    // Set the quadrature of size dim and 1D for sum-factorization.
    dealii::QGauss<dim> quad_extra(this->max_degree+1+overintegrate);
    dealii::QGauss<1> quad_extra_1D(this->max_degree+1+overintegrate);

    const unsigned int n_quad_pts = quad_extra.size();
    const unsigned int grid_degree = this->high_order_grid->fe_system.tensor_degree();
    const unsigned int poly_degree = this->max_degree;
    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, grid_degree); 
    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, poly_degree, grid_degree);
    // Build basis function volume operator and gradient operator from 1D finite element for 1 state.
    soln_basis.build_1D_volume_operator(this->oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    soln_basis.build_1D_gradient_operator(this->oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    // Build mapping shape functions operators using the oneD high_ordeR_grid finite element
    mapping_basis.build_1D_shape_functions_at_grid_nodes(this->high_order_grid->oneD_fe_system, this->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(this->high_order_grid->oneD_fe_system, quad_extra_1D, this->oneD_face_quadrature);
    const std::vector<double> &quad_weights = quad_extra.get_weights();
    // If in the future we need the physical quadrature node location, turn these flags to true and the constructor will
    // automatically compute it for you. Currently set to false as to not compute extra unused terms.
    const bool store_vol_flux_nodes = false;//currently doesn't need the volume physical nodal position
    const bool store_surf_flux_nodes = false;//currently doesn't need the surface physical nodal position

    const unsigned int n_dofs = this->fe_collection[poly_degree].n_dofs_per_cell();
    const unsigned int n_shape_fns = n_dofs / nstate;
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs);
    auto metric_cell = this->high_order_grid->dof_handler_grid.begin_active();
    // Changed for loop to update metric_cell.
    for (auto cell = this->dof_handler.begin_active(); cell!= this->dof_handler.end(); ++cell, ++metric_cell) {
        if (!(cell->is_locally_owned() || cell->is_ghost())) continue;
        cell->get_dof_indices (dofs_indices);

        // Initialize the strain rate tensor integral (for computing the mean) to zero
        dealii::Tensor<2,dim,double> cell_strain_rate_tensor_integral;
        for (int d1=0; d1<dim; ++d1) {
            for (int d2=0; d2<dim; ++d2) {
                cell_strain_rate_tensor_integral[d1][d2] = 0.0;
            }
        }

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, grid_degree, store_vol_flux_nodes, store_surf_flux_nodes);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            this->all_parameters->use_invariant_curl_form);

        // Fetch the modal soln coefficients
        // We immediately separate them by state as to be able to use sum-factorization
        // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
        // mult would sum the states at the quadrature point.
        // That is why the basis functions are based off the 1state oneD fe_collection.
        std::array<std::vector<double>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs; ++idof) {
            const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
            }
         
            soln_coeff[istate][ishape] = this->solution(dofs_indices[idof]);
        }
        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<std::vector<double>,nstate> soln_at_q_vect;
        std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> soln_grad_at_q_vect;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q_vect[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_vect[istate],
                                             soln_basis.oneD_vol_operator);
            // We need to first compute the reference gradient of the solution, then transform that to a physical gradient.
            dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
            for(int idim=0; idim<dim; idim++){
                ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
                soln_grad_at_q_vect[istate][idim].resize(n_quad_pts);
            }
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.
            soln_basis.gradient_matrix_vector_mult_1D(soln_coeff[istate], ref_gradient_basis_fns_times_soln,
                                                      soln_basis.oneD_vol_operator,
                                                      soln_basis.oneD_grad_operator);
            // Transform the reference gradient into a physical gradient operator.
            for(int idim=0; idim<dim; idim++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(int jdim=0; jdim<dim; jdim++){
                        //transform into the physical gradient
                        soln_grad_at_q_vect[istate][idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                                  * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                                  / metric_oper.det_Jac_vol[iquad];
                    }
                }
            }
        }

        // -- Solution at legendre poly
        std::array<std::vector<real>,nstate> legendre_soln_at_q_vect;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_q_vect; // legendre auxiliary sol at flux nodes
        if(this->do_compute_filtered_solution) {
            const unsigned int p_min_filtered = this->poly_degree_max_large_scales + 1;
            //==================================================
            // GET THE PRIMITIVE SOLUTION
            //==================================================
            std::array<std::vector<real>,nstate> primitive_soln_at_q;
            std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_q; // primitive auxiliary sol at flux nodes
            // Resize the primitive soln arrays
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_q[istate].resize(n_quad_pts);
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_q[istate][idim].resize(n_quad_pts);
                }
            }
            // Compute the primitive soln at all iquad and fill arrays
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                // extract conservative soln state
                std::array<real,nstate> soln_state;
                std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
                for(int istate=0; istate<nstate; istate++){
                    soln_state[istate] = soln_at_q_vect[istate][iquad];
                    for(int idim=0; idim<dim; idim++){
                        aux_soln_state[istate][idim] = soln_grad_at_q_vect[istate][idim][iquad];
                    }
                }
                // compute primitive soln state from conservative
                std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
                std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
                // store primitive soln at quadrature point
                for(int istate=0; istate<nstate; istate++){
                    primitive_soln_at_q[istate][iquad] = primitive_soln_state[istate];
                    for(int idim=0; idim<dim; idim++){
                        primitive_aux_soln_at_q[istate][idim][iquad] = primitive_aux_soln_state[istate][idim];
                    }
                }
            }
            
            //==================================================
            // PROJECT TO LEGENDRE BASIS AND MODALLY FILTER
            //==================================================
            // -- Primitive solution at legendre poly
            std::array<std::vector<real>,nstate> primitive_legendre_soln_at_q;
            std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_q; // legendre auxiliary sol at quad points

            // Details: this projects to Legendre basis, truncates, then interpolates back to quad nodes.
            // -- Constructor for tensor product polynomials based on Polynomials::Legendre interpolation. 
            dealii::FE_DGQLegendre<1,1> legendre_poly_1D(poly_degree);
            // -- Projection operator for legendre basis
            OPERATOR::vol_projection_operator<dim,2*dim> legendre_soln_basis_projection_oper(1, poly_degree, grid_degree);
            legendre_soln_basis_projection_oper.build_1D_volume_operator(legendre_poly_1D, quad_extra_1D);
            // -- Legendre basis functions 
            OPERATOR::basis_functions<dim,2*dim> legendre_soln_basis(1, poly_degree, grid_degree);
            legendre_soln_basis.build_1D_volume_operator(legendre_poly_1D, quad_extra_1D);
            legendre_soln_basis.build_1D_gradient_operator(legendre_poly_1D, quad_extra_1D);
            for(int istate=0; istate<nstate; istate++){
                //==================================================
                // Solution and Solution Gradient
                //==================================================
                // -- (1) Project to Legendre basis
                std::vector<real> legendre_soln_coeff(n_shape_fns);
                legendre_soln_basis_projection_oper.matrix_vector_mult_1D(primitive_soln_at_q[istate], legendre_soln_coeff,
                                                                          legendre_soln_basis_projection_oper.oneD_vol_operator);
                // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
                if(this->apply_modal_high_pass_filter_on_filtered_solution && (istate!=0 && istate!=(nstate-1))) {
                    for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                        if(ishape < p_min_filtered){
                            legendre_soln_coeff[ishape] = 0.0;
                        }
                    }    
                }
                // -- (3) Interpolate filtered solution back to quadrature points
                primitive_legendre_soln_at_q[istate].resize(n_quad_pts);
                legendre_soln_basis.matrix_vector_mult_1D(legendre_soln_coeff, primitive_legendre_soln_at_q[istate],
                                                          legendre_soln_basis.oneD_vol_operator);

                // We need to first compute the reference gradient of the solution, then transform that to a physical gradient.
                dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
                for(int idim=0; idim<dim; idim++){
                    ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
                    primitive_legendre_aux_soln_at_q[istate][idim].resize(n_quad_pts);
                }
                // Apply gradient of reference basis functions on the solution at volume cubature nodes.
                legendre_soln_basis.gradient_matrix_vector_mult_1D(legendre_soln_coeff, ref_gradient_basis_fns_times_soln,
                                                                   legendre_soln_basis.oneD_vol_operator,
                                                                   legendre_soln_basis.oneD_grad_operator);
                // Transform the reference gradient into a physical gradient operator.
                for(int idim=0; idim<dim; idim++){
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        for(int jdim=0; jdim<dim; jdim++){
                            //transform into the physical gradient
                            primitive_legendre_aux_soln_at_q[istate][idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                                                   * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                                                   / metric_oper.det_Jac_vol[iquad];
                        }
                    }
                }
                //==================================================
            }
            //=======================================================
            // CONVERT PRIMITIVE LEGENDRE SOLUTION TO CONSERVATIVE
            //=======================================================
            // Resize the conservative soln arrays
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_q_vect[istate].resize(n_quad_pts);
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_q_vect[istate][idim].resize(n_quad_pts);
                }
            }
            // Compute the primitive soln at all iquad and fill arrays
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                // extract conservative soln state
                std::array<real,nstate> primitive_legendre_soln_state;
                std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
                for(int istate=0; istate<nstate; istate++){
                    primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_q[istate][iquad];
                    for(int idim=0; idim<dim; idim++){
                        primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_q[istate][idim][iquad];
                    }
                }
                // compute conservative soln state from primitive
                std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
                std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
                // store conservative soln at quadrature point
                for(int istate=0; istate<nstate; istate++){
                    legendre_soln_at_q_vect[istate][iquad] = legendre_soln_state[istate];
                    for(int idim=0; idim<dim; idim++){
                        legendre_aux_soln_at_q_vect[istate][idim][iquad] = legendre_aux_soln_state[istate][idim];
                    }
                }
            }
        }

        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_at_q;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;
            // Extract solution and gradient in a way that the physics can use them.
            for(int istate=0; istate<nstate; istate++){
                if(this->do_compute_filtered_solution) soln_at_q[istate] = legendre_soln_at_q_vect[istate][iquad];
                else soln_at_q[istate] = soln_at_q_vect[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    if(this->do_compute_filtered_solution) soln_grad_at_q[istate][idim] = legendre_aux_soln_at_q_vect[istate][idim][iquad];
                    else soln_grad_at_q[istate][idim] = soln_grad_at_q_vect[istate][idim][iquad];
                }
            }

            // Get strain rate tensor
            const dealii::Tensor<2,dim,double> strain_rate_tensor = this->pde_model_les_double->navier_stokes_physics->compute_strain_rate_tensor_from_conservative(soln_at_q,soln_grad_at_q);
            for (int d1=0; d1<dim; ++d1) {
                for (int d2=0; d2<dim; ++d2) {
                    cell_strain_rate_tensor_integral[d1][d2] += strain_rate_tensor[d1][d2] * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
                }
            }
        }
        
        // get cell index
        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        // get mean strain rate tensor
        dealii::Tensor<2,dim,double> cell_mean_strain_rate_tensor;
        for (int d1=0; d1<dim; ++d1) {
            for (int d2=0; d2<dim; ++d2) {
                cell_mean_strain_rate_tensor[d1][d2] = cell_strain_rate_tensor_integral[d1][d2];
                cell_mean_strain_rate_tensor[d1][d2] /= this->pde_model_double->cellwise_volume[cell_index]; // divide by current cell volume
            }
        }
        // update the cellwise mean strain rate tensor magnitude at the current cell
        const double cell_mean_strain_rate_tensor_magnitude = this->pde_model_les_double->navier_stokes_physics->get_tensor_magnitude(cell_mean_strain_rate_tensor);
        this->pde_model_double->cellwise_mean_strain_rate_tensor_magnitude[cell_index] = cell_mean_strain_rate_tensor_magnitude;
    }
    // update ghost values
    this->pde_model_double->cellwise_mean_strain_rate_tensor_magnitude.update_ghost_values();
}

template <int dim, int nstate, typename real, typename MeshType>
DGStrongLES_DynamicSmagorinsky<dim,nstate,real,MeshType>::DGStrongLES_DynamicSmagorinsky(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGStrongLES<dim,nstate,real,MeshType>::DGStrongLES(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
{ 
    // do nothing
}

// Destructor
template <int dim, int nstate, typename real, typename MeshType>
DGStrongLES_DynamicSmagorinsky<dim,nstate,real,MeshType>::~DGStrongLES_DynamicSmagorinsky()
{
    pcout << "Destructing DGStrongLES_DynamicSmagorinsky..." << std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrongLES_DynamicSmagorinsky<dim,nstate,real,MeshType>::allocate_model_variables()
{
    // allocate all model variables for each ModelBase object
    // -- double
    this->pde_model_double->cellwise_poly_degree.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    this->pde_model_double->cellwise_volume.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
    // allocate the cellwise dynamic Smagorinsky model constant filter width squared
    // -- double
    this->pde_model_double->dynamic_smagorinsky_model_constant_times_filter_width_sqr.reinit(this->triangulation->n_active_cells(), this->mpi_communicator);
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrongLES_DynamicSmagorinsky<dim,nstate,real,MeshType>::update_cellwise_mean_quantities()
{
    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10; // set to zero to reduce computational cost; currently set to 10 for peace of mind

    // Set the quadrature of size dim and 1D for sum-factorization.
    dealii::QGauss<dim> quad_extra(this->max_degree+1+overintegrate);
    dealii::QGauss<1> quad_extra_1D(this->max_degree+1+overintegrate);

    const unsigned int n_quad_pts = quad_extra.size();
    const unsigned int grid_degree = this->high_order_grid->fe_system.tensor_degree();
    const unsigned int poly_degree = this->max_degree;
    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, grid_degree); 
    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, poly_degree, grid_degree);
    // Build basis function volume operator and gradient operator from 1D finite element for 1 state.
    soln_basis.build_1D_volume_operator(this->oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    soln_basis.build_1D_gradient_operator(this->oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    // Build mapping shape functions operators using the oneD high_ordeR_grid finite element
    mapping_basis.build_1D_shape_functions_at_grid_nodes(this->high_order_grid->oneD_fe_system, this->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(this->high_order_grid->oneD_fe_system, quad_extra_1D, this->oneD_face_quadrature);
    const std::vector<double> &quad_weights = quad_extra.get_weights();
    // If in the future we need the physical quadrature node location, turn these flags to true and the constructor will
    // automatically compute it for you. Currently set to false as to not compute extra unused terms.
    const bool store_vol_flux_nodes = false;//currently doesn't need the volume physical nodal position
    const bool store_surf_flux_nodes = false;//currently doesn't need the surface physical nodal position

    const unsigned int n_dofs = this->fe_collection[poly_degree].n_dofs_per_cell();
    const unsigned int n_shape_fns = n_dofs / nstate;
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs);
    auto metric_cell = this->high_order_grid->dof_handler_grid.begin_active();
    // Changed for loop to update metric_cell.
    for (auto cell = this->dof_handler.begin_active(); cell!= this->dof_handler.end(); ++cell, ++metric_cell) {
        if (!(cell->is_locally_owned() || cell->is_ghost())) continue;
        cell->get_dof_indices (dofs_indices);

        // Initialize the matrix product integrals (for computing the means) to zero
        real cell_matrix_L_times_matrix_M_integral = 0.0;
        real cell_matrix_M_times_matrix_M_integral = 0.0;

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, grid_degree, store_vol_flux_nodes, store_surf_flux_nodes);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            this->all_parameters->use_invariant_curl_form);

        // Fetch the modal soln coefficients
        // We immediately separate them by state as to be able to use sum-factorization
        // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
        // mult would sum the states at the quadrature point.
        // That is why the basis functions are based off the 1state oneD fe_collection.
        std::array<std::vector<double>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs; ++idof) {
            const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
            }
         
            soln_coeff[istate][ishape] = this->solution(dofs_indices[idof]);
        }
        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<std::vector<double>,nstate> soln_at_q_vect;
        std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> soln_grad_at_q_vect;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q_vect[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_vect[istate],
                                             soln_basis.oneD_vol_operator);
            // We need to first compute the reference gradient of the solution, then transform that to a physical gradient.
            dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
            for(int idim=0; idim<dim; idim++){
                ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
                soln_grad_at_q_vect[istate][idim].resize(n_quad_pts);
            }
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.
            soln_basis.gradient_matrix_vector_mult_1D(soln_coeff[istate], ref_gradient_basis_fns_times_soln,
                                                      soln_basis.oneD_vol_operator,
                                                      soln_basis.oneD_grad_operator);
            // Transform the reference gradient into a physical gradient operator.
            for(int idim=0; idim<dim; idim++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(int jdim=0; jdim<dim; jdim++){
                        //transform into the physical gradient
                        soln_grad_at_q_vect[istate][idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                                  * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                                  / metric_oper.det_Jac_vol[iquad];
                    }
                }
            }
        }

        // -- Solution at legendre poly
        std::array<std::vector<real>,nstate> legendre_soln_at_q_vect;
        std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> legendre_aux_soln_at_q_vect; // legendre auxiliary sol at flux nodes
        dealii::Tensor<2,dim,std::vector<real>> legendre_matrix_L_component_at_q_vect;
        dealii::Tensor<2,dim,std::vector<real>> legendre_matrix_M_component_at_q_vect;
        /*if(this->do_compute_filtered_solution) {*/
            const unsigned int p_min_filtered = this->poly_degree_max_large_scales + 1;
            //==================================================
            // GET THE PRIMITIVE SOLUTION AND DSM MATRICES
            //==================================================
            std::array<std::vector<real>,nstate> primitive_soln_at_q;
            std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_aux_soln_at_q; // primitive auxiliary sol at flux nodes
            dealii::Tensor<2,dim,std::vector<real>> matrix_L_component_at_q;
            dealii::Tensor<2,dim,std::vector<real>> matrix_M_component_at_q;
            
            // Resize the primitive soln arrays
            for(int istate=0; istate<nstate; istate++){
                primitive_soln_at_q[istate].resize(n_quad_pts);
                for(int idim=0; idim<dim; idim++){
                    primitive_aux_soln_at_q[istate][idim].resize(n_quad_pts);
                }
            }
            for(int jdim=0; jdim<dim; jdim++){
                for(int idim=0; idim<dim; idim++){
                    matrix_L_component_at_q[jdim][idim].resize(n_quad_pts);
                    matrix_M_component_at_q[jdim][idim].resize(n_quad_pts);
                }
            }
            
            // Compute the primitive soln at all iquad and fill arrays
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                // extract conservative soln state
                std::array<real,nstate> soln_state;
                std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
                for(int istate=0; istate<nstate; istate++){
                    soln_state[istate] = soln_at_q_vect[istate][iquad];
                    for(int idim=0; idim<dim; idim++){
                        aux_soln_state[istate][idim] = soln_grad_at_q_vect[istate][idim][iquad];
                    }
                }
                // compute primitive soln state from conservative
                const std::array<real,nstate> primitive_soln_state = this->pde_physics_double->convert_conservative_to_primitive(soln_state);
                const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_aux_soln_state = this->pde_physics_double->convert_conservative_gradient_to_primitive_gradient(soln_state,aux_soln_state);
                // store primitive soln at quadrature point
                for(int istate=0; istate<nstate; istate++){
                    primitive_soln_at_q[istate][iquad] = primitive_soln_state[istate];
                    for(int idim=0; idim<dim; idim++){
                        primitive_aux_soln_at_q[istate][idim][iquad] = primitive_aux_soln_state[istate][idim];
                    }
                }
                // compute the DMS matrices
                // -- matrix L
                const dealii::Tensor<2,dim,real> matrix_L_component_state = this->pde_model_les_double->navier_stokes_physics->compute_germano_idendity_matrix_L_component(soln_state);
                // store the DMS matrix L
                for(int jdim=0; jdim<dim; jdim++){
                    for(int idim=0; idim<dim; idim++){
                        matrix_L_component_at_q[jdim][idim][iquad] = matrix_L_component_state[jdim][idim];
                    }
                }
                // -- matrix M
                const dealii::Tensor<2,dim,real> matrix_M_component_state = this->pde_model_les_double->navier_stokes_physics->compute_germano_idendity_matrix_M_component(soln_state,aux_soln_state);
                // store the DMS matrix M
                for(int jdim=0; jdim<dim; jdim++){
                    for(int idim=0; idim<dim; idim++){
                        matrix_M_component_at_q[jdim][idim][iquad] = matrix_M_component_state[jdim][idim];
                    }
                }
            }
            
            //==================================================
            // PROJECT TO LEGENDRE BASIS AND MODALLY FILTER
            //==================================================
            // -- Primitive solution at legendre poly
            std::array<std::vector<real>,nstate> primitive_legendre_soln_at_q;
            std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> primitive_legendre_aux_soln_at_q; // legendre auxiliary sol at quad points
            // Details: this projects to Legendre basis, truncates, then interpolates back to quad nodes.
            // -- Constructor for tensor product polynomials based on Polynomials::Legendre interpolation. 
            dealii::FE_DGQLegendre<1,1> legendre_poly_1D(poly_degree);
            // -- Projection operator for legendre basis
            OPERATOR::vol_projection_operator<dim,2*dim> legendre_soln_basis_projection_oper(1, poly_degree, grid_degree);
            legendre_soln_basis_projection_oper.build_1D_volume_operator(legendre_poly_1D, quad_extra_1D);
            // -- Legendre basis functions 
            OPERATOR::basis_functions<dim,2*dim> legendre_soln_basis(1, poly_degree, grid_degree);
            legendre_soln_basis.build_1D_volume_operator(legendre_poly_1D, quad_extra_1D);
            legendre_soln_basis.build_1D_gradient_operator(legendre_poly_1D, quad_extra_1D);
            for(int istate=0; istate<nstate; istate++){
                //==================================================
                // Solution and Solution Gradient
                //==================================================
                // -- (1) Project to Legendre basis
                std::vector<real> legendre_soln_coeff(n_shape_fns);
                legendre_soln_basis_projection_oper.matrix_vector_mult_1D(primitive_soln_at_q[istate], legendre_soln_coeff,
                                                                          legendre_soln_basis_projection_oper.oneD_vol_operator);
                // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
                if(/*this->apply_modal_high_pass_filter_on_filtered_solution && */(istate!=0 && istate!=(nstate-1))) {
                    for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                        if(ishape < p_min_filtered){
                            legendre_soln_coeff[ishape] = 0.0;
                        }
                    }    
                }
                // -- (3) Interpolate filtered solution back to quadrature points
                primitive_legendre_soln_at_q[istate].resize(n_quad_pts);
                legendre_soln_basis.matrix_vector_mult_1D(legendre_soln_coeff, primitive_legendre_soln_at_q[istate],
                                                          legendre_soln_basis.oneD_vol_operator);

                // We need to first compute the reference gradient of the solution, then transform that to a physical gradient.
                dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
                for(int idim=0; idim<dim; idim++){
                    ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
                    primitive_legendre_aux_soln_at_q[istate][idim].resize(n_quad_pts);
                }
                // Apply gradient of reference basis functions on the solution at volume cubature nodes.
                legendre_soln_basis.gradient_matrix_vector_mult_1D(legendre_soln_coeff, ref_gradient_basis_fns_times_soln,
                                                                   legendre_soln_basis.oneD_vol_operator,
                                                                   legendre_soln_basis.oneD_grad_operator);
                // Transform the reference gradient into a physical gradient operator.
                for(int idim=0; idim<dim; idim++){
                    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                        for(int jdim=0; jdim<dim; jdim++){
                            //transform into the physical gradient
                            primitive_legendre_aux_soln_at_q[istate][idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                                                   * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                                                   / metric_oper.det_Jac_vol[iquad];
                        }
                    }
                }
                //==================================================
            }

            
            //==================================================
            // PROJECT TO LEGENDRE BASIS AND MODALLY FILTER
            //==================================================
            // Compute the filtered DSM matrix components
            dealii::Tensor<2,dim,std::vector<real>> legendre_matrix_L_component_at_q;
            dealii::Tensor<2,dim,std::vector<real>> legendre_matrix_M_component_at_q;
            for(int jdim=0; jdim<dim; jdim++){
                dealii::Tensor<1,dim,std::vector<real>> legendre_matrix_L_component_coeff;
                dealii::Tensor<1,dim,std::vector<real>> legendre_matrix_M_component_coeff;
                for(int idim=0; idim<dim; idim++){
                    // -- (1) Project to Legendre basis
                    legendre_matrix_L_component_coeff[idim].resize(n_shape_fns);
                    legendre_soln_basis_projection_oper.matrix_vector_mult_1D(matrix_L_component_at_q[jdim][idim], legendre_matrix_L_component_coeff[idim],
                                                                              legendre_soln_basis_projection_oper.oneD_vol_operator);
                    legendre_matrix_M_component_coeff[idim].resize(n_shape_fns);
                    legendre_soln_basis_projection_oper.matrix_vector_mult_1D(matrix_M_component_at_q[jdim][idim], legendre_matrix_M_component_coeff[idim],
                                                                              legendre_soln_basis_projection_oper.oneD_vol_operator);
                    // -- (2) Truncate modes for high-pass filter (i.e. DG-VMS like)
                    if(/*this->apply_modal_high_pass_filter_on_filtered_solution*/) {
                        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                            if(ishape < p_min_filtered){
                                legendre_matrix_L_component_coeff[idim][ishape] = 0.0;
                                legendre_matrix_M_component_coeff[idim][ishape] = 0.0;
                            }
                        }
                    }
                    
                    // -- (3) Interpolate filtered solution back to quadrature points
                    legendre_matrix_L_component_at_q[jdim][idim].resize(n_quad_pts);
                    legendre_soln_basis.matrix_vector_mult_1D(legendre_matrix_L_component_coeff[idim], legendre_matrix_L_component_at_q[jdim][idim],
                                                              legendre_soln_basis.oneD_vol_operator);
                    legendre_matrix_M_component_at_q[jdim][idim].resize(n_quad_pts);
                    legendre_soln_basis.matrix_vector_mult_1D(legendre_matrix_M_component_coeff[idim], legendre_matrix_M_component_at_q[jdim][idim],
                                                              legendre_soln_basis.oneD_vol_operator);
                }
                //==================================================
                //==================================================
            }

            //=======================================================
            // CONVERT PRIMITIVE LEGENDRE SOLUTION TO CONSERVATIVE
            //=======================================================
            // Resize the conservative soln arrays
            for(int istate=0; istate<nstate; istate++){
                legendre_soln_at_q_vect[istate].resize(n_quad_pts);
                for(int idim=0; idim<dim; idim++){
                    legendre_aux_soln_at_q_vect[istate][idim].resize(n_quad_pts);
                }
            }
            for(int jdim=0; jdim<dim; jdim++){
                for(int idim=0; idim<dim; idim++){
                    legendre_matrix_L_component_at_q_vect[jdim][idim].resize(n_quad_pts);
                    legendre_matrix_M_component_at_q_vect[jdim][idim].resize(n_quad_pts);
                }
            }
            // Compute the primitive soln at all iquad and fill arrays
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                // extract conservative soln state
                std::array<real,nstate> primitive_legendre_soln_state;
                std::array<dealii::Tensor<1,dim,real>,nstate> primitive_legendre_aux_soln_state;
                for(int istate=0; istate<nstate; istate++){
                    primitive_legendre_soln_state[istate] = primitive_legendre_soln_at_q[istate][iquad];
                    for(int idim=0; idim<dim; idim++){
                        primitive_legendre_aux_soln_state[istate][idim] = primitive_legendre_aux_soln_at_q[istate][idim][iquad];
                    }
                }
                // compute conservative soln state from primitive
                std::array<real,nstate> legendre_soln_state = this->pde_physics_double->convert_primitive_to_conservative(primitive_legendre_soln_state);
                std::array<dealii::Tensor<1,dim,real>,nstate> legendre_aux_soln_state = this->pde_physics_double->convert_primitive_gradient_to_conservative_gradient(primitive_legendre_soln_state,primitive_legendre_aux_soln_state);
                // store conservative soln at quadrature point
                for(int istate=0; istate<nstate; istate++){
                    legendre_soln_at_q_vect[istate][iquad] = legendre_soln_state[istate];
                    for(int idim=0; idim<dim; idim++){
                        legendre_aux_soln_at_q_vect[istate][idim][iquad] = legendre_aux_soln_state[istate][idim];
                    }
                }
                for(int jdim=0; jdim<dim; jdim++){
                    for(int idim=0; idim<dim; idim++){
                        legendre_matrix_L_component_at_q_vect[jdim][idim][iquad] = legendre_matrix_L_component_at_q[jdim][idim][iquad];
                        legendre_matrix_M_component_at_q_vect[jdim][idim][iquad] = legendre_matrix_M_component_at_q[jdim][idim][iquad];
                    }
                }
            }
        /*}*/

        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_at_q;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;
            std::array<double,nstate> filtered_soln_at_q;
            std::array<dealii::Tensor<1,dim,double>,nstate> filtered_soln_grad_at_q;
            // Extract solution and gradient in a way that the physics can use them.
            for(int istate=0; istate<nstate; istate++){
                filtered_soln_at_q[istate] = legendre_soln_at_q_vect[istate][iquad];
                soln_at_q[istate] = soln_at_q_vect[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    filtered_soln_grad_at_q[istate][idim] = legendre_aux_soln_at_q_vect[istate][idim][iquad];
                    soln_grad_at_q[istate][idim] = soln_grad_at_q_vect[istate][idim][iquad];
                }
            }

            // Get strain rate tensor
            const dealii::Tensor<2,dim,real> matrix_L_component_state_from_filtered_soln = this->pde_model_les_double->navier_stokes_physics->compute_germano_idendity_matrix_L_component(filtered_soln_at_q);
            const dealii::Tensor<2,dim,real> matrix_M_component_state_from_filtered_soln = this->pde_model_les_double->navier_stokes_physics->compute_germano_idendity_matrix_M_component(filtered_soln_at_q,filtered_soln_grad_at_q);
            
            dealii::Tensor<2,dim,real> filtered_matrix_L_component_state;
            dealii::Tensor<2,dim,real> filtered_matrix_M_component_state;
            for(int jdim=0; jdim<dim; jdim++){
                for(int idim=0; idim<dim; idim++){
                    filtered_matrix_L_component_state[jdim][idim] = legendre_matrix_L_component_at_q_vect[jdim][idim][iquad];
                    filtered_matrix_M_component_state[jdim][idim] = legendre_matrix_M_component_at_q_vect[jdim][idim][iquad];
                }
            }

            // get cell index
            const dealii::types::global_dof_index cell_index = cell->active_cell_index();

            dealii::Tensor<2,dim,real> matrix_L; // Leonard stress tensor associated with the test filter
            dealii::Tensor<2,dim,real> matrix_M;
            const real filter_width = this->pde_model_les_double->get_filter_width(cell_index);
            const real test_filter_width = this->pde_model_les_double->get_filter_width_from_poly_degree(cell_index,(int)this->poly_degree_max_large_scales);
            for (int d1=0; d1<dim; ++d1) {
                for (int d2=0; d2<dim; ++d2) {
                    matrix_L[d1][d2] = filtered_matrix_L_component_state[d1][d2] - matrix_L_component_state_from_filtered_soln[d1][d2];
                    matrix_M[d1][d2] = filter_width*filter_width*filtered_matrix_M_component_state[d1][d2] - test_filter_width*test_filter_width*matrix_M_component_state_from_filtered_soln[d1][d2];
                }
            }

            const real matrix_L_times_matrix_M = this->pde_model_les_double->navier_stokes_physics->get_tensor_product_magnitude_sqr(matrix_L,matrix_M);
            const real matrix_M_times_matrix_M = this->pde_model_les_double->navier_stokes_physics->get_tensor_product_magnitude_sqr(matrix_M,matrix_M);

            cell_matrix_L_times_matrix_M_integral += matrix_L_times_matrix_M * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            cell_matrix_M_times_matrix_M_integral += matrix_M_times_matrix_M * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
        }
        
        // get the mean
        const real cell_volume = this->pde_model_double->cellwise_volume[cell_index];
        const real cell_averaged_matrix_L_times_matrix_M = cell_matrix_L_times_matrix_M_integral/cell_volume;
        const real cell_averaged_matrix_M_times_matrix_M = cell_matrix_M_times_matrix_M_integral/cell_volume;
        // update the DSM constant times filter width (all) squared
        real dynamic_smagorinsky_model_constant = -0.5*cell_averaged_matrix_L_times_matrix_M/cell_averaged_matrix_M_times_matrix_M;
        if(dynamic_smagorinsky_model_constant < 0.0) dynamic_smagorinsky_model_constant = 0.0;// clip values less than zero
        this->pde_model_double->dynamic_smagorinsky_model_constant_times_filter_width_sqr[cell_index] = dynamic_smagorinsky_model_constant*filter_width*filter_width;
    }
    // update ghost values
    this->pde_model_double->dynamic_smagorinsky_model_constant_times_filter_width_sqr.update_ghost_values();
}

#if PHILIP_DIM==3
template class DGStrongLES <PHILIP_DIM, PHILIP_DIM+2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrongLES <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrongLES <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrongLES_ShearImproved <PHILIP_DIM, PHILIP_DIM+2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrongLES_ShearImproved <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrongLES_ShearImproved <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrongLES_DynamicSmagorinsky <PHILIP_DIM, PHILIP_DIM+2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrongLES_DynamicSmagorinsky <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrongLES_DynamicSmagorinsky <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
