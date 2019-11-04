// includes
#include <vector>

#include <Sacado.hpp>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

//#include "dg/high_order_grid.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "dg/high_order_grid.h"
#include "functional.h"

namespace PHiLiP {

template <int dim, int nstate, typename real>
real Functional<dim, nstate, real>::evaluate_function(
    DGBase<dim,real> &dg, 
    const Physics::PhysicsBase<dim,nstate,real> &physics)
{
    real local_sum = 0;

    // allocating vectors for local calculations
    // could these also be indexSets?
    const unsigned int max_dofs_per_cell = dg.dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);
    std::vector<real> local_solution(max_dofs_per_cell);

    const auto mapping = (*(dg.high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);

    dg.solution.update_ghost_values();
    for(auto cell = dg.dof_handler.begin_active(); cell != dg.dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        // setting up the volume integration
        const unsigned int mapping_index = 0; // *** ask doug if this will ever be 
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const unsigned int quad_index = fe_index_curr_cell;
        const dealii::FESystem<dim,dim> &current_fe_ref = dg.fe_collection[fe_index_curr_cell];
        //const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        
        // reinitialize the volume integration
        fe_values_collection_volume.reinit(cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

        // // number of quadrature points
        // const unsigned int n_quad_points = fe_values_volume.n_quadrature_points;

        // getting the indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);

        // getting solution values
        local_solution.resize(n_dofs_curr_cell);
        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof){
            local_solution[idof] = dg.solution[current_dofs_indices[idof]];
        }

        // adding the contribution from the current volume, also need to pass the solution vector on these points
        local_sum += this->evaluate_cell_volume(physics, fe_values_volume, local_solution);

        // next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = cell->face(iface);
            
            if(face->at_boundary()){
                fe_values_collection_face.reinit(cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();

                const unsigned int boundary_id = face->boundary_id();

                local_sum += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, local_solution);
            }

        }
    }

    return dealii::Utilities::MPI::sum(local_sum, MPI_COMM_WORLD);
}

template <int dim, int nstate, typename real>
dealii::LinearAlgebra::distributed::Vector<real> Functional<dim, nstate, real>::evaluate_dIdw(
    DGBase<dim,real> &dg, 
    const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics)
{
    // for the AD'd return variable
    using ADType = Sacado::Fad::DFad<real>;

    // for taking the local derivatives
    ADType local_sum;

    // vector for storing the derivatives with respect to each DOF
    dealii::LinearAlgebra::distributed::Vector<real> dIdw;
 
    // allocating the vector
    dealii::IndexSet locally_owned_dofs = dg.dof_handler.locally_owned_dofs();
    dIdw.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
    const unsigned int max_dofs_per_cell = dg.dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);
    std::vector<ADType> local_solution(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real>   local_dIdw(max_dofs_per_cell);

    const auto mapping = (*(dg.high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);

    dg.solution.update_ghost_values();
    for(auto cell = dg.dof_handler.begin_active(); cell != dg.dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        // setting up the volume integration
        const unsigned int mapping_index = 0; // *** ask doug if this will ever be 
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const unsigned int quad_index = fe_index_curr_cell;
        const dealii::FESystem<dim,dim> &current_fe_ref = dg.fe_collection[fe_index_curr_cell];
        //const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // reinitialize the volume integration
        fe_values_collection_volume.reinit(cell, quad_index, mapping_index, fe_index_curr_cell);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();

        // getting the indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);

        // getting their values and setting up 
        local_solution.resize(n_dofs_curr_cell);
        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof){
            local_solution[idof] = dg.solution[current_dofs_indices[idof]];
            local_solution[idof].diff(idof, n_dofs_curr_cell);
        }

        // adding the contribution from the current volume, also need to pass the solution vector on these points
        local_sum = this->evaluate_cell_volume(physics, fe_values_volume, local_solution);

        // next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = cell->face(iface);
            
            if(face->at_boundary()){
                fe_values_collection_face.reinit(cell, iface, quad_index, mapping_index, fe_index_curr_cell);
                const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();

                const unsigned int boundary_id = face->boundary_id();

                local_sum += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, local_solution);
            }

        }

        // now getting the values and adding them to the derivaitve vector
        local_dIdw.resize(n_dofs_curr_cell);
        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof){
            local_dIdw[idof] = local_sum.dx(idof);
        }
        dIdw.add(current_dofs_indices, local_dIdw);
    }
    // compress before the return
    dIdw.compress(dealii::VectorOperation::add);
    
    return dIdw;
}

template class Functional <PHILIP_DIM, 1, double>;
template class Functional <PHILIP_DIM, 2, double>;
template class Functional <PHILIP_DIM, 3, double>;
template class Functional <PHILIP_DIM, 4, double>;
template class Functional <PHILIP_DIM, 5, double>;

} // PHiLiP namespace