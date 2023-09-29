#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "model.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Models Base Class
//================================================================
template <int dim, int nstate, typename real>
ModelBase<dim, nstate, real>::ModelBase(
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input):
        manufactured_solution_function(manufactured_solution_function_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> ModelBase<dim, nstate, real>
::physical_source_term (
    const dealii::Point<dim,real> &/*pos*/,
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    std::array<real,nstate> physical_source;
    physical_source.fill(0.0);
    return physical_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim,nstate,real>
::boundary_manufactured_solution (
    const dealii::Point<dim, real> &/*pos*/,
    const dealii::Tensor<1,dim,real> &/*normal_int*/,
    const std::array<real,nstate> &/*soln_int*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
    std::array<real,nstate> &/*soln_bc*/,
    std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // Do nothing for nstate==(dim+2)
    if constexpr(nstate>(dim+2)) {
        pcout << "Error: boundary_manufactured_solution() not implemented in class derived from ModelBase with nstate>(dim+2)." << std::endl;
        pcout << "Aborting..." << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim,nstate,real>
::boundary_wall (
   std::array<real,nstate> &/*soln_bc*/,
   std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // Do nothing for nstate==(dim+2)
    if constexpr(nstate>(dim+2)) {
        pcout << "Error: boundary_wall() not implemented in class derived from ModelBase with nstate>(dim+2)." << std::endl;
        pcout << "Aborting..." << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim,nstate,real>
::boundary_outflow (
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
   std::array<real,nstate> &/*soln_bc*/,
   std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // Do nothing for nstate==(dim+2)
    if constexpr(nstate>(dim+2)) {
        pcout << "Error: boundary_outflow() not implemented in class derived from ModelBase with nstate>(dim+2)." << std::endl;
        pcout << "Aborting..." << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim,nstate,real>
::boundary_inflow (
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
   std::array<real,nstate> &/*soln_bc*/,
   std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // Do nothing for nstate==(dim+2)
    if constexpr(nstate>(dim+2)) {
        pcout << "Error: boundary_inflow() not implemented in class derived from ModelBase with nstate>(dim+2)." << std::endl;
        pcout << "Aborting..." << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim,nstate,real>
::boundary_farfield (
   std::array<real,nstate> &/*soln_bc*/) const
{
    // Do nothing for nstate==(dim+2)
    if constexpr(nstate>(dim+2)) {
        pcout << "Error: boundary_farfield() not implemented in class derived from ModelBase with nstate>(dim+2)." << std::endl;
        pcout << "Aborting..." << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim,nstate,real>
::boundary_slip_wall (
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
   std::array<real,nstate> &/*soln_bc*/,
   std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // Do nothing for nstate==(dim+2)
    if constexpr(nstate>(dim+2)) {
        pcout << "Error: boundary_slip_wall() not implemented in class derived from ModelBase with nstate>(dim+2)." << std::endl;
        pcout << "Aborting..." << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim,nstate,real>
::boundary_riemann (
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &/*soln_int*/,
   std::array<real,nstate> &/*soln_bc*/) const
{
    // Do nothing for nstate==(dim+2)
    if constexpr(nstate>(dim+2)) {
        pcout << "Error: boundary_riemann() not implemented in class derived from ModelBase with nstate>(dim+2)." << std::endl;
        pcout << "Aborting..." << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void ModelBase<dim,nstate,real>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    if (boundary_type == 1000) {
        // Manufactured solution boundary condition
        boundary_manufactured_solution (pos, normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } 
    else if (boundary_type == 1001) {
        // Wall boundary condition for working variables of RANS turbulence model
        boundary_wall (soln_bc, soln_grad_bc);
    } 
    else if (boundary_type == 1002) {
        // Outflow boundary condition 
        boundary_outflow (soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } 
    else if (boundary_type == 1003) {
        // Inflow boundary condition
        boundary_inflow (soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } 
    else if (boundary_type == 1004) {
        // Riemann-based farfield boundary condition
        boundary_riemann (normal_int, soln_int, soln_bc);
    } 
    else if (boundary_type == 1005) {
        // Simple farfield boundary condition
        boundary_farfield(soln_bc);
    } 
    else if (boundary_type == 1006) {
        // Slip wall boundary condition
        boundary_slip_wall (normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } 
    else {
        pcout << "Invalid boundary_type: " << boundary_type << " in ModelBase.cpp" << std::endl;
        std::abort();
    }
    // Note: this does not get called when nstate==dim+2 since baseline physics takes care of it
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Vector<double> ModelBase<dim, nstate, real>
::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &/*duh*/,
    const std::vector<dealii::Tensor<2,dim> > &/*dduh*/,
    const dealii::Tensor<1,dim>               &/*normals*/,
    const dealii::Point<dim>                  &/*evaluation_points*/) const
{
    dealii::Vector<double> computed_quantities(nstate-(dim+2));
    for (unsigned int s=dim+2; s<nstate; ++s) {
        computed_quantities(s-(dim+2)) = uh(s);
    }
    return computed_quantities;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::vector<std::string> ModelBase<dim, nstate, real>
::post_get_names () const
{
    std::vector<std::string> names;
    for (unsigned int s=dim+2; s<nstate; ++s) {
        std::string varname = "state" + dealii::Utilities::int_to_string(s,1);
        names.push_back(varname);
    }
    return names;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> ModelBase<dim, nstate, real>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation;
    for (unsigned int s=dim+2; s<nstate; ++s) {
        interpretation.push_back (DCI::component_is_scalar);
    }
    return interpretation;
}

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
template class ModelBase<PHILIP_DIM, 1, double>;
template class ModelBase<PHILIP_DIM, 2, double>;
template class ModelBase<PHILIP_DIM, 3, double>;
template class ModelBase<PHILIP_DIM, 4, double>;
template class ModelBase<PHILIP_DIM, 5, double>;
template class ModelBase<PHILIP_DIM, 6, double>;
template class ModelBase<PHILIP_DIM, 8, double>;

template class ModelBase<PHILIP_DIM, 1, FadType>;
template class ModelBase<PHILIP_DIM, 2, FadType>;
template class ModelBase<PHILIP_DIM, 3, FadType>;
template class ModelBase<PHILIP_DIM, 4, FadType>;
template class ModelBase<PHILIP_DIM, 5, FadType>;
template class ModelBase<PHILIP_DIM, 6, FadType>;
template class ModelBase<PHILIP_DIM, 8, FadType>;

template class ModelBase<PHILIP_DIM, 1, RadType>;
template class ModelBase<PHILIP_DIM, 2, RadType>;
template class ModelBase<PHILIP_DIM, 3, RadType>;
template class ModelBase<PHILIP_DIM, 4, RadType>;
template class ModelBase<PHILIP_DIM, 5, RadType>;
template class ModelBase<PHILIP_DIM, 6, RadType>;
template class ModelBase<PHILIP_DIM, 8, RadType>;

template class ModelBase<PHILIP_DIM, 1, FadFadType>;
template class ModelBase<PHILIP_DIM, 2, FadFadType>;
template class ModelBase<PHILIP_DIM, 3, FadFadType>;
template class ModelBase<PHILIP_DIM, 4, FadFadType>;
template class ModelBase<PHILIP_DIM, 5, FadFadType>;
template class ModelBase<PHILIP_DIM, 6, FadFadType>;
template class ModelBase<PHILIP_DIM, 8, FadFadType>;

template class ModelBase<PHILIP_DIM, 1, RadFadType>;
template class ModelBase<PHILIP_DIM, 2, RadFadType>;
template class ModelBase<PHILIP_DIM, 3, RadFadType>;
template class ModelBase<PHILIP_DIM, 4, RadFadType>;
template class ModelBase<PHILIP_DIM, 5, RadFadType>;
template class ModelBase<PHILIP_DIM, 6, RadFadType>;
template class ModelBase<PHILIP_DIM, 8, RadFadType>;

} // Physics namespace
} // PHiLiP namespace