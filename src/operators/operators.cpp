#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1_eulerian.h>


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <Epetra_RowMatrixTransposer.h>
#include <AztecOO.h>

#include "ADTypes.hpp"
#include <Sacado.hpp>
#include <CoDiPack/include/codi.hpp>

#include "operators.h"

namespace PHiLiP {
namespace OPERATOR {

//Constructor
template <int dim, typename real, int n_faces>
OperatorsBase<dim,real,n_faces>::OperatorsBase(
    const Parameters::AllParameters *const parameters_input,
    const int nstate_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : OperatorsBase<dim,real,n_faces>(parameters_input, nstate_input, degree, max_degree_input, grid_degree_input, this->create_collection_tuple(max_degree_input, nstate_input, parameters_input))
{ }

template <int dim, typename real, int n_faces>
OperatorsBase<dim,real,n_faces>::OperatorsBase(
    const Parameters::AllParameters *const parameters_input,
    const int nstate_input,
    const unsigned int /*degree*/,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const MassiveCollectionTuple collection_tuple)
    : all_parameters(parameters_input)
    , max_degree(max_degree_input)
    , max_grid_degree(grid_degree_input)
    , nstate(nstate_input)
    , max_grid_degree_check(grid_degree_input)
    , fe_collection_basis(std::get<0>(collection_tuple))
    , volume_quadrature_collection(std::get<1>(collection_tuple))
    , face_quadrature_collection(std::get<2>(collection_tuple))
    , oned_quadrature_collection(std::get<3>(collection_tuple))
    , fe_collection_flux_basis(std::get<4>(collection_tuple))
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{ 
    //Allocate and construct the base operators.
    pcout<<" Constructing Operators ..."<<std::endl;
    //setup volume operators
    pcout<<" ... allocating volume operators ..."<<std::flush;
    allocate_volume_operators();
    pcout<<" done."<<std::endl;
    pcout<<" ... creating volume basis operators ..."<<std::flush;
    create_vol_basis_operators();
    pcout<<" done."<<std::endl;
    pcout<<" ... building mass matrix operators ..."<<std::flush;
    build_Mass_Matrix_operators();
    pcout<<" done."<<std::endl;
    pcout<<" ... building stiffness matrix operators ..."<<std::flush;
    build_Stiffness_Matrix_operators ();
    pcout<<" done."<<std::endl;
    pcout<<" ... getting higher derivatives ..."<<std::flush;
    get_higher_derivatives();
    pcout<<" done."<<std::endl;
    pcout<<" ... building flux reconstruction operators ..."<<std::flush;
    build_Flux_Reconstruction_operators();
    pcout<<" done."<<std::endl;
    pcout<<" ... getting volume projection operators ..."<<std::flush;
    get_vol_projection_operators();
    pcout<<" done."<<std::endl;
    //setup surface operators
    pcout<<" ... allocating surface operators ..."<<std::flush;
    allocate_surface_operators();
    pcout<<" done."<<std::endl;
    pcout<<" ... creating surface basis operators ..."<<std::flush;
    create_surface_basis_operators();
    pcout<<" done."<<std::endl;
    pcout<<" ... getting surface lifting operators ..."<<std::flush;
    get_surface_lifting_operators ();
    pcout<<" done."<<std::endl;
    pcout<<" ... allocating metric operators ..."<<std::flush;
    //setup metric operators
    allocate_metric_operators(max_grid_degree);
    pcout<<" done."<<std::endl;
    pcout<<" ... creating metric basis basis operators ..."<<std::flush;
    create_metric_basis_operators(max_grid_degree);
    pcout<<" done."<<std::endl;
    pcout<<" done."<<std::endl;
}
// Destructor
template <int dim, typename real, int n_faces>
OperatorsBase<dim,real,n_faces>::~OperatorsBase ()
{
    pcout << "Destructing Operators..." << std::endl;
}
template <int dim, typename real, int n_faces>
std::tuple<
        dealii::hp::FECollection<dim>, // Solution FE basis functions
        dealii::hp::QCollection<dim>,  // Volume quadrature
        dealii::hp::QCollection<dim-1>, // Face quadrature
        dealii::hp::QCollection<1>, // 1D quadrature for strong form
        dealii::hp::FECollection<dim> >   // Flux Basis polynomials for strong form
OperatorsBase<dim,real,n_faces>::create_collection_tuple(const unsigned int max_degree, const int nstate, const Parameters::AllParameters *const parameters_input) const
{
    dealii::hp::FECollection<dim>      fe_coll;//basis functions collection
    dealii::hp::QCollection<dim>       volume_quad_coll;//volume flux nodes
    dealii::hp::QCollection<dim-1>     face_quad_coll;//facet flux nodes
    dealii::hp::QCollection<1>         oned_quad_coll;//1D flux nodes

    dealii::hp::FECollection<dim>      fe_coll_lagr;//flux basis collocated on flux nodes

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
            dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1);
            dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
//Commented out, to be able to handle mixed integration weights in future.
//            dealii::QGaussChebyshev<1> oned_quad_Gauss_Legendre (degree+1);
//            dealii::QGaussChebyshev<dim> vol_quad_Gauss_Legendre (degree+1);
//            if(dim == 1) {
//                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
//            face_quad = face_quad_Gauss_Legendre;
//            } else {
//                dealii::QGaussChebyshev<dim-1> face_quad_Gauss_Legendre (degree+1);
//            face_quad = face_quad_Gauss_Legendre;
//            }

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
            dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1+overintegration);
            dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1+overintegration);
            dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
//            dealii::QGaussChebyshev<1> oned_quad_Gauss_Legendre (degree+1+overintegration);
//            dealii::QGaussChebyshev<dim> vol_quad_Gauss_Legendre (degree+1+overintegration);
//            if(dim == 1) {
//                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
//            face_quad = face_quad_Gauss_Legendre;
//            } else {
//                dealii::QGaussChebyshev<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
//            face_quad = face_quad_Gauss_Legendre;
//            }
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


/*******************************************
 *
 *      VOLUME OPERATORS FUNCTIONS
 *
 *
 *      *****************************************/
template <int dim, typename real, int n_faces>
double OperatorsBase<dim,real,n_faces>::compute_factorial(double n)
{
    if ((n==0)||(n==1))
      return 1;
   else
      return n*compute_factorial(n-1);
}
template <int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::allocate_volume_operators ()
{

    //basis functions evaluated at volume cubature (flux) nodes
    basis_at_vol_cubature.resize(this->max_degree+1);
    vol_integral_basis.resize(this->max_degree+1);
    modal_basis_differential_operator.resize(this->max_degree+1);
    local_mass.resize(this->max_degree+1);
    local_basis_stiffness.resize(this->max_degree+1);
    derivative_p.resize(this->max_degree+1);
    derivative_2p.resize(this->max_degree+1);
    derivative_3p.resize(this->max_degree+1);
    local_Flux_Reconstruction_operator.resize(this->max_degree+1);
    c_param_FR.resize(this->max_degree+1);
    local_Flux_Reconstruction_operator_aux.resize(this->max_degree+1);
    k_param_FR.resize(this->max_degree+1);
    vol_projection_operator.resize(this->max_degree+1);
    vol_projection_operator_FR.resize(this->max_degree+1);
    FR_mass_inv.resize(this->max_degree+1);
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        basis_at_vol_cubature[idegree].reinit(n_quad_pts, n_dofs);
        vol_integral_basis[idegree].reinit(n_quad_pts, n_dofs);
        local_mass[idegree].reinit(n_dofs, n_dofs);
        derivative_3p[idegree].reinit(n_dofs, n_dofs);
        local_Flux_Reconstruction_operator[idegree].reinit(n_dofs, n_dofs);
        vol_projection_operator[idegree].reinit(n_dofs, n_quad_pts);
        vol_projection_operator_FR[idegree].reinit(n_dofs, n_quad_pts);
        FR_mass_inv[idegree].reinit(n_dofs, n_dofs);
        for(int idim=0; idim<dim; idim++){
            modal_basis_differential_operator[idegree][idim].reinit(n_dofs, n_dofs);
            local_basis_stiffness[idegree][idim].reinit(n_dofs, n_dofs);
            derivative_p[idegree][idim].reinit(n_dofs, n_dofs);
            derivative_2p[idegree][idim].reinit(n_dofs, n_dofs);
            local_Flux_Reconstruction_operator_aux[idegree][idim].reinit(n_dofs, n_dofs);
        }
    }

}

template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>::create_vol_basis_operators ()
{

    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            const dealii::Point<dim> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
            const std::vector<real> &quad_weights = this->volume_quadrature_collection[idegree].get_weights ();
            for(unsigned int idof=0; idof<n_dofs; idof++){
                const int istate = this->fe_collection_basis[idegree].system_to_component_index(idof).first;
                //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
                basis_at_vol_cubature[idegree][iquad][idof] = this->fe_collection_basis[idegree].shape_value_component(idof,qpoint,istate);
                //Basis function idof of poly degree idegree evaluated at cubature node qpoint multiplied by quad weight.
                vol_integral_basis[idegree][iquad][idof] = quad_weights[iquad] * basis_at_vol_cubature[idegree][iquad][idof];
            }
        }
    }
}


template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>::build_local_Mass_Matrix (
                                const std::vector<real> &quad_weights,
                                const unsigned int n_dofs_cell, const unsigned int n_quad_pts,
                                const int current_fe_index,
                                dealii::FullMatrix<real> &Mass_Matrix)
{
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        const int istate_test = this->fe_collection_basis[current_fe_index].system_to_component_index(itest).first;

        for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {

            const int istate_trial = this->fe_collection_basis[current_fe_index].system_to_component_index(itrial).first;

            real value = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                value +=
                        basis_at_vol_cubature[current_fe_index][iquad][itest] 
                    *   basis_at_vol_cubature[current_fe_index][iquad][itrial] 
                    *   quad_weights[iquad];//note that for mass matrix with metric Jacobian dependence pass JxW for quad_weights                            
            }

            Mass_Matrix[itrial][itest] = 0.0;
            Mass_Matrix[itest][itrial] = 0.0;
            if(istate_test==istate_trial) {
                Mass_Matrix[itrial][itest] = value;
                Mass_Matrix[itest][itrial] = value;
            }
        }
    }
}

template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>::build_Mass_Matrix_operators ()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs_cell = this->fe_collection_basis[idegree].dofs_per_cell;
        const std::vector<real> &quad_weights = this->volume_quadrature_collection[idegree].get_weights ();
        build_local_Mass_Matrix(quad_weights, n_dofs_cell, n_quad_pts, idegree, local_mass[idegree]);
    }
}
template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>::build_Stiffness_Matrix_operators ()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        const std::vector<real> &quad_weights = this->volume_quadrature_collection[idegree].get_weights ();
        for(unsigned int itest=0; itest<n_dofs; itest++){
            const int istate_test = this->fe_collection_basis[idegree].system_to_component_index(itest).first;
            for(unsigned int idof=0; idof<n_dofs; idof++){
                const int istate = this->fe_collection_basis[idegree].system_to_component_index(idof).first;
                dealii::Tensor<1,dim,real> value;
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    const dealii::Point<dim> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
                    dealii::Tensor<1,dim,real> derivative;
                    derivative = this->fe_collection_basis[idegree].shape_grad_component(idof, qpoint, istate);
                    value += basis_at_vol_cubature[idegree][iquad][itest] * quad_weights[iquad] * derivative;
                }
                if(istate == istate_test){
                    for(int idim=0; idim<dim; idim++){
                        local_basis_stiffness[idegree][idim][itest][idof] = value[idim]; 
                    }
                }
            }
        }
        for(int idim=0; idim<dim; idim++){
            dealii::FullMatrix<real> inv_mass(n_dofs);
            inv_mass.invert(local_mass[idegree]);
            inv_mass.mmult(modal_basis_differential_operator[idegree][idim],local_basis_stiffness[idegree][idim]);
        }
    }

}
template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_higher_derivatives ()
{

    for(unsigned int curr_cell_degree=0;curr_cell_degree<=this->max_degree; curr_cell_degree++){
        unsigned int degree_index = curr_cell_degree;
        unsigned int n_dofs_cell = this->fe_collection_basis[degree_index].dofs_per_cell;
        //write each deriv p to identity
        for(int idim=0; idim<dim; idim++){
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
               for(unsigned int idof2=0; idof2<n_dofs_cell; idof2++){
                   if(idof == idof2){
                       derivative_p[degree_index][idim][idof][idof2] = 1.0;//set it equal to identity
                   }
               }
            } 
        }
        for(int idim=0; idim<dim; idim++){
            for(unsigned int idegree=0; idegree< curr_cell_degree; idegree++){
               dealii::FullMatrix<real> derivative_p_temp(n_dofs_cell, n_dofs_cell);
               derivative_p_temp.add(1, derivative_p[degree_index][idim]);
               modal_basis_differential_operator[degree_index][idim].mmult(derivative_p[degree_index][idim], derivative_p_temp);
            }
        }
        if(dim >= 2){
            derivative_p[degree_index][0].mmult(derivative_2p[degree_index][0],derivative_p[degree_index][1]);
        }
        if(dim==3){
            //derivative_p[degree_index][0].mmult(derivative_2p[degree_index][0],derivative_p[degree_index][1]);
            derivative_p[degree_index][0].mmult(derivative_2p[degree_index][1],derivative_p[degree_index][2]);
            derivative_p[degree_index][1].mmult(derivative_2p[degree_index][2],derivative_p[degree_index][2]);
            derivative_p[degree_index][0].mmult(derivative_3p[degree_index],derivative_2p[degree_index][2]);
        }
    }
}

template <int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_Huynh_g2_parameter (
                                const unsigned int curr_cell_degree,
                                real &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));//since ref element [0,1]
    c = 2.0 * (curr_cell_degree+1)/( curr_cell_degree*((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2))));  
    c/=2.0;//since orthonormal
}
template <int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_spectral_difference_parameter (
                                const unsigned int curr_cell_degree,
                                real &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));
    c = 2.0 * (curr_cell_degree)/( (curr_cell_degree+1.0)*((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2))));  
    c/=2.0;//since orthonormal
}
template <int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_c_negative_FR_parameter (
                                const unsigned int curr_cell_degree,
                                real &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));
    c = - 2.0 / ( pow((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2)),1.0));  
    c/=2.0;//since orthonormal
}
template <int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_c_negative_divided_by_two_FR_parameter (
                                const unsigned int curr_cell_degree,
                                real &c)
{
    get_c_negative_FR_parameter(curr_cell_degree, c); 
    c/=2.0;
}
template <int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_c_plus_parameter (
                                const unsigned int curr_cell_degree,
                                real &c)
{
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

template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_FR_correction_parameter (
                                const unsigned int curr_cell_degree,
                                real &c, real &k)
{
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    using FR_Aux_enum = Parameters::AllParameters::Flux_Reconstruction_Aux;
    FR_enum c_input = this->all_parameters->flux_reconstruction_type; 
    FR_Aux_enum k_input = this->all_parameters->flux_reconstruction_aux_type; 
    if(c_input == FR_enum::cHU || c_input == FR_enum::cHULumped){ 
        get_Huynh_g2_parameter(curr_cell_degree, c); 
    }
    else if(c_input == FR_enum::cSD){ 
        get_spectral_difference_parameter(curr_cell_degree, c); 
    }
    else if(c_input == FR_enum::cNegative){ 
        get_c_negative_FR_parameter(curr_cell_degree, c); 
    }
    else if(c_input == FR_enum::cNegative2){ 
        get_c_negative_divided_by_two_FR_parameter(curr_cell_degree, c); 
    }
    else if(c_input == FR_enum::cDG){ 
        //DG case is the 0.0 case.
        c = 0.0;
    }
    else if(c_input == FR_enum::c10Thousand){ 
        //Set the value to 10000 for arbitrary high-numbers.
        c = 10000.0;
    }
    else if(c_input == FR_enum::cPlus){ 
        get_c_plus_parameter(curr_cell_degree, c); 
    }

    if(k_input == FR_Aux_enum::kHU){ 
        get_Huynh_g2_parameter(curr_cell_degree, k); 
    }
    else if(k_input == FR_Aux_enum::kSD){ 
        get_spectral_difference_parameter(curr_cell_degree, k); 
    }
    else if(k_input == FR_Aux_enum::kNegative){ 
        get_c_negative_FR_parameter(curr_cell_degree, k); 
    }
    else if(k_input == FR_Aux_enum::kNegative2){//knegative divided by 2 
        get_c_negative_divided_by_two_FR_parameter(curr_cell_degree, k); 
    }
    else if(k_input == FR_Aux_enum::kDG){ 
        k = 0.0;
    }
    else if(k_input == FR_Aux_enum::k10Thousand){ 
        k = 10000.0;
    }
    else if(k_input == FR_Aux_enum::kPlus){ 
        get_c_plus_parameter(curr_cell_degree, k); 
    }
}
template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>
::build_local_Flux_Reconstruction_operator(
                                const dealii::FullMatrix<real> &local_Mass_Matrix,
                                const unsigned int  n_dofs_cell, const unsigned int degree_index, 
                                dealii::FullMatrix<real> &Flux_Reconstruction_operator)
{
    real c = 0.0;
//get the Flux_Reconstruction operator
    c = c_param_FR[degree_index];
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<real> derivative_p_temp(n_dofs_cell, n_dofs_cell);
        derivative_p_temp.add(c, derivative_p[degree_index][idim]);
        dealii::FullMatrix<real> Flux_Reconstruction_operator_temp(n_dofs_cell);
        derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
        Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, derivative_p[degree_index][idim], true);
    }
    if(dim>=2){
        const int deriv_2p_loop = (dim==2) ? 1 : dim; 
        double c_2 = pow(c,2.0);
        for(int idim=0; idim<deriv_2p_loop; idim++){
            dealii::FullMatrix<real> derivative_p_temp(n_dofs_cell, n_dofs_cell);
            derivative_p_temp.add(c_2, derivative_2p[degree_index][idim]);
            dealii::FullMatrix<real> Flux_Reconstruction_operator_temp(n_dofs_cell);
            derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
            Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, derivative_2p[degree_index][idim], true);
        }
    }
    if(dim == 3){
        double c_3 = pow(c,3.0);
        dealii::FullMatrix<real> derivative_p_temp(n_dofs_cell, n_dofs_cell);
        derivative_p_temp.add(c_3, derivative_3p[degree_index]);
        dealii::FullMatrix<real> Flux_Reconstruction_operator_temp(n_dofs_cell);
        derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
        Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, derivative_3p[degree_index], true);
    }
    
}
template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>
::build_local_Flux_Reconstruction_operator_AUX(
                                const dealii::FullMatrix<real> &local_Mass_Matrix,
                                const unsigned int n_dofs_cell, const unsigned int degree_index, 
                                std::array<dealii::FullMatrix<real>,dim> &Flux_Reconstruction_operator_aux)
{
    real k = 0.0;
//get the Flux_Reconstruction AUX operator
    k = k_param_FR[degree_index];
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<real> derivative_p_temp2(n_dofs_cell, n_dofs_cell);
        dealii::FullMatrix<real> Flux_Reconstruction_operator_temp(n_dofs_cell);
        derivative_p_temp2.add(k,derivative_p[degree_index][idim]);
        derivative_p_temp2.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
        Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator_aux[idim], derivative_p[degree_index][idim]);
    }
    
}
template <  int dim, typename real,
            int n_faces>  
void OperatorsBase<dim,real,n_faces>::build_Flux_Reconstruction_operators ()
{
    for(unsigned int degree_index=0; degree_index<=this->max_degree; degree_index++){
        unsigned int n_dofs_cell = this->fe_collection_basis[degree_index].dofs_per_cell;
        unsigned int curr_cell_degree = degree_index; 
        get_FR_correction_parameter(curr_cell_degree, c_param_FR[degree_index], k_param_FR[degree_index]);
        build_local_Flux_Reconstruction_operator(local_mass[degree_index], n_dofs_cell, degree_index, local_Flux_Reconstruction_operator[degree_index]);
        build_local_Flux_Reconstruction_operator_AUX(local_mass[degree_index], n_dofs_cell, degree_index, local_Flux_Reconstruction_operator_aux[degree_index]);
    }
}

template <int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::compute_local_vol_projection_operator(
                                const unsigned int degree_index, 
                                const unsigned int n_dofs_cell,
                                const dealii::FullMatrix<real> &norm_matrix, 
                                dealii::FullMatrix<real> &volume_projection)
{
    dealii::FullMatrix<real> norm_inv(n_dofs_cell);
    norm_inv.invert(norm_matrix);
    norm_inv.mTmult(volume_projection, vol_integral_basis[degree_index]);
}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_vol_projection_operators ()
{
    for(unsigned int degree_index=0; degree_index<=this->max_degree; degree_index++){
        unsigned int n_dofs = this->fe_collection_basis[degree_index].dofs_per_cell;
        compute_local_vol_projection_operator(degree_index, n_dofs, local_mass[degree_index], vol_projection_operator[degree_index]);
        dealii::FullMatrix<real> M_plus_Flux_Reconstruction(n_dofs);
        M_plus_Flux_Reconstruction.add(1.0, local_mass[degree_index], 1.0, local_Flux_Reconstruction_operator[degree_index]);
        FR_mass_inv[degree_index].invert(M_plus_Flux_Reconstruction);
        compute_local_vol_projection_operator(degree_index, n_dofs, M_plus_Flux_Reconstruction, vol_projection_operator_FR[degree_index]);
    }
}

/*******************************************
 *
 *      SURFACE OPERATORS FUNCTIONS
 *
 *
 *      *****************************************/
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::allocate_surface_operators ()
{
    basis_at_facet_cubature.resize(this->max_degree+1);
    face_integral_basis.resize(this->max_degree+1);
    lifting_operator.resize(this->max_degree+1);
    lifting_operator_FR.resize(this->max_degree+1);
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_face_pts = this->face_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        for(unsigned int iface=0; iface<n_faces; iface++){
            basis_at_facet_cubature[idegree][iface].reinit(n_quad_face_pts, n_dofs);
            face_integral_basis[idegree][iface].reinit(n_quad_face_pts, n_dofs);
            lifting_operator[idegree][iface].reinit(n_dofs, n_quad_face_pts);
            lifting_operator_FR[idegree][iface].reinit(n_dofs, n_quad_face_pts);
        }
    }
}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::create_surface_basis_operators ()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        unsigned int n_quad_face_pts = this->face_quadrature_collection[idegree].size();
        const std::vector<real> &quad_weights = this->face_quadrature_collection[idegree].get_weights ();
        for(unsigned int iface=0; iface<n_faces; iface++){ 
            const dealii::Quadrature<dim> quadrature = dealii::QProjector<dim>::project_to_face(dealii::ReferenceCell::get_hypercube(dim),
                                                                                                this->face_quadrature_collection[idegree],
                                                                                                iface);
            for(unsigned int iquad=0; iquad<n_quad_face_pts; iquad++){
                for(unsigned int idof=0; idof<n_dofs; idof++){
                    const int istate = this->fe_collection_basis[idegree].system_to_component_index(idof).first;
                    basis_at_facet_cubature[idegree][iface][iquad][idof] = this->fe_collection_basis[idegree].shape_value_component(idof,quadrature.point(iquad),istate);
                    face_integral_basis[idegree][iface][iquad][idof] = 
                                basis_at_facet_cubature[idegree][iface][iquad][idof] 
                            *   quad_weights[iquad];
                }
            }
        }
    }

}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::build_local_surface_lifting_operator (
                                const unsigned int degree_index, 
                                const unsigned int n_dofs_cell, 
                                const unsigned int face_number, 
                                const dealii::FullMatrix<real> &norm_matrix, 
                                dealii::FullMatrix<real> &lifting)
{
    dealii::FullMatrix<real> norm_inv(n_dofs_cell);
    norm_inv.invert(norm_matrix);
    norm_matrix.mTmult(lifting, face_integral_basis[degree_index][face_number]);
}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::get_surface_lifting_operators ()
{
    for(unsigned int degree_index=0; degree_index<=this->max_degree; degree_index++){
        unsigned int n_dofs = this->fe_collection_basis[degree_index].dofs_per_cell;
        for(unsigned int iface=0; iface<n_faces; iface++){
            build_local_surface_lifting_operator(degree_index, n_dofs, iface, local_mass[degree_index], lifting_operator[degree_index][iface]);
            dealii::FullMatrix<real> M_plus_Flux_Reconstruction(n_dofs);
            M_plus_Flux_Reconstruction.add(1.0, local_mass[degree_index], 1.0, local_Flux_Reconstruction_operator[degree_index]);
            build_local_surface_lifting_operator(degree_index, n_dofs, iface, M_plus_Flux_Reconstruction, lifting_operator_FR[degree_index][iface]);
        }
    }

}
/*********************************************************************
 *
 *              METRIC OPERATOR FUNCTIONS
 *
 *              ******************************************************/
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::allocate_metric_operators (
                                                        const unsigned int max_grid_degree_local)
{
    mapping_shape_functions_grid_nodes.resize(max_grid_degree_local+1);
    gradient_mapping_shape_functions_grid_nodes.resize(max_grid_degree_local+1);
    mapping_shape_functions_vol_flux_nodes.resize(max_grid_degree_local+1);
    mapping_shape_functions_face_flux_nodes.resize(max_grid_degree_local+1);
    gradient_mapping_shape_functions_vol_flux_nodes.resize(max_grid_degree_local+1);
    gradient_mapping_shape_functions_face_flux_nodes.resize(max_grid_degree_local+1);
    for(unsigned int idegree=0; idegree<=max_grid_degree_local; idegree++){
        unsigned int n_dofs = pow(idegree+1,dim);
        mapping_shape_functions_grid_nodes[idegree].reinit(n_dofs, n_dofs);
        for(int idim=0; idim<dim; idim++){
            gradient_mapping_shape_functions_grid_nodes[idegree][idim].reinit(n_dofs, n_dofs);
        }
        //initialize flux sets
        mapping_shape_functions_vol_flux_nodes[idegree].resize(this->max_degree+1);
        mapping_shape_functions_face_flux_nodes[idegree].resize(this->max_degree+1);
        gradient_mapping_shape_functions_vol_flux_nodes[idegree].resize(this->max_degree+1);
        gradient_mapping_shape_functions_face_flux_nodes[idegree].resize(this->max_degree+1);
        for(unsigned int iflux_degree=0; iflux_degree<=this->max_degree; iflux_degree++){
            const unsigned int n_quad_pts = this->volume_quadrature_collection[iflux_degree].size();
            mapping_shape_functions_vol_flux_nodes[idegree][iflux_degree].reinit(n_quad_pts, n_dofs);
            for(int idim=0; idim<dim; idim++){
                gradient_mapping_shape_functions_vol_flux_nodes[idegree][iflux_degree][idim].reinit(n_quad_pts, n_dofs);
            }
            const unsigned int n_face_quad_pts = this->face_quadrature_collection[iflux_degree].size();
            for(unsigned int iface=0; iface<n_faces; iface++){
                mapping_shape_functions_face_flux_nodes[idegree][iflux_degree][iface].reinit(n_face_quad_pts, n_dofs);
                for(int idim=0; idim<dim; idim++){
                    gradient_mapping_shape_functions_face_flux_nodes[idegree][iflux_degree][iface][idim].reinit(n_face_quad_pts, n_dofs);
                }
            }

        }
         
    }
}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::create_metric_basis_operators (
                                                            const unsigned int max_grid_degree_local)
{
    //degree >=1
    for(unsigned int idegree=1; idegree<=max_grid_degree_local; idegree++){
       dealii::QGaussLobatto<1> GLL (idegree+1);
       dealii::FE_DGQArbitraryNodes<dim,dim> feq(GLL);
        dealii::FESystem<dim,dim> fe(feq, 1);
        dealii::QGaussLobatto<dim> vol_GLL(idegree +1);
        const unsigned int n_dofs = fe.dofs_per_cell;
        for(unsigned int iquad_GN=0; iquad_GN<n_dofs; iquad_GN++){
            const dealii::Point<dim> grid_node = vol_GLL.point(iquad_GN); 
            for(unsigned int idof=0; idof<n_dofs; idof++){
                mapping_shape_functions_grid_nodes[idegree][iquad_GN][idof] = fe.shape_value_component(idof,grid_node,0);
                dealii::Tensor<1,dim,real> derivative;
                derivative = fe.shape_grad_component(idof, grid_node, 0);
                for(int idim=0; idim<dim; idim++){
                    gradient_mapping_shape_functions_grid_nodes[idegree][idim][iquad_GN][idof] = derivative[idim];
                }
            }
        }
        for(unsigned int ipoly=0; ipoly<=this->max_degree; ipoly++){
            const unsigned int n_flux_quad_pts = this->volume_quadrature_collection[ipoly].size();
            for(unsigned int iquad=0; iquad<n_flux_quad_pts; iquad++){
                const dealii::Point<dim> flux_node = this->volume_quadrature_collection[ipoly].point(iquad); 
                for(unsigned int idof=0; idof<n_dofs; idof++){
                    mapping_shape_functions_vol_flux_nodes[idegree][ipoly][iquad][idof] = fe.shape_value_component(idof,flux_node,0);
                    dealii::Tensor<1,dim,real> derivative_flux;
                    derivative_flux = fe.shape_grad_component(idof, flux_node, 0);
                    for(int idim=0; idim<dim; idim++){
                        gradient_mapping_shape_functions_vol_flux_nodes[idegree][ipoly][idim][iquad][idof] = derivative_flux[idim];
                    }
                }
            }
            for(unsigned int iface=0; iface<n_faces; iface++){
                const dealii::Quadrature<dim> quadrature = dealii::QProjector<dim>::project_to_face(dealii::ReferenceCell::get_hypercube(dim),
                                                                                                    this->face_quadrature_collection[ipoly],
                                                                                                    iface);
                const unsigned int n_quad_face_pts = this->face_quadrature_collection[ipoly].size();
                for(unsigned int iquad=0; iquad<n_quad_face_pts; iquad++){
                    const dealii::Point<dim> flux_node = quadrature.point(iquad); 
                    for(unsigned int idof=0; idof<n_dofs; idof++){
                        mapping_shape_functions_face_flux_nodes[idegree][ipoly][iface][iquad][idof] = fe.shape_value_component(idof,flux_node,0);
                        dealii::Tensor<1,dim,real> derivative_flux;
                        derivative_flux = fe.shape_grad_component(idof, flux_node, 0);
                        for(int idim=0; idim<dim; idim++){
                            gradient_mapping_shape_functions_face_flux_nodes[idegree][ipoly][iface][idim][iquad][idof] = derivative_flux[idim];
                        }
                    }
                }
            }
        }
    }
}

template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::build_local_vol_determinant_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree, 
                                    const unsigned int n_quad_pts,//number volume quad pts
                                    const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
                                    const std::array<std::vector<real>,dim> &mapping_support_points,
                                    std::vector<real> &determinant_Jacobian)
{
    //mapping support points must be passed as a vector[dim][n_metric_dofs]
    assert(pow(grid_degree+1,dim) == mapping_support_points[0].size());
    assert(pow(grid_degree+1,dim) == n_metric_dofs);
    //check that the grid_degree is within the range of the metric basis
    is_the_grid_higher_order_than_initialized(grid_degree);

    std::vector<dealii::FullMatrix<real>> Jacobian(n_quad_pts);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        Jacobian[iquad].reinit(dim,dim);
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                for(unsigned int idof=0; idof<n_metric_dofs; idof++){//assume n_dofs_cell==n_quad_points
                    Jacobian[iquad][idim][jdim] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][jdim][iquad][idof]//This is wrong due to FEQ indexing 
                                                *       mapping_support_points[idim][idof];  
                }
                determinant_Jacobian[iquad] = Jacobian[iquad].determinant();
            }
        }
    }
}


template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::build_local_vol_metric_cofactor_matrix_and_det_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree, 
                                    const unsigned int n_quad_pts,//number volume quad pts
                                    const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
                                    const std::array<std::vector<real>,dim> &mapping_support_points,
                                    std::vector<real> &determinant_Jacobian,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor)
{
    //mapping support points must be passed as a vector[dim][n_metric_dofs]
    assert(pow(grid_degree+1,dim) == mapping_support_points[0].size());
    assert(pow(grid_degree+1,dim) == n_metric_dofs);
    is_the_grid_higher_order_than_initialized(grid_degree);

    std::vector<dealii::FullMatrix<real>> Jacobian(n_quad_pts);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        Jacobian[iquad].reinit(dim,dim);
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                Jacobian[iquad][idim][jdim] = 0.0;
                for(unsigned int idof=0; idof<n_metric_dofs; idof++){//assume n_dofs_cell==n_quad_points
                    Jacobian[iquad][idim][jdim] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][jdim][iquad][idof] 
                                                *       mapping_support_points[idim][idof];  
                }
            }
        }
        determinant_Jacobian[iquad] = Jacobian[iquad].determinant();
    }

    if(dim == 1){
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            metric_cofactor[iquad][0][0] = 1.0;
        }
    }
    if(dim == 2){
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            dealii::FullMatrix<real> temp(dim);
            temp.invert(Jacobian[iquad]);
            metric_cofactor[iquad].Tadd(1.0, temp);
            metric_cofactor[iquad] *= determinant_Jacobian[iquad];
        }
    }
    if(dim == 3){
        compute_local_3D_cofactor_vol(grid_degree, poly_degree, n_quad_pts, n_metric_dofs, mapping_support_points, metric_cofactor);
    }
}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::compute_local_3D_cofactor_vol(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts,
                                    const unsigned int n_metric_dofs,
                                    const std::array<std::vector<real>,dim> &mapping_support_points,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor)
{
//Invariant curl form commented out because not freestream preserving
//        std::vector<dealii::DerivativeForm<1,dim,dim>> Xl_grad_Xm(n_metric_dofs);//(x_l * \nabla(x_m)) evaluated at GRID NODES
//        compute_Xl_grad_Xm(grid_degree, n_metric_dofs, mapping_support_points, Xl_grad_Xm);
//
//        // Evaluate the physical (Y grad Z), (Z grad X), (X grad
//        std::vector<real> Ta(n_metric_dofs); 
//        std::vector<real> Tb(n_metric_dofs); 
//        std::vector<real> Tc(n_metric_dofs);
//
//        std::vector<real> Td(n_metric_dofs);
//        std::vector<real> Te(n_metric_dofs);
//        std::vector<real> Tf(n_metric_dofs);
//
//        std::vector<real> Tg(n_metric_dofs);
//        std::vector<real> Th(n_metric_dofs);
//        std::vector<real> Ti(n_metric_dofs);
//
//        for(unsigned int igrid=0; igrid<n_metric_dofs; igrid++) {
//            Ta[igrid] = 0.5*(Xl_grad_Xm[igrid][1][1] * mapping_support_points[2][igrid] - Xl_grad_Xm[igrid][2][1] * mapping_support_points[1][igrid]);
//            Tb[igrid] = 0.5*(Xl_grad_Xm[igrid][1][2] * mapping_support_points[2][igrid] - Xl_grad_Xm[igrid][2][2] * mapping_support_points[1][igrid]);
//            Tc[igrid] = 0.5*(Xl_grad_Xm[igrid][1][0] * mapping_support_points[2][igrid] - Xl_grad_Xm[igrid][2][0] * mapping_support_points[1][igrid]);
//                                                                                                                                                    
//            Td[igrid] = 0.5*(Xl_grad_Xm[igrid][2][1] * mapping_support_points[0][igrid] - Xl_grad_Xm[igrid][0][1] * mapping_support_points[2][igrid]);
//            Te[igrid] = 0.5*(Xl_grad_Xm[igrid][2][2] * mapping_support_points[0][igrid] - Xl_grad_Xm[igrid][0][2] * mapping_support_points[2][igrid]);
//            Tf[igrid] = 0.5*(Xl_grad_Xm[igrid][2][0] * mapping_support_points[0][igrid] - Xl_grad_Xm[igrid][0][0] * mapping_support_points[2][igrid]);
//                                                                                                                                                    
//            Tg[igrid] = 0.5*(Xl_grad_Xm[igrid][0][1] * mapping_support_points[1][igrid] - Xl_grad_Xm[igrid][1][1] * mapping_support_points[0][igrid]);
//            Th[igrid] = 0.5*(Xl_grad_Xm[igrid][0][2] * mapping_support_points[1][igrid] - Xl_grad_Xm[igrid][1][2] * mapping_support_points[0][igrid]);
//            Ti[igrid] = 0.5*(Xl_grad_Xm[igrid][0][0] * mapping_support_points[1][igrid] - Xl_grad_Xm[igrid][1][0] * mapping_support_points[0][igrid]);
//        }
//
//        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++) {
//
//            for(unsigned int igrid=0; igrid<n_metric_dofs; igrid++) {
//
//                metric_cofactor[iquad][0][0] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][2][iquad][igrid] * Ta[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][1][iquad][igrid] * Tb[igrid];
//
//                metric_cofactor[iquad][1][0] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][2][iquad][igrid] * Td[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][1][iquad][igrid] * Te[igrid];
//
//                metric_cofactor[iquad][2][0] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][2][iquad][igrid] * Tg[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][1][iquad][igrid] * Th[igrid];
//
//
//                metric_cofactor[iquad][0][1] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][0][iquad][igrid] * Tb[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][2][iquad][igrid] * Tc[igrid];
//
//                metric_cofactor[iquad][1][1] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][0][iquad][igrid] * Te[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][2][iquad][igrid] * Tf[igrid];
//
//                metric_cofactor[iquad][2][1] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][0][iquad][igrid] * Th[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][2][iquad][igrid] * Ti[igrid];
//
//
//                metric_cofactor[iquad][0][2] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][1][iquad][igrid] * Tc[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][0][iquad][igrid] * Ta[igrid];
//
//                metric_cofactor[iquad][1][2] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][1][iquad][igrid] * Tf[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][0][iquad][igrid] * Td[igrid];
//
//                metric_cofactor[iquad][2][2] += gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][1][iquad][igrid] * Ti[igrid] 
//                                                - gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][0][iquad][igrid] * Tg[igrid];
//
//            }
//
//        }


///compute transpose of conservative curl below
            std::vector<dealii::DerivativeForm<2,dim,dim>> grad_Xl_grad_Xm(n_quad_pts);//gradient of gradient of mapping support points at Flux nodes
                                                                                            //for the curl of interp at flux nodes
                                                                                            //ie/ \nabla ( x_l * \nabla(x_m))
            std::vector<dealii::DerivativeForm<1,dim,dim>> Xl_grad_Xm(n_metric_dofs);//(x_l * \nabla(x_m)) evaluated at GRID NODES
            compute_Xl_grad_Xm(grid_degree, n_metric_dofs, mapping_support_points, Xl_grad_Xm);
            //now get the derivative of X_l*nabla(X_m) evaluated at the quadrature/flux nodes
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        for(int kdim=0; kdim<dim; kdim++){
                            grad_Xl_grad_Xm[iquad][idim][jdim][kdim] = 0.0;
                            for(unsigned int idof=0; idof<n_metric_dofs; idof++){
                                grad_Xl_grad_Xm[iquad][idim][jdim][kdim] += 
                                            gradient_mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][kdim][iquad][idof]
                                        *   Xl_grad_Xm[idof][idim][jdim];
                            }
                        }
                    }
                }
            }
            do_curl_loop_metric_cofactor(n_quad_pts, grad_Xl_grad_Xm, metric_cofactor);

}

template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::build_local_face_metric_cofactor_matrix_and_det_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int iface,
                                    const unsigned int n_quad_pts, const unsigned int n_metric_dofs,
                                    const std::array<std::vector<real>,dim> &mapping_support_points,
                                    std::vector<real> &determinant_Jacobian,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor)
{
    //mapping support points must be passed as a vector[dim][n_metric_dofs]
    assert(pow(grid_degree+1,dim) == mapping_support_points[0].size());
    assert(pow(grid_degree+1,dim) == n_metric_dofs);
    is_the_grid_higher_order_than_initialized(grid_degree);

    std::vector<dealii::FullMatrix<real>> Jacobian(n_quad_pts);
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        Jacobian[iquad].reinit(dim,dim);
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                Jacobian[iquad][idim][jdim] = 0.0;
                for(unsigned int idof=0; idof<n_metric_dofs; idof++){//assume n_dofs_cell==n_quad_points
                    Jacobian[iquad][idim][jdim] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][jdim][iquad][idof] 
                                                *       mapping_support_points[idim][idof];  
                }
            }
        }
        determinant_Jacobian[iquad] = Jacobian[iquad].determinant();
    }

    if(dim == 1){
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            metric_cofactor[iquad][0][0] = 1.0;
        }
    }
    if(dim == 2){
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            dealii::FullMatrix<real> temp(dim);
            temp.invert(Jacobian[iquad]);
            metric_cofactor[iquad].Tadd(1.0, temp);
            metric_cofactor[iquad] *= determinant_Jacobian[iquad];
        }
    }
    if(dim == 3){
        compute_local_3D_cofactor_face(grid_degree, poly_degree, n_quad_pts, n_metric_dofs, iface, mapping_support_points, metric_cofactor);
    }

}

template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::compute_local_3D_cofactor_face(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts,
                                    const unsigned int n_metric_dofs, const unsigned int iface,
                                    const std::array<std::vector<real>,dim> &mapping_support_points,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor)
{
//compute invariant curl form on surface, commented out bc not freestream preserving
//        std::vector<dealii::DerivativeForm<1,dim,dim>> Xl_grad_Xm(n_metric_dofs);//(x_l * \nabla(x_m)) evaluated at GRID NODES
//        compute_Xl_grad_Xm(grid_degree, n_metric_dofs, mapping_support_points, Xl_grad_Xm);
//
//        // Evaluate the physical (Y grad Z), (Z grad X), (X grad
//        std::vector<real> Ta(n_metric_dofs); 
//        std::vector<real> Tb(n_metric_dofs); 
//        std::vector<real> Tc(n_metric_dofs);
//
//        std::vector<real> Td(n_metric_dofs);
//        std::vector<real> Te(n_metric_dofs);
//        std::vector<real> Tf(n_metric_dofs);
//
//        std::vector<real> Tg(n_metric_dofs);
//        std::vector<real> Th(n_metric_dofs);
//        std::vector<real> Ti(n_metric_dofs);
//
//        for(unsigned int igrid=0; igrid<n_metric_dofs; igrid++) {
//            Ta[igrid] = 0.5*(Xl_grad_Xm[igrid][1][1] * mapping_support_points[2][igrid] - Xl_grad_Xm[igrid][2][1] * mapping_support_points[1][igrid]);
//            Tb[igrid] = 0.5*(Xl_grad_Xm[igrid][1][2] * mapping_support_points[2][igrid] - Xl_grad_Xm[igrid][2][2] * mapping_support_points[1][igrid]);
//            Tc[igrid] = 0.5*(Xl_grad_Xm[igrid][1][0] * mapping_support_points[2][igrid] - Xl_grad_Xm[igrid][2][0] * mapping_support_points[1][igrid]);
//                                                                                                                                                    
//            Td[igrid] = 0.5*(Xl_grad_Xm[igrid][2][1] * mapping_support_points[0][igrid] - Xl_grad_Xm[igrid][0][1] * mapping_support_points[2][igrid]);
//            Te[igrid] = 0.5*(Xl_grad_Xm[igrid][2][2] * mapping_support_points[0][igrid] - Xl_grad_Xm[igrid][0][2] * mapping_support_points[2][igrid]);
//            Tf[igrid] = 0.5*(Xl_grad_Xm[igrid][2][0] * mapping_support_points[0][igrid] - Xl_grad_Xm[igrid][0][0] * mapping_support_points[2][igrid]);
//                                                                                                                                                    
//            Tg[igrid] = 0.5*(Xl_grad_Xm[igrid][0][1] * mapping_support_points[1][igrid] - Xl_grad_Xm[igrid][1][1] * mapping_support_points[0][igrid]);
//            Th[igrid] = 0.5*(Xl_grad_Xm[igrid][0][2] * mapping_support_points[1][igrid] - Xl_grad_Xm[igrid][1][2] * mapping_support_points[0][igrid]);
//            Ti[igrid] = 0.5*(Xl_grad_Xm[igrid][0][0] * mapping_support_points[1][igrid] - Xl_grad_Xm[igrid][1][0] * mapping_support_points[0][igrid]);
//        }
//
//        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++) {
//
//            for(unsigned int igrid=0; igrid<n_metric_dofs; igrid++) {
//
//                metric_cofactor[iquad][0][0] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][2][iquad][igrid] * Ta[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][1][iquad][igrid] * Tb[igrid];
//
//                metric_cofactor[iquad][1][0] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][2][iquad][igrid] * Td[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][1][iquad][igrid] * Te[igrid];
//
//                metric_cofactor[iquad][2][0] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][2][iquad][igrid] * Tg[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][1][iquad][igrid] * Th[igrid];
//
//
//                metric_cofactor[iquad][0][1] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][0][iquad][igrid] * Tb[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][2][iquad][igrid] * Tc[igrid];
//
//                metric_cofactor[iquad][1][1] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][0][iquad][igrid] * Te[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][2][iquad][igrid] * Tf[igrid];
//
//                metric_cofactor[iquad][2][1] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][0][iquad][igrid] * Th[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][2][iquad][igrid] * Ti[igrid];
//
//
//                metric_cofactor[iquad][0][2] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][1][iquad][igrid] * Tc[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][0][iquad][igrid] * Ta[igrid];
//
//                metric_cofactor[iquad][1][2] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][1][iquad][igrid] * Tf[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][0][iquad][igrid] * Td[igrid];
//
//                metric_cofactor[iquad][2][2] += gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][1][iquad][igrid] * Ti[igrid] 
//                                                - gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][0][iquad][igrid] * Tg[igrid];
//
//            }
//
//        }


///compute transpose of conservative curl below

            std::vector<dealii::DerivativeForm<2,dim,dim>> grad_Xl_grad_Xm(n_quad_pts);//gradient of gradient of mapping support points at Flux nodes
                                                                                            //for the curl of interp at flux nodes
                                                                                            //ie/ \nabla ( x_l * \nabla(x_m))
            std::vector<dealii::DerivativeForm<1,dim,dim>> Xl_grad_Xm(n_metric_dofs);//(x_l * \nabla(x_m)) evaluated at GRID NODES
            compute_Xl_grad_Xm(grid_degree, n_metric_dofs, mapping_support_points, Xl_grad_Xm);
            //now get the derivative of X_l*nabla(X_m) evaluated at the quadrature/flux nodes
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        for(int kdim=0; kdim<dim; kdim++){
                            grad_Xl_grad_Xm[iquad][idim][jdim][kdim] = 0.0;
                            for(unsigned int idof=0; idof<n_metric_dofs; idof++){
                                grad_Xl_grad_Xm[iquad][idim][jdim][kdim] += 
                                            gradient_mapping_shape_functions_face_flux_nodes[grid_degree][poly_degree][iface][kdim][iquad][idof]
                                        *   Xl_grad_Xm[idof][idim][jdim];
                            }
                        }
                    }
                }
            }
            do_curl_loop_metric_cofactor(n_quad_pts, grad_Xl_grad_Xm, metric_cofactor);

}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::compute_Xl_grad_Xm(
                                    const unsigned int grid_degree,
                                    const unsigned int n_metric_dofs, 
                                    const std::array<std::vector<real>,dim> &mapping_support_points,
                                    std::vector<dealii::DerivativeForm<1,dim,dim>> &Xl_grad_Xm)
{
    std::vector<dealii::DerivativeForm<1,dim,dim>> grad_Xm(n_metric_dofs);//gradient of mapping support points at Grid nodes
    for(unsigned int iquad=0; iquad<n_metric_dofs; iquad++){
        for(int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    grad_Xm[iquad][idim][jdim] =0.0;
                    for(unsigned int idof=0; idof<n_metric_dofs; idof++){
                        grad_Xm[iquad][idim][jdim] += 
                                    gradient_mapping_shape_functions_grid_nodes[grid_degree][jdim][iquad][idof]
                                *   mapping_support_points[idim][idof];
                    }
                }
        }
    }
    // X_l * \nabla(X_m) applied first at mapping support points as to have consistent normals/water-tight mesh
    for(unsigned int iquad=0; iquad<n_metric_dofs; iquad++){
        for(int ndim=0; ndim<dim; ndim++){
            int mdim, ldim;//ndim, mdim, ldim cyclic indices
            if(ndim == dim-1){
                mdim = 0;
            }
            else{
                mdim = ndim + 1;
            }
            if(ndim == 0){
                ldim = dim - 1;
            }
            else{
                ldim = ndim - 1;
            }//this computed the cyclic index loop
            for(int i=0; i<dim; ++i){
                Xl_grad_Xm[iquad][ndim][i] = mapping_support_points[ldim][iquad]
                                           * grad_Xm[iquad][mdim][i];
            }
        }
    }
}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::do_curl_loop_metric_cofactor(
                                    const unsigned int n_quad_pts,
                                    const std::vector<dealii::DerivativeForm<2,dim,dim>> grad_Xl_grad_Xm,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor)
{
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(int ndim=0; ndim<dim; ndim++){
            for(int idim=0; idim<dim; ++idim){
            int jdim, kdim;//ndim, mdim, ldim cyclic
            if(idim == dim-1){
                jdim = 0;
            }
            else{
                jdim = idim + 1;
            }
            if(idim == 0){
                kdim = dim - 1;
            }
            else{
                kdim = idim - 1;
            }//computed cyclic index loop
                metric_cofactor[iquad][ndim][idim] = - (grad_Xl_grad_Xm[iquad][ndim][kdim][jdim] - grad_Xl_grad_Xm[iquad][ndim][jdim][kdim]);
                //index is idim then ndim to be consistent with inverse of Jacobian dealii notation and 2D equivalent
            }
        }
    }
}


template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::compute_physical_to_reference(
                                    const dealii::Tensor<1,dim,real> &phys,
                                    const dealii::FullMatrix<real> &metric_cofactor,
                                    dealii::Tensor<1,dim,real> &ref)
{
    for(int idim=0; idim<dim; idim++){
        ref[idim] = 0.0;
        for(int idim2=0; idim2<dim; idim2++){
            ref[idim] += metric_cofactor[idim2][idim] * phys[idim2];
        }
    }

}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::compute_reference_to_physical(
                                    const dealii::Tensor<1,dim,real> &ref,
                                    const dealii::FullMatrix<real> &metric_cofactor,
                                    dealii::Tensor<1,dim,real> &phys)
{
    for(int idim=0; idim<dim; idim++){
        phys[idim] = 0.0;
        for(int idim2=0; idim2<dim; idim2++){
            phys[idim] += metric_cofactor[idim][idim2] * ref[idim2];
        }
    }

}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::is_the_grid_higher_order_than_initialized(
                                    const unsigned int grid_degree)
{
    if(grid_degree > this->max_grid_degree_check){
        this->pcout<<"Updating the metric basis for grid degree "<<grid_degree<<std::endl;
        allocate_metric_operators(grid_degree);
        create_metric_basis_operators(grid_degree);
        this->max_grid_degree_check = grid_degree; 
    }
}

template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::sum_factorization_matrix_vector_mult(
                                    const std::vector<real> &input_vect,
                                    std::vector<real> &output_vect,
                                    const unsigned int rows, const unsigned int columns,
                                    const unsigned int dimension,
                                    const dealii::FullMatrix<real> &basis_x,
                                    const dealii::FullMatrix<real> &basis_y,
                                    const dealii::FullMatrix<real> &basis_z)
{
    //assert that each basis matrix is of size (rows x columns)
    assert(basis_x.m() == rows);
    assert(basis_y.m() == rows);
    if(dimension == 3)
        assert(basis_z.m() == rows);
    assert(basis_x.n() == columns);
    assert(basis_y.n() == columns);
    if(dimension == 3)
        assert(basis_z.n() == columns);
    //assert the input vector is of size columns^{dim}
    assert(input_vect.size() == pow(columns, dimension));
    assert(output_vect.size() == pow(rows, dimension));

    if(dimension==2){
        //convert the input vector to matrix
        dealii::FullMatrix<real> input_mat(columns, columns);
        for(unsigned int idof=0; idof<columns; idof++){ 
            for(unsigned int jdof=0; jdof<columns; jdof++){ 
                input_mat[jdof][idof] = input_vect[idof * columns + jdof];//jdof runs fastest (x) idof slowest (y)
            }
        }
        dealii::FullMatrix<real> temp(rows, columns);
        basis_x.mmult(temp, input_mat);//apply x tensor product
        dealii::FullMatrix<real> output_mat(rows);
        basis_y.mTmult(output_mat, temp);//apply y tensor product
        //convert mat back to vect
        for(unsigned int iquad=0; iquad<rows; iquad++){
            for(unsigned int jquad=0; jquad<rows; jquad++){
                output_vect[iquad*rows + jquad] = output_mat[iquad][jquad];
            }
        }

    }
    if(dimension==3){
        //convert vect to mat first
        dealii::FullMatrix<real> input_mat(columns, columns*columns);
        for(unsigned int idof=0; idof<columns; idof++){ 
            for(unsigned int jdof=0; jdof<columns; jdof++){ 
                for(unsigned int kdof=0; kdof<columns; kdof++){
                    const unsigned int dof_index = idof*pow(columns,2) + jdof*columns + kdof;
                    input_mat[kdof][idof*columns + jdof] = input_vect[dof_index];//kdof runs fastest (x) idof slowest (z)
                }
            }
        }
        dealii::FullMatrix<real> temp(rows, columns*columns);
        basis_x.mmult(temp, input_mat);//apply x tensor product
        //convert to have y dofs ie/ change the stride
        dealii::FullMatrix<real> temp2(columns, rows * columns);
        for(unsigned int iquad=0; iquad<rows; iquad++){
            for(unsigned int idof=0; idof<columns; idof++){
                for(unsigned int jdof=0; jdof<columns; jdof++){
                    temp2[jdof][iquad*rows + idof] = temp[iquad][idof*columns + jdof];//extract y runs second fastest
                }
            }
        }
        dealii::FullMatrix<real> temp3(rows, rows * columns);
        basis_y.mmult(temp3, temp2);//apply y tensor product
        dealii::FullMatrix<real> temp4(columns, rows * rows);
        //convert to have z dofs ie/ change the stride
        for(unsigned int iquad=0; iquad<rows; iquad++){
            for(unsigned int idof=0; idof<columns; idof++){
                for(unsigned int jquad=0; jquad<rows; jquad++){
                    temp4[idof][iquad*rows + jquad] = temp3[jquad][iquad*rows + idof];//extract z runs slowest
                }
            }
        }
        dealii::FullMatrix<real> output_mat(rows, rows*rows);
        basis_z.mmult(output_mat, temp4);
        //convert mat to vect
        for(unsigned int iquad=0; iquad<rows; iquad++){
            for(unsigned int jquad=0; jquad<rows; jquad++){
                for(unsigned int kquad=0; kquad<rows; kquad++){
                    const unsigned int quad_index = iquad*pow(rows,2) + jquad*rows + kquad;
                    output_vect[quad_index] = output_mat[iquad][kquad*rows + jquad];
                }
            }
        }
    }

}
template <  int dim, typename real, int n_faces>  
void OperatorsBase<dim,real,n_faces>::sum_factorization_inner_product(
                                    const std::vector<real> &input_vect,
                                    const std::vector<real> &weight_vect,
                                    std::vector<real> &output_vect,
                                    const unsigned int rows, const unsigned int columns,
                                    const unsigned int dimension,
                                    const dealii::FullMatrix<real> &basis_x,
                                    const dealii::FullMatrix<real> &basis_y,
                                    const dealii::FullMatrix<real> &basis_z)
{
    //assert that each basis matrix is of size (rows x columns)
    assert(basis_x.m() == rows);
    assert(basis_y.m() == rows);
    if(dimension == 3)
        assert(basis_z.m() == rows);
    assert(basis_x.n() == columns);
    assert(basis_y.n() == columns);
    if(dimension == 3)
        assert(basis_z.n() == columns);
    //assert the input vector is of size columns^{dim}
    assert(input_vect.size() == pow(rows, dimension));//rows since matrix transpose
    assert(weight_vect.size() == pow(rows, dimension));
    assert(output_vect.size() == pow(rows, dimension));

    dealii::FullMatrix<real> basis_x_trans(columns, rows);
    dealii::FullMatrix<real> basis_y_trans(columns, rows);
    dealii::FullMatrix<real> basis_z_trans(columns, rows);

    //set as the transpose as inputed basis
    basis_x_trans.Tadd(1.0, basis_x);
    basis_y_trans.Tadd(1.0, basis_y);
    if(dimension==3)
        basis_z_trans.Tadd(1.0, basis_z);

    std::vector<real> new_input_vect(pow(rows,dimension));
    for(unsigned int iquad=0; iquad<pow(rows,dimension); iquad++){
        new_input_vect[iquad] = input_vect[iquad] * weight_vect[iquad];
    }

    sum_factorization_matrix_vector_mult(new_input_vect, output_vect, rows, columns, dimension, basis_x_trans, basis_y_trans, basis_z_trans);

}
/**********************************************************
    * Operators Base State below constructing the operators
    *******************************************************/
template <  int dim, typename real, int nstate,
            int n_faces>  
OperatorsBaseState<dim,real,nstate,n_faces>
::OperatorsBaseState(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : OperatorsBase<dim,real,n_faces>::OperatorsBase(parameters_input, nstate, max_degree_input, max_degree_input, grid_degree_input)
{
    this->pcout<<" Constructing Operators state ..."<<std::endl;
    this->pcout<<" ... allocating volume operators state ..."<<std::flush;
    allocate_volume_operators_state();
    this->pcout<<" done."<<std::endl;
    this->pcout<<" ... creating volume basis operators state ..."<<std::flush;
    create_vol_basis_operators_state();
    //setup surface operators
    this->pcout<<" done."<<std::endl;
    this->pcout<<" ... allocating surface operators state ..."<<std::flush;
    allocate_surface_operators_state();
    this->pcout<<" done."<<std::endl;
    this->pcout<<" ... creating surface basis operators state ..."<<std::flush;
    create_surface_basis_operators_state();
    this->pcout<<" done."<<std::endl;
    this->pcout<<" done."<<std::endl;
}
// Destructor
template <  int dim, typename real, int nstate,
            int n_faces>  
OperatorsBaseState<dim,real,nstate,n_faces>::~OperatorsBaseState ()
{
    this->pcout << "Destructing Operators state ..." << std::endl;
}

template <  int dim, typename real, int nstate,
            int n_faces>  
void OperatorsBaseState<dim,real,nstate,n_faces>::allocate_volume_operators_state()
{
    flux_basis_at_vol_cubature.resize(this->max_degree+1);
    gradient_flux_basis.resize(this->max_degree+1);
    local_flux_basis_stiffness.resize(this->max_degree+1);
    vol_integral_gradient_basis.resize(this->max_degree+1);
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        //flux basis allocator
        unsigned int n_dofs_flux = this->fe_collection_flux_basis[idegree].dofs_per_cell;
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        if(n_dofs_flux != n_quad_pts)
           this->pcout<<"flux basis not collocated on quad points"<<std::endl;
        //Note flux basis is collocated on the volume cubature nodes.
        for(int istate=0; istate<nstate; istate++){
            flux_basis_at_vol_cubature[idegree][istate].reinit(n_quad_pts, n_dofs_flux);
            for(int idim=0; idim<dim; idim++){
                gradient_flux_basis[idegree][istate][idim].reinit(n_quad_pts, n_dofs_flux);
                local_flux_basis_stiffness[idegree][istate][idim].reinit(n_dofs, n_dofs_flux);
                vol_integral_gradient_basis[idegree][istate][idim].reinit(n_quad_pts, n_dofs);
            }
        }
    }

}
template <  int dim, typename real, int nstate,
            int n_faces>  
void OperatorsBaseState<dim,real,nstate,n_faces>::create_vol_basis_operators_state()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_dofs_flux = this->fe_collection_flux_basis[idegree].dofs_per_cell;
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            const dealii::Point<dim> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
            for(int istate=0; istate<nstate; istate++){
                for(unsigned int idof=0; idof<n_dofs_flux; idof++){
                    //Flux basis function idof of poly degree idegree evaluated at cubature node qpoint.
                    flux_basis_at_vol_cubature[idegree][istate][iquad][idof] = this->fe_collection_flux_basis[idegree].shape_value_component(idof,qpoint,0);
                    dealii::Tensor<1,dim,real> derivative;
                    derivative = this->fe_collection_flux_basis[idegree].shape_grad_component(idof, qpoint, 0);
                    for(int idim=0; idim<dim; idim++){
                        gradient_flux_basis[idegree][istate][idim][iquad][idof] = derivative[idim];
                    }
                }
            }
        }
        const std::vector<real> &quad_weights = this->volume_quadrature_collection[idegree].get_weights ();
        for(unsigned int itest=0; itest<n_dofs; itest++){
            const int istate_test = this->fe_collection_basis[idegree].system_to_component_index(itest).first;
            for(unsigned int idof=0; idof<n_dofs_flux; idof++){
                dealii::Tensor<1,dim,real> value;
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    const dealii::Point<dim> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
                    dealii::Tensor<1,dim,real> derivative;
                    derivative = this->fe_collection_flux_basis[idegree].shape_grad_component(idof, qpoint, 0);
                    value += this->basis_at_vol_cubature[idegree][iquad][itest] * quad_weights[iquad] * derivative;
                }
                const int test_shape = this->fe_collection_basis[idegree].system_to_component_index(itest).second;
                    for(int idim=0; idim<dim; idim++){
                        local_flux_basis_stiffness[idegree][istate_test][idim][test_shape][idof] = value[idim]; 
                    }
            }
            const int ishape_test = this->fe_collection_basis[idegree].system_to_component_index(itest).second;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){ 
                const dealii::Point<dim> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
                dealii::Tensor<1,dim,real> derivative;
                derivative = this->fe_collection_basis[idegree].shape_grad_component(itest, qpoint, istate_test);
                for(int idim=0; idim<dim; idim++){
                    vol_integral_gradient_basis[idegree][istate_test][idim][iquad][ishape_test] = derivative[idim] * quad_weights[iquad]; 
                }
            }
        }
    }

}
template <  int dim, typename real, int nstate,
            int n_faces>  
void OperatorsBaseState<dim,real,nstate,n_faces>::allocate_surface_operators_state()
{
    flux_basis_at_facet_cubature.resize(this->max_degree+1);
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_face_pts = this->face_quadrature_collection[idegree].size();
        unsigned int n_dofs_flux = this->fe_collection_flux_basis[idegree].dofs_per_cell;
        //for flux basis by nstate
        for(int istate=0; istate<nstate; istate++){
            for(unsigned int iface=0; iface<n_faces; iface++){
                flux_basis_at_facet_cubature[idegree][istate][iface].reinit(n_quad_face_pts, n_dofs_flux);
            }
        }
    }

}
template <  int dim, typename real, int nstate,
            int n_faces>  
void OperatorsBaseState<dim,real,nstate,n_faces>::create_surface_basis_operators_state()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_dofs_flux = this->fe_collection_flux_basis[idegree].dofs_per_cell;
        unsigned int n_quad_face_pts = this->face_quadrature_collection[idegree].size();
        for(unsigned int iface=0; iface<n_faces; iface++){ 
            const dealii::Quadrature<dim> quadrature = dealii::QProjector<dim>::project_to_face(dealii::ReferenceCell::get_hypercube(dim),
                                                                                                this->face_quadrature_collection[idegree],
                                                                                                iface);
            for(unsigned int iquad=0; iquad<n_quad_face_pts; iquad++){
                for(int istate=0; istate<nstate; istate++){
                    for(unsigned int idof=0; idof<n_dofs_flux; idof++){
                        flux_basis_at_facet_cubature[idegree][istate][iface][iquad][idof] = this->fe_collection_flux_basis[idegree].shape_value_component(idof,quadrature.point(iquad),0);
                    }
                }
            }
        }
    }

}
template <  int dim, typename real, int nstate, int n_faces>  
void OperatorsBaseState<dim,real,nstate,n_faces>::get_Jacobian_scaled_physical_gradient(
                                    const bool use_conservative_divergence,
                                    const std::array<std::array<dealii::FullMatrix<real>,dim>,nstate> &ref_gradient,
                                    const std::vector<dealii::FullMatrix<real>> &metric_cofactor,
                                    const unsigned int n_quad_pts,
                                    std::array<std::array<dealii::FullMatrix<real>,dim>,nstate> &physical_gradient)
{

    for(int istate=0; istate<nstate; istate++){
        for(int idim=0; idim<dim; idim++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(unsigned int iquad2=0; iquad2<n_quad_pts; iquad2++){
                    physical_gradient[istate][idim][iquad][iquad2] = 0.0;
                    for(int jdim=0; jdim<dim; jdim++){
                        if(this->all_parameters->use_curvilinear_split_form == false){
                            if(use_conservative_divergence){//Build gradient such that when applied computes conservative divergence operator.
                                physical_gradient[istate][idim][iquad][iquad2] += metric_cofactor[iquad2][idim][jdim] * ref_gradient[istate][jdim][iquad][iquad2];
                            }
                            else{//Build gradient such that when applied computes the gradient of a scalar function.
                                physical_gradient[istate][idim][iquad][iquad2] += metric_cofactor[iquad][idim][jdim] * ref_gradient[istate][jdim][iquad][iquad2];
                            }
                        }
                        else{//Split form is half of the two forms above.
                            physical_gradient[istate][idim][iquad][iquad2] += 0.5 * ( metric_cofactor[iquad][idim][jdim] 
                                                                                   + metric_cofactor[iquad2][idim][jdim] ) 
                                                                            * ref_gradient[istate][jdim][iquad][iquad2];
                        }
                    }
                }
            }
        }
    }
    
}

template class OperatorsBase <PHILIP_DIM, double, 2*PHILIP_DIM>;

template class OperatorsBaseState <PHILIP_DIM, double, 1, 2*PHILIP_DIM>;
template class OperatorsBaseState <PHILIP_DIM, double, 2, 2*PHILIP_DIM>;
template class OperatorsBaseState <PHILIP_DIM, double, 3, 2*PHILIP_DIM>;
template class OperatorsBaseState <PHILIP_DIM, double, 4, 2*PHILIP_DIM>;
template class OperatorsBaseState <PHILIP_DIM, double, 5, 2*PHILIP_DIM>;

} // OPERATOR namespace
} // PHiLiP namespace

