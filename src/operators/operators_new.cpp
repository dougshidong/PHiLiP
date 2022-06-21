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

#include "operators_new.h"

namespace PHiLiP {
namespace OPERATOR {

//Constructor
//template <int dim, int n_faces>
//OperatorsBase<dim,n_faces>::OperatorsBase(
//    const int nstate_input,
//    const unsigned int max_degree_input,
//    const unsigned int grid_degree_input)
//    : OperatorsBase<dim,n_faces>(parameters_input, nstate_input, degree, max_degree_input, grid_degree_input, this->create_collection_tuple(max_degree_input, nstate_input, parameters_input))
//{ }
//
template <int dim, int n_faces>
OperatorsBaseNEW<dim,n_faces>::OperatorsBaseNEW(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)//,
//    const MassiveCollectionTuple collection_tuple)
    : max_degree(max_degree_input)
    , max_grid_degree(grid_degree_input)
    , nstate(nstate_input)
    , max_grid_degree_check(grid_degree_input)
//    , fe_collection_basis(std::get<0>(collection_tuple))
//    , volume_quadrature_collection(std::get<1>(collection_tuple))
//    , face_quadrature_collection(std::get<2>(collection_tuple))
//    , oned_quadrature_collection(std::get<3>(collection_tuple))
//    , fe_collection_flux_basis(std::get<4>(collection_tuple))
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{ 
}
// Destructor
template <int dim, int n_faces>
OperatorsBaseNEW<dim,n_faces>::~OperatorsBaseNEW ()
{
}

template <int dim, int n_faces>
dealii::FullMatrix<double> OperatorsBaseNEW<dim,n_faces>::tensor_product(
    const unsigned int rows, const unsigned int columns,
    const dealii::FullMatrix<double> &basis_x,
    const dealii::FullMatrix<double> &basis_y,
    const dealii::FullMatrix<double> &basis_z)
{
    //assert that each basis matrix is of size (rows x columns)
    assert(basis_x.m() == rows);
    assert(basis_y.m() == rows);
    if(dim == 3)
        assert(basis_z.m() == rows);
    assert(basis_x.n() == columns);
    assert(basis_y.n() == columns);
    if(dim == 3)
        assert(basis_z.n() == columns);

    if(dim==1)
        return basis_x;
    if(dim==2){
        dealii::FullMatrix<double> tens_prod(pow(rows,dim),pow(columns,dim));
        for(unsigned int jdof=0; jdof<rows; jdof++){
            for(unsigned int kdof=0; kdof<rows; kdof++){
                for(unsigned int ndof=0; ndof<columns; ndof++){
                    for(unsigned int odof=0; odof<columns; odof++){
                        const unsigned int index_row = rows*jdof + kdof;
                        const unsigned int index_col = columns*ndof + odof;
                        tens_prod[index_row][index_col] = basis_x[jdof][ndof] * basis_y[kdof][odof];
                    }
                }
            }
        }
        return tens_prod;
    }
    if(dim==3){
        dealii::FullMatrix<double> tens_prod(pow(rows,dim),pow(columns,dim));
        for(unsigned int idof=0; idof<rows; idof++){
            for(unsigned int jdof=0; jdof<rows; jdof++){
                for(unsigned int kdof=0; kdof<rows; kdof++){
                    for(unsigned int mdof=0; mdof<columns; mdof++){
                        for(unsigned int ndof=0; ndof<columns; ndof++){
                            for(unsigned int odof=0; odof<columns; odof++){
                                const unsigned int index_row = pow(rows,2)*idof + rows*jdof + kdof;
                                const unsigned int index_col = pow(columns,2)*mdof + columns*ndof + odof;
                                tens_prod[index_row][index_col] = basis_x[idof][mdof] * basis_y[jdof][ndof] * basis_z[kdof][odof];
                            }
                        }
                    }
                }
            }
        }
        return tens_prod;
    }
}

template <int dim, int n_faces>
dealii::FullMatrix<double> OperatorsBaseNEW<dim,n_faces>::tensor_product_state(
    const unsigned int rows, const unsigned int columns,
    const int nstate,
    const dealii::FullMatrix<double> &basis_x,
    const dealii::FullMatrix<double> &basis_y,
    const dealii::FullMatrix<double> &basis_z)
{
    //assert that each basis matrix is of size (rows x columns)
    assert(basis_x.m() == rows);
    assert(basis_y.m() == rows);
    if(dim == 3)
        assert(basis_z.m() == rows);
    assert(basis_x.n() == columns);
    assert(basis_y.n() == columns);
    if(dim == 3)
        assert(basis_z.n() == columns);

    const unsigned int rows_1state = rows / nstate;
    const unsigned int columns_1state = columns / nstate;
    dealii::FullMatrix<double> tens_prod(pow(rows,dim)*nstate, pow(columns,dim)*nstate);
    for(int istate=0; istate<nstate; istate++){
        dealii::FullMatrix<double> basis_x_1state(rows_1state, columns_1state);
        dealii::FullMatrix<double> basis_y_1state(rows_1state, columns_1state);
        dealii::FullMatrix<double> basis_z_1state(rows_1state, columns_1state);
        for(unsigned int r=0; r<rows_1state; r++){
            for(unsigned int c=0; c<columns_1state; c++){
                if(dim>=1)
                    basis_x_1state[r][c] = basis_x[istate*rows_1state + r][istate*columns_1state + c];
                if(dim>=2)
                    basis_y_1state[r][c] = basis_y[istate*rows_1state + r][istate*columns_1state + c];
                if(dim>=3)
                    basis_z_1state[r][c] = basis_z[istate*rows_1state + r][istate*columns_1state + c];
            }
        }
        dealii::FullMatrix<double> tens_prod_1state(pow(rows_1state,dim), pow(columns_1state,dim));
        tens_prod_1state = tensor_product(rows_1state, columns_1state, basis_x_1state, basis_y_1state, basis_z_1state);
        for(unsigned int r=0; r<pow(rows_1state,dim); r++){
            for(unsigned int c=0; c<pow(columns_1state,dim); c++){
                tens_prod[istate*pow(rows_1state,dim) + r][istate*pow(columns_1state,dim) + c] = tens_prod_1state[r][c];
            }
        }
    }
    return tens_prod;
}

//template <int dim, int n_faces>
//std::tuple<
//        dealii::hp::FECollection<dim>, // Solution FE basis functions
//        dealii::hp::QCollection<dim>,  // Volume quadrature
//        dealii::hp::QCollection<dim-1>, // Face quadrature
//        dealii::hp::QCollection<1>, // 1D quadrature for strong form
//        dealii::hp::FECollection<dim> >   // Flux Basis polynomials for strong form
//OperatorsBase<dim,n_faces>::create_collection_tuple(const unsigned int max_degree, const int nstate, const Parameters::AllParameters *const parameters_input) const
//{
//    dealii::hp::FECollection<dim>      fe_coll;//basis functions collection
//    dealii::hp::QCollection<dim>       volume_quad_coll;//volume flux nodes
//    dealii::hp::QCollection<dim-1>     face_quad_coll;//facet flux nodes
//    dealii::hp::QCollection<1>         oned_quad_coll;//1D flux nodes
//
//    dealii::hp::FECollection<dim>      fe_coll_lagr;//flux basis collocated on flux nodes
//
//    // for p=0, we use a p=1 FE for collocation, since there's no p=0 quadrature for Gauss Lobatto
//    if (parameters_input->use_collocated_nodes==true)
//    {
//        int degree = 1;
//
//        const dealii::FE_DGQ<dim> fe_dg(degree);
//        const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
//        fe_coll.push_back (fe_system);
//
//        dealii::Quadrature<1>     oned_quad(degree+1);
//        dealii::Quadrature<dim>   volume_quad(degree+1);
//        dealii::Quadrature<dim-1> face_quad(degree+1); //removed const
//
//        if (parameters_input->use_collocated_nodes) {
//
//            dealii::QGaussLobatto<1> oned_quad_Gauss_Lobatto (degree+1);
//            dealii::QGaussLobatto<dim> vol_quad_Gauss_Lobatto (degree+1);
//            oned_quad = oned_quad_Gauss_Lobatto;
//            volume_quad = vol_quad_Gauss_Lobatto;
//
//            if(dim == 1) {
//                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
//                face_quad = face_quad_Gauss_Legendre;
//            } else {
//                dealii::QGaussLobatto<dim-1> face_quad_Gauss_Lobatto (degree+1);
//                face_quad = face_quad_Gauss_Lobatto;
//            }
//        } else {
//            dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1);
//            dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1);
//            dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
////Commented out, to be able to handle mixed integration weights in future.
////            dealii::QGaussChebyshev<1> oned_quad_Gauss_Legendre (degree+1);
////            dealii::QGaussChebyshev<dim> vol_quad_Gauss_Legendre (degree+1);
////            if(dim == 1) {
////                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
////            face_quad = face_quad_Gauss_Legendre;
////            } else {
////                dealii::QGaussChebyshev<dim-1> face_quad_Gauss_Legendre (degree+1);
////            face_quad = face_quad_Gauss_Legendre;
////            }
//
//            oned_quad = oned_quad_Gauss_Legendre;
//            volume_quad = vol_quad_Gauss_Legendre;
//            face_quad = face_quad_Gauss_Legendre;
//        }
//
//        volume_quad_coll.push_back (volume_quad);
//        face_quad_coll.push_back (face_quad);
//        oned_quad_coll.push_back (oned_quad);
//
//        dealii::FE_DGQArbitraryNodes<dim,dim> lagrange_poly(oned_quad);
//        fe_coll_lagr.push_back (lagrange_poly);
//    }
//
//    int minimum_degree = (parameters_input->use_collocated_nodes==true) ?  1 :  0;
//    for (unsigned int degree=minimum_degree; degree<=max_degree; ++degree) {
//
//        // Solution FECollection
//        const dealii::FE_DGQ<dim> fe_dg(degree);
//        const dealii::FESystem<dim,dim> fe_system(fe_dg, nstate);
//        fe_coll.push_back (fe_system);
//
//        dealii::Quadrature<1>     oned_quad(degree+1);
//        dealii::Quadrature<dim>   volume_quad(degree+1);
//        dealii::Quadrature<dim-1> face_quad(degree+1); //removed const
//
//        if (parameters_input->use_collocated_nodes) {
//            dealii::QGaussLobatto<1> oned_quad_Gauss_Lobatto (degree+1);
//            dealii::QGaussLobatto<dim> vol_quad_Gauss_Lobatto (degree+1);
//            oned_quad = oned_quad_Gauss_Lobatto;
//            volume_quad = vol_quad_Gauss_Lobatto;
//
//            if(dim == 1)
//            {
//                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1);
//                face_quad = face_quad_Gauss_Legendre;
//            }
//            else
//            {
//                dealii::QGaussLobatto<dim-1> face_quad_Gauss_Lobatto (degree+1);
//                face_quad = face_quad_Gauss_Lobatto;
//            }
//        } else {
//            const unsigned int overintegration = parameters_input->overintegration;
//            dealii::QGauss<1> oned_quad_Gauss_Legendre (degree+1+overintegration);
//            dealii::QGauss<dim> vol_quad_Gauss_Legendre (degree+1+overintegration);
//            dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
////            dealii::QGaussChebyshev<1> oned_quad_Gauss_Legendre (degree+1+overintegration);
////            dealii::QGaussChebyshev<dim> vol_quad_Gauss_Legendre (degree+1+overintegration);
////            if(dim == 1) {
////                dealii::QGauss<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
////            face_quad = face_quad_Gauss_Legendre;
////            } else {
////                dealii::QGaussChebyshev<dim-1> face_quad_Gauss_Legendre (degree+1+overintegration);
////            face_quad = face_quad_Gauss_Legendre;
////            }
//            oned_quad = oned_quad_Gauss_Legendre;
//            volume_quad = vol_quad_Gauss_Legendre;
//            face_quad = face_quad_Gauss_Legendre;
//        }
//
//        volume_quad_coll.push_back (volume_quad);
//        face_quad_coll.push_back (face_quad);
//        oned_quad_coll.push_back (oned_quad);
//
//        dealii::FE_DGQArbitraryNodes<dim,dim> lagrange_poly(oned_quad);
//        fe_coll_lagr.push_back (lagrange_poly);
//    }
//
//    return std::make_tuple(fe_coll, volume_quad_coll, face_quad_coll, oned_quad_coll, fe_coll_lagr);
//}
template <int dim, int n_faces>
double OperatorsBaseNEW<dim,n_faces>::compute_factorial(double n)
{
    if ((n==0)||(n==1))
      return 1;
   else
      return n*compute_factorial(n-1);
}

/**********************************
*
* Sum Factorization class
*
**********************************/
//Constructor
template <int dim, int n_faces>
SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : OperatorsBaseNEW<dim,n_faces>::OperatorsBaseNEW(nstate_input, max_degree_input, grid_degree_input)
//    , oneD_vol_operator        (build_1D_volume_operator()) 
//    , oneD_surf_operator       (build_1D_surface_operator()) 
//    , oneD_grad_operator       (build_1D_gradient_operator())
//    , oneD_surf_grad_operator  (build_1D_surface_gradient_operator())
//    , oneD_diag_operator       (build_1D_diagonal_operator())
//    , oneD_diag_tensor_operator(build_1D_diagonal_tensor_operator())
{ 
}
// Destructor
template <int dim, int n_faces>
SumFactorizedOperators<dim,n_faces>::~SumFactorizedOperators()
{
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::matrix_vector_mult(
    const std::vector<double> &input_vect,
    std::vector<double> &output_vect,
    const dealii::FullMatrix<double> &basis_x,
    const dealii::FullMatrix<double> &basis_y,
    const dealii::FullMatrix<double> &basis_z,
    const bool adding,
    const double factor) 
{
    //assert that each basis matrix is of size (rows x columns)
    const unsigned int rows_x    = basis_x.m();
    const unsigned int rows_y    = basis_y.m();
    const unsigned int rows_z    = basis_z.m();
    const unsigned int columns_x = basis_x.n();
    const unsigned int columns_y = basis_y.n();
    const unsigned int columns_z = basis_z.n();
    if(dim == 1){
        assert(rows_x    == output_vect.size());
        assert(columns_x == input_vect.size());
    }
    if(dim == 2){
        assert(rows_x * rows_y       == output_vect.size());
        assert(columns_x * columns_y == input_vect.size());
    }
    if(dim == 3){
        assert(rows_x * rows_y * rows_z          == output_vect.size());
        assert(columns_x * columns_y * columns_z == input_vect.size());
    }

    if(dim==1){
        for(unsigned int iquad=0; iquad<rows_x; iquad++){
            if(!adding)
                output_vect[iquad] = 0.0;
            for(unsigned int jquad=0; jquad<columns_x; jquad++){
                output_vect[iquad] += factor * basis_x[iquad][jquad] * input_vect[jquad];
            }
        }
    }
    if(dim==2){
        //convert the input vector to matrix
        dealii::FullMatrix<double> input_mat(columns_x, columns_y);
        for(unsigned int idof=0; idof<columns_y; idof++){ 
            for(unsigned int jdof=0; jdof<columns_x; jdof++){ 
                input_mat[jdof][idof] = input_vect[idof * columns_x + jdof];//jdof runs fastest (x) idof slowest (y)
            }
        }
        dealii::FullMatrix<double> temp(rows_x, columns_y);
        basis_x.mmult(temp, input_mat);//apply x tensor product
        dealii::FullMatrix<double> output_mat(rows_y, rows_x);
        basis_y.mTmult(output_mat, temp);//apply y tensor product
        //convert mat back to vect
        for(unsigned int iquad=0; iquad<rows_y; iquad++){
            for(unsigned int jquad=0; jquad<rows_x; jquad++){
                if(adding)
                    output_vect[iquad * rows_x + jquad] += factor * output_mat[iquad][jquad];
                else
                    output_vect[iquad * rows_x + jquad] = factor * output_mat[iquad][jquad];
            }
        }

    }
    if(dim==3){
        //convert vect to mat first
        dealii::FullMatrix<double> input_mat(columns_x, columns_y * columns_z);
        for(unsigned int idof=0; idof<columns_z; idof++){ 
            for(unsigned int jdof=0; jdof<columns_y; jdof++){ 
                for(unsigned int kdof=0; kdof<columns_x; kdof++){
                    const unsigned int dof_index = idof * columns_x * columns_y + jdof * columns_x + kdof;
                    input_mat[kdof][idof * columns_y + jdof] = input_vect[dof_index];//kdof runs fastest (x) idof slowest (z)
                }
            }
        }
        dealii::FullMatrix<double> temp(rows_x, columns_y * columns_z);
        basis_x.mmult(temp, input_mat);//apply x tensor product
        //convert to have y dofs ie/ change the stride
        dealii::FullMatrix<double> temp2(columns_y, rows_x * columns_z);
        for(unsigned int iquad=0; iquad<rows_x; iquad++){
            for(unsigned int idof=0; idof<columns_z; idof++){
                for(unsigned int jdof=0; jdof<columns_y; jdof++){
                    temp2[jdof][iquad * columns_z + idof] = temp[iquad][idof * columns_y + jdof];//extract y runs second fastest
                }
            }
        }
        dealii::FullMatrix<double> temp3(rows_y, rows_x * columns_z);
        basis_y.mmult(temp3, temp2);//apply y tensor product
        dealii::FullMatrix<double> temp4(columns_z, rows_x * rows_y);
        //convert to have z dofs ie/ change the stride
        for(unsigned int iquad=0; iquad<rows_x; iquad++){
            for(unsigned int idof=0; idof<columns_z; idof++){
                for(unsigned int jquad=0; jquad<rows_y; jquad++){
                    temp4[idof][iquad * rows_y + jquad] = temp3[jquad][iquad * columns_z + idof];//extract z runs slowest
                }
            }
        }
        dealii::FullMatrix<double> output_mat(rows_z, rows_x * rows_y);
        basis_z.mmult(output_mat, temp4);
        //convert mat to vect
        for(unsigned int iquad=0; iquad<rows_z; iquad++){
            for(unsigned int jquad=0; jquad<rows_y; jquad++){
                for(unsigned int kquad=0; kquad<rows_x; kquad++){
                    const unsigned int quad_index = iquad * rows_x * rows_y + jquad * rows_x + kquad;
                    if(adding)
                        output_vect[quad_index] += factor * output_mat[iquad][kquad * rows_y + jquad];
                    else
                        output_vect[quad_index] = factor * output_mat[iquad][kquad * rows_y + jquad];
                }
            }
        }
    }

}
template <  int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::matrix_vector_mult_1D(
    const std::vector<double> &input_vect,
    std::vector<double> &output_vect,
    const dealii::FullMatrix<double> &basis_x) 
{
    this->matrix_vector_mult(input_vect, output_vect, basis_x, basis_x, basis_x);
}
template <  int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::divergence_matrix_vector_mult(
    const dealii::Tensor<1,dim,std::vector<double>> &input_vect,
    std::vector<double> &output_vect,
    const dealii::FullMatrix<double> &basis_x,
    const dealii::FullMatrix<double> &basis_y,
    const dealii::FullMatrix<double> &basis_z,
    const dealii::FullMatrix<double> &gradient_basis_x,
    const dealii::FullMatrix<double> &gradient_basis_y,
    const dealii::FullMatrix<double> &gradient_basis_z)
{
    for(int idim=0; idim<dim;idim++){
        if(idim==0)
            this->matrix_vector_mult(input_vect[idim], output_vect, 
                                     gradient_basis_x, 
                                     basis_y, 
                                     basis_z,
                                     false);//first one doesn't add in the divergence
        if(idim==1)
            this->matrix_vector_mult(input_vect[idim], output_vect, 
                                     basis_x, 
                                     gradient_basis_y, 
                                     basis_z,
                                     true);
        if(idim==2)
            this->matrix_vector_mult(input_vect[idim], output_vect, 
                                     basis_x, 
                                     basis_y,
                                     gradient_basis_z,
                                     true);
    } 
}

template <  int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::inner_product(
    const std::vector<double> &input_vect,
    const std::vector<double> &weight_vect,
    std::vector<double> &output_vect,
    const dealii::FullMatrix<double> &basis_x,
    const dealii::FullMatrix<double> &basis_y,
    const dealii::FullMatrix<double> &basis_z,
    const bool adding,
    const double factor) 
{
    //assert that each basis matrix is of size (rows x columns)
    const unsigned int rows_x    = basis_x.m();
    const unsigned int rows_y    = basis_y.m();
    const unsigned int rows_z    = basis_z.m();
    const unsigned int columns_x = basis_x.n();
    const unsigned int columns_y = basis_y.n();
    const unsigned int columns_z = basis_z.n();
    if(dim == 1){
        assert(rows_x    == output_vect.size());
        assert(columns_x == input_vect.size());
    }
    if(dim == 2){
        assert(rows_x * rows_y       == output_vect.size());
        assert(columns_x * columns_y == input_vect.size());
    }
    if(dim == 3){
        assert(rows_x * rows_y * rows_z          == output_vect.size());
        assert(columns_x * columns_y * columns_z == input_vect.size());
    }
    assert(weight_vect.size() == input_size); 

    dealii::FullMatrix<double> basis_x_trans(columns_x, rows_x);
    dealii::FullMatrix<double> basis_y_trans(columns_y, rows_y);
    dealii::FullMatrix<double> basis_z_trans(columns_z, rows_z);

    //set as the transpose as inputed basis
    basis_x_trans.Tadd(1.0, basis_x);
    basis_y_trans.Tadd(1.0, basis_y);
    basis_z_trans.Tadd(1.0, basis_z);

    std::vector<double> new_input_vect(input_vect.size());
    for(unsigned int iquad=0; iquad<input_vect.size(); iquad++){
        new_input_vect[iquad] = input_vect[iquad] * weight_vect[iquad];
    }

    this->matrix_vector_mult(new_input_vect, output_vect, basis_x_trans, basis_y_trans, basis_z_trans, adding, factor);

}
template <  int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::inner_product_1D(
    const std::vector<double> &input_vect,
    const std::vector<double> &weight_vect,
    std::vector<double> &output_vect,
    const dealii::FullMatrix<double> &basis_x) 
{
    this->inner_product(input_vect, weight_vect, output_vect, basis_x, basis_x, basis_x);
}

/*******************************************
 *
 *      VOLUME OPERATORS FUNCTIONS
 *
 *
 *      *****************************************/

template <int dim, int n_faces>  
basis_functions<dim,n_faces>::basis_functions(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
basis_functions<dim,n_faces>::~basis_functions()
{
}
template <int dim, int n_faces>  
void basis_functions<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    //allocate the basis at volume cubature
    this->oneD_vol_operator.reinit(n_quad_pts, n_dofs);
    //loop and store
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<1> qpoint  = quadrature.point(iquad);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            const int istate = finite_element.system_to_component_index(idof).first;
            //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
            this->oneD_vol_operator[iquad][idof] = finite_element.shape_value_component(idof,qpoint,istate);
        }
    }
}

template <int dim, int n_faces>  
void basis_functions<dim,n_faces>::build_1D_gradient_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    //allocate the basis at volume cubature
    this->oneD_grad_operator.reinit(n_quad_pts, n_dofs);
    //loop and store
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<1> qpoint  = quadrature.point(iquad);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            const int istate = finite_element.system_to_component_index(idof).first;
            //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
            this->oneD_grad_operator[iquad][idof] = finite_element.shape_grad_component(idof,qpoint,istate)[0];
        }
    }
}

template <int dim, int n_faces>  
void basis_functions<dim,n_faces>::build_1D_surface_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<0> &face_quadrature)
{
    const unsigned int n_face_quad_pts = face_quadrature.size();
    const unsigned int n_dofs          = finite_element.dofs_per_cell;
    const unsigned int n_faces_1D      = n_faces / dim;
    //loop and store
    for(unsigned int iface=0; iface<n_faces_1D; iface++){ 
        //allocate the facet operator
        this->oneD_surf_operator[iface].reinit(n_face_quad_pts, n_dofs);
        //sum factorized operators use a 1D element.
        const dealii::Quadrature<1> quadrature = dealii::QProjector<1>::project_to_face(dealii::ReferenceCell::get_hypercube(1),
                                                                                                face_quadrature,
                                                                                                iface);
        for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
            const dealii::Point<1> qpoint  = quadrature.point(iquad);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                const int istate = finite_element.system_to_component_index(idof).first;
                //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
                this->oneD_surf_operator[iface][iquad][idof] = finite_element.shape_value_component(idof,qpoint,istate);
            }
        }
    }
}

template <int dim, int n_faces>  
void basis_functions<dim,n_faces>::build_1D_surface_gradient_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<0> &face_quadrature)
{
    const unsigned int n_face_quad_pts = face_quadrature.size();
    const unsigned int n_dofs          = finite_element.dofs_per_cell;
    const unsigned int n_faces_1D      = n_faces / dim;
    //loop and store
    for(unsigned int iface=0; iface<n_faces_1D; iface++){ 
        //allocate the facet operator
        this->oneD_surf_grad_operator[iface].reinit(n_face_quad_pts, n_dofs);
        //sum factorized operators use a 1D element.
        const dealii::Quadrature<1> quadrature = dealii::QProjector<1>::project_to_face(dealii::ReferenceCell::get_hypercube(1),
                                                                                                face_quadrature,
                                                                                                iface);
        for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
            const dealii::Point<1> qpoint  = quadrature.point(iquad);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                const int istate = finite_element.system_to_component_index(idof).first;
                //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
                this->oneD_surf_grad_operator[iface][iquad][idof] = finite_element.shape_grad_component(idof,qpoint,istate)[0];
            }
        }
    }
}

template <int dim, int n_faces>  
vol_integral_basis<dim,n_faces>::vol_integral_basis(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
vol_integral_basis<dim,n_faces>::~vol_integral_basis()
{
}
template <int dim, int n_faces>  
void vol_integral_basis<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    //allocate
    this->oneD_vol_operator.reinit(n_quad_pts, n_dofs);
    //loop and store
    const std::vector<double> &quad_weights = quadrature.get_weights ();
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<1> qpoint  = quadrature.point(iquad);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            const int istate = finite_element.system_to_component_index(idof).first;
            //Basis function idof of poly degree idegree evaluated at cubature node qpoint multiplied by quad weight.
            this->oneD_vol_operator[iquad][idof] = quad_weights[iquad] * finite_element.shape_value_component(idof,qpoint,istate);
        }
    }
}

template <int dim, int n_faces>  
local_mass<dim,n_faces>::local_mass(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
local_mass<dim,n_faces>::~local_mass()
{
}
template <int dim, int n_faces>  
void local_mass<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    const std::vector<double> &quad_weights = quadrature.get_weights ();
    //allocate
    this->oneD_vol_operator.reinit(n_dofs,n_dofs);
    //loop and store
    for (unsigned int itest=0; itest<n_dofs; ++itest) {
        const int istate_test = finite_element.system_to_component_index(itest).first;
        for (unsigned int itrial=itest; itrial<n_dofs; ++itrial) {
            const int istate_trial = finite_element.system_to_component_index(itrial).first;
            double value = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                const dealii::Point<1> qpoint  = quadrature.point(iquad);
                value +=
                         finite_element.shape_value_component(itest,qpoint,istate_test)
                       * finite_element.shape_value_component(itrial,qpoint,istate_trial)
                       * quad_weights[iquad];                            
            }

            this->oneD_vol_operator[itrial][itest] = 0.0;
            this->oneD_vol_operator[itest][itrial] = 0.0;
            if(istate_test==istate_trial) {
                this->oneD_vol_operator[itrial][itest] = value;
                this->oneD_vol_operator[itest][itrial] = value;
            }
        }
    }
}

template <int dim, int n_faces>  
local_basis_stiffness<dim,n_faces>::local_basis_stiffness(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
local_basis_stiffness<dim,n_faces>::~local_basis_stiffness()
{
}
template <int dim, int n_faces>  
void local_basis_stiffness<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    const std::vector<double> &quad_weights = quadrature.get_weights ();
    //allocate
    this->oneD_vol_operator.reinit(n_dofs,n_dofs);
    //loop and store
    for(unsigned int itest=0; itest<n_dofs; itest++){
        const int istate_test = finite_element.system_to_component_index(itest).first;
        for(unsigned int idof=0; idof<n_dofs; idof++){
            const int istate = finite_element.system_to_component_index(idof).first;
            double value = 0.0;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                const dealii::Point<1> qpoint  = quadrature.point(iquad);
                value += finite_element.shape_value_component(itest,qpoint,istate_test)
                       * finite_element.shape_grad_component(idof, qpoint, istate)[0]//since it's a 1D operator
                       * quad_weights[iquad];
            }
            if(istate == istate_test){
                this->oneD_vol_operator[itest][idof] = value; 
            }
        }
    }
}

template <int dim, int n_faces>  
modal_basis_differential_operator<dim,n_faces>::modal_basis_differential_operator(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
modal_basis_differential_operator<dim,n_faces>::~modal_basis_differential_operator()
{
}
template <int dim, int n_faces>  
void modal_basis_differential_operator<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    local_mass<dim,n_faces> mass_matrix(this->nstate, this->max_degree, this->max_grid_degree);
    mass_matrix.build_1D_volume_operator(finite_element, quadrature);
    local_basis_stiffness<dim,n_faces> stiffness(this->nstate, this->max_degree, this->max_grid_degree);
    stiffness.build_1D_volume_operator(finite_element, quadrature);
    //allocate
    this->oneD_vol_operator.reinit(n_dofs,n_dofs);
    dealii::FullMatrix<double> inv_mass(n_dofs);
    inv_mass.invert(mass_matrix.oneD_vol_operator);
    //solves
    inv_mass.mmult(this->oneD_vol_operator, stiffness.oneD_vol_operator);
}

template <int dim, int n_faces>  
derivative_p<dim,n_faces>::derivative_p(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
derivative_p<dim,n_faces>::~derivative_p()
{
}
template <int dim, int n_faces>  
void derivative_p<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    //allocate
    this->oneD_vol_operator.reinit(n_dofs,n_dofs);
    //set as identity
    for(unsigned int idof=0; idof<n_dofs; idof++){
        this->oneD_vol_operator[idof][idof] = 1.0;//set it equal to identity
    } 
    //get modal basis differential operator
    modal_basis_differential_operator<dim,n_faces> diff_oper(this->nstate, this->max_degree, this->max_grid_degree);
    diff_oper.build_1D_volume_operator(finite_element, quadrature);
    //loop and solve
    for(unsigned int idegree=0; idegree< this->max_degree; idegree++){
       dealii::FullMatrix<double> derivative_p_temp(n_dofs, n_dofs);
       derivative_p_temp.add(1.0, this->oneD_vol_operator);
       diff_oper.oneD_vol_operator.mmult(this->oneD_vol_operator, derivative_p_temp);
    }
}

template <int dim, int n_faces>  
local_Flux_Reconstruction_operator<dim,n_faces>::local_Flux_Reconstruction_operator(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction FR_param_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , FR_param_type(FR_param_input)
{
    //get the FR corrcetion parameter value
    get_FR_correction_parameter(this->max_degree, FR_param);
}
template <int dim, int n_faces>  
local_Flux_Reconstruction_operator<dim,n_faces>::~local_Flux_Reconstruction_operator()
{
}
template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator<dim,n_faces>::get_Huynh_g2_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));//since ref element [0,1]
    c = 2.0 * (curr_cell_degree+1)/( curr_cell_degree*((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2))));  
    c/=2.0;//since orthonormal
}
template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator<dim,n_faces>::get_spectral_difference_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));
    c = 2.0 * (curr_cell_degree)/( (curr_cell_degree+1.0)*((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2))));  
    c/=2.0;//since orthonormal
}
template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator<dim,n_faces>::get_c_negative_FR_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));
    c = - 2.0 / ( pow((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2)),1.0));  
    c/=2.0;//since orthonormal
}
template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator<dim,n_faces>::get_c_negative_divided_by_two_FR_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    get_c_negative_FR_parameter(curr_cell_degree, c); 
    c/=2.0;
}
template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator<dim,n_faces>::get_c_plus_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
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

template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator<dim,n_faces>::get_FR_correction_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    if(FR_param_type == FR_enum::cHU || FR_param_type == FR_enum::cHULumped){ 
        get_Huynh_g2_parameter(curr_cell_degree, FR_param); 
    }
    else if(FR_param_type == FR_enum::cSD){ 
        get_spectral_difference_parameter(curr_cell_degree, c); 
    }
    else if(FR_param_type == FR_enum::cNegative){ 
        get_c_negative_FR_parameter(curr_cell_degree, c); 
    }
    else if(FR_param_type == FR_enum::cNegative2){ 
        get_c_negative_divided_by_two_FR_parameter(curr_cell_degree, c); 
    }
    else if(FR_param_type == FR_enum::cDG){ 
        //DG case is the 0.0 case.
        c = 0.0;
    }
    else if(FR_param_type == FR_enum::c10Thousand){ 
        //Set the value to 10000 for arbitrary high-numbers.
        c = 10000.0;
    }
    else if(FR_param_type == FR_enum::cPlus){ 
        get_c_plus_parameter(curr_cell_degree, c); 
    }
}
template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator<dim,n_faces>::build_local_Flux_Reconstruction_operator(
    const dealii::FullMatrix<double> &local_Mass_Matrix,
    const dealii::FullMatrix<double> &pth_derivative,
    const unsigned int n_dofs, 
    const double c, 
    dealii::FullMatrix<double> &Flux_Reconstruction_operator)
{
    dealii::FullMatrix<double> derivative_p_temp(n_dofs);
    derivative_p_temp.add(c, pth_derivative);
    dealii::FullMatrix<double> Flux_Reconstruction_operator_temp(n_dofs);
    derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
    Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, pth_derivative);
}


template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //build the FR correction operator
    derivative_p<dim,n_faces> pth_derivative(this->nstate, this->max_degree, this->max_grid_degree);
    pth_derivative.build_1D_volume_operator(finite_element, quadrature);

    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    //solves
    build_local_Flux_Reconstruction_operator(local_Mass_Matrix.oneD_vol_operator, pth_derivative.oneD_vol_operator, n_dofs, FR_param, this->oneD_vol_operator);
}

template <int dim, int n_faces>  
dealii::FullMatrix<double> local_Flux_Reconstruction_operator<dim,n_faces>::build_dim_Flux_Reconstruction_operator(
    const dealii::FullMatrix<double> &local_Mass_Matrix,
    const int nstate,
    const unsigned int n_dofs)
{
    const unsigned int n_dofs_1D = local_Mass_Matrix.m();
    assert(n_dofs == pow(n_dofs_1D, dim));//n_dofs passed has to be the dim sized
    dealii::FullMatrix<double> dim_FR_operator(n_dofs);
    if(dim == 1){
        dim_FR_operator = this->oneD_vol_operator;
    }
    if(dim >= 2){
        dealii::FullMatrix<double> FR1(n_dofs);
        FR1 = this->tensor_product_state(n_dofs_1D, n_dofs_1D, nstate, this->oneD_vol_operator, local_Mass_Matrix, local_Mass_Matrix);
        dealii::FullMatrix<double> FR2(n_dofs);
        FR2 = this->tensor_product_state(n_dofs_1D, n_dofs_1D, nstate, local_Mass_Matrix, this->oneD_vol_operator, local_Mass_Matrix);
        dealii::FullMatrix<double> FR_cross1(n_dofs);
        FR_cross1 = this->tensor_product_state(n_dofs_1D, n_dofs_1D, nstate, this->oneD_vol_operator, this->oneD_vol_operator, local_Mass_Matrix);
        dim_FR_operator.add(1.0, FR1, 1.0, FR2, 1.0, FR_cross1);
    }
    if(dim == 3){
        dealii::FullMatrix<double> FR3(n_dofs);
        FR3 = this->tensor_product_state(n_dofs_1D, n_dofs_1D, nstate, local_Mass_Matrix, local_Mass_Matrix, this->oneD_vol_operator);
        dealii::FullMatrix<double> FR_cross2(n_dofs);
        FR_cross2 = this->tensor_product_state(n_dofs_1D, n_dofs_1D, nstate, this->oneD_vol_operator, local_Mass_Matrix, this->oneD_vol_operator);
        dealii::FullMatrix<double> FR_cross3(n_dofs);
        FR_cross3 = this->tensor_product_state(n_dofs_1D, n_dofs_1D, nstate, local_Mass_Matrix, this->oneD_vol_operator, this->oneD_vol_operator);
        dealii::FullMatrix<double> FR_triple(n_dofs);
        FR_triple = this->tensor_product_state(n_dofs_1D, n_dofs_1D, nstate, this->oneD_vol_operator, this->oneD_vol_operator, this->oneD_vol_operator);
        dim_FR_operator.add(1.0, FR3, 1.0, FR_cross2, 1.0, FR_cross3); 
        dim_FR_operator.add(1.0, FR_triple); 
    }
    return dim_FR_operator;
    
}


template <int dim, int n_faces>  
local_Flux_Reconstruction_operator_aux<dim,n_faces>::local_Flux_Reconstruction_operator_aux(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_aux_input)
    : local_Flux_Reconstruction_operator<dim,n_faces>::local_Flux_Reconstruction_operator(nstate_input, max_degree_input, grid_degree_input, Parameters::AllParameters::Flux_Reconstruction::cDG)
    , FR_param_aux_type(FR_param_aux_input)
{
    //get the FR corrcetion parameter value
    get_FR_aux_correction_parameter(this->max_degree, FR_param_aux);
}
template <int dim, int n_faces>  
local_Flux_Reconstruction_operator_aux<dim,n_faces>::~local_Flux_Reconstruction_operator_aux()
{
}
template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator_aux<dim,n_faces>::get_FR_aux_correction_parameter (
                                const unsigned int curr_cell_degree,
                                double &k)
{
    using FR_Aux_enum = Parameters::AllParameters::Flux_Reconstruction_Aux;
    if(FR_param_aux_type == FR_Aux_enum::kHU){ 
        this->get_Huynh_g2_parameter(curr_cell_degree, k); 
    }
    else if(FR_param_aux_type == FR_Aux_enum::kSD){ 
        this->get_spectral_difference_parameter(curr_cell_degree, k); 
    }
    else if(FR_param_aux_type == FR_Aux_enum::kNegative){ 
        this->get_c_negative_FR_parameter(curr_cell_degree, k); 
    }
    else if(FR_param_aux_type == FR_Aux_enum::kNegative2){//knegative divided by 2 
        this->get_c_negative_divided_by_two_FR_parameter(curr_cell_degree, k); 
    }
    else if(FR_param_aux_type == FR_Aux_enum::kDG){ 
        k = 0.0;
    }
    else if(FR_param_aux_type == FR_Aux_enum::k10Thousand){ 
        k = 10000.0;
    }
    else if(FR_param_aux_type == FR_Aux_enum::kPlus){ 
        this->get_c_plus_parameter(curr_cell_degree, k); 
    }
}
template <int dim, int n_faces>  
void local_Flux_Reconstruction_operator_aux<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    //build the FR correction operator
    derivative_p<dim,n_faces> pth_derivative(this->nstate, this->max_degree, this->max_grid_degree);
    pth_derivative.build_1D_volume_operator(finite_element, quadrature);
    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //solves
    this->build_local_Flux_Reconstruction_operator(local_Mass_Matrix.oneD_vol_operator, pth_derivative.oneD_vol_operator, n_dofs, FR_param_aux, this->oneD_vol_operator);
}

template <int dim, int n_faces>  
vol_projection_operator<dim,n_faces>::vol_projection_operator(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
vol_projection_operator<dim,n_faces>::~vol_projection_operator()
{
}
template <int dim, int n_faces>  
void vol_projection_operator<dim,n_faces>::compute_local_vol_projection_operator(
                                const dealii::FullMatrix<double> &norm_matrix_inverse, 
                                const dealii::FullMatrix<double> &integral_vol_basis, 
                                dealii::FullMatrix<double> &volume_projection)
{
    norm_matrix_inverse.mTmult(volume_projection, integral_vol_basis);
}
template <int dim, int n_faces>  
void vol_projection_operator<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    vol_integral_basis<dim,n_faces> integral_vol_basis(this->nstate, this->max_degree, this->max_grid_degree);
    integral_vol_basis.build_1D_volume_operator(finite_element, quadrature);
    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    dealii::FullMatrix<double> mass_inv(n_dofs);
    mass_inv.invert(local_Mass_Matrix.oneD_vol_operator);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //solves
    compute_local_vol_projection_operator(mass_inv, integral_vol_basis.oneD_vol_operator, this->oneD_vol_operator);
}

template <int dim, int n_faces>  
vol_projection_operator_FR<dim,n_faces>::vol_projection_operator_FR(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction FR_param_input)
    : vol_projection_operator<dim,n_faces>::vol_projection_operator(nstate_input, max_degree_input, grid_degree_input)
    , FR_param_type(FR_param_input)
{
}
template <int dim, int n_faces>  
vol_projection_operator_FR<dim,n_faces>::~vol_projection_operator_FR()
{
}

template <int dim, int n_faces>  
void vol_projection_operator_FR<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    vol_integral_basis<dim,n_faces> integral_vol_basis(this->nstate, this->max_degree, this->max_grid_degree);
    integral_vol_basis.build_1D_volume_operator(finite_element, quadrature);
    FR_mass_inv<dim,n_faces> local_FR_Mass_Matrix_inv(this->nstate, this->max_degree, this->max_grid_degree, FR_param_type);
    local_FR_Mass_Matrix_inv.build_1D_volume_operator(finite_element, quadrature);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //solves
    this->compute_local_vol_projection_operator(local_FR_Mass_Matrix_inv.oneD_vol_operator, integral_vol_basis.oneD_vol_operator, this->oneD_vol_operator);
}

template <int dim, int n_faces>  
FR_mass_inv<dim,n_faces>::FR_mass_inv(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction FR_param_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , FR_param_type(FR_param_input)
{
}
template <int dim, int n_faces>  
FR_mass_inv<dim,n_faces>::~FR_mass_inv()
{
}
template <int dim, int n_faces>  
void FR_mass_inv<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    local_Flux_Reconstruction_operator<dim,n_faces> local_FR_oper(this->nstate, this->max_degree, this->max_grid_degree, FR_param_type);
    local_FR_oper.build_1D_volume_operator(finite_element, quadrature);
    dealii::FullMatrix<double> FR_mass_matrix(n_dofs);
    FR_mass_matrix.add(1.0, local_Mass_Matrix.oneD_vol_operator, 1.0, local_FR_oper.oneD_vol_operator);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //solves
    this->oneD_vol_operator.invert(FR_mass_matrix);
}

template <int dim, int n_faces>
vol_integral_gradient_basis<dim,n_faces>::vol_integral_gradient_basis(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>
vol_integral_gradient_basis<dim,n_faces>::~vol_integral_gradient_basis()
{
}
template <int dim, int n_faces>
void vol_integral_gradient_basis<dim,n_faces>::build_1D_gradient_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts  = quadrature.size();
    const unsigned int n_dofs      = finite_element.dofs_per_cell;
    //allocate
    this->oneD_grad_operator.reinit(n_quad_pts, n_dofs);
    //loop and store
    const std::vector<double> &quad_weights = quadrature.get_weights ();
    for(unsigned int itest=0; itest<n_dofs; itest++){
        const int istate_test = finite_element.system_to_component_index(itest).first;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){ 
            const dealii::Point<1> qpoint  = quadrature.point(iquad);
            this->oneD_grad_operator[iquad][itest] = finite_element.shape_grad_component(itest, qpoint, istate_test)[0]
                                                        * quad_weights[iquad]; 
        }
    }
}

/*************************************
*
*  SURFACE OPERATORS
*
*************************************/

//template <int dim, int n_faces>  
//basis_at_facet_cubature<dim,n_faces>::basis_at_facet_cubature(
//    const int nstate_input,
//    const unsigned int max_degree_input,
//    const unsigned int grid_degree_input)
//    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
//{
//}
//template <int dim, int n_faces>  
//basis_at_facet_cubature<dim,n_faces>::~basis_at_facet_cubature()
//{
//}
//template <int dim, int n_faces>  
//void basis_at_facet_cubature<dim,n_faces>::build_1D_surface_operator(
//    const dealii::FESystem<1,1> &finite_element,
//    const dealii::Quadrature<0> &face_quadrature)
//{
//    const unsigned int n_face_quad_pts = face_quadrature.size();
//    const unsigned int n_dofs          = finite_element.dofs_per_cell;
//    const unsigned int n_faces_1D      = n_faces / dim;
//    //loop and store
//    for(unsigned int iface=0; iface<n_faces_1D; iface++){ 
//        //allocate the facet operator
//        this->oneD_surf_operator[iface].reinit(n_face_quad_pts, n_dofs);
//        //sum factorized operators use a 1D element.
//        const dealii::Quadrature<1> quadrature = dealii::QProjector<1>::project_to_face(dealii::ReferenceCell::get_hypercube(1),
//                                                                                                face_quadrature,
//                                                                                                iface);
//        for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
//            const dealii::Point<1> qpoint  = quadrature.point(iquad);
//            for(unsigned int idof=0; idof<n_dofs; idof++){
//                const int istate = finite_element.system_to_component_index(idof).first;
//                //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
//                this->oneD_surf_operator[iface][iquad][idof] = finite_element.shape_value_component(idof,qpoint,istate);
//            }
//        }
//    }
//}

template <int dim, int n_faces>  
face_integral_basis<dim,n_faces>::face_integral_basis(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
face_integral_basis<dim,n_faces>::~face_integral_basis()
{
}
template <int dim, int n_faces>  
void face_integral_basis<dim,n_faces>::build_1D_surface_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<0> &face_quadrature)
{
    const unsigned int n_face_quad_pts = face_quadrature.size();
    const unsigned int n_dofs          = finite_element.dofs_per_cell;
    const unsigned int n_faces_1D      = n_faces / dim;
    const std::vector<double> &quad_weights = face_quadrature.get_weights ();
    //loop and store
    for(unsigned int iface=0; iface<n_faces_1D; iface++){ 
        //allocate the facet operator
        this->oneD_surf_operator[iface].reinit(n_face_quad_pts, n_dofs);
        //sum factorized operators use a 1D element.
        const dealii::Quadrature<1> quadrature = dealii::QProjector<1>::project_to_face(dealii::ReferenceCell::get_hypercube(1),
                                                                                                face_quadrature,
                                                                                                iface);
        for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
            const dealii::Point<1> qpoint  = quadrature.point(iquad);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                const int istate = finite_element.system_to_component_index(idof).first;
                //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
                this->oneD_surf_operator[iface][iquad][idof] = finite_element.shape_value_component(idof,qpoint,istate)
                                                       * quad_weights[iquad];
            }
        }
    }
}

template <int dim, int n_faces>  
lifting_operator<dim,n_faces>::lifting_operator(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
lifting_operator<dim,n_faces>::~lifting_operator()
{
}
template <int dim, int n_faces>  
void lifting_operator<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs = finite_element.dofs_per_cell;
    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //solves
    this->oneD_vol_operator.add(1.0, local_Mass_Matrix.oneD_vol_operator);
}
template <int dim, int n_faces>  
void lifting_operator<dim,n_faces>::build_local_surface_lifting_operator(
    const unsigned int n_dofs, 
    const dealii::FullMatrix<double> &norm_matrix, 
    const dealii::FullMatrix<double> &face_integral, 
    dealii::FullMatrix<double> &lifting)
{
    dealii::FullMatrix<double> norm_inv(n_dofs);
    norm_inv.invert(norm_matrix);
    norm_inv.mTmult(lifting, face_integral);
}
template <int dim, int n_faces>  
void lifting_operator<dim,n_faces>::build_1D_surface_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<0> &face_quadrature)
{
    const unsigned int n_face_quad_pts = face_quadrature.size();
    const unsigned int n_dofs          = finite_element.dofs_per_cell;
    const unsigned int n_faces_1D      = n_faces / dim;
    //create surface integral of basis functions
    face_integral_basis<dim,n_faces> basis_int_facet(this->nstate, this->max_degree, this->max_grid_degree);
    basis_int_facet.build_1D_surface_operator(finite_element, face_quadrature);
    //loop and store
    for(unsigned int iface=0; iface<n_faces_1D; iface++){
        //allocate the facet operator
        this->oneD_surf_operator[iface].reinit(n_dofs, n_face_quad_pts);
        build_local_surface_lifting_operator(n_dofs, this->oneD_vol_operator, basis_int_facet.oneD_surf_operator[iface], this->oneD_surf_operator[iface]);
    }
}

template <int dim, int n_faces>  
lifting_operator_FR<dim,n_faces>::lifting_operator_FR(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction FR_param_input)
    : lifting_operator<dim,n_faces>::lifting_operator(nstate_input, max_degree_input, grid_degree_input)
    , FR_param_type(FR_param_input)
{
}
template <int dim, int n_faces>  
lifting_operator_FR<dim,n_faces>::~lifting_operator_FR()
{
}
template <int dim, int n_faces>  
void lifting_operator_FR<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs = finite_element.dofs_per_cell;
    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    local_Flux_Reconstruction_operator<dim,n_faces> local_FR(this->nstate, this->max_degree, this->max_grid_degree, FR_param_type);
    local_FR.build_1D_volume_operator(finite_element, quadrature);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //solves
    this->oneD_vol_operator.add(1.0, local_Mass_Matrix.oneD_vol_operator);
    this->oneD_vol_operator.add(1.0, local_FR.oneD_vol_operator);
}
template <int dim, int n_faces>  
void lifting_operator_FR<dim,n_faces>::build_1D_surface_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<0> &face_quadrature)
{
    const unsigned int n_face_quad_pts = face_quadrature.size();
    const unsigned int n_dofs          = finite_element.dofs_per_cell;
    const unsigned int n_faces_1D      = n_faces / dim;
    //create surface integral of basis functions
    face_integral_basis<dim,n_faces> basis_int_facet(this->nstate, this->max_degree, this->max_grid_degree);
    basis_int_facet.build_1D_surface_operator(finite_element, face_quadrature);
    //loop and store
    for(unsigned int iface=0; iface<n_faces_1D; iface++){
        //allocate the facet operator
        this->oneD_surf_operator[iface].reinit(n_dofs, n_face_quad_pts);
        this->build_local_surface_lifting_operator(n_dofs, this->oneD_vol_operator, basis_int_facet.oneD_surf_operator[iface], this->oneD_surf_operator[iface]);
    }
}


/******************************************************************************
*
*          METRIC MAPPING OPERATORS
*
******************************************************************************/

template <int dim, int n_faces>  
mapping_shape_functions<dim,n_faces>::mapping_shape_functions(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , mapping_shape_functions_grid_nodes(nstate_input, max_degree_input, grid_degree_input)
    , mapping_shape_functions_flux_nodes(nstate_input, max_degree_input, grid_degree_input)
{
}
template <int dim, int n_faces>  
mapping_shape_functions<dim,n_faces>::~mapping_shape_functions()
{
}

template <int dim, int n_faces>  
void mapping_shape_functions<dim,n_faces>::build_1D_shape_functions_at_grid_nodes(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    assert(finite_element.dofs_per_cell == quadrature.size());//checks collocation
    mapping_shape_functions_grid_nodes.build_1D_volume_operator(finite_element, quadrature);
    mapping_shape_functions_grid_nodes.build_1D_gradient_operator(finite_element, quadrature);
}
template <int dim, int n_faces>  
void mapping_shape_functions<dim,n_faces>::build_1D_shape_functions_at_flux_nodes(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature,
    const dealii::Quadrature<0> &face_quadrature)
{
    mapping_shape_functions_flux_nodes.build_1D_volume_operator(finite_element, quadrature);
    mapping_shape_functions_flux_nodes.build_1D_gradient_operator(finite_element, quadrature);
    mapping_shape_functions_flux_nodes.build_1D_surface_operator(finite_element, face_quadrature);
    mapping_shape_functions_flux_nodes.build_1D_surface_gradient_operator(finite_element, face_quadrature);

}

//template <int dim, int n_faces>  
//void mapping_shape_functions<dim,n_faces>::build_1D_volume_operator(
//    const dealii::FESystem<1,1> &finite_element,
//    const dealii::Quadrature<1> &quadrature)
//{
//    const unsigned int n_grid_nodes  = quadrature.size();
//    const unsigned int n_dofs_metric = finite_element.dofs_per_cell;
//    //allocate
//    this->oneD_vol_operator.reinit(n_grid_nodes,n_dofs_metric);
//    //solve
//    for(unsigned int iquad=0; iquad<n_grid_nodes; iquad++){
//        const dealii::Point<1> grid_node = quadrature.point(iquad); 
//        for(unsigned int idof=0; idof<n_dofs_metric; idof++){
//            this->oneD_vol_operator[iquad][idof] = finite_element.shape_value_component(idof,grid_node,0);//metric shpae functions only 1 state always.
//        }
//    }
//}
//template <int dim, int n_faces>  
//void mapping_shape_functions<dim,n_faces>::build_1D_gradient_operator(
//    const dealii::FESystem<1,1> &finite_element,
//    const dealii::Quadrature<1> &quadrature)
//{
//    const unsigned int n_grid_nodes  = quadrature.size();
//    const unsigned int n_dofs_metric = finite_element.dofs_per_cell;
//    //allocate
//    this->oneD_grad_operator.reinit(n_grid_nodes,n_dofs_metric);
//    //solve
//    for(unsigned int iquad_GN=0; iquad_GN<n_grid_nodes; iquad_GN++){
//        const dealii::Point<1> grid_node = quadrature.point(iquad_GN); 
//        for(unsigned int idof=0; idof<n_dofs_metric; idof++){
//            this->oneD_grad_operator[iquad_GN][idof] = finite_element.shape_grad_component(idof, grid_node, 0)[0];
//        }
//    }
//}
//template <int dim, int n_faces>  
//void mapping_shape_functions<dim,n_faces>::build_1D_surface_operator(
//    const dealii::FESystem<1,1> &finite_element,
//    const dealii::Quadrature<0> &face_quadrature)
//{
//    const unsigned int n_face_flux_nodes  = face_quadrature.size();
//    const unsigned int n_dofs_metric      = finite_element.dofs_per_cell;
//    const unsigned int n_faces_1D         = n_faces / dim;
//    for(unsigned int iface=0; iface<n_faces_1D; iface++){
//        //allocate
//        this->oneD_surf_operator[iface].reinit(n_face_flux_nodes, n_dofs_metric);
//        const dealii::Quadrature<1> vol_quadrature = 
//                    dealii::QProjector<1>::project_to_face(dealii::ReferenceCell::get_hypercube(1),
//                                                             face_quadrature,
//                                                             iface);
//        for(unsigned int iquad=0; iquad<n_face_flux_nodes; iquad++){
//            const dealii::Point<1> flux_node = vol_quadrature.point(iquad); 
//            for(unsigned int idof=0; idof<n_dofs_metric; idof++){
//                this->oneD_surf_operator[iface][iquad][idof] = finite_element.shape_value_component(idof,flux_node,0);
//            }
//        }
//    }
//}
//template <int dim, int n_faces>  
//void mapping_shape_functions<dim,n_faces>::build_1D_surface_gradient_operator(
//    const dealii::FESystem<1,1> &finite_element,
//    const dealii::Quadrature<0> &face_quadrature)
//{
//    const unsigned int n_face_flux_nodes  = face_quadrature.size();
//    const unsigned int n_dofs_metric      = finite_element.dofs_per_cell;
//    const unsigned int n_faces_1D         = n_faces / dim;
//    for(unsigned int iface=0; iface<n_faces_1D; iface++){
//        //allocate
//        this->oneD_surf_grad_operator[iface].reinit(n_face_flux_nodes, n_dofs_metric);
//        const dealii::Quadrature<1> vol_quadrature = 
//                    dealii::QProjector<1>::project_to_face(dealii::ReferenceCell::get_hypercube(1),
//                                                             face_quadrature,
//                                                             iface);
//        for(unsigned int iquad=0; iquad<n_face_flux_nodes; iquad++){
//            const dealii::Point<1> flux_node = vol_quadrature.point(iquad); 
//            for(unsigned int idof=0; idof<n_dofs_metric; idof++){
//                this->oneD_surf_grad_operator[iface][iquad][idof] = finite_element.shape_grad_component(idof, flux_node, 0)[0];
//            }
//        }
//    }
//}

/***********************************************
*
*       METRIC DET JAC AND COFACTOR
*
************************************************/
//Constructor
template <typename real, int dim, int n_faces>  
metric_operators<real,dim,n_faces>::metric_operators(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const bool store_Jacobian_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , store_Jacobian(store_Jacobian_input)
{
}
//Destructor
template <typename real, int dim, int n_faces>  
metric_operators<real,dim,n_faces>::~metric_operators()
{
}
template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::transform_physical_to_reference(
    const dealii::Tensor<1,dim,real> &phys,
    const dealii::Tensor<2,dim,real> &metric_cofactor,
    dealii::Tensor<1,dim,real> &ref)
{
    for(int idim=0; idim<dim; idim++){
        ref[idim] = 0.0;
        for(int idim2=0; idim2<dim; idim2++){
            ref[idim] += metric_cofactor[idim2][idim] * phys[idim2];
        }
    }

}
template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::transform_reference_to_physical(
    const dealii::Tensor<1,dim,real> &ref,
    const dealii::Tensor<2,dim,real> &metric_cofactor,
    dealii::Tensor<1,dim,real> &phys)
{
    for(int idim=0; idim<dim; idim++){
//        phys[idim] = 0.0;
//        for(int idim2=0; idim2<dim; idim2++){
//            phys[idim] += metric_cofactor[idim][idim2] * ref[idim2];
//        }
        phys[idim] = metric_cofactor[idim] * ref;
    }

}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::build_volume_metric_operators(
    const unsigned int n_quad_pts,
    const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
    const std::array<std::vector<real>,dim> &mapping_support_points,
    mapping_shape_functions<dim,n_faces> &mapping_basis,
    const bool use_invariant_curl_form)
{
    det_Jac_vol.resize(n_quad_pts);
    for(int idim=0; idim<dim; idim++){
        for(int jdim=0; jdim<dim; jdim++){
            metric_cofactor_vol[idim][jdim].resize(n_quad_pts);
        }
    }
    //compute determinant of metric Jacobian
    build_determinant_metric_Jacobian(
        n_quad_pts,
        mapping_support_points,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator,
        det_Jac_vol);
    //compute the metric cofactor
    build_local_metric_cofactor_matrix(
        n_quad_pts,
        n_metric_dofs,
        mapping_support_points,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator,
        metric_cofactor_vol,
        use_invariant_curl_form);
}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::build_facet_metric_operators(
    const unsigned int iface,
    const unsigned int n_quad_pts,
    const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
    const std::array<std::vector<real>,dim> &mapping_support_points,
    mapping_shape_functions<dim,n_faces> &mapping_basis,
    const bool use_invariant_curl_form)
{
    det_Jac_vol.resize(n_quad_pts);
    for(int idim=0; idim<dim; idim++){
        for(int jdim=0; jdim<dim; jdim++){
            metric_cofactor_surf[idim][jdim].resize(n_quad_pts);
        }
    }
    //compute determinant of metric Jacobian
    build_determinant_metric_Jacobian(
        n_quad_pts,
        mapping_support_points,
        (iface == 0) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
            ((iface == 1) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator),
        (iface == 2) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
            ((iface == 3) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator),
        (iface == 4) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
            ((iface == 5) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator),
        (iface == 0) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[0] : 
            ((iface == 1) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator),
        (iface == 2) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[0] : 
            ((iface == 3) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator),
        (iface == 4) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[0] : 
            ((iface == 5) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator),
        det_Jac_vol);
    //compute the metric cofactor
    build_local_metric_cofactor_matrix(
        n_quad_pts,
        n_metric_dofs,
        mapping_support_points,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_vol_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_vol_operator,
        (iface == 0) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
            ((iface == 1) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator),
        (iface == 2) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
            ((iface == 3) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator),
        (iface == 4) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
            ((iface == 5) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator),
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_grad_operator,
        mapping_basis.mapping_shape_functions_grid_nodes.oneD_grad_operator,
        (iface == 0) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[0] : 
            ((iface == 1) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator),
        (iface == 2) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[0] : 
            ((iface == 3) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator),
        (iface == 4) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[0] : 
            ((iface == 5) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_grad_operator[1] : 
                mapping_basis.mapping_shape_functions_flux_nodes.oneD_grad_operator),
        metric_cofactor_surf,
        use_invariant_curl_form);
}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::build_metric_Jacobian(
    const unsigned int n_quad_pts,
    const std::array<std::vector<real>,dim> &mapping_support_points,
    const dealii::FullMatrix<double> &basis_x_flux_nodes,
    const dealii::FullMatrix<double> &basis_y_flux_nodes,
    const dealii::FullMatrix<double> &basis_z_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_x_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_y_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_z_flux_nodes,
    std::vector<dealii::Tensor<2,dim,real>> &local_Jac)
{
    for(int idim=0; idim<dim; idim++){
        for(int jdim=0; jdim<dim; jdim++){
           // std::vector<real> output_vect(pow(n_quad_pts_1D,dim));
            std::vector<real> output_vect(n_quad_pts);
            if(jdim == 0)
                this->matrix_vector_mult(mapping_support_points[idim], output_vect,
                                        grad_basis_x_flux_nodes, 
                                        basis_y_flux_nodes, 
                                        basis_z_flux_nodes);
            if(jdim == 1)
                this->matrix_vector_mult(mapping_support_points[idim], output_vect,
                                        basis_x_flux_nodes, 
                                        grad_basis_y_flux_nodes, 
                                        basis_z_flux_nodes);
            if(jdim == 2)
                this->matrix_vector_mult(mapping_support_points[idim], output_vect,
                                        basis_x_flux_nodes, 
                                        basis_y_flux_nodes, 
                                        grad_basis_z_flux_nodes);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                local_Jac[iquad][idim][jdim] = output_vect[iquad];
            }
        }
    }
}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::build_determinant_metric_Jacobian(
    const unsigned int n_quad_pts,//number volume quad pts
    const std::array<std::vector<real>,dim> &mapping_support_points,
    const dealii::FullMatrix<double> &basis_x_flux_nodes,
    const dealii::FullMatrix<double> &basis_y_flux_nodes,
    const dealii::FullMatrix<double> &basis_z_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_x_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_y_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_z_flux_nodes,
    std::vector<real> &det_metric_Jac)
{
    //mapping support points must be passed as a vector[dim][n_metric_dofs]
    assert(pow(this->max_grid_degree+1,dim) == mapping_support_points[0].size());

    std::vector<dealii::Tensor<2,dim,double>> Jacobian_flux_nodes(n_quad_pts);
    this->build_metric_Jacobian(n_quad_pts,
                                mapping_support_points, 
                                basis_x_flux_nodes, 
                                basis_y_flux_nodes, 
                                basis_z_flux_nodes, 
                                grad_basis_x_flux_nodes,
                                grad_basis_y_flux_nodes,
                                grad_basis_z_flux_nodes,
                                Jacobian_flux_nodes);
    if(store_Jacobian){
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_Jacobian_vol_cubature[idim][jdim].resize(n_quad_pts);
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    metric_Jacobian_vol_cubature[idim][jdim][iquad] = Jacobian_flux_nodes[iquad][idim][jdim];
                }
            }
        }
    }

    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        det_metric_Jac[iquad] = dealii::determinant(Jacobian_flux_nodes[iquad]);
    }
}
template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::build_local_metric_cofactor_matrix(
    const unsigned int n_quad_pts,//number flux pts
    const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
    const std::array<std::vector<real>,dim> &mapping_support_points,
    const dealii::FullMatrix<double> &basis_x_grid_nodes,
    const dealii::FullMatrix<double> &basis_y_grid_nodes,
    const dealii::FullMatrix<double> &basis_z_grid_nodes,
    const dealii::FullMatrix<double> &basis_x_flux_nodes,
    const dealii::FullMatrix<double> &basis_y_flux_nodes,
    const dealii::FullMatrix<double> &basis_z_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_x_grid_nodes,
    const dealii::FullMatrix<double> &grad_basis_y_grid_nodes,
    const dealii::FullMatrix<double> &grad_basis_z_grid_nodes,
    const dealii::FullMatrix<double> &grad_basis_x_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_y_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_z_flux_nodes,
    dealii::Tensor<2,dim,std::vector<real>> &metric_cofactor,
    const bool use_invariant_curl_form)
{
    //mapping support points must be passed as a vector[dim][n_metric_dofs]
    //Solve for Cofactor
    if(dim == 1){//constant for 1D
        std::fill(metric_cofactor[0][0].begin(), metric_cofactor[0][0].end(), 1.0);
    }
    if(dim == 2){
        std::vector<dealii::Tensor<2,dim,double>> Jacobian_flux_nodes(n_quad_pts);
        this->build_metric_Jacobian(n_quad_pts,
                                    mapping_support_points, 
                                    basis_x_flux_nodes, 
                                    basis_y_flux_nodes, 
                                    basis_z_flux_nodes, 
                                    grad_basis_x_flux_nodes,
                                    grad_basis_y_flux_nodes,
                                    grad_basis_z_flux_nodes,
                                    Jacobian_flux_nodes);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            metric_cofactor[0][0][iquad] =   Jacobian_flux_nodes[iquad][1][1];
            metric_cofactor[1][0][iquad] = - Jacobian_flux_nodes[iquad][0][1];
            metric_cofactor[0][1][iquad] = - Jacobian_flux_nodes[iquad][1][0];
            metric_cofactor[1][1][iquad] =   Jacobian_flux_nodes[iquad][0][0];
        }
    }
    if(dim == 3){
        compute_local_3D_cofactor(n_metric_dofs, 
                                  n_quad_pts,
                                  mapping_support_points, 
                                  basis_x_grid_nodes, 
                                  basis_y_grid_nodes, 
                                  basis_z_grid_nodes, 
                                  basis_x_flux_nodes,
                                  basis_y_flux_nodes,
                                  basis_z_flux_nodes,
                                  grad_basis_x_grid_nodes, 
                                  grad_basis_y_grid_nodes, 
                                  grad_basis_z_grid_nodes, 
                                  grad_basis_x_flux_nodes,
                                  grad_basis_y_flux_nodes,
                                  grad_basis_z_flux_nodes,
                                  metric_cofactor,
                                  use_invariant_curl_form);
    }
}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::compute_local_3D_cofactor(
    const unsigned int n_metric_dofs,
    const unsigned int /*n_quad_pts*/,
    const std::array<std::vector<real>,dim> &mapping_support_points,
    const dealii::FullMatrix<double> &basis_x_grid_nodes,
    const dealii::FullMatrix<double> &basis_y_grid_nodes,
    const dealii::FullMatrix<double> &basis_z_grid_nodes,
    const dealii::FullMatrix<double> &basis_x_flux_nodes,
    const dealii::FullMatrix<double> &basis_y_flux_nodes,
    const dealii::FullMatrix<double> &basis_z_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_x_grid_nodes,
    const dealii::FullMatrix<double> &grad_basis_y_grid_nodes,
    const dealii::FullMatrix<double> &grad_basis_z_grid_nodes,
    const dealii::FullMatrix<double> &grad_basis_x_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_y_flux_nodes,
    const dealii::FullMatrix<double> &grad_basis_z_flux_nodes,
    dealii::Tensor<2,dim,std::vector<real>> &metric_cofactor,
    const bool use_invariant_curl_form)
{
    //Conservative Curl Form

    //compute grad Xm, NOTE: it is the same as the Jacobian at Grid Nodes
    std::vector<dealii::Tensor<2,dim,real>> grad_Xm(n_metric_dofs);//gradient of mapping support points at Grid Nodes
    this->build_metric_Jacobian(n_metric_dofs,
                                mapping_support_points, 
                                basis_x_grid_nodes, 
                                basis_y_grid_nodes, 
                                basis_z_grid_nodes, 
                                grad_basis_x_grid_nodes,
                                grad_basis_y_grid_nodes,
                                grad_basis_z_grid_nodes,
                                grad_Xm);

    //Store Xl * grad Xm using fact the mapping shape functions collocated on Grid Nodes.
    //Also, we only store the ones needed, not all to reduce computational cost.
    std::vector<real> z_dy_dxi(n_metric_dofs);
    std::vector<real> z_dy_deta(n_metric_dofs);
    std::vector<real> z_dy_dzeta(n_metric_dofs);

    std::vector<real> x_dz_dxi(n_metric_dofs);
    std::vector<real> x_dz_deta(n_metric_dofs);
    std::vector<real> x_dz_dzeta(n_metric_dofs);

    std::vector<real> y_dx_dxi(n_metric_dofs);
    std::vector<real> y_dx_deta(n_metric_dofs);
    std::vector<real> y_dx_dzeta(n_metric_dofs);

    for(unsigned int grid_node=0; grid_node<n_metric_dofs; grid_node++){
        z_dy_dxi[grid_node]   = mapping_support_points[2][grid_node] * grad_Xm[grid_node][1][0];
        if(use_invariant_curl_form){
            z_dy_dxi[grid_node] = 0.5 * z_dy_dxi[grid_node]  
                                - 0.5 * mapping_support_points[1][grid_node] * grad_Xm[grid_node][2][0];
        }
        z_dy_deta[grid_node]  = mapping_support_points[2][grid_node] * grad_Xm[grid_node][1][1];
        if(use_invariant_curl_form){
            z_dy_deta[grid_node] = 0.5 * z_dy_deta[grid_node]  
                                 - 0.5 * mapping_support_points[1][grid_node] * grad_Xm[grid_node][2][1];
        }
        z_dy_dzeta[grid_node] = mapping_support_points[2][grid_node] * grad_Xm[grid_node][1][2];
        if(use_invariant_curl_form){
            z_dy_dzeta[grid_node] = 0.5 * z_dy_dzeta[grid_node]  
                                  - 0.5 * mapping_support_points[1][grid_node] * grad_Xm[grid_node][2][2];
        }
                                                                                         
        x_dz_dxi[grid_node]   = mapping_support_points[0][grid_node] * grad_Xm[grid_node][2][0];
        if(use_invariant_curl_form){
            x_dz_dxi[grid_node] = 0.5 * x_dz_dxi[grid_node]  
                                - 0.5 * mapping_support_points[2][grid_node] * grad_Xm[grid_node][1][0];
        }
        x_dz_deta[grid_node]  = mapping_support_points[0][grid_node] * grad_Xm[grid_node][2][1];
        if(use_invariant_curl_form){
            x_dz_deta[grid_node] = 0.5 * x_dz_deta[grid_node]  
                                 - 0.5 * mapping_support_points[2][grid_node] * grad_Xm[grid_node][1][1];
        }
        x_dz_dzeta[grid_node] = mapping_support_points[0][grid_node] * grad_Xm[grid_node][2][2];
        if(use_invariant_curl_form){
            x_dz_dzeta[grid_node] = 0.5 * x_dz_dzeta[grid_node]  
                                  - 0.5 * mapping_support_points[2][grid_node] * grad_Xm[grid_node][1][2];
        }
                                                                                         
        y_dx_dxi[grid_node]   = mapping_support_points[1][grid_node] * grad_Xm[grid_node][0][0];
        if(use_invariant_curl_form){
            y_dx_dxi[grid_node] = 0.5 * y_dx_dxi[grid_node]  
                                - 0.5 * mapping_support_points[0][grid_node] * grad_Xm[grid_node][1][0];
        }
        y_dx_deta[grid_node]  = mapping_support_points[1][grid_node] * grad_Xm[grid_node][0][1];
        if(use_invariant_curl_form){
            y_dx_deta[grid_node] = 0.5 * y_dx_deta[grid_node]  
                                 - 0.5 * mapping_support_points[0][grid_node] * grad_Xm[grid_node][1][1];
        }
        y_dx_dzeta[grid_node] = mapping_support_points[1][grid_node] * grad_Xm[grid_node][0][2];
        if(use_invariant_curl_form){
            y_dx_dzeta[grid_node] = 0.5 * y_dx_dzeta[grid_node]  
                                  - 0.5 * mapping_support_points[0][grid_node] * grad_Xm[grid_node][1][2];
        }
    }

    //Compute metric Cofactor via conservative curl form at flux nodes.
    //C11
    this->matrix_vector_mult(z_dy_dzeta, metric_cofactor[0][0],
                             basis_x_flux_nodes, grad_basis_y_flux_nodes, basis_z_flux_nodes, false, -1.0);
    this->matrix_vector_mult(z_dy_deta, metric_cofactor[0][0],
                             basis_x_flux_nodes, basis_y_flux_nodes, grad_basis_z_flux_nodes, true);
    //C12
    this->matrix_vector_mult(z_dy_dzeta, metric_cofactor[0][1],
                             grad_basis_x_flux_nodes, basis_y_flux_nodes, basis_z_flux_nodes);
    this->matrix_vector_mult(z_dy_dxi, metric_cofactor[0][1],
                             basis_x_flux_nodes, basis_y_flux_nodes, grad_basis_z_flux_nodes, true, -1.0);
    //C13
    this->matrix_vector_mult(z_dy_deta, metric_cofactor[0][2],
                             grad_basis_x_flux_nodes, basis_y_flux_nodes, basis_z_flux_nodes, false, -1.0);
    this->matrix_vector_mult(z_dy_dxi, metric_cofactor[0][2],
                             basis_x_flux_nodes, grad_basis_y_flux_nodes, basis_z_flux_nodes, true);

    //C21
    this->matrix_vector_mult(x_dz_dzeta, metric_cofactor[1][0],
                             basis_x_flux_nodes, grad_basis_y_flux_nodes, basis_z_flux_nodes, false, -1.0);
    this->matrix_vector_mult(x_dz_deta, metric_cofactor[1][0],
                             basis_x_flux_nodes, basis_y_flux_nodes, grad_basis_z_flux_nodes, true);
    //C22
    this->matrix_vector_mult(x_dz_dzeta, metric_cofactor[1][1],
                             grad_basis_x_flux_nodes, basis_y_flux_nodes, basis_z_flux_nodes);
    this->matrix_vector_mult(x_dz_dxi, metric_cofactor[1][1],
                             basis_x_flux_nodes, basis_y_flux_nodes, grad_basis_z_flux_nodes, true, -1.0);
    //C23
    this->matrix_vector_mult(x_dz_deta, metric_cofactor[1][2],
                             grad_basis_x_flux_nodes, basis_y_flux_nodes, basis_z_flux_nodes, false, -1.0);
    this->matrix_vector_mult(x_dz_dxi, metric_cofactor[1][2],
                             basis_x_flux_nodes, grad_basis_y_flux_nodes, basis_z_flux_nodes, true);

    //C31
    this->matrix_vector_mult(y_dx_dzeta, metric_cofactor[2][0],
                             basis_x_flux_nodes, grad_basis_y_flux_nodes, basis_z_flux_nodes, false, -1.0);
    this->matrix_vector_mult(y_dx_deta, metric_cofactor[2][0],
                             basis_x_flux_nodes, basis_y_flux_nodes, grad_basis_z_flux_nodes, true);
    //C32
    this->matrix_vector_mult(y_dx_dzeta, metric_cofactor[2][1],
                             grad_basis_x_flux_nodes, basis_y_flux_nodes, basis_z_flux_nodes);
    this->matrix_vector_mult(y_dx_dxi, metric_cofactor[2][1],
                             basis_x_flux_nodes, basis_y_flux_nodes, grad_basis_z_flux_nodes, true, -1.0);
    //C33
    this->matrix_vector_mult(y_dx_deta, metric_cofactor[2][2],
                             grad_basis_x_flux_nodes, basis_y_flux_nodes, basis_z_flux_nodes, false, -1.0);
    this->matrix_vector_mult(y_dx_dxi, metric_cofactor[2][2],
                             basis_x_flux_nodes, grad_basis_y_flux_nodes, basis_z_flux_nodes, true);

}

/**********************************
*
* Sum Factorization STATE class
*
**********************************/
//Constructor
template <int dim, int nstate, int n_faces>
SumFactorizedOperatorsState<dim,nstate,n_faces>::SumFactorizedOperatorsState(
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate, max_degree_input, grid_degree_input)
//    , oneD_vol_state_operator        (build_1D_volume_state_operator()) 
//    , oneD_surf_state_operator       (build_1D_surface_state_operator()) 
//    , oneD_grad_state_operator       (build_1D_gradient_state_operator())
//    , oneD_surf_grad_state_operator  (build_1D_surface_gradient_state_operator())
{ 
}
// Destructor
template <int dim, int nstate, int n_faces>
SumFactorizedOperatorsState<dim,nstate,n_faces>::~SumFactorizedOperatorsState()
{
}


template <int dim, int nstate, int n_faces>
flux_basis<dim,nstate,n_faces>::flux_basis(
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperatorsState<dim,nstate,n_faces>::SumFactorizedOperatorsState(max_degree_input, grid_degree_input)
{
}
// Destructor
template <int dim, int nstate, int n_faces>
flux_basis<dim,nstate,n_faces>::~flux_basis()
{
}
template <int dim, int nstate, int n_faces>
void flux_basis<dim,nstate,n_faces>::build_1D_volume_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    assert(n_quad_pts == n_dofs);//flux basis constructed on flux nodes
    //Note thate the flux basis should only have one state in the finite element.
    //loop and store
    for(int istate=0; istate<nstate; istate++){
        //allocate the basis at volume cubature
        this->oneD_vol_state_operator[istate].reinit(n_quad_pts, n_dofs);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            const dealii::Point<1> qpoint  = quadrature.point(iquad);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
                this->oneD_vol_state_operator[istate][iquad][idof] = finite_element.shape_value_component(idof,qpoint,0);
            }
        }
    }
}

template <int dim, int nstate, int n_faces>
void flux_basis<dim,nstate,n_faces>::build_1D_gradient_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    assert(n_quad_pts == n_dofs);//flux basis constructed on flux nodes
    //loop and store
    for(int istate=0; istate<nstate; istate++){
        //allocate 
        this->oneD_grad_state_operator[istate].reinit(n_quad_pts, n_dofs);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            const dealii::Point<1> qpoint  = quadrature.point(iquad);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                this->oneD_grad_state_operator[istate][iquad][idof] = finite_element.shape_grad_component(idof, qpoint, 0)[0];
            }
        }
    }
}
template <int dim, int nstate, int n_faces>
void flux_basis<dim,nstate,n_faces>::build_1D_surface_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<0> &face_quadrature)
{
    const unsigned int n_face_quad_pts = face_quadrature.size();
    const unsigned int n_dofs          = finite_element.dofs_per_cell;
    const unsigned int n_faces_1D      = n_faces / dim;
    //loop and store
    for(unsigned int iface=0; iface<n_faces_1D; iface++){ 
        const dealii::Quadrature<1> quadrature = dealii::QProjector<1>::project_to_face(dealii::ReferenceCell::get_hypercube(1),
                                                                                            face_quadrature,
                                                                                            iface);
        for(int istate=0; istate<nstate; istate++){
            this->oneD_surf_state_operator[istate][iface].reinit(n_face_quad_pts, n_dofs);
            for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
                for(unsigned int idof=0; idof<n_dofs; idof++){
                    this->oneD_surf_state_operator[istate][iface][iquad][idof] = finite_element.shape_value_component(idof,quadrature.point(iquad),0);
                }
            }
        }
    }
}

template <int dim, int nstate, int n_faces>
local_flux_basis_stiffness<dim,nstate,n_faces>::local_flux_basis_stiffness(
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : flux_basis<dim,nstate,n_faces>::flux_basis(max_degree_input, grid_degree_input)
{
}
// Destructor
template <int dim, int nstate, int n_faces>
local_flux_basis_stiffness<dim,nstate,n_faces>::~local_flux_basis_stiffness()
{
}
template <int dim, int nstate, int n_faces>
void local_flux_basis_stiffness<dim,nstate,n_faces>::build_1D_volume_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts  = quadrature.size();
    const unsigned int n_dofs_flux = quadrature.size();
    const unsigned int n_dofs      = finite_element.dofs_per_cell;
    //loop and store
    const std::vector<double> &quad_weights = quadrature.get_weights ();
    for(int istate_flux=0; istate_flux<nstate; istate_flux++){
        //allocate
        this->oneD_vol_state_operator[istate_flux].reinit(n_dofs, n_quad_pts);
        for(unsigned int itest=0; itest<n_dofs; itest++){
            const int istate_test = finite_element.system_to_component_index(itest).first;
            for(unsigned int idof=0; idof<n_dofs_flux; idof++){
                double value = 0.0;
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    const dealii::Point<1> qpoint  = quadrature.point(iquad);
                    value += finite_element.shape_value_component(itest, qpoint, istate_test) 
                           * quad_weights[iquad] 
                           * this->oneD_grad_state_operator[istate_flux][iquad][idof];
                }
                this->oneD_vol_state_operator[istate_flux][itest][idof] = value; 
            }
        }
    }
}









/**************************************************
*
*   OLD STUFF BELOW LOL
*
**************************************************/

#if 0



template <int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::allocate_volume_operators ()
{

    //basis functions evaluated at volume cubature (flux) nodes
    basis_functions.resize(this->max_degree+1);
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
        basis_functions[idegree].reinit(n_quad_pts, n_dofs);
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

template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>::create_vol_basis_operators ()
{

    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            const dealii::Point<1> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
            const std::vector<double> &quad_weights = this->volume_quadrature_collection[idegree].get_weights ();
            for(unsigned int idof=0; idof<n_dofs; idof++){
                const int istate = this->fe_collection_basis[idegree].system_to_component_index(idof).first;
                //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
                basis_functions[idegree][iquad][idof] = this->fe_collection_basis[idegree].shape_value_component(idof,qpoint,istate);
                //Basis function idof of poly degree idegree evaluated at cubature node qpoint multiplied by quad weight.
                vol_integral_basis[idegree][iquad][idof] = quad_weights[iquad] * basis_functions[idegree][iquad][idof];
            }
        }
    }
}


template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>::build_local_Mass_Matrix (
                                const std::vector<double> &quad_weights,
                                const unsigned int n_dofs_cell, const unsigned int n_quad_pts,
                                const int current_fe_index,
                                dealii::FullMatrix<double> &Mass_Matrix)
{
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        const int istate_test = this->fe_collection_basis[current_fe_index].system_to_component_index(itest).first;

        for (unsigned int itrial=itest; itrial<n_dofs_cell; ++itrial) {

            const int istate_trial = this->fe_collection_basis[current_fe_index].system_to_component_index(itrial).first;

            double value = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                value +=
                        basis_functions[current_fe_index][iquad][itest] 
                    *   basis_functions[current_fe_index][iquad][itrial] 
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

template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>::build_Mass_Matrix_operators ()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs_cell = this->fe_collection_basis[idegree].dofs_per_cell;
        const std::vector<double> &quad_weights = this->volume_quadrature_collection[idegree].get_weights ();
        build_local_Mass_Matrix(quad_weights, n_dofs_cell, n_quad_pts, idegree, local_mass[idegree]);
    }
}
template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>::build_Stiffness_Matrix_operators ()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        const std::vector<double> &quad_weights = this->volume_quadrature_collection[idegree].get_weights ();
        for(unsigned int itest=0; itest<n_dofs; itest++){
            const int istate_test = this->fe_collection_basis[idegree].system_to_component_index(itest).first;
            for(unsigned int idof=0; idof<n_dofs; idof++){
                const int istate = this->fe_collection_basis[idegree].system_to_component_index(idof).first;
                dealii::Tensor<1,dim,double> value;
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    const dealii::Point<1> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
                    dealii::Tensor<1,dim,double> derivative;
                    derivative = this->fe_collection_basis[idegree].shape_grad_component(idof, qpoint, istate);
                    value += basis_functions[idegree][iquad][itest] * quad_weights[iquad] * derivative;
                }
                if(istate == istate_test){
                    for(int idim=0; idim<dim; idim++){
                        local_basis_stiffness[idegree][idim][itest][idof] = value[idim]; 
                    }
                }
            }
        }
        for(int idim=0; idim<dim; idim++){
            dealii::FullMatrix<double> inv_mass(n_dofs);
            inv_mass.invert(local_mass[idegree]);
            inv_mass.mmult(modal_basis_differential_operator[idegree][idim],local_basis_stiffness[idegree][idim]);
        }
    }

}
template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>::get_higher_derivatives ()
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
               dealii::FullMatrix<double> derivative_p_temp(n_dofs_cell, n_dofs_cell);
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

template <int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::get_Huynh_g2_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));//since ref element [0,1]
    c = 2.0 * (curr_cell_degree+1)/( curr_cell_degree*((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2))));  
    c/=2.0;//since orthonormal
}
template <int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::get_spectral_difference_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));
    c = 2.0 * (curr_cell_degree)/( (curr_cell_degree+1.0)*((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2))));  
    c/=2.0;//since orthonormal
}
template <int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::get_c_negative_FR_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    const double pfact = this->compute_factorial(curr_cell_degree);
    const double pfact2 = this->compute_factorial(2.0 * curr_cell_degree);
    double cp = pfact2/(pow(pfact,2));
    c = - 2.0 / ( pow((2.0*curr_cell_degree+1.0)*(pow(pfact*cp,2)),1.0));  
    c/=2.0;//since orthonormal
}
template <int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::get_c_negative_divided_by_two_FR_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
{
    get_c_negative_FR_parameter(curr_cell_degree, c); 
    c/=2.0;
}
template <int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::get_c_plus_parameter (
                                const unsigned int curr_cell_degree,
                                double &c)
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

template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>::get_FR_correction_parameter (
                                const unsigned int curr_cell_degree,
                                double &c, real &k)
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
template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>
::build_local_Flux_Reconstruction_operator(
                                const dealii::FullMatrix<double> &local_Mass_Matrix,
                                const unsigned int  n_dofs_cell, const unsigned int degree_index, 
                                dealii::FullMatrix<double> &Flux_Reconstruction_operator)
{
    double c = 0.0;
//get the Flux_Reconstruction operator
    c = c_param_FR[degree_index];
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<double> derivative_p_temp(n_dofs_cell, n_dofs_cell);
        derivative_p_temp.add(c, derivative_p[degree_index][idim]);
        dealii::FullMatrix<double> Flux_Reconstruction_operator_temp(n_dofs_cell);
        derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
        Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, derivative_p[degree_index][idim], true);
    }
    if(dim>=2){
        const int deriv_2p_loop = (dim==2) ? 1 : dim; 
        double c_2 = pow(c,2.0);
        for(int idim=0; idim<deriv_2p_loop; idim++){
            dealii::FullMatrix<double> derivative_p_temp(n_dofs_cell, n_dofs_cell);
            derivative_p_temp.add(c_2, derivative_2p[degree_index][idim]);
            dealii::FullMatrix<double> Flux_Reconstruction_operator_temp(n_dofs_cell);
            derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
            Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, derivative_2p[degree_index][idim], true);
        }
    }
    if(dim == 3){
        double c_3 = pow(c,3.0);
        dealii::FullMatrix<double> derivative_p_temp(n_dofs_cell, n_dofs_cell);
        derivative_p_temp.add(c_3, derivative_3p[degree_index]);
        dealii::FullMatrix<double> Flux_Reconstruction_operator_temp(n_dofs_cell);
        derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
        Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, derivative_3p[degree_index], true);
    }
    
}
template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>
::build_local_Flux_Reconstruction_operator_AUX(
                                const dealii::FullMatrix<double> &local_Mass_Matrix,
                                const unsigned int n_dofs_cell, const unsigned int degree_index, 
                                std::array<dealii::FullMatrix<double>,dim> &Flux_Reconstruction_operator_aux)
{
    double k = 0.0;
//get the Flux_Reconstruction AUX operator
    k = k_param_FR[degree_index];
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<double> derivative_p_temp2(n_dofs_cell, n_dofs_cell);
        dealii::FullMatrix<double> Flux_Reconstruction_operator_temp(n_dofs_cell);
        derivative_p_temp2.add(k,derivative_p[degree_index][idim]);
        derivative_p_temp2.Tmmult(Flux_Reconstruction_operator_temp, local_Mass_Matrix);
        Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator_aux[idim], derivative_p[degree_index][idim]);
    }
    
}
template <  int dim, typename double,
            int n_faces>  
void OperatorsBase<dim,n_faces>::build_Flux_Reconstruction_operators ()
{
    for(unsigned int degree_index=0; degree_index<=this->max_degree; degree_index++){
        unsigned int n_dofs_cell = this->fe_collection_basis[degree_index].dofs_per_cell;
        unsigned int curr_cell_degree = degree_index; 
        get_FR_correction_parameter(curr_cell_degree, c_param_FR[degree_index], k_param_FR[degree_index]);
        build_local_Flux_Reconstruction_operator(local_mass[degree_index], n_dofs_cell, degree_index, local_Flux_Reconstruction_operator[degree_index]);
        build_local_Flux_Reconstruction_operator_AUX(local_mass[degree_index], n_dofs_cell, degree_index, local_Flux_Reconstruction_operator_aux[degree_index]);
    }
}

template <int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::compute_local_vol_projection_operator(
                                const unsigned int degree_index, 
                                const unsigned int n_dofs_cell,
                                const dealii::FullMatrix<double> &norm_matrix, 
                                dealii::FullMatrix<double> &volume_projection)
{
    dealii::FullMatrix<double> norm_inv(n_dofs_cell);
    norm_inv.invert(norm_matrix);
    norm_inv.mTmult(volume_projection, vol_integral_basis[degree_index]);
}
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::get_vol_projection_operators ()
{
    for(unsigned int degree_index=0; degree_index<=this->max_degree; degree_index++){
        unsigned int n_dofs = this->fe_collection_basis[degree_index].dofs_per_cell;
        compute_local_vol_projection_operator(degree_index, n_dofs, local_mass[degree_index], vol_projection_operator[degree_index]);
        dealii::FullMatrix<double> M_plus_Flux_Reconstruction(n_dofs);
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
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::allocate_surface_operators ()
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
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::create_surface_basis_operators ()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        unsigned int n_quad_face_pts = this->face_quadrature_collection[idegree].size();
        const std::vector<double> &quad_weights = this->face_quadrature_collection[idegree].get_weights ();
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
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::build_local_surface_lifting_operator (
                                const unsigned int degree_index, 
                                const unsigned int n_dofs_cell, 
                                const unsigned int face_number, 
                                const dealii::FullMatrix<double> &norm_matrix, 
                                dealii::FullMatrix<double> &lifting)
{
    dealii::FullMatrix<double> norm_inv(n_dofs_cell);
    norm_inv.invert(norm_matrix);
    norm_matrix.mTmult(lifting, face_integral_basis[degree_index][face_number]);
}
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::get_surface_lifting_operators ()
{
    for(unsigned int degree_index=0; degree_index<=this->max_degree; degree_index++){
        unsigned int n_dofs = this->fe_collection_basis[degree_index].dofs_per_cell;
        for(unsigned int iface=0; iface<n_faces; iface++){
            build_local_surface_lifting_operator(degree_index, n_dofs, iface, local_mass[degree_index], lifting_operator[degree_index][iface]);
            dealii::FullMatrix<double> M_plus_Flux_Reconstruction(n_dofs);
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
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::allocate_metric_operators (
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
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::create_metric_basis_operators (
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
            const dealii::Point<1> grid_node = vol_GLL.point(iquad_GN); 
            for(unsigned int idof=0; idof<n_dofs; idof++){
                mapping_shape_functions_grid_nodes[idegree][iquad_GN][idof] = fe.shape_value_component(idof,grid_node,0);
                dealii::Tensor<1,dim,double> derivative;
                derivative = fe.shape_grad_component(idof, grid_node, 0);
                for(int idim=0; idim<dim; idim++){
                    gradient_mapping_shape_functions_grid_nodes[idegree][idim][iquad_GN][idof] = derivative[idim];
                }
            }
        }
        for(unsigned int ipoly=0; ipoly<=this->max_degree; ipoly++){
            const unsigned int n_flux_quad_pts = this->volume_quadrature_collection[ipoly].size();
            for(unsigned int iquad=0; iquad<n_flux_quad_pts; iquad++){
                const dealii::Point<1> flux_node = this->volume_quadrature_collection[ipoly].point(iquad); 
                for(unsigned int idof=0; idof<n_dofs; idof++){
                    mapping_shape_functions_vol_flux_nodes[idegree][ipoly][iquad][idof] = fe.shape_value_component(idof,flux_node,0);
                    dealii::Tensor<1,dim,double> derivative_flux;
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
                    const dealii::Point<1> flux_node = quadrature.point(iquad); 
                    for(unsigned int idof=0; idof<n_dofs; idof++){
                        mapping_shape_functions_face_flux_nodes[idegree][ipoly][iface][iquad][idof] = fe.shape_value_component(idof,flux_node,0);
                        dealii::Tensor<1,dim,double> derivative_flux;
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

template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::build_local_vol_determinant_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree, 
                                    const unsigned int n_quad_pts,//number volume quad pts
                                    const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
                                    const std::array<std::vector<double>,dim> &mapping_support_points,
                                    std::vector<double> &determinant_Jacobian)
{
    //mapping support points must be passed as a vector[dim][n_metric_dofs]
    assert(pow(grid_degree+1,dim) == mapping_support_points[0].size());
    assert(pow(grid_degree+1,dim) == n_metric_dofs);
    //check that the grid_degree is within the range of the metric basis
    is_the_grid_higher_order_than_initialized(grid_degree);

    std::vector<dealii::FullMatrix<double>> Jacobian(n_quad_pts);
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


template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::build_local_vol_metric_cofactor_matrix_and_det_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree, 
                                    const unsigned int n_quad_pts,//number volume quad pts
                                    const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
                                    const std::array<std::vector<double>,dim> &mapping_support_points,
                                    std::vector<double> &determinant_Jacobian,
                                    std::vector<dealii::FullMatrix<double>> &metric_cofactor)
{
    //mapping support points must be passed as a vector[dim][n_metric_dofs]
    assert(pow(grid_degree+1,dim) == mapping_support_points[0].size());
    assert(pow(grid_degree+1,dim) == n_metric_dofs);
    is_the_grid_higher_order_than_initialized(grid_degree);

    std::vector<dealii::FullMatrix<double>> Jacobian(n_quad_pts);
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
            dealii::FullMatrix<double> temp(dim);
            temp.invert(Jacobian[iquad]);
            metric_cofactor[iquad].Tadd(1.0, temp);
            metric_cofactor[iquad] *= determinant_Jacobian[iquad];
        }
    }
    if(dim == 3){
        compute_local_3D_cofactor_vol(grid_degree, poly_degree, n_quad_pts, n_metric_dofs, mapping_support_points, metric_cofactor);
    }
}
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::compute_local_3D_cofactor_vol(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts,
                                    const unsigned int n_metric_dofs,
                                    const std::array<std::vector<double>,dim> &mapping_support_points,
                                    std::vector<dealii::FullMatrix<double>> &metric_cofactor)
{
//Invariant curl form commented out because not freestream preserving
//        std::vector<dealii::DerivativeForm<1,dim,dim>> Xl_grad_Xm(n_metric_dofs);//(x_l * \nabla(x_m)) evaluated at GRID NODES
//        compute_Xl_grad_Xm(grid_degree, n_metric_dofs, mapping_support_points, Xl_grad_Xm);
//
//        // Evaluate the physical (Y grad Z), (Z grad X), (X grad
//        std::vector<double> Ta(n_metric_dofs); 
//        std::vector<double> Tb(n_metric_dofs); 
//        std::vector<double> Tc(n_metric_dofs);
//
//        std::vector<double> Td(n_metric_dofs);
//        std::vector<double> Te(n_metric_dofs);
//        std::vector<double> Tf(n_metric_dofs);
//
//        std::vector<double> Tg(n_metric_dofs);
//        std::vector<double> Th(n_metric_dofs);
//        std::vector<double> Ti(n_metric_dofs);
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

template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::build_local_face_metric_cofactor_matrix_and_det_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int iface,
                                    const unsigned int n_quad_pts, const unsigned int n_metric_dofs,
                                    const std::array<std::vector<double>,dim> &mapping_support_points,
                                    std::vector<double> &determinant_Jacobian,
                                    std::vector<dealii::FullMatrix<double>> &metric_cofactor)
{
    //mapping support points must be passed as a vector[dim][n_metric_dofs]
    assert(pow(grid_degree+1,dim) == mapping_support_points[0].size());
    assert(pow(grid_degree+1,dim) == n_metric_dofs);
    is_the_grid_higher_order_than_initialized(grid_degree);

    std::vector<dealii::FullMatrix<double>> Jacobian(n_quad_pts);
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
            dealii::FullMatrix<double> temp(dim);
            temp.invert(Jacobian[iquad]);
            metric_cofactor[iquad].Tadd(1.0, temp);
            metric_cofactor[iquad] *= determinant_Jacobian[iquad];
        }
    }
    if(dim == 3){
        compute_local_3D_cofactor_face(grid_degree, poly_degree, n_quad_pts, n_metric_dofs, iface, mapping_support_points, metric_cofactor);
    }

}

template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::compute_local_3D_cofactor_face(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts,
                                    const unsigned int n_metric_dofs, const unsigned int iface,
                                    const std::array<std::vector<double>,dim> &mapping_support_points,
                                    std::vector<dealii::FullMatrix<double>> &metric_cofactor)
{
//compute invariant curl form on surface, commented out bc not freestream preserving
//        std::vector<dealii::DerivativeForm<1,dim,dim>> Xl_grad_Xm(n_metric_dofs);//(x_l * \nabla(x_m)) evaluated at GRID NODES
//        compute_Xl_grad_Xm(grid_degree, n_metric_dofs, mapping_support_points, Xl_grad_Xm);
//
//        // Evaluate the physical (Y grad Z), (Z grad X), (X grad
//        std::vector<double> Ta(n_metric_dofs); 
//        std::vector<double> Tb(n_metric_dofs); 
//        std::vector<double> Tc(n_metric_dofs);
//
//        std::vector<double> Td(n_metric_dofs);
//        std::vector<double> Te(n_metric_dofs);
//        std::vector<double> Tf(n_metric_dofs);
//
//        std::vector<double> Tg(n_metric_dofs);
//        std::vector<double> Th(n_metric_dofs);
//        std::vector<double> Ti(n_metric_dofs);
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
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::compute_Xl_grad_Xm(
                                    const unsigned int grid_degree,
                                    const unsigned int n_metric_dofs, 
                                    const std::array<std::vector<double>,dim> &mapping_support_points,
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
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::do_curl_loop_metric_cofactor(
                                    const unsigned int n_quad_pts,
                                    const std::vector<dealii::DerivativeForm<2,dim,dim>> grad_Xl_grad_Xm,
                                    std::vector<dealii::FullMatrix<double>> &metric_cofactor)
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


template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::transform_physical_to_reference(
                                    const dealii::Tensor<1,dim,double> &phys,
                                    const dealii::FullMatrix<double> &metric_cofactor,
                                    dealii::Tensor<1,dim,double> &ref)
{
    for(int idim=0; idim<dim; idim++){
        ref[idim] = 0.0;
        for(int idim2=0; idim2<dim; idim2++){
            ref[idim] += metric_cofactor[idim2][idim] * phys[idim2];
        }
    }

}
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::transform_reference_to_physical(
                                    const dealii::Tensor<1,dim,double> &ref,
                                    const dealii::FullMatrix<double> &metric_cofactor,
                                    dealii::Tensor<1,dim,double> &phys)
{
    for(int idim=0; idim<dim; idim++){
        phys[idim] = 0.0;
        for(int idim2=0; idim2<dim; idim2++){
            phys[idim] += metric_cofactor[idim][idim2] * ref[idim2];
        }
    }

}
template <  int dim, int n_faces>  
void OperatorsBase<dim,n_faces>::is_the_grid_higher_order_than_initialized(
                                    const unsigned int grid_degree)
{
    if(grid_degree > this->max_grid_degree_check){
        this->pcout<<"Updating the metric basis for grid degree "<<grid_degree<<std::endl;
        allocate_metric_operators(grid_degree);
        create_metric_basis_operators(grid_degree);
        this->max_grid_degree_check = grid_degree; 
    }
}

/**********************************************************
    * Operators Base State below constructing the operators
    *******************************************************/
template <  int dim, int nstate,
            int n_faces>  
OperatorsBaseState<dim,nstate,n_faces>
::OperatorsBaseState(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : OperatorsBase<dim,n_faces>::OperatorsBase(parameters_input, nstate, max_degree_input, max_degree_input, grid_degree_input)
{
    allocate_volume_operators_state();
    create_vol_basis_operators_state();
    //setup surface operators
    allocate_surface_operators_state();
    create_surface_basis_operators_state();
}
// Destructor
template <  int dim, int nstate,
            int n_faces>  
OperatorsBaseState<dim,nstate,n_faces>::~OperatorsBaseState ()
{
}

template <  int dim, int nstate,
            int n_faces>  
void OperatorsBaseState<dim,nstate,n_faces>::allocate_volume_operators_state()
{
    flux_basis_functions.resize(this->max_degree+1);
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
            flux_basis_functions[idegree][istate].reinit(n_quad_pts, n_dofs_flux);
            for(int idim=0; idim<dim; idim++){
                gradient_flux_basis[idegree][istate][idim].reinit(n_quad_pts, n_dofs_flux);
                local_flux_basis_stiffness[idegree][istate][idim].reinit(n_dofs, n_dofs_flux);
                vol_integral_gradient_basis[idegree][istate][idim].reinit(n_quad_pts, n_dofs);
            }
        }
    }

}
template <  int dim, int nstate,
            int n_faces>  
void OperatorsBaseState<dim,nstate,n_faces>::create_vol_basis_operators_state()
{
    for(unsigned int idegree=0; idegree<=this->max_degree; idegree++){
        unsigned int n_dofs_flux = this->fe_collection_flux_basis[idegree].dofs_per_cell;
        unsigned int n_quad_pts = this->volume_quadrature_collection[idegree].size();
        unsigned int n_dofs = this->fe_collection_basis[idegree].dofs_per_cell;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            const dealii::Point<1> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
            for(int istate=0; istate<nstate; istate++){
                for(unsigned int idof=0; idof<n_dofs_flux; idof++){
                    //Flux basis function idof of poly degree idegree evaluated at cubature node qpoint.
                    flux_basis_functions[idegree][istate][iquad][idof] = this->fe_collection_flux_basis[idegree].shape_value_component(idof,qpoint,0);
                    dealii::Tensor<1,dim,double> derivative;
                    derivative = this->fe_collection_flux_basis[idegree].shape_grad_component(idof, qpoint, 0);
                    for(int idim=0; idim<dim; idim++){
                        gradient_flux_basis[idegree][istate][idim][iquad][idof] = derivative[idim];
                    }
                }
            }
        }
        const std::vector<double> &quad_weights = this->volume_quadrature_collection[idegree].get_weights ();
        for(unsigned int itest=0; itest<n_dofs; itest++){
            const int istate_test = this->fe_collection_basis[idegree].system_to_component_index(itest).first;
            for(unsigned int idof=0; idof<n_dofs_flux; idof++){
                dealii::Tensor<1,dim,double> value;
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    const dealii::Point<1> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
                    dealii::Tensor<1,dim,double> derivative;
                    derivative = this->fe_collection_flux_basis[idegree].shape_grad_component(idof, qpoint, 0);
                    value += this->basis_functions[idegree][iquad][itest] * quad_weights[iquad] * derivative;
                }
                const int test_shape = this->fe_collection_basis[idegree].system_to_component_index(itest).second;
                    for(int idim=0; idim<dim; idim++){
                        local_flux_basis_stiffness[idegree][istate_test][idim][test_shape][idof] = value[idim]; 
                    }
            }
            const int ishape_test = this->fe_collection_basis[idegree].system_to_component_index(itest).second;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){ 
                const dealii::Point<1> qpoint  = this->volume_quadrature_collection[idegree].point(iquad);
                dealii::Tensor<1,dim,double> derivative;
                derivative = this->fe_collection_basis[idegree].shape_grad_component(itest, qpoint, istate_test);
                for(int idim=0; idim<dim; idim++){
                    vol_integral_gradient_basis[idegree][istate_test][idim][iquad][ishape_test] = derivative[idim] * quad_weights[iquad]; 
                }
            }
        }
    }

}
template <  int dim, int nstate,
            int n_faces>  
void OperatorsBaseState<dim,nstate,n_faces>::allocate_surface_operators_state()
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
template <  int dim, int nstate,
            int n_faces>  
void OperatorsBaseState<dim,nstate,n_faces>::create_surface_basis_operators_state()
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
template <  int dim, int nstate, int n_faces>  
void OperatorsBaseState<dim,nstate,n_faces>::get_Jacobian_scaled_physical_gradient(
                                    const bool use_conservative_divergence,
                                    const std::array<std::array<dealii::FullMatrix<double>,dim>,nstate> &ref_gradient,
                                    const std::vector<dealii::FullMatrix<double>> &metric_cofactor,
                                    const unsigned int n_quad_pts,
                                    std::array<std::array<dealii::FullMatrix<double>,dim>,nstate> &physical_gradient)
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

template class OperatorsBaseState <PHILIP_DIM, double, 1, 2*PHILIP_DIM>;
template class OperatorsBaseState <PHILIP_DIM, double, 2, 2*PHILIP_DIM>;
template class OperatorsBaseState <PHILIP_DIM, double, 3, 2*PHILIP_DIM>;
template class OperatorsBaseState <PHILIP_DIM, double, 4, 2*PHILIP_DIM>;
template class OperatorsBaseState <PHILIP_DIM, double, 5, 2*PHILIP_DIM>;

#endif
//end of old stuff

template class OperatorsBaseNEW <PHILIP_DIM, 2*PHILIP_DIM>;

template class SumFactorizedOperators <PHILIP_DIM, 2*PHILIP_DIM>;

template class SumFactorizedOperatorsState <PHILIP_DIM, 1, 2*PHILIP_DIM>;
template class SumFactorizedOperatorsState <PHILIP_DIM, 2, 2*PHILIP_DIM>;
template class SumFactorizedOperatorsState <PHILIP_DIM, 3, 2*PHILIP_DIM>;
template class SumFactorizedOperatorsState <PHILIP_DIM, 4, 2*PHILIP_DIM>;
template class SumFactorizedOperatorsState <PHILIP_DIM, 5, 2*PHILIP_DIM>;

template class basis_functions <PHILIP_DIM, 2*PHILIP_DIM>;
template class vol_integral_basis <PHILIP_DIM, 2*PHILIP_DIM>;
template class local_mass <PHILIP_DIM, 2*PHILIP_DIM>;
template class local_basis_stiffness <PHILIP_DIM, 2*PHILIP_DIM>;
template class modal_basis_differential_operator <PHILIP_DIM, 2*PHILIP_DIM>;
template class derivative_p <PHILIP_DIM, 2*PHILIP_DIM>;
template class local_Flux_Reconstruction_operator <PHILIP_DIM, 2*PHILIP_DIM>;
template class local_Flux_Reconstruction_operator_aux <PHILIP_DIM, 2*PHILIP_DIM>;
template class vol_projection_operator <PHILIP_DIM, 2*PHILIP_DIM>;
template class vol_projection_operator_FR <PHILIP_DIM, 2*PHILIP_DIM>;
template class FR_mass_inv <PHILIP_DIM, 2*PHILIP_DIM>;
template class vol_integral_gradient_basis <PHILIP_DIM, 2*PHILIP_DIM>;

//template class basis_at_facet_cubature <PHILIP_DIM, 2*PHILIP_DIM>;
template class face_integral_basis <PHILIP_DIM, 2*PHILIP_DIM>;
template class lifting_operator <PHILIP_DIM, 2*PHILIP_DIM>;
template class lifting_operator_FR <PHILIP_DIM, 2*PHILIP_DIM>;

template class mapping_shape_functions <PHILIP_DIM, 2*PHILIP_DIM>;

template class metric_operators <double,PHILIP_DIM, 2*PHILIP_DIM>;
//template class vol_metric_operators <double,PHILIP_DIM, 2*PHILIP_DIM>;
//template class vol_determinant_metric_Jacobian<PHILIP_DIM, 2*PHILIP_DIM>;
//template class vol_metric_cofactor<PHILIP_DIM, 2*PHILIP_DIM>;
//template class surface_metric_cofactor<PHILIP_DIM, 2*PHILIP_DIM>;
//
template class flux_basis <PHILIP_DIM, 1, 2*PHILIP_DIM>;
template class flux_basis <PHILIP_DIM, 2, 2*PHILIP_DIM>;
template class flux_basis <PHILIP_DIM, 3, 2*PHILIP_DIM>;
template class flux_basis <PHILIP_DIM, 4, 2*PHILIP_DIM>;
template class flux_basis <PHILIP_DIM, 5, 2*PHILIP_DIM>;

template class local_flux_basis_stiffness <PHILIP_DIM, 1, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 2, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 3, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 4, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 5, 2*PHILIP_DIM>;

} // OPERATOR namespace
} // PHiLiP namespace

