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
#include <deal.II/lac/identity_matrix.h>

#include <Epetra_RowMatrixTransposer.h>
#include <AztecOO.h>

#include "ADTypes.hpp"
#include <Sacado.hpp>
#include <CoDiPack/include/codi.hpp>

#include "operators.h"

namespace PHiLiP {
namespace OPERATOR {

//Constructor
template <int dim, int n_faces>
OperatorsBase<dim,n_faces>::OperatorsBase(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : max_degree(max_degree_input)
    , max_grid_degree(grid_degree_input)
    , nstate(nstate_input)
    , max_grid_degree_check(grid_degree_input)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{}

template <int dim, int n_faces>
dealii::FullMatrix<double> OperatorsBase<dim,n_faces>::tensor_product(
    const dealii::FullMatrix<double> &basis_x,
    const dealii::FullMatrix<double> &basis_y,
    const dealii::FullMatrix<double> &basis_z)
{
    const unsigned int rows_x    = basis_x.m();
    const unsigned int columns_x = basis_x.n();
    const unsigned int rows_y    = basis_y.m();
    const unsigned int columns_y = basis_y.n();
    const unsigned int rows_z    = basis_z.m();
    const unsigned int columns_z = basis_z.n();

    if constexpr (dim==1)
        return basis_x;
    if constexpr (dim==2){
        dealii::FullMatrix<double> tens_prod(rows_x * rows_y, columns_x * columns_y);
        for(unsigned int jdof=0; jdof<rows_y; jdof++){
            for(unsigned int kdof=0; kdof<rows_x; kdof++){
                for(unsigned int ndof=0; ndof<columns_y; ndof++){
                    for(unsigned int odof=0; odof<columns_x; odof++){
                        const unsigned int index_row = rows_x * jdof + kdof;
                        const unsigned int index_col = columns_x * ndof + odof;
                        tens_prod[index_row][index_col] = basis_x[kdof][odof] * basis_y[jdof][ndof];
                    }
                }
            }
        }
        return tens_prod;
    }
    if constexpr (dim==3){
        dealii::FullMatrix<double> tens_prod(rows_x * rows_y * rows_z, columns_x * columns_y * columns_z);
        for(unsigned int idof=0; idof<rows_z; idof++){
            for(unsigned int jdof=0; jdof<rows_y; jdof++){
                for(unsigned int kdof=0; kdof<rows_x; kdof++){
                    for(unsigned int mdof=0; mdof<columns_z; mdof++){
                        for(unsigned int ndof=0; ndof<columns_y; ndof++){
                            for(unsigned int odof=0; odof<columns_x; odof++){
                                const unsigned int index_row = rows_x * rows_y * idof + rows_x * jdof + kdof;
                                const unsigned int index_col = columns_x * columns_y * mdof + columns_x * ndof + odof;
                                tens_prod[index_row][index_col] = basis_x[kdof][odof] * basis_y[jdof][ndof] * basis_z[idof][mdof];
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
dealii::FullMatrix<double> OperatorsBase<dim,n_faces>::tensor_product_state(
    const int nstate,
    const dealii::FullMatrix<double> &basis_x,
    const dealii::FullMatrix<double> &basis_y,
    const dealii::FullMatrix<double> &basis_z)
{
    //assert that each basis matrix is of size (rows x columns)
    const unsigned int rows_x    = basis_x.m();
    const unsigned int columns_x = basis_x.n();
    const unsigned int rows_y    = basis_y.m();
    const unsigned int columns_y = basis_y.n();
    const unsigned int rows_z    = basis_z.m();
    const unsigned int columns_z = basis_z.n();

    const unsigned int rows_1state_x    = rows_x / nstate;
    const unsigned int columns_1state_x = columns_x / nstate;
    const unsigned int rows_1state_y    = rows_y / nstate;
    const unsigned int columns_1state_y = columns_y / nstate;
    const unsigned int rows_1state_z    = rows_z / nstate;
    const unsigned int columns_1state_z = columns_z / nstate;

    const unsigned int rows_all_states    = (dim == 1) ? rows_1state_x * nstate : 
                                                ( (dim == 2) ? rows_1state_x * rows_1state_y * nstate : 
                                                    rows_1state_x * rows_1state_y * rows_1state_z * nstate);
    const unsigned int columns_all_states = (dim == 1) ? columns_1state_x * nstate : 
                                                ( (dim == 2) ? columns_1state_x * columns_1state_y * nstate : 
                                                    columns_1state_x * columns_1state_y * columns_1state_z * nstate);
    dealii::FullMatrix<double> tens_prod(rows_all_states, columns_all_states);


    for(int istate=0; istate<nstate; istate++){
        dealii::FullMatrix<double> basis_x_1state(rows_1state_x, columns_1state_x);
        dealii::FullMatrix<double> basis_y_1state(rows_1state_y, columns_1state_y);
        dealii::FullMatrix<double> basis_z_1state(rows_1state_z, columns_1state_z);
        for(unsigned int r=0; r<rows_1state_x; r++){
            for(unsigned int c=0; c<columns_1state_x; c++){
                basis_x_1state[r][c] = basis_x[istate*rows_1state_x + r][istate*columns_1state_x + c];
            }
        }
        if constexpr(dim>=2){
            for(unsigned int r=0; r<rows_1state_y; r++){
                for(unsigned int c=0; c<columns_1state_y; c++){
                    basis_y_1state[r][c] = basis_y[istate*rows_1state_y + r][istate*columns_1state_y + c];
                }
            }
        }
        if constexpr(dim>=3){
            for(unsigned int r=0; r<rows_1state_z; r++){
                for(unsigned int c=0; c<columns_1state_z; c++){
                    basis_z_1state[r][c] = basis_z[istate*rows_1state_z + r][istate*columns_1state_z + c];
                }
            }
        }
        const unsigned int r1state = (dim == 1) ? rows_1state_x : ( (dim==2) ? rows_1state_x * rows_1state_y : rows_1state_x * rows_1state_y * rows_1state_z);
        const unsigned int c1state = (dim == 1) ? columns_1state_x : ( (dim==2) ? columns_1state_x * columns_1state_y : columns_1state_x * columns_1state_y * columns_1state_z);
        dealii::FullMatrix<double> tens_prod_1state(r1state, c1state);
        tens_prod_1state = tensor_product(basis_x_1state, basis_y_1state, basis_z_1state);
        for(unsigned int r=0; r<r1state; r++){
            for(unsigned int c=0; c<c1state; c++){
                tens_prod[istate*r1state + r][istate*c1state + c] = tens_prod_1state[r][c];
            }
        }
    }
    return tens_prod;
}

template <int dim, int n_faces>
double OperatorsBase<dim,n_faces>::compute_factorial(double n)
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
    : OperatorsBase<dim,n_faces>::OperatorsBase(nstate_input, max_degree_input, grid_degree_input)
{}

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
    if constexpr (dim == 1){
        assert(rows_x    == output_vect.size());
        assert(columns_x == input_vect.size());
    }
    if constexpr (dim == 2){
        assert(rows_x * rows_y       == output_vect.size());
        assert(columns_x * columns_y == input_vect.size());
    }
    if constexpr (dim == 3){
        assert(rows_x * rows_y * rows_z          == output_vect.size());
        assert(columns_x * columns_y * columns_z == input_vect.size());
    }

    if constexpr (dim==1){
        for(unsigned int iquad=0; iquad<rows_x; iquad++){
            if(!adding)
                output_vect[iquad] = 0.0;
            for(unsigned int jquad=0; jquad<columns_x; jquad++){
                output_vect[iquad] += factor * basis_x[iquad][jquad] * input_vect[jquad];
            }
        }
    }
    if constexpr (dim==2){
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
    if constexpr (dim==3){
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

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::matrix_vector_mult_1D(
    const std::vector<double> &input_vect,
    std::vector<double> &output_vect,
    const dealii::FullMatrix<double> &basis_x,
    const bool adding,
    const double factor)
{
    this->matrix_vector_mult(input_vect, output_vect, basis_x, basis_x, basis_x, adding, factor);
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::matrix_vector_mult_surface_1D(
    const unsigned int face_number,
    const std::vector<double> &input_vect,
    std::vector<double> &output_vect,
    const std::array<dealii::FullMatrix<double>,2> &basis_surf,
    const dealii::FullMatrix<double> &basis_vol,
    const bool adding,
    const double factor)
{
    if(face_number == 0)
        this->matrix_vector_mult(input_vect, output_vect, basis_surf[0], basis_vol, basis_vol, adding, factor);
    if(face_number == 1)
        this->matrix_vector_mult(input_vect, output_vect, basis_surf[1], basis_vol, basis_vol, adding, factor);
    if(face_number == 2)
        this->matrix_vector_mult(input_vect, output_vect, basis_vol, basis_surf[0], basis_vol, adding, factor);
    if(face_number == 3)
        this->matrix_vector_mult(input_vect, output_vect, basis_vol, basis_surf[1], basis_vol, adding, factor);
    if(face_number == 4)
        this->matrix_vector_mult(input_vect, output_vect, basis_vol, basis_vol, basis_surf[0], adding, factor);
    if(face_number == 5)
        this->matrix_vector_mult(input_vect, output_vect, basis_vol, basis_vol, basis_surf[1], adding, factor);
}


template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::inner_product_surface_1D(
    const unsigned int face_number,
    const std::vector<double> &input_vect,
    const std::vector<double> &weight_vect,
    std::vector<double> &output_vect,
    const std::array<dealii::FullMatrix<double>,2> &basis_surf,
    const dealii::FullMatrix<double> &basis_vol,
    const bool adding,
    const double factor)
{
    if(face_number == 0)
        this->inner_product(input_vect, weight_vect, output_vect, basis_surf[0], basis_vol, basis_vol, adding, factor);
    if(face_number == 1)
        this->inner_product(input_vect, weight_vect, output_vect, basis_surf[1], basis_vol, basis_vol, adding, factor);
    if(face_number == 2)
        this->inner_product(input_vect, weight_vect, output_vect, basis_vol, basis_surf[0], basis_vol, adding, factor);
    if(face_number == 3)
        this->inner_product(input_vect, weight_vect, output_vect, basis_vol, basis_surf[1], basis_vol, adding, factor);
    if(face_number == 4)
        this->inner_product(input_vect, weight_vect, output_vect, basis_vol, basis_vol, basis_surf[0], adding, factor);
    if(face_number == 5)
        this->inner_product(input_vect, weight_vect, output_vect, basis_vol, basis_vol, basis_surf[1], adding, factor);
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::divergence_matrix_vector_mult_1D(
    const dealii::Tensor<1,dim,std::vector<double>> &input_vect,
    std::vector<double> &output_vect,
    const dealii::FullMatrix<double> &basis,
    const dealii::FullMatrix<double> &gradient_basis)
{
    divergence_matrix_vector_mult(input_vect, output_vect,
                                  basis, basis, basis,
                                  gradient_basis, gradient_basis, gradient_basis);
}

template <int dim, int n_faces>  
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

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::gradient_matrix_vector_mult_1D(
    const std::vector<double> &input_vect,
    dealii::Tensor<1,dim,std::vector<double>> &output_vect,
    const dealii::FullMatrix<double> &basis,
    const dealii::FullMatrix<double> &gradient_basis)
{
    gradient_matrix_vector_mult(input_vect, output_vect,
                                basis, basis, basis,
                                gradient_basis, gradient_basis, gradient_basis);
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::gradient_matrix_vector_mult(
    const std::vector<double> &input_vect,
    dealii::Tensor<1,dim,std::vector<double>> &output_vect,
    const dealii::FullMatrix<double> &basis_x,
    const dealii::FullMatrix<double> &basis_y,
    const dealii::FullMatrix<double> &basis_z,
    const dealii::FullMatrix<double> &gradient_basis_x,
    const dealii::FullMatrix<double> &gradient_basis_y,
    const dealii::FullMatrix<double> &gradient_basis_z)
{
    for(int idim=0; idim<dim;idim++){
//        output_vect[idim].resize(input_vect.size());
        if(idim==0)
            this->matrix_vector_mult(input_vect, output_vect[idim],
                                     gradient_basis_x, 
                                     basis_y, 
                                     basis_z,
                                     false);//first one doesn't add in the divergence
        if(idim==1)
            this->matrix_vector_mult(input_vect, output_vect[idim],
                                     basis_x, 
                                     gradient_basis_y, 
                                     basis_z,
                                     false);
        if(idim==2)
            this->matrix_vector_mult(input_vect, output_vect[idim],
                                     basis_x, 
                                     basis_y,
                                     gradient_basis_z,
                                     false);
    } 
}

template <int dim, int n_faces>  
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
    //Note the assertion has columns to output and rows to input
    //bc we transpose the basis inputted for the inner product
    if constexpr (dim == 1){
        assert(rows_x    == input_vect.size());
        assert(columns_x == output_vect.size());
    }
    if constexpr (dim == 2){
        assert(rows_x * rows_y       == input_vect.size());
        assert(columns_x * columns_y == output_vect.size());
    }
    if constexpr (dim == 3){
        assert(rows_x * rows_y * rows_z          == input_vect.size());
        assert(columns_x * columns_y * columns_z == output_vect.size());
    }
    assert(weight_vect.size() == input_vect.size()); 

    dealii::FullMatrix<double> basis_x_trans(columns_x, rows_x);
    dealii::FullMatrix<double> basis_y_trans(columns_y, rows_y);
    dealii::FullMatrix<double> basis_z_trans(columns_z, rows_z);

    //set as the transpose as inputed basis
    //found an issue with Tadd for arbitrary size so I manually do it here.
    for(unsigned int row=0; row<rows_x; row++){
        for(unsigned int col=0; col<columns_x; col++){
            basis_x_trans[col][row] = basis_x[row][col];
        }
    }
    for(unsigned int row=0; row<rows_y; row++){
        for(unsigned int col=0; col<columns_y; col++){
            basis_y_trans[col][row] = basis_y[row][col];
        }
    }
    for(unsigned int row=0; row<rows_z; row++){
        for(unsigned int col=0; col<columns_z; col++){
            basis_z_trans[col][row] = basis_z[row][col];
        }
    }

    std::vector<double> new_input_vect(input_vect.size());
    for(unsigned int iquad=0; iquad<input_vect.size(); iquad++){
        new_input_vect[iquad] = input_vect[iquad] * weight_vect[iquad];
    }

    this->matrix_vector_mult(new_input_vect, output_vect, basis_x_trans, basis_y_trans, basis_z_trans, adding, factor);
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::inner_product_1D(
    const std::vector<double> &input_vect,
    const std::vector<double> &weight_vect,
    std::vector<double> &output_vect,
    const dealii::FullMatrix<double> &basis_x,
    const bool adding,
    const double factor) 
{
    this->inner_product(input_vect, weight_vect, output_vect, basis_x, basis_x, basis_x, adding, factor);
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::divergence_two_pt_flux_Hadamard_product(
    const dealii::Tensor<1,dim,dealii::FullMatrix<double>> &input_mat,
    std::vector<double> &output_vect,
    const std::vector<double> &weights,
    const dealii::FullMatrix<double> &basis,
    const double scaling)
{
    assert(input_mat[0].m() == output_vect.size());

    dealii::FullMatrix<double> output_mat(input_mat[0].m(), input_mat[0].n());
    for(int idim=0; idim<dim; idim++){
        two_pt_flux_Hadamard_product(input_mat[idim], output_mat, basis, weights, idim);
        if constexpr(dim==1){
            for(unsigned int row=0; row<input_mat[0].m(); row++){//n^d rows
                for(unsigned int col=0; col<basis.m(); col++){//only need to sum n columns
                    const unsigned int col_index = col; 
                    output_vect[row] += scaling * output_mat[row][col_index];//scaled by 2.0 for 2pt flux
                }
            }
        }
        if constexpr(dim==2){
            const unsigned int size_1D = basis.m();
            for(unsigned int irow=0; irow<size_1D; irow++){
                for(unsigned int jrow=0; jrow<size_1D; jrow++){
                    const unsigned int row_index = irow * size_1D + jrow;
                    for(unsigned int col=0; col<size_1D; col++){
                        if(idim==0){
                            const unsigned int col_index = col + irow * size_1D;
                            output_vect[row_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                        }
                        if(idim==1){
                            const unsigned int col_index = col * size_1D + jrow;
                            output_vect[row_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                        }
                    }
                }
            }
        }
        if constexpr(dim==3){
            const unsigned int size_1D = basis.m();
            for(unsigned int irow=0; irow<size_1D; irow++){
                for(unsigned int jrow=0; jrow<size_1D; jrow++){
                    for(unsigned int krow=0; krow<size_1D; krow++){
                        const unsigned int row_index = irow * size_1D * size_1D + jrow * size_1D + krow;
                        for(unsigned int col=0; col<size_1D; col++){
                            if(idim==0){
                                const unsigned int col_index = col + irow * size_1D * size_1D + jrow * size_1D;
                                output_vect[row_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                            }
                            if(idim==1){
                                const unsigned int col_index = col * size_1D + krow + irow * size_1D * size_1D;
                                output_vect[row_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                            }
                            if(idim==2){
                                const unsigned int col_index = col * size_1D * size_1D + krow + jrow * size_1D;
                                output_vect[row_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                            }
                        }
                    }
                }
            }
        }
    }
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::surface_two_pt_flux_Hadamard_product(
    const dealii::FullMatrix<double> &input_mat,
    std::vector<double> &output_vect_vol,
    std::vector<double> &output_vect_surf,
    const std::vector<double> &weights,
    const std::array<dealii::FullMatrix<double>,2> &surf_basis,
    const unsigned int iface,
    const unsigned int dim_not_zero,
    const double scaling)//scaling is unit_ref_normal[dim_not_zero]
{
    assert(input_mat.n() == output_vect_vol.size());
    assert(input_mat.m() == output_vect_surf.size());
    const unsigned int iface_1D = iface % 2;

    dealii::FullMatrix<double> output_mat(input_mat.m(), input_mat.n());
    two_pt_flux_Hadamard_product(input_mat, output_mat, surf_basis[iface_1D], weights, dim_not_zero);
    if constexpr(dim==1){
        for(unsigned int row=0; row<surf_basis[iface_1D].m(); row++){//n rows
            for(unsigned int col=0; col<surf_basis[iface_1D].n(); col++){//only need to sum n columns
                output_vect_vol[col] += scaling 
                                      * output_mat[row][col];//scaled by 2.0 for 2pt flux
                output_vect_surf[row] -= scaling //minus because skew-symm form
                                       * output_mat[row][col];//scaled by 2.0 for 2pt flux
            }
        }
    }
    if constexpr(dim==2){
        const unsigned int size_1D_row = surf_basis[iface_1D].m();
        const unsigned int size_1D_col = surf_basis[iface_1D].n();
        for(unsigned int irow=0; irow<size_1D_col; irow++){
            for(unsigned int jrow=0; jrow<size_1D_row; jrow++){
                const unsigned int row_index = irow * size_1D_row + jrow;
                for(unsigned int col=0; col<size_1D_col; col++){
                    if(dim_not_zero==0){
                        const unsigned int col_index = col + irow * size_1D_col;
                        output_vect_vol[col_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                        output_vect_surf[row_index] -= scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                    }
                    if(dim_not_zero==1){
                        const unsigned int col_index = col * size_1D_col + irow;
                        output_vect_vol[col_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                        output_vect_surf[row_index] -= scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                    }
                }
            }
        }
    }
    if constexpr(dim==3){
        const unsigned int size_1D_row = surf_basis[iface_1D].m();
        const unsigned int size_1D_col = surf_basis[iface_1D].n();
        for(unsigned int irow=0; irow<size_1D_col; irow++){
            for(unsigned int jrow=0; jrow<size_1D_col; jrow++){
                for(unsigned int krow=0; krow<size_1D_row; krow++){
                    const unsigned int row_index = irow * size_1D_row * size_1D_col + jrow * size_1D_row + krow;
                    for(unsigned int col=0; col<size_1D_col; col++){
                        if(dim_not_zero==0){
                            const unsigned int col_index = col + irow * size_1D_col * size_1D_col + jrow * size_1D_col;
                            output_vect_vol[col_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                            output_vect_surf[row_index] -= scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                        }
                        if(dim_not_zero==1){
                            const unsigned int col_index = col * size_1D_col + jrow + irow * size_1D_col * size_1D_col;
                            output_vect_vol[col_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                            output_vect_surf[row_index] -= scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                        }
                        if(dim_not_zero==2){
                            const unsigned int col_index = col * size_1D_col * size_1D_col + jrow + irow * size_1D_col;
                            output_vect_vol[col_index] += scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                            output_vect_surf[row_index] -= scaling * output_mat[row_index][col_index];//scaled by 2.0 for 2pt flux
                        }
                    }
                }
            }
        }
    }
}



template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::two_pt_flux_Hadamard_product(
    const dealii::FullMatrix<double> &input_mat,
    dealii::FullMatrix<double> &output_mat,
    const dealii::FullMatrix<double> &basis,
    const std::vector<double> &weights,
    const int direction)
{
    assert(input_mat.size() == output_mat.size());
    const unsigned int size = basis.n();
    assert(size == weights.size());

    if constexpr(dim == 1){
        Hadamard_product(input_mat, basis, output_mat);  
    }
    if constexpr(dim == 2){
        //In the general case, the basis is non-square (think surface lifting functions).
        //We assume the other directions are square but variable in the basis of interest. 
        const unsigned int rows = basis.m();
        assert(rows == input_mat.m());
        if(direction == 0){
            for(unsigned int idiag=0; idiag<size; idiag++){
                dealii::FullMatrix<double> local_block(rows, size);
                std::vector<unsigned int> row_index(rows);
                std::vector<unsigned int> col_index(size);
                //fill index range for diagonal blocks of rize rows_x x columns_x
                std::iota(row_index.begin(), row_index.end(), idiag*rows);
                std::iota(col_index.begin(), col_index.end(), idiag*size);
                //extract diagonal block from input matrix
                local_block.extract_submatrix_from(input_mat, row_index, col_index);
                dealii::FullMatrix<double> local_Hadamard(rows, size);
                Hadamard_product(local_block, basis, local_Hadamard);
                //scale by the diagonal weight from tensor product
                local_Hadamard *= weights[idiag];
                //write the values into diagonal block of output
                local_Hadamard.scatter_matrix_to(row_index, col_index, output_mat);
            }
        }
        if(direction == 1){
            for(unsigned int idiag=0; idiag<rows; idiag++){
                const unsigned int row_index = idiag * size;
                for(unsigned int jdiag=0; jdiag<size; jdiag++){
                    const unsigned int col_index = jdiag * size;
                    for(unsigned int kdiag=0; kdiag<size; kdiag++){
                        output_mat[row_index + kdiag][col_index + kdiag] = basis[idiag][jdiag]
                                                                         * input_mat[row_index + kdiag][col_index + kdiag]
                                                                         * weights[kdiag];
                    }
                }
            }

        }
    }
    if constexpr(dim == 3){
        const unsigned int rows = basis.m();
        if(direction == 0){
            unsigned int kdiag=0;
            for(unsigned int idiag=0; idiag< size * size; idiag++){
                if(kdiag==size) kdiag = 0;
                dealii::FullMatrix<double> local_block(rows, size);
                std::vector<unsigned int> row_index(rows);
                std::vector<unsigned int> col_index(size);
                //fill index range for diagonal blocks of rize rows_x x columns_x
                std::iota(row_index.begin(), row_index.end(), idiag*rows);
                std::iota(col_index.begin(), col_index.end(), idiag*size);
                //extract diagonal block from input matrix
                local_block.extract_submatrix_from(input_mat, row_index, col_index);
                dealii::FullMatrix<double> local_Hadamard(rows, size);
                Hadamard_product(local_block, basis, local_Hadamard);
                //scale by the diagonal weight from tensor product
                local_Hadamard *= weights[kdiag];
                kdiag++;
                const unsigned int jdiag = idiag / size;
                local_Hadamard *= weights[jdiag];
                //write the values into diagonal block of output
                local_Hadamard.scatter_matrix_to(row_index, col_index, output_mat);
            }
        }
        if(direction == 1){
            for(unsigned int zdiag=0; zdiag<size; zdiag++){ 
                for(unsigned int idiag=0; idiag<rows; idiag++){
                    const unsigned int row_index = zdiag * size * rows + idiag * size;
                    for(unsigned int jdiag=0; jdiag<size; jdiag++){
                        const unsigned int col_index = zdiag * size * size + jdiag * size;
                        for(unsigned int kdiag=0; kdiag<size; kdiag++){
                            output_mat[row_index + kdiag][col_index + kdiag] = basis[idiag][jdiag]
                                                                             * input_mat[row_index + kdiag][col_index + kdiag]
                                                                             * weights[zdiag]
                                                                             * weights[kdiag];
                        }
                    }
                }
            }
        }
        if(direction == 2){
            for(unsigned int row_block=0; row_block<rows; row_block++){
                for(unsigned int col_block=0; col_block<size; col_block++){
                    const unsigned int row_index = row_block * size * size;
                    const unsigned int col_index = col_block * size * size;
                    for(unsigned int jdiag=0; jdiag<size; jdiag++){
                        for(unsigned int kdiag=0; kdiag<size; kdiag++){
                            output_mat[row_index + jdiag * size + kdiag][col_index + jdiag * size  + kdiag] = basis[row_block][col_block]
                                                                                                            * input_mat[row_index + jdiag * size  + kdiag][col_index + jdiag * size  + kdiag]
                                                                                                            * weights[kdiag]
                                                                                                            * weights[jdiag];
                        }
                    }
                }
            }
        }
    }
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::Hadamard_product(
    const dealii::FullMatrix<double> &input_mat1,
    const dealii::FullMatrix<double> &input_mat2,
    dealii::FullMatrix<double> &output_mat)
{
    const unsigned int rows    = input_mat1.m();
    const unsigned int columns = input_mat1.n();
    assert(rows    == input_mat2.m());
    assert(columns == input_mat2.n());
    
    for(unsigned int irow=0; irow<rows; irow++){
        for(unsigned int icol=0; icol<columns; icol++){
            output_mat[irow][icol] = input_mat1[irow][icol] 
                                   * input_mat2[irow][icol];
        }
    }
}

template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::sum_factorized_Hadamard_sparsity_pattern(
    const unsigned int rows_size,
    const unsigned int columns_size,
    std::vector<std::array<unsigned int,dim>> &rows,
    std::vector<std::array<unsigned int,dim>> &columns)
{
    //Note that for all directions, the rows vector should always be the same.
    if constexpr(dim == 1){
        for(unsigned int irow=0; irow<rows_size; irow++){
            for(unsigned int icol=0; icol<columns_size; icol++){
                const unsigned int array_index = irow * rows_size + icol;
                rows[array_index][0]    = irow;
                columns[array_index][0] = icol;
            }
        }
    }
    if constexpr(dim == 2){
        for(unsigned int idiag=0; idiag<rows_size; idiag++){
            for(unsigned int jdiag=0; jdiag<columns_size; jdiag++){
                for(unsigned int kdiag=0; kdiag<columns_size; kdiag++){
                    const unsigned int array_index = idiag * rows_size * columns_size + jdiag * columns_size + kdiag;
                    const unsigned int row_index = idiag * rows_size;
                    rows[array_index][0] = row_index + jdiag;
                    rows[array_index][1] = row_index + jdiag;
                    //direction 0
                    const unsigned int col_index_0 = idiag * columns_size;
                    columns[array_index][0] = col_index_0 + kdiag;
                    //direction 1
                    const unsigned int col_index_1 = kdiag * columns_size;
                    columns[array_index][1] = col_index_1 + jdiag;
                }
            }
        }
    }
    if constexpr(dim == 3){
        for(unsigned int idiag=0; idiag<rows_size; idiag++){
            for(unsigned int jdiag=0; jdiag<columns_size; jdiag++){
                for(unsigned int kdiag=0; kdiag<columns_size; kdiag++){
                    for(unsigned int ldiag=0; ldiag<columns_size; ldiag++){
                        const unsigned int array_index = idiag * rows_size * columns_size * columns_size 
                                                       + jdiag * columns_size * columns_size 
                                                       + kdiag * columns_size
                                                       + ldiag;
                        const unsigned int row_index = idiag * rows_size * columns_size
                                                     + jdiag * columns_size;
                        rows[array_index][0] = row_index + kdiag;
                        rows[array_index][1] = row_index + kdiag;
                        rows[array_index][2] = row_index + kdiag;
                        //direction 0
                        const unsigned int col_index_0 = idiag * columns_size * columns_size
                                                     + jdiag * columns_size;
                        columns[array_index][0] = col_index_0 + ldiag;
                        //direction 1
                        const unsigned int col_index_1 = ldiag * columns_size;
                        columns[array_index][1] = col_index_1 + kdiag + idiag * columns_size * columns_size;
                        //direction 2
                        const unsigned int col_index_2 = ldiag * columns_size * columns_size;
                        columns[array_index][2] = col_index_2 + kdiag + jdiag * columns_size;
                    }
                }
            }
        }
    }
}
template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::sum_factorized_Hadamard_basis_assembly(
    const unsigned int rows_size_1D,
    const unsigned int columns_size_1D,
    const std::vector<std::array<unsigned int,dim>> &rows,
    const std::vector<std::array<unsigned int,dim>> &columns,
    const dealii::FullMatrix<double> &basis,
    const std::vector<double> &weights,
    std::array<dealii::FullMatrix<double>,dim> &basis_sparse)
{
    if constexpr(dim == 1){
        for(unsigned int irow=0; irow<rows_size_1D; irow++){
            for(unsigned int icol=0; icol<columns_size_1D; icol++){
                basis_sparse[0][irow][icol] = basis[irow][icol];
            }
        }
    }
    if constexpr(dim == 2){
        const unsigned int total_size = rows.size();
        for(unsigned int index=0, counter=0; index<total_size; index++, counter++){
            if(counter == columns_size_1D){
                counter = 0;
            }
            //direction 0
            basis_sparse[0][rows[index][0]][counter] = basis[rows[index][0]%rows_size_1D][columns[index][0]%columns_size_1D]
                                                     * weights[rows[index][1]/columns_size_1D];
            //direction 1
            basis_sparse[1][rows[index][1]][counter] = basis[rows[index][1]/rows_size_1D][columns[index][1]/columns_size_1D]
                                                     * weights[rows[index][0]%columns_size_1D];
        }
    }
    if constexpr(dim == 3){
        const unsigned int total_size = rows.size();
        for(unsigned int index=0, counter=0; index<total_size; index++, counter++){
            if(counter == columns_size_1D){
                counter = 0;
            }
            //direction 0
            basis_sparse[0][rows[index][0]][counter] = basis[rows[index][0]%rows_size_1D][columns[index][0]%columns_size_1D]
                                                     * weights[(rows[index][1]/columns_size_1D)%columns_size_1D]
                                                     * weights[rows[index][2]/columns_size_1D/columns_size_1D];
            //direction 1
            basis_sparse[1][rows[index][1]][counter] = basis[(rows[index][1]/rows_size_1D)%rows_size_1D][(columns[index][1]/columns_size_1D)%columns_size_1D]
                                                     * weights[rows[index][0]%columns_size_1D]
                                                     * weights[rows[index][2]/columns_size_1D/columns_size_1D];
            //direction 2
            basis_sparse[2][rows[index][2]][counter] = basis[rows[index][2]/rows_size_1D/rows_size_1D][columns[index][2]/columns_size_1D/columns_size_1D]
                                                     * weights[rows[index][0]%columns_size_1D]
                                                     * weights[(rows[index][1]/columns_size_1D)%columns_size_1D];
        }
    }

}
template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::sum_factorized_Hadamard_surface_sparsity_pattern(
    const unsigned int rows_size,
    const unsigned int columns_size,
    std::vector<unsigned int> &rows,
    std::vector<unsigned int> &columns,
    const int dim_not_zero)
{
    //Note that for all directions, the rows vector should always be the same.
    if constexpr(dim == 1){
        for(unsigned int irow=0; irow<rows_size; irow++){
            for(unsigned int icol=0; icol<columns_size; icol++){
                const unsigned int array_index = irow * columns_size + icol;
                rows[array_index]    = irow;
                columns[array_index] = icol;
            }
        }
    }
    if constexpr(dim == 2){
        for(unsigned int idiag=0; idiag<rows_size; idiag++){
            for(unsigned int jdiag=0; jdiag<columns_size; jdiag++){
                const unsigned int array_index = idiag * columns_size + jdiag ;

                const unsigned int row_index = idiag;
                rows[array_index] = row_index;
                //direction 0
                if(dim_not_zero == 0){
                    const unsigned int col_index_0 = idiag * columns_size;
                    columns[array_index] = col_index_0 + jdiag;
                }
                //direction 1
                if(dim_not_zero == 1){
                    const unsigned int col_index_1 = jdiag * columns_size;
                    columns[array_index] = col_index_1 + idiag;
                }
            }
        }
    }
    if constexpr(dim == 3){
        for(unsigned int idiag=0; idiag<columns_size; idiag++){
            for(unsigned int jdiag=0; jdiag<columns_size; jdiag++){
                for(unsigned int kdiag=0; kdiag<columns_size; kdiag++){
                    const unsigned int array_index = idiag * columns_size * columns_size 
                                                   + jdiag * columns_size
                                                   + kdiag;
                    const unsigned int row_index = idiag * columns_size + jdiag;
                    rows[array_index] = row_index;
                    //direction 0
                    if(dim_not_zero == 0){
                        const unsigned int col_index_0 = idiag * columns_size * columns_size
                                                       + jdiag * columns_size;
                        columns[array_index] = col_index_0 + kdiag;
                    }
                    //direction 1
                    if(dim_not_zero == 1){
                        const unsigned int col_index_1 = kdiag * columns_size;
                        columns[array_index] = col_index_1 + jdiag + idiag * columns_size * columns_size;
                    }
                    //direction 2
                    if(dim_not_zero == 2){
                        const unsigned int col_index_2 = kdiag * columns_size * columns_size;
                        columns[array_index] = col_index_2 + jdiag + idiag * columns_size;
                    }
                }
            }
        }
    }
}
template <int dim, int n_faces>  
void SumFactorizedOperators<dim,n_faces>::sum_factorized_Hadamard_surface_basis_assembly(
    const unsigned int rows_size,
    const unsigned int columns_size_1D,
    const std::vector<unsigned int> &rows,
    const std::vector<unsigned int> &columns,
    const dealii::FullMatrix<double> &basis,
    const std::vector<double> &weights,
    dealii::FullMatrix<double> &basis_sparse,
    const int dim_not_zero)
{
    if constexpr(dim == 1){
        for(unsigned int irow=0; irow<rows_size; irow++){
            for(unsigned int icol=0; icol<columns_size_1D; icol++){
                basis_sparse[irow][icol] = basis[irow][icol];
            }
        }
    }
    if constexpr(dim == 2){
        const unsigned int total_size = rows.size();
        for(unsigned int index=0, counter=0; index<total_size; index++, counter++){
            if(counter == columns_size_1D){
                counter = 0;
            }
            //direction 0
            if(dim_not_zero == 0){
                basis_sparse[rows[index]][counter] = basis[0][columns[index]%columns_size_1D]//oneD surf basis only 1 point
                                                         * weights[rows[index]%columns_size_1D];
            }
            //direction 1
            if(dim_not_zero == 1){
                basis_sparse[rows[index]][counter] = basis[0][columns[index]/columns_size_1D]
                                                         * weights[rows[index]%columns_size_1D];
            }
        }
    }
    if constexpr(dim == 3){
        const unsigned int total_size = rows.size();
        for(unsigned int index=0, counter=0; index<total_size; index++, counter++){
            if(counter == columns_size_1D){
                counter = 0;
            }
            //direction 0
            if(dim_not_zero == 0){
                basis_sparse[rows[index]][counter] = basis[0][columns[index]%columns_size_1D]//oneD surf basis only 1 point
                                                   * weights[rows[index]%columns_size_1D]
                                                   * weights[rows[index]/columns_size_1D];
            }
            //direction 1
            if(dim_not_zero == 1){
                basis_sparse[rows[index]][counter] = basis[0][(columns[index]/columns_size_1D)%columns_size_1D]
                                                   * weights[rows[index]%columns_size_1D]
                                                   * weights[rows[index]/columns_size_1D];
            }
            //direction 2
            if(dim_not_zero == 2){
                basis_sparse[rows[index]][counter] = basis[0][columns[index]/columns_size_1D/columns_size_1D]
                                                   * weights[rows[index]%columns_size_1D]
                                                   * weights[rows[index]/columns_size_1D];
            }
        }
    }
}

template <int dim, int n_faces>  
unsigned int SumFactorizedOperators<dim,n_faces>::reference_face_number(
    const unsigned int iface,
    const bool /*face_orientation*/,
    const bool face_flip,
    const bool face_rotation)
{
    unsigned int face_number = 100;
    if(!face_flip && !face_rotation){
        face_number = iface;
    }
    else if(!face_flip && face_rotation){//90 degree rotation
        //Not fully verified yet
        std::cout<<"Cannot handle reference cell face rotations!. Aborting..."<<std::endl;
        std::abort();
    }
    else if(face_flip && !face_rotation){//180 degree rotation
        //Not fully verified yet
	if(iface%2==0)
            face_number = iface + 1;
        if(iface%2==1)
            face_number = iface - 1;
       // std::cout<<"Cannot handle reference cell face rotations!. Aborting..."<<std::endl;
       // std::abort();
    }
    else if(face_flip && face_rotation){//270 degree rotation
        //Not fully verified yet
        std::cout<<"Cannot handle reference cell face rotations!. Aborting..."<<std::endl;
        std::abort();
    }
    if(face_number == 100){
        std::cout<<"face rotated condition not yet considered."<<std::endl;
        std::abort();
    }

//    if(face_orientation){
//        face_number = iface;
//       // std::cout<<"Correct face orientation. Face id is:"<<iface<<std::endl;
//    }
//    else{
////	std::cout<<"face_orientation:"<<face_orientation<<std::endl;
//      //  std::cout<<"Cell face was rotated by Deal.ii! Face id is:"<<iface<<std::endl;	
////	std::cout<<"face orientation"<<face_orientation<<std::endl;
////	std::cout<<"face flip"<<face_flip<<std::endl;
////	std::cout<<"face rotation"<<face_rotation<<std::endl;
//     //   if(!face_flip && !face_rotation){
////         //   std::cout<<"Cannot handle reference cell face flip!"<<std::endl;   
////	    if(iface%2==0)
////                face_number = iface + 1;
////            if(iface%2==1)
////                face_number = iface - 1;
////	   // std::abort();
//           face_number = iface;
//        }
//        if(!face_flip && face_rotation){
//        std::cout<<"Cell face was rotated by Deal.ii! Face id is:"<<iface<<std::endl;	
//        std::cout<<"face orientation"<<face_orientation<<std::endl;
//        std::cout<<"face flip"<<face_flip<<std::endl;
//        std::cout<<"face rotation"<<face_rotation<<std::endl;
//            std::cout<<"Cannot handle reference cell face rotations!. Aborting..."<<std::endl;
//            std::abort();
//        }
//        if(face_flip && face_rotation){
//            std::cout<<"Cannot handle reference cell face flip and rotations!. Aborting..."<<std::endl;
//        std::cout<<"Cell face was rotated by Deal.ii! Face id is:"<<iface<<std::endl;	
//        std::cout<<"face orientation"<<face_orientation<<std::endl;
//        std::cout<<"face flip"<<face_flip<<std::endl;
//        std::cout<<"face rotation"<<face_rotation<<std::endl;
//            std::abort();
//        }
//    }
//    if(face_number == 100){
//            std::cout<<"Cannot handle reference cell no face orientation and no rotations!. Aborting..."<<std::endl;
//            //std::cout<<"face_flip:"<<face_flip<<std::endl;
//            //std::cout<<"face_rotation:"<<face_rotation<<std::endl;
//            //std::abort();
//    }
//
    return face_number;

}

/*******************************************
 *
 *      VOLUME OPERATORS FUNCTIONS
 *
 *
 ******************************************/

template <int dim, int n_faces>  
basis_functions<dim,n_faces>::basis_functions(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
    //Initialize to the max degrees
    current_degree = max_degree_input;
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
dealii::FullMatrix<double> local_mass<dim,n_faces>::build_dim_mass_matrix(
    const int nstate,
    const unsigned int n_dofs, const unsigned int n_quad_pts,
    basis_functions<dim,n_faces> &basis,
    const std::vector<double> &det_Jac,
    const std::vector<double> &quad_weights)

{
    const unsigned int n_shape_fns = n_dofs / nstate;
    assert(nstate*pow(basis.oneD_vol_operator.m() / nstate, dim) == n_dofs);
    dealii::FullMatrix<double> mass_matrix_dim(n_dofs);
    dealii::FullMatrix<double> basis_dim(n_dofs);
    basis_dim = basis.tensor_product_state(
                    nstate,
                    basis.oneD_vol_operator,
                    basis.oneD_vol_operator,
                    basis.oneD_vol_operator);
    //loop and store
    for(int istate=0; istate<nstate; istate++){
        for(unsigned int itest=0; itest<n_shape_fns; ++itest){
            for (unsigned int itrial=itest; itrial<n_shape_fns; ++itrial) {
                double value = 0.0;
                const unsigned int trial_index = istate*n_shape_fns + itrial;
                const unsigned int test_index  = istate*n_shape_fns + itest;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    value += basis_dim[iquad][test_index]
                           * basis_dim[iquad][trial_index]
                           * det_Jac[iquad]
                           * quad_weights[iquad];
                }
                mass_matrix_dim[trial_index][test_index] = value;
                mass_matrix_dim[test_index][trial_index] = value;
            }
        }
    }
    return mass_matrix_dim;
}

template <int dim, int n_faces>  
local_basis_stiffness<dim,n_faces>::local_basis_stiffness(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const bool store_skew_symmetric_form_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , store_skew_symmetric_form(store_skew_symmetric_form_input)
{
    //Initialize to the max degrees
    current_degree = max_degree_input;
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
    if(store_skew_symmetric_form){
        //allocate
        oneD_skew_symm_vol_oper.reinit(n_dofs,n_dofs);
        //solve
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int jdof=0; jdof<n_dofs; jdof++){
                oneD_skew_symm_vol_oper[idof][jdof] = this->oneD_vol_operator[idof][jdof]
                                                    - this->oneD_vol_operator[jdof][idof];
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
    //get the FR corrcetion parameter value
    get_FR_correction_parameter(this->max_degree, FR_param);
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
dealii::FullMatrix<double> local_Flux_Reconstruction_operator<dim,n_faces>::build_dim_Flux_Reconstruction_operator_directly(
    const int nstate,
    const unsigned int n_dofs,
    dealii::FullMatrix<double> &pth_deriv,
    dealii::FullMatrix<double> &mass_matrix)
{
    dealii::FullMatrix<double> Flux_Reconstruction_operator(n_dofs);
    dealii::FullMatrix<double> identity (dealii::IdentityMatrix(pth_deriv.m()));
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<double> pth_deriv_dim(n_dofs);
        if(idim==0){
            pth_deriv_dim = this->tensor_product_state(nstate, pth_deriv, identity, identity);
        }
        if(idim==1){
            pth_deriv_dim = this->tensor_product_state(nstate, identity, pth_deriv, identity);
        }
        if(idim==2){
            pth_deriv_dim = this->tensor_product_state(nstate, identity, identity, pth_deriv);
        }
        dealii::FullMatrix<double> derivative_p_temp(n_dofs);
        derivative_p_temp.add(FR_param, pth_deriv_dim);
        dealii::FullMatrix<double> Flux_Reconstruction_operator_temp(n_dofs);
        derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, mass_matrix);
        Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, pth_deriv_dim, true);
    }
    if constexpr (dim>=2){
        const int deriv_2p_loop = (dim==2) ? 1 : dim; 
        double FR_param_sqrd = pow(FR_param,2.0);
        for(int idim=0; idim<deriv_2p_loop; idim++){
            dealii::FullMatrix<double> pth_deriv_dim(n_dofs);
            if(idim==0){
                pth_deriv_dim = this->tensor_product_state(nstate, pth_deriv, pth_deriv, identity);
            }
            if(idim==1){
                pth_deriv_dim = this->tensor_product_state(nstate, identity, pth_deriv, pth_deriv);
            }
            if(idim==2){
                pth_deriv_dim = this->tensor_product_state(nstate, pth_deriv, identity, pth_deriv);
            }
            dealii::FullMatrix<double> derivative_p_temp(n_dofs);
            derivative_p_temp.add(FR_param_sqrd, pth_deriv_dim);
            dealii::FullMatrix<double> Flux_Reconstruction_operator_temp(n_dofs);
            derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, mass_matrix);
            Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, pth_deriv_dim, true);
        }
    }
    if constexpr (dim == 3){
        double FR_param_cubed = pow(FR_param,3.0);
        dealii::FullMatrix<double> pth_deriv_dim(n_dofs);
        pth_deriv_dim = this->tensor_product_state(nstate, pth_deriv, pth_deriv, pth_deriv);
        dealii::FullMatrix<double> derivative_p_temp(n_dofs);
        derivative_p_temp.add(FR_param_cubed, pth_deriv_dim);
        dealii::FullMatrix<double> Flux_Reconstruction_operator_temp(n_dofs);
        derivative_p_temp.Tmmult(Flux_Reconstruction_operator_temp, mass_matrix);
        Flux_Reconstruction_operator_temp.mmult(Flux_Reconstruction_operator, pth_deriv_dim, true);
    }    

    return Flux_Reconstruction_operator;
}        

template <int dim, int n_faces>  
dealii::FullMatrix<double> local_Flux_Reconstruction_operator<dim,n_faces>::build_dim_Flux_Reconstruction_operator(
    const dealii::FullMatrix<double> &local_Mass_Matrix,
    const int nstate,
    const unsigned int n_dofs)
{
    dealii::FullMatrix<double> dim_FR_operator(n_dofs);
    if constexpr (dim == 1){
        dim_FR_operator = this->oneD_vol_operator;
    }
    if (dim >= 2){
        dealii::FullMatrix<double> FR1(n_dofs);
        FR1 = this->tensor_product_state(nstate, this->oneD_vol_operator, local_Mass_Matrix, local_Mass_Matrix);
        dealii::FullMatrix<double> FR2(n_dofs);
        FR2 = this->tensor_product_state(nstate, local_Mass_Matrix, this->oneD_vol_operator, local_Mass_Matrix);
        dealii::FullMatrix<double> FR_cross1(n_dofs);
        FR_cross1 = this->tensor_product_state(nstate, this->oneD_vol_operator, this->oneD_vol_operator, local_Mass_Matrix);
        dim_FR_operator.add(1.0, FR1, 1.0, FR2, 1.0, FR_cross1);
    }
    if constexpr (dim == 3){
        dealii::FullMatrix<double> FR3(n_dofs);
        FR3 = this->tensor_product_state(nstate, local_Mass_Matrix, local_Mass_Matrix, this->oneD_vol_operator);
        dealii::FullMatrix<double> FR_cross2(n_dofs);
        FR_cross2 = this->tensor_product_state(nstate, this->oneD_vol_operator, local_Mass_Matrix, this->oneD_vol_operator);
        dealii::FullMatrix<double> FR_cross3(n_dofs);
        FR_cross3 = this->tensor_product_state(nstate, local_Mass_Matrix, this->oneD_vol_operator, this->oneD_vol_operator);
        dealii::FullMatrix<double> FR_triple(n_dofs);
        FR_triple = this->tensor_product_state(nstate, this->oneD_vol_operator, this->oneD_vol_operator, this->oneD_vol_operator);
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
    //get the FR corrcetion parameter value
    get_FR_aux_correction_parameter(this->max_degree, FR_param_aux);
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
    const unsigned int n_quad_pts = quadrature.size();
    vol_integral_basis<dim,n_faces> integral_vol_basis(this->nstate, this->max_degree, this->max_grid_degree);
    integral_vol_basis.build_1D_volume_operator(finite_element, quadrature);
    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    dealii::FullMatrix<double> mass_inv(n_dofs);
    mass_inv.invert(local_Mass_Matrix.oneD_vol_operator);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_quad_pts);
    //solves
    compute_local_vol_projection_operator(mass_inv, integral_vol_basis.oneD_vol_operator, this->oneD_vol_operator);
}

template <int dim, int n_faces>  
vol_projection_operator_FR<dim,n_faces>::vol_projection_operator_FR(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction FR_param_input,
    const bool store_transpose_input)
    : vol_projection_operator<dim,n_faces>::vol_projection_operator(nstate_input, max_degree_input, grid_degree_input)
    , store_transpose(store_transpose_input)
    , FR_param_type(FR_param_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
}

template <int dim, int n_faces>  
void vol_projection_operator_FR<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    const unsigned int n_quad_pts = quadrature.size();
    vol_integral_basis<dim,n_faces> integral_vol_basis(this->nstate, this->max_degree, this->max_grid_degree);
    integral_vol_basis.build_1D_volume_operator(finite_element, quadrature);
    FR_mass_inv<dim,n_faces> local_FR_Mass_Matrix_inv(this->nstate, this->max_degree, this->max_grid_degree, FR_param_type);
    local_FR_Mass_Matrix_inv.build_1D_volume_operator(finite_element, quadrature);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_quad_pts);
    //solves
    this->compute_local_vol_projection_operator(local_FR_Mass_Matrix_inv.oneD_vol_operator, integral_vol_basis.oneD_vol_operator, this->oneD_vol_operator);
    
    if(store_transpose){
        oneD_transpose_vol_operator.reinit(n_quad_pts, n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                oneD_transpose_vol_operator[iquad][idof] = this->oneD_vol_operator[idof][iquad];
            }
        }
    }
}
template <int dim, int n_faces>  
vol_projection_operator_FR_aux<dim,n_faces>::vol_projection_operator_FR_aux(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_input,
    const bool store_transpose_input)
    : vol_projection_operator<dim,n_faces>::vol_projection_operator(nstate_input, max_degree_input, grid_degree_input)
    , store_transpose(store_transpose_input)
    , FR_param_type(FR_param_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
}

template <int dim, int n_faces>  
void vol_projection_operator_FR_aux<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    const unsigned int n_quad_pts = quadrature.size();
    vol_integral_basis<dim,n_faces> integral_vol_basis(this->nstate, this->max_degree, this->max_grid_degree);
    integral_vol_basis.build_1D_volume_operator(finite_element, quadrature);
    FR_mass_inv_aux<dim,n_faces> local_FR_Mass_Matrix_inv(this->nstate, this->max_degree, this->max_grid_degree, FR_param_type);
    local_FR_Mass_Matrix_inv.build_1D_volume_operator(finite_element, quadrature);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_quad_pts);
    //solves
    this->compute_local_vol_projection_operator(local_FR_Mass_Matrix_inv.oneD_vol_operator, integral_vol_basis.oneD_vol_operator, this->oneD_vol_operator);
    
    if(store_transpose){
        oneD_transpose_vol_operator.reinit(n_quad_pts, n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                oneD_transpose_vol_operator[iquad][idof] = this->oneD_vol_operator[idof][iquad];
            }
        }
    }
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
FR_mass_inv_aux<dim,n_faces>::FR_mass_inv_aux(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , FR_param_type(FR_param_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
}

template <int dim, int n_faces>  
void FR_mass_inv_aux<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    local_Flux_Reconstruction_operator_aux<dim,n_faces> local_FR_oper(this->nstate, this->max_degree, this->max_grid_degree, FR_param_type);
    local_FR_oper.build_1D_volume_operator(finite_element, quadrature);
    dealii::FullMatrix<double> FR_mass_matrix(n_dofs);
    FR_mass_matrix.add(1.0, local_Mass_Matrix.oneD_vol_operator, 1.0, local_FR_oper.oneD_vol_operator);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //solves
    this->oneD_vol_operator.invert(FR_mass_matrix);
}
template <int dim, int n_faces>  
FR_mass<dim,n_faces>::FR_mass(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction FR_param_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , FR_param_type(FR_param_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
}

template <int dim, int n_faces>  
void FR_mass<dim,n_faces>::build_1D_volume_operator(
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
    this->oneD_vol_operator.add(1.0, FR_mass_matrix);
}
template <int dim, int n_faces>  
FR_mass_aux<dim,n_faces>::FR_mass_aux(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , FR_param_type(FR_param_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
}

template <int dim, int n_faces>  
void FR_mass_aux<dim,n_faces>::build_1D_volume_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    local_mass<dim,n_faces> local_Mass_Matrix(this->nstate, this->max_degree, this->max_grid_degree);
    local_Mass_Matrix.build_1D_volume_operator(finite_element, quadrature);
    local_Flux_Reconstruction_operator_aux<dim,n_faces> local_FR_oper(this->nstate, this->max_degree, this->max_grid_degree, FR_param_type);
    local_FR_oper.build_1D_volume_operator(finite_element, quadrature);
    dealii::FullMatrix<double> FR_mass_matrix(n_dofs);
    FR_mass_matrix.add(1.0, local_Mass_Matrix.oneD_vol_operator, 1.0, local_FR_oper.oneD_vol_operator);
    //allocate the volume operator
    this->oneD_vol_operator.reinit(n_dofs, n_dofs);
    //solves
    this->oneD_vol_operator.add(1.0, FR_mass_matrix);
}

template <int dim, int n_faces>
vol_integral_gradient_basis<dim,n_faces>::vol_integral_gradient_basis(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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

template <int dim, int n_faces>  
face_integral_basis<dim,n_faces>::face_integral_basis(
    const int nstate_input,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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
    //Initialize to the max degrees
    current_degree      = max_degree_input;
    current_grid_degree = grid_degree_input;
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
template <int dim, int n_faces>  
void mapping_shape_functions<dim,n_faces>::build_1D_shape_functions_at_volume_flux_nodes(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    mapping_shape_functions_flux_nodes.build_1D_volume_operator(finite_element, quadrature);
    mapping_shape_functions_flux_nodes.build_1D_gradient_operator(finite_element, quadrature);
}

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
    const bool store_vol_flux_nodes_input,
    const bool store_surf_flux_nodes_input,
    const bool store_Jacobian_input)
    : SumFactorizedOperators<dim,n_faces>::SumFactorizedOperators(nstate_input, max_degree_input, grid_degree_input)
    , store_Jacobian(store_Jacobian_input)
    , store_vol_flux_nodes(store_vol_flux_nodes_input)
    , store_surf_flux_nodes(store_surf_flux_nodes_input)
{}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::transform_physical_to_reference(
    const dealii::Tensor<1,dim,real> &phys,
    const dealii::Tensor<2,dim,real> &metric_cofactor,
    dealii::Tensor<1,dim,real> &ref)
{
    for(int idim=0; idim<dim; idim++){
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
        phys[idim] = metric_cofactor[idim] * ref;
//        phys[idim] = 0.0;
//        for(int idim2=0; idim2<dim; idim2++){
//            phys[idim] += metric_cofactor[idim][idim2] 
//                               * ref[idim2];
//        }
    }
}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::transform_physical_to_reference_vector(
    const dealii::Tensor<1,dim,std::vector<real>> &phys,
    const dealii::Tensor<2,dim,std::vector<real>> &metric_cofactor,
    dealii::Tensor<1,dim,std::vector<real>> &ref)
{
    assert(phys[0].size() == metric_cofactor[0][0].size());
    const unsigned int n_pts = phys[0].size();
    for(int idim=0; idim<dim; idim++){
        ref[idim].resize(n_pts);
        for(int idim2=0; idim2<dim; idim2++){
            for(unsigned int ipt=0; ipt<n_pts; ipt++){
                ref[idim][ipt] += metric_cofactor[idim2][idim][ipt] * phys[idim2][ipt];
            }
        }
    }

}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::transform_reference_unit_normal_to_physical_unit_normal(
    const unsigned int n_quad_pts,
    const dealii::Tensor<1,dim,real> &ref,
    const dealii::Tensor<2,dim,std::vector<real>> &metric_cofactor,
    std::vector<dealii::Tensor<1,dim,real>> &phys)
{
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        for(int idim=0; idim<dim; idim++){
            phys[iquad][idim] = 0.0;
            for(int idim2=0; idim2<dim; idim2++){
                phys[iquad][idim] += metric_cofactor[idim][idim2][iquad] 
                                   * ref[idim2];
            }
        }
        phys[iquad] /= phys[iquad].norm();
    } 
}

template <typename real, int dim, int n_faces>  
void metric_operators<real,dim,n_faces>::build_determinant_volume_metric_Jacobian(
    const unsigned int n_quad_pts,
    const unsigned int /*n_metric_dofs*/,//dofs of metric basis. NOTE: this is the number of mapping support points
    const std::array<std::vector<real>,dim> &mapping_support_points,
    mapping_shape_functions<dim,n_faces> &mapping_basis)
{
    det_Jac_vol.resize(n_quad_pts);
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

    if(store_vol_flux_nodes){
        for(int idim=0; idim<dim; idim++){
            flux_nodes_vol[idim].resize(n_quad_pts);
            this->matrix_vector_mult(mapping_support_points[idim],
                                     flux_nodes_vol[idim],
                                     mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator,
                                     mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator,
                                     mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator);
        }
    }
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
    det_Jac_surf.resize(n_quad_pts);
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
        det_Jac_surf);
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

    if(store_surf_flux_nodes){
        for(int iface=0; iface<n_faces; iface++){
            for(int idim=0; idim<dim; idim++){
                flux_nodes_surf[iface][idim].resize(n_quad_pts);
                this->matrix_vector_mult(mapping_support_points[idim],
                                         flux_nodes_surf[iface][idim],
                                         (iface == 0) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
                                             ((iface == 1) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                                                 mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator),
                                         (iface == 2) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
                                             ((iface == 3) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                                                 mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator),
                                         (iface == 4) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[0] : 
                                             ((iface == 5) ? mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator[1] : 
                                                 mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator));
            }
        }
    }
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
        //check for valid cell
        if(det_metric_Jac[iquad] <= 1e-14){
            std::cout<<"The determinant of the Jacobian is negative. Aborting..."<<std::endl;
            std::abort();
        }
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
    if (dim == 1){//constant for 1D
        std::fill(metric_cofactor[0][0].begin(), metric_cofactor[0][0].end(), 1.0);
    }
    if (dim == 2){
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
    if (dim == 3){
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
{}

template <int dim, int nstate, int n_faces>
basis_functions_state<dim,nstate,n_faces>::basis_functions_state(
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperatorsState<dim,nstate,n_faces>::SumFactorizedOperatorsState(max_degree_input, grid_degree_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
}

template <int dim, int nstate, int n_faces>
void basis_functions_state<dim,nstate,n_faces>::build_1D_volume_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts  = quadrature.size();
    const unsigned int n_dofs      = finite_element.dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs / nstate;
    //Note thate the flux basis should only have one state in the finite element.
    //loop and store
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<1> qpoint  = quadrature.point(iquad);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            const unsigned int istate = finite_element.system_to_component_index(idof).first;
            const unsigned int ishape = finite_element.system_to_component_index(idof).second;
            if(ishape == 0)
                this->oneD_vol_state_operator[istate].reinit(n_quad_pts, n_shape_fns);

            //Basis function idof of poly degree idegree evaluated at cubature node qpoint.
            this->oneD_vol_state_operator[istate][iquad][ishape] = finite_element.shape_value_component(idof,qpoint,istate);
        }
    }
}

template <int dim, int nstate, int n_faces>
void basis_functions_state<dim,nstate,n_faces>::build_1D_gradient_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts  = quadrature.size();
    const unsigned int n_dofs      = finite_element.dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs / nstate;
    //loop and store
    for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        const dealii::Point<1> qpoint  = quadrature.point(iquad);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            const unsigned int istate = finite_element.system_to_component_index(idof).first;
            const unsigned int ishape = finite_element.system_to_component_index(idof).second;
            if(ishape == 0)
                this->oneD_grad_state_operator[istate].reinit(n_quad_pts, n_shape_fns);

            this->oneD_grad_state_operator[istate][iquad][ishape] = finite_element.shape_grad_component(idof, qpoint, istate)[0];
        }
    }
}
template <int dim, int nstate, int n_faces>
void basis_functions_state<dim,nstate,n_faces>::build_1D_surface_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<0> &face_quadrature)
{
    const unsigned int n_face_quad_pts = face_quadrature.size();
    const unsigned int n_dofs          = finite_element.dofs_per_cell;
    const unsigned int n_faces_1D      = n_faces / dim;
    const unsigned int n_shape_fns     = n_dofs / nstate;
    //loop and store
    for(unsigned int iface=0; iface<n_faces_1D; iface++){ 
        const dealii::Quadrature<1> quadrature = dealii::QProjector<1>::project_to_face(dealii::ReferenceCell::get_hypercube(1),
                                                                                            face_quadrature,
                                                                                            iface);
        for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
            for(unsigned int idof=0; idof<n_dofs; idof++){
                const unsigned int istate = finite_element.system_to_component_index(idof).first;
                const unsigned int ishape = finite_element.system_to_component_index(idof).second;
                if(ishape == 0)
                    this->oneD_surf_state_operator[istate][iface].reinit(n_face_quad_pts, n_shape_fns);

                this->oneD_surf_state_operator[istate][iface][iquad][ishape] = finite_element.shape_value_component(idof,quadrature.point(iquad),istate);
            }
        }
    }
}

template <int dim, int nstate, int n_faces>
flux_basis_functions_state<dim,nstate,n_faces>::flux_basis_functions_state(
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input)
    : SumFactorizedOperatorsState<dim,nstate,n_faces>::SumFactorizedOperatorsState(max_degree_input, grid_degree_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
}

template <int dim, int nstate, int n_faces>
void flux_basis_functions_state<dim,nstate,n_faces>::build_1D_volume_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    assert(n_dofs == n_quad_pts);
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
void flux_basis_functions_state<dim,nstate,n_faces>::build_1D_gradient_state_operator(
    const dealii::FESystem<1,1> &finite_element,
    const dealii::Quadrature<1> &quadrature)
{
    const unsigned int n_quad_pts = quadrature.size();
    const unsigned int n_dofs     = finite_element.dofs_per_cell;
    assert(n_dofs == n_quad_pts);
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
void flux_basis_functions_state<dim,nstate,n_faces>::build_1D_surface_state_operator(
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
    : flux_basis_functions_state<dim,nstate,n_faces>::flux_basis_functions_state(max_degree_input, grid_degree_input)
{
    //Initialize to the max degrees
    current_degree      = max_degree_input;
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


template class OperatorsBase <PHILIP_DIM, 2*PHILIP_DIM>;

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
template class vol_projection_operator_FR_aux <PHILIP_DIM, 2*PHILIP_DIM>;
template class FR_mass_inv <PHILIP_DIM, 2*PHILIP_DIM>;
template class FR_mass_inv_aux <PHILIP_DIM, 2*PHILIP_DIM>;
template class FR_mass <PHILIP_DIM, 2*PHILIP_DIM>;
template class FR_mass_aux <PHILIP_DIM, 2*PHILIP_DIM>;
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
template class basis_functions_state <PHILIP_DIM, 1, 2*PHILIP_DIM>;
template class basis_functions_state <PHILIP_DIM, 2, 2*PHILIP_DIM>;
template class basis_functions_state <PHILIP_DIM, 3, 2*PHILIP_DIM>;
template class basis_functions_state <PHILIP_DIM, 4, 2*PHILIP_DIM>;
template class basis_functions_state <PHILIP_DIM, 5, 2*PHILIP_DIM>;

template class flux_basis_functions_state <PHILIP_DIM, 1, 2*PHILIP_DIM>;
template class flux_basis_functions_state <PHILIP_DIM, 2, 2*PHILIP_DIM>;
template class flux_basis_functions_state <PHILIP_DIM, 3, 2*PHILIP_DIM>;
template class flux_basis_functions_state <PHILIP_DIM, 4, 2*PHILIP_DIM>;
template class flux_basis_functions_state <PHILIP_DIM, 5, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 1, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 2, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 3, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 4, 2*PHILIP_DIM>;
template class local_flux_basis_stiffness <PHILIP_DIM, 5, 2*PHILIP_DIM>;

} // OPERATOR namespace
} // PHiLiP namespace

