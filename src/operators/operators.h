#ifndef __OPERATORS_H__
#define __OPERATORS_H__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/qprojector.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/fe_q.h>


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


#include "parameters/all_parameters.h"
#include "parameters/parameters.h"

namespace PHiLiP {
namespace OPERATOR {
///Operator base class
/**
 *This base class constructs 4 different types of operators: volume, surface, flux and metric. In addition it has corresponding functions to compute the operators.
 * The general operator form order is:
 * vector of degree, vector of state (for flux operators), vector of face (for surface operators), vector of dim (for gradient operators), FullMatrix of quadrature points or degrees of freedom. 
 * In more detail for each type of operator:
 * (1) Volume Operators: Always start as a vector of polynomial degree, then if its a flux vector of state variables, then dimension, then Matrix where the evaluations happen on Volume Cubature nodes. 
 * (2) Surface Operators: Always start as a vector of polynomial degree, then if its a flux vector of state variables, then the face index which goes from face 0 clockwise as-per dealii standard to nfaces, then dimension, then Matrix where the evaluations happen on Volume Cubature nodes. Lastly, currently assume tensor-product elements so \f$n_{faces} = 2.0 * dim\f$. 
 * (3) Flux Operators: See above. It is important to note that since flux operators are "collocated" on the cubature set, the number of degrees of freedom of the flux basis MUST equal the number of cubature nodes. Importantly on the surface, the flux basis interpolates from the volume to the surface, thus corresponds to volume cubature nodes collocation.  
 * (4) Metric Operators: Since the solution polynomial degree for each state varies, along with the local grid element's polynomial degree varying, the operators distinguish between polynomial degree (for solution or flux) and grid degree (for element). Explicitly, they first go vector of grid_degree, then vector of polynomial degree for the ones that are applied on the flux nodes. The mapping-support-points follow the standard from dealii, where they are always Gauss-Legendre-Lobatto quadrature nodes making the polynomial elements continuous in a sense. 
 */

template <int dim, int n_faces>
class OperatorsBase
{
public:
    ///Constructor
    OperatorsBase(
          const int nstate_input,//number of states input
          const unsigned int max_degree_input,//max poly degree for operators
          const unsigned int grid_degree_input);//max grid degree for operators
    ///Destructor
    ~OperatorsBase();

    ///Max polynomial degree.
    const unsigned int max_degree;
    ///Max grid degree.
    const unsigned int max_grid_degree;
    ///Number of states.
    const int nstate;

protected:
    ///Check to see if the metrics used are a higher order then the initialized grid.
    unsigned int max_grid_degree_check;

public:

    ///Returns the tensor product of matrices passed.
    dealii::FullMatrix<double> tensor_product(
                                    const dealii::FullMatrix<double> &basis_x,
                                    const dealii::FullMatrix<double> &basis_y,
                                    const dealii::FullMatrix<double> &basis_z);

    ///Returns the tensor product of matrices passed, but makes it sparse diagonal by state.
    /** An example for this would be a multi-state mass matrix. When the mass matrix
    * is constructed, it is not templated by nstate. Also, if each 1D mass matrix is multi-state
    * then their tensor product does not match the dim multi-state mass matrix.
    * Instead, only if the states match, then do the tensor product. This results in a diagonal
    * sparse matrix by state number, with each state's block being a dim-ordered tensor product.
    */
    dealii::FullMatrix<double> tensor_product_state(
                                    const int nstate,
                                    const dealii::FullMatrix<double> &basis_x,
                                    const dealii::FullMatrix<double> &basis_y,
                                    const dealii::FullMatrix<double> &basis_z);

    ///Standard function to compute factorial of a number.
    double compute_factorial(double n);

    ///virtual function to be defined.
    virtual void matrix_vector_mult(
                const std::vector<double> &input_vect,
                std::vector<double> &output_vect,
                const dealii::FullMatrix<double> &basis_x,
                const dealii::FullMatrix<double> &basis_y,
                const dealii::FullMatrix<double> &basis_z,
                const bool adding = false,
                const double factor = 1.0) = 0;
    ///virtual function to be defined.
    virtual void inner_product(
                const std::vector<double> &input_vect,
                const std::vector<double> &weight_vect,
                std::vector<double> &output_vect,
                const dealii::FullMatrix<double> &basis_x,
                const dealii::FullMatrix<double> &basis_y,
                const dealii::FullMatrix<double> &basis_z,
                const bool adding = false,
                const double factor = 1.0) = 0;
protected:

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
};//End of OperatorsBase

///Sum Factorization derived class.
/* All dim-sized operators are constructed by their one dimensional equivalent, and we use 
* sum factorization to perform their operations.
* Note that we assume tensor product elements in this operators class.
*/
template<int dim, int n_faces>
class SumFactorizedOperators : public OperatorsBase<dim,n_faces>
{

public:
    /// Precompute 1D operator in constructor
    SumFactorizedOperators(
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);
    ///Destructor
    ~SumFactorizedOperators ();

    ///Computes a matrix-vector product using sum-factorization. Pass the one-dimensional basis, where x runs the fastest, then y, and z runs the slowest. Also, assume each one-dimensional basis is the same size.
    /** Uses sum-factorization with BLAS techniques to solve the the matrix-vector multiplication, where the matrix is the tensor product of three one-dimensional matrices. We use the standard notation that x runs the fastest, then y, and z runs the slowest.
    * For an operator \f$\mathbf{A}\f$ of size \f$n^d\f$ with \f$n\f$ the one dimensional dofs
    * and \f$ d\f$ the dim, for ouput row vector \f$ v \f$ and input row vector \f$ u \f$,
    * we compute \f$v^T=\mathbf{A}u^T\f$.
    * Lastly, the adding allows the result to add onto the previous output_vect scaled by "factor".
    */
    void matrix_vector_mult(
            const std::vector<double> &input_vect,
            std::vector<double> &output_vect,
            const dealii::FullMatrix<double> &basis_x,
            const dealii::FullMatrix<double> &basis_y,
            const dealii::FullMatrix<double> &basis_z,
            const bool adding = false,
            const double factor = 1.0) override;
    ///Computes the divergence using the sum factorization matrix-vector multiplication.
    /** Often, we compute a dot product in dim, where each matrix multiplictaion uses
    * sum factorization. Example, consider taking the reference divergence of the reference flux:
    * \f[ 
    * \nabla^r \cdot f^r = \frac{\partial f^r}{\partial \xi} + \frac{\partial f^r}{\partial \eta} +
    * \frac{\partial f^r}{\partial \zeta} = 
    *   \left( \frac{d \mathbf{\chi}(\mathbf{\xi})}{d\xi} \otimes \mathbf{\chi}(\mathbf{\eta}) \otimes \mathbf{\chi}(\mathbf{\zeta}) \right) \left(\hat{\mathbf{f}^r}\right)^T
    * + \left( \mathbf{\chi}(\mathbf{\xi}) \otimes \frac{d\mathbf{\chi}(\mathbf{\eta})}{d\eta} \otimes \mathbf{\chi}(\mathbf{\zeta}) \right) \left(\hat{\mathbf{f}^r}\right)^T
    * + \left( \mathbf{\chi}(\mathbf{\xi}) \otimes \mathbf{\chi}(\mathbf{\eta} \otimes \frac{d\mathbf{\chi}(\mathbf{\zeta})}{d\zeta})\right) \left(\hat{\mathbf{f}^r}\right)^T,
    * \f] where we use sum factorization to evaluate each matrix-vector multiplication in each dim direction.
    */
    void divergence_matrix_vector_mult(
            const dealii::Tensor<1,dim,std::vector<double>> &input_vect,
            std::vector<double> &output_vect,
            const dealii::FullMatrix<double> &basis_x,
            const dealii::FullMatrix<double> &basis_y,
            const dealii::FullMatrix<double> &basis_z,
            const dealii::FullMatrix<double> &gradient_basis_x,
            const dealii::FullMatrix<double> &gradient_basis_y,
            const dealii::FullMatrix<double> &gradient_basis_z);

    ///Computes the divergence using sum-factorization where the basis are the same in each direction.
    void divergence_matrix_vector_mult_1D(
            const dealii::Tensor<1,dim,std::vector<double>> &input_vect,
            std::vector<double> &output_vect,
            const dealii::FullMatrix<double> &basis,
            const dealii::FullMatrix<double> &gradient_basis);

    ///Computes the gradient of a scalar using sum-factorization.
    void gradient_matrix_vector_mult(
            const std::vector<double> &input_vect,
            dealii::Tensor<1,dim,std::vector<double>> &output_vect,
            const dealii::FullMatrix<double> &basis_x,
            const dealii::FullMatrix<double> &basis_y,
            const dealii::FullMatrix<double> &basis_z,
            const dealii::FullMatrix<double> &gradient_basis_x,
            const dealii::FullMatrix<double> &gradient_basis_y,
            const dealii::FullMatrix<double> &gradient_basis_z);
    ///Computes the gradient of a scalar using sum-factorization where the basis are the same in each direction.
    void gradient_matrix_vector_mult_1D(
            const std::vector<double> &input_vect,
            dealii::Tensor<1,dim,std::vector<double>> &output_vect,
            const dealii::FullMatrix<double> &basis,
            const dealii::FullMatrix<double> &gradient_basis);

    ///Computes the inner product between a matrix and a vector multiplied by some weight function.  
    /** That is, we compute \f$ \int Awu d\mathbf{\Omega}_r = \mathbf{A}^T \text{diag}(w) \mathbf{u}^T \f$. When using this function, pass \f$ \mathbf{A} \f$ and NOT it's transpose--the function transposes it in the first few lines.
    */
    void inner_product(
            const std::vector<double> &input_vect,
            const std::vector<double> &weight_vect,
            std::vector<double> &output_vect,
            const dealii::FullMatrix<double> &basis_x,
            const dealii::FullMatrix<double> &basis_y,
            const dealii::FullMatrix<double> &basis_z,
            const bool adding = false,
            const double factor = 1.0) override;


    ///Computes the divergence of the 2pt flux Hadamard products, then sums the rows.
    /** Note that we have the factor of 2.0 definied in this function.
    * We also make use of the structure of the flux basis to get the matrix vector product after the Hadamard product
    * to be \f$ \mathcal{O}(n^{d+1})\f$.
    */
    void divergence_two_pt_flux_Hadamard_product(
            const dealii::Tensor<1,dim,dealii::FullMatrix<double>> &input_mat,
            std::vector<double> &output_vect,
            const dealii::FullMatrix<double> &basis);//the only direction that isn't identity

    ///Computes the Hadamard product ONLY for 2pt flux calculations.
    /**
    * The Hadamard product comes up naturally in calculcating 2-point fluxes, so we needed an efficient way to compute it.
    * Using the commutative property of Hadamard products: \f$ (A \otimes B) \circ ( C \otimes D) = (A\circ C) \otimes (B\circ D) \f$,
    * we can find a "sum-factorization" type expression for \f$ A \circ U \f$, where here \f$ A = A_x \otimes A_y \otimes A_z \f$
    * and \f$ U \f$ is an \f$ n \times n \f$ matrix. <br>
    * We make use of the flux basis being collocated on flux nodes, so the directions that aren't the derivative are identity.
    * This results in the Hadamard product only needing \f$ \mathcal{O}(n^{d+1})\f$ flops to compute non-zero entries.<br>
    * This is NOT for GENERAL Hadamard products since those are \f$ \mathcal{O}(n^{2d})\f$ .
    */
    void two_pt_flux_Hadamard_product(
            const dealii::FullMatrix<double> &input_mat,
            dealii::FullMatrix<double> &output_mat,
            const dealii::FullMatrix<double> &basis,//the only direction that isn't identity
            const int direction);//direction for the derivative that corresponds to basis

    /// Apply the matrix vector operation using the 1D operator in each direction
    /** This is for the case where the operator of size dim is the dyadic product of
    * the same 1D operator in each direction
    */
    void matrix_vector_mult_1D(
            const std::vector<double> &input_vect,
            std::vector<double> &output_vect,
            const dealii::FullMatrix<double> &basis_x,
            const bool adding = false,
            const double factor = 1.0);

    /// Apply the inner product operation using the 1D operator in each direction
    /* This is for the case where the operator of size dim is the dyadic product of
    * the same 1D operator in each direction
    */
    void inner_product_1D(
            const std::vector<double> &input_vect,
            const std::vector<double> &weight_vect,
            std::vector<double> &output_vect,
            const dealii::FullMatrix<double> &basis_x,
            const bool adding  = false,
            const double factor = 1.0);

    /// Apply sum-factorization matrix vector multiplication on a surface.
    /** Often times we have to interpolate to a surface, where in multiple dimensions,
    * that's the tensor product of a surface operator with volume operators. This simplifies
    * the function call.
    * Explicitly, this passes basis_surf in the direction by face_number, and basis_vol
    * in all other directions.
    */
    void matrix_vector_mult_surface_1D(
            const unsigned int face_number,
            const std::vector<double> &input_vect,
            std::vector<double> &output_vect,
            const std::array<dealii::FullMatrix<double>,2> &basis_surf,//only 2 faces in 1D
            const dealii::FullMatrix<double> &basis_vol,
            const bool adding = false,
            const double factor = 1.0);

    /// Apply sum-factorization inner product on a surface.
    void inner_product_surface_1D(
            const unsigned int face_number,
            const std::vector<double> &input_vect,
            const std::vector<double> &weight_vect,
            std::vector<double> &output_vect,
            const std::array<dealii::FullMatrix<double>,2> &basis_surf,//only 2 faces in 1D
            const dealii::FullMatrix<double> &basis_vol,
            const bool adding = false,
            const double factor = 1.0);

protected:

    ///Computes a single Hadamard product. 
    /** For input mat1 \f$ A \f$ and input mat2 \f$ B \f$, this computes
    * \f$ A \circ B = C \implies \left( C \right)_{ij} = \left( A \right)_{ij}\left( B \right)_{ij}\f$.
    */
    void Hadamard_product(
        const dealii::FullMatrix<double> &input_mat1,
        const dealii::FullMatrix<double> &input_mat2,
        dealii::FullMatrix<double> &output_mat);

//protected:
public:
    ///Stores the one dimensional volume operator.
    dealii::FullMatrix<double>  oneD_vol_operator;

    ///Stores the one dimensional surface operator.
    /** Note that in 1D there are only 2 faces
    */
    std::array<dealii::FullMatrix<double>,2>  oneD_surf_operator;

    ///Stores the one dimensional gradient operator.
    dealii::FullMatrix<double> oneD_grad_operator;

    ///Stores the one dimensional surface gradient operator.
    std::array<dealii::FullMatrix<double>,2>  oneD_surf_grad_operator;

};//End of SumFactorizedOperators Class

/************************************************************************
*
*      VOLUME OPERATORS
*
************************************************************************/

///Basis functions.
/* This class stores the basis functions evaluated at volume and facet
* cubature nodes, as well as it's gradient in REFERENCE space.
*/
template<int dim, int n_faces>
class basis_functions : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    basis_functions (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~basis_functions ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_gradient_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_surface_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<0> &quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_surface_gradient_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<0> &quadrature);
};

///\f$ \mathbf{W}*\mathbf{\chi}(\mathbf{\xi}_v^r) \f$  That is Quadrature Weights multiplies with basis_at_vol_cubature.
template<int dim, int n_faces>
class vol_integral_basis : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    vol_integral_basis (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~vol_integral_basis ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

///Local mass matrix without jacobian dependence.
template<int dim, int n_faces>
class local_mass : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    local_mass (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~local_mass ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature); //override;

    ///Assemble the dim mass matrix on the fly with metric Jacobian dependence.
    /** Note, n_shape_functions is not the same as n_dofs, but the number of 
    * shape functions for the state.
    */
    dealii::FullMatrix<double> build_dim_mass_matrix(
        const int nstate,
        const unsigned int n_dofs, const unsigned int n_quad_pts,
        basis_functions<dim,n_faces> &basis,
        const std::vector<double> &det_Jac,
        const std::vector<double> &quad_weights);
};

///Local stiffness matrix without jacobian dependence.
/**NOTE: this is not used in DG volume integral since that needs to use the derivative of the flux basis and is multiplied by flux at volume cubature nodes this is strictly for consturtcing D operator
*\f[
        (\mathbf{S}_\xi)_{ij}  = \int_\mathbf{{\Omega}_r} \mathbf{\chi}_i(\mathbf{\xi}^r) \frac{\mathbf{\chi}_{j}(\mathbf{\xi}^r)}{\partial \xi} d\mathbf{\Omega}_r
        \f]
*/
template<int dim, int n_faces>
class local_basis_stiffness : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    local_basis_stiffness (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~local_basis_stiffness ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

///This is the solution basis \f$\mathbf{D}_i\f$, the modal differential opertaor commonly seen in DG defined as \f$\mathbf{D}_i=\mathbf{M}^{-1}*\mathbf{S}_i\f$.
template<int dim, int n_faces>
class modal_basis_differential_operator : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    modal_basis_differential_operator (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~modal_basis_differential_operator ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

///\f$ p\f$ -th order modal derivative of basis fuctions, ie/\f$ [D_\xi^p, D_\eta^p, D_\zeta^p]\f$
template<int dim, int n_faces>
class derivative_p : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    derivative_p (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~derivative_p ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

/// ESFR correction matrix without jac dependence
template<int dim, int n_faces>
class local_Flux_Reconstruction_operator : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    local_Flux_Reconstruction_operator (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction FR_param_input);
    ///Destructor.
    ~local_Flux_Reconstruction_operator ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction FR_param_type;

    ///Flux reconstruction paramater value.
    double FR_param;

    ///Evaluates Huynh's g2 parameter for flux reconstruction.
    /* This parameter recovers Huynh, Hung T. "A flux reconstruction approach to high-order schemes including discontinuous Galerkin methods." 18th AIAA computational fluid dynamics conference. 2007.
    */
    void get_Huynh_g2_parameter (
            const unsigned int curr_cell_degree,
            double &c);

    ///Evaluates the spectral difference parameter for flux reconstruction.
    /**Value from Allaneau, Y., and Antony Jameson. "Connections between the filtered discontinuous Galerkin method and the flux reconstruction approach to high order discretizations." Computer Methods in Applied Mechanics and Engineering 200.49-52 (2011): 3628-3636. 
    */
    void get_spectral_difference_parameter (
            const unsigned int curr_cell_degree,
            double &c);

    ///Evaluates the flux reconstruction parameter at the bottom limit where the scheme is unstable.
    /**Value from Allaneau, Y., and Antony Jameson. "Connections between the filtered discontinuous Galerkin method and the flux reconstruction approach to high order discretizations." Computer Methods in Applied Mechanics and Engineering 200.49-52 (2011): 3628-3636. 
    */
    void get_c_negative_FR_parameter (
            const unsigned int curr_cell_degree,
            double &c);

    ///Evaluates the flux reconstruction parameter at the bottom limit where the scheme is unstable, divided by 2.
    /**Value from Allaneau, Y., and Antony Jameson. "Connections between the filtered discontinuous Galerkin method and the flux reconstruction approach to high order discretizations." Computer Methods in Applied Mechanics and Engineering 200.49-52 (2011): 3628-3636. 
    * Commonly in the lterature we use this value to show the approach to the bottom limit of stability.
    */
    void get_c_negative_divided_by_two_FR_parameter (
            const unsigned int curr_cell_degree,
            double &c);

    ///Gets the FR correction parameter corresponding to the maximum timestep.
    /** Note that this parameter is also a good approximation for when the FR scheme begins to
    * lose an order of accuracy, but the original definition is that it corresponds to the maximum timestep.
    * Value from Table 3.4 in Castonguay, Patrice. High-order energy stable flux reconstruction schemes for fluid flow simulations on unstructured grids. Stanford University, 2012.
    */
    void get_c_plus_parameter (
            const unsigned int curr_cell_degree,
            double &c);

    ///Gets the FR correction parameter for the primary equation and stores.
    /**These values are name specified in parameters/all_parameters.h, passed through control file/or test and here converts/stores as value.
    * Please note that in all the functions within this that evaluate the parameter, we divide the value in the literature by 2.0
    * because our basis are contructed by an orthonormal Legendre basis rather than the orthogonal basis in the literature. 
    * Also, we have the additional scaling by pow(pow(2.0,curr_cell_degree),2) because our basis functions are defined on
    * a reference element between [0,1], whereas the values in the literature are based on [-1,1].
    * For further details please refer to Cicchino, Alexander, and Siva Nadarajah. "A new norm and stability condition for tensor product flux reconstruction schemes." Journal of Computational Physics 429 (2021): 110025.
    */
    void get_FR_correction_parameter (
            const unsigned int curr_cell_degree,
            double &c);

   ///Computes a single local Flux_Reconstruction operator (ESFR correction operator) on the fly for a local element.
   /**Note that this is dependent on the Mass Matrix, so for metric Jacobian dependent \f$K_m\f$,
   *pass the metric Jacobian dependent Mass Matrix \f$M_m\f$.
    */
    void build_local_Flux_Reconstruction_operator(
            const dealii::FullMatrix<double> &local_Mass_Matrix,
            const dealii::FullMatrix<double> &pth_derivative,
            const unsigned int n_dofs, 
            const double c,
            dealii::FullMatrix<double> &Flux_Reconstruction_operator);

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

   ///Computes the dim sized flux reconstruction operator with simplified tensor product form.
   /** The formula for the dim sized flux reconstruction operator is
   * \f$ \mathbf{K}_m = \sum_{s,v,w} c_{(s,v,w)} \Big( \mathbf{D}_\xi^s\mathbf{D}_\eta^v\mathbf{D}_\zeta^w \Big)^T \mathbf{M}_m \Big( \mathbf{D}_\xi^s\mathbf{D}_\eta^v\mathbf{D}_\zeta^w \Big) \f$,
   * where \f$ c_{(s,v,w)} = c^{\frac{s}{p} + \frac{v}{p} +\frac{w}{p}}\f$.
   * Please pass the number of dofs for the dim sized operator.
   */
    dealii::FullMatrix<double> build_dim_Flux_Reconstruction_operator(
            const dealii::FullMatrix<double> &local_Mass_Matrix,
            const int nstate,
            const unsigned int n_dofs);

   ///Computes the dim sized flux reconstruction operator for general Mass Matrix (needed for curvilinear).
   /** The formula for the dim sized flux reconstruction operator is
   * \f$ \mathbf{K}_m = \sum_{s,v,w} c_{(s,v,w)} \Big( \mathbf{D}_\xi^s\mathbf{D}_\eta^v\mathbf{D}_\zeta^w \Big)^T \mathbf{M}_m \Big( \mathbf{D}_\xi^s\mathbf{D}_\eta^v\mathbf{D}_\zeta^w \Big) \f$,
   * where \f$ c_{(s,v,w)} = c^{\frac{s}{p} + \frac{v}{p} +\frac{w}{p}}\f$.
   * Please pass the number of dofs for the dim sized operator.
   * For pth deriv, please pass the 1D operator.
   */
    dealii::FullMatrix<double> build_dim_Flux_Reconstruction_operator_directly(
        const int nstate,
        const unsigned int n_dofs,
        dealii::FullMatrix<double> &pth_deriv,
        dealii::FullMatrix<double> &mass_matrix);
};

/// ESFR correction matrix for AUX EQUATION without jac dependence
/** NOTE Auxiliary equation is a vector in dim, so theres an ESFR correction for each dim -> Flux_Reconstruction_aux also a vector of dim
* ie/ local_Flux_Reconstruction_operator_aux[degree_index][dimension_index] = Flux_Reconstruction_operator for AUX eaquation in direction dimension_index for
* polynomial degree of degree_index+1
*/
template<int dim, int n_faces>
class local_Flux_Reconstruction_operator_aux : public local_Flux_Reconstruction_operator<dim,n_faces>
{
public:
    ///Constructor.
    local_Flux_Reconstruction_operator_aux (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_aux_input);

    ///Destructor.
    ~local_Flux_Reconstruction_operator_aux ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_aux_type;

    ///Flux reconstruction paramater value.
    double FR_param_aux;

    ///Gets the FR correction parameter for the auxiliary equations and stores.
    /**These values are name specified in parameters/all_parameters.h, passed through control file/or test and here converts/stores as value.
    * Please note that in all the functions within this that evaluate the parameter, we divide the value in the literature by 2.0
    * because our basis are contructed by an orthonormal Legendre basis rather than the orthogonal basis in the literature. 
    * Also, we have the additional scaling by pow(pow(2.0,curr_cell_degree),2) because our basis functions are defined on
    * a reference element between [0,1], whereas the values in the literature are based on [-1,1].
    * For further details please refer to Cicchino, Alexander, and Siva Nadarajah. "A new norm and stability condition for tensor product flux reconstruction schemes." Journal of Computational Physics 429 (2021): 110025.
    */
    void get_FR_aux_correction_parameter (
            const unsigned int curr_cell_degree,
            double &k);

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

///Projection operator corresponding to basis functions onto M-norm (L2).
template<int dim, int n_faces>
class vol_projection_operator : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    vol_projection_operator (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~vol_projection_operator ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Computes a single local projection operator on some space (norm).
    void compute_local_vol_projection_operator(
            const dealii::FullMatrix<double> &norm_matrix_inverse, 
            const dealii::FullMatrix<double> &integral_vol_basis, 
            dealii::FullMatrix<double> &volume_projection);

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

///Projection operator corresponding to basis functions onto \f$(M+K)\f$-norm.
template<int dim, int n_faces>
class vol_projection_operator_FR : public vol_projection_operator<dim,n_faces>
{
public:
    ///Constructor.
    vol_projection_operator_FR (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction FR_param_input,
        const bool store_transpose_input = false);

    ///Destructor.
    ~vol_projection_operator_FR ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flag is store transpose operator.
    bool store_transpose;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction FR_param_type;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Stores the transpose of the operator for fast weight-adjusted solves.
    dealii::FullMatrix<double> oneD_transpose_vol_operator;
};

///Projection operator corresponding to basis functions onto \f$(M+K)\f$-norm for auxiliary equation.
template<int dim, int n_faces>
class vol_projection_operator_FR_aux : public vol_projection_operator<dim,n_faces>
{
public:
    ///Constructor.
    vol_projection_operator_FR_aux (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_input,
        const bool store_transpose_input = false);

    ///Destructor.
    ~vol_projection_operator_FR_aux ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flag is store transpose operator.
    bool store_transpose;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_type;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Stores the transpose of the operator for fast weight-adjusted solves.
    dealii::FullMatrix<double> oneD_transpose_vol_operator;
};

///The metric independent inverse of the FR mass matrix \f$(M+K)^{-1}\f$.
template<int dim, int n_faces>
class FR_mass_inv : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    FR_mass_inv (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction FR_param_input);

    ///Destructor.
    ~FR_mass_inv ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction FR_param_type;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};
///The metric independent inverse of the FR mass matrix for auxiliary equation \f$(M+K)^{-1}\f$.
template<int dim, int n_faces>
class FR_mass_inv_aux : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    FR_mass_inv_aux (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_input);

    ///Destructor.
    ~FR_mass_inv_aux ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_type;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};
///The metric independent FR mass matrix \f$(M+K)\f$.
template<int dim, int n_faces>
class FR_mass : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    FR_mass (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction FR_param_input);

    ///Destructor.
    ~FR_mass ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction FR_param_type;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

///The metric independent FR mass matrix for auxiliary equation \f$(M+K)\f$.
template<int dim, int n_faces>
class FR_mass_aux : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    FR_mass_aux (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_input);

    ///Destructor.
    ~FR_mass_aux ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction_Aux FR_param_type;

    ///Assembles the one dimensional operator.
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

///The integration of gradient of solution basis.
/**Please note that for the weak form volume integral with arbitrary integration strength, use the transpose of the following operator.
*Note: that it is also a vector of nstate since it will be applied to a flux vector per state.
*Lastly its gradient of basis functions NOT flux basis because in the weak form the test function is the basis function not the flux basis (technically the flux is spanned by the flux basis at quadrature nodes).
 *   \f[
 *           \mathbf{W}\nabla\Big(\chi_i(\mathbf{\xi}^r)\Big)  
 *   \f]
 */
template <int dim, int n_faces>  
class vol_integral_gradient_basis : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    vol_integral_gradient_basis (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~vol_integral_gradient_basis ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_gradient_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);
};

/************************************************************************
*
*      SURFACE OPERATORS
*
************************************************************************/


///The surface integral of test functions.
/**\f[
*     \mathbf{W}_f \mathbf{\chi}(\mathbf{\xi}_f^r) 
* \f]
*ie/ diag of REFERENCE unit normal times facet quadrature weights times solution basis functions evaluated on that face
*in DG surface integral would be transpose(face_integral_basis) times flux_on_face
*/
template<int dim, int n_faces>
class face_integral_basis : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    face_integral_basis (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~face_integral_basis ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_surface_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<0> &face_quadrature);
};

/// The DG lifting operator is defined as the operator that lifts inner products of polynomials of some order \f$p\f$ onto the L2-space.
/**In DG lifting operator is \f$L=\mathbf{M}^{-1}*(\text{face_integral_basis})^T\f$.
*So DG surface is \f$L*\text{flux_interpolated_to_face}\f$.
*NOTE this doesn't have metric Jacobian dependence, for DG solver
*we build that using the functions below on the fly!
*/
template<int dim, int n_faces>
class lifting_operator : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    lifting_operator (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~lifting_operator ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Builds the local lifting operator. 
    void build_local_surface_lifting_operator (
            const unsigned int n_dofs, 
            const dealii::FullMatrix<double> &norm_matrix, 
            const dealii::FullMatrix<double> &face_integral,
            dealii::FullMatrix<double> &lifting);

    ///Assembles the one dimensional norm operator that it is lifted onto.
    /** Note that the norm is the DG mass matrix in this case. This has to be called before build_1D_surface_operator.
    */
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &face_quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_surface_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<0> &face_quadrature);
};

///The ESFR lifting operator. 
/**
*Consider the broken Sobolev-space \f$W_{\delta}^{dim*p,2}(\mathbf{\Omega}_r)\f$ (which is the ESFR norm)
* for \f$u \in W_{\delta}^{dim*p,2}(\mathbf{\Omega}_r)\f$,
* \f[
* L_{FR}:\: <L_{FR} u,v>_{\mathbf{\Omega}_r} = <u,v>_{\mathbf{\Gamma}_2}, \forall v\in P^p(\mathbf{\Omega}_r)
* \f].
*/
template<int dim, int n_faces>
class lifting_operator_FR : public lifting_operator<dim,n_faces>
{
public:
    ///Constructor.
    lifting_operator_FR (
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const Parameters::AllParameters::Flux_Reconstruction FR_param_input);

    ///Destructor.
    ~lifting_operator_FR ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Flux reconstruction parameter type.
    const Parameters::AllParameters::Flux_Reconstruction FR_param_type;

    ///Assembles the one dimensional norm operator that it is lifted onto.
    /** Note that the norm is the FR mass matrix in this case. This has to be called before build_1D_surface_operator.
    */
    void build_1D_volume_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &face_quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_surface_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<0> &face_quadrature);
};


/************************************************************************
*
*      METRIC MAPPING OPERATORS
*
************************************************************************/


///The mapping shape functions evaluated at the desired nodes (facet set included in volume grid nodes for consistency).
/**
* The finite element passed has to be the metric finite element. That is the one
* collocated on the mapping support points.
* By default, we use Gauss-Lobatto-Legendre as the mapping support points.
*/
template<int dim, int n_faces>
class mapping_shape_functions: public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    mapping_shape_functions(
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~mapping_shape_functions();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Stores the degree of the current grid degree.
    unsigned int current_grid_degree;

    ///Object of mapping shape functions evaluated at grid nodes.
    basis_functions<dim,n_faces> mapping_shape_functions_grid_nodes;

    ///Object of mapping shape functions evaluated at flux nodes.
    basis_functions<dim,n_faces> mapping_shape_functions_flux_nodes;

    ///Constructs the volume operator and gradient operator.
    /**
    * This function assures that the shape
    * functions are collocated on the grid nodes.
    * Also, makes for easier function calls.
    * Function builds the 1D operators in mapping_shape_functions_grid_nodes.
    * Note that at grid nodes, we do NOT need any surface information.
    */
    void build_1D_shape_functions_at_grid_nodes(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Constructs the volume, gradient, surface, and surface gradient operator.
    /**
    * Function builds the 1D operators in mapping_shape_functions_flux_nodes.
    */
    void build_1D_shape_functions_at_flux_nodes(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature,
            const dealii::Quadrature<0> &face_quadrature);

    ///Constructs the volume and volume gradient operator.
    /**
    * Often times, we only need the values at just the volume flux nodes.
    * Function builds the 1D operators in mapping_shape_functions_flux_nodes.
    */
    void build_1D_shape_functions_at_volume_flux_nodes(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

};

/*****************************************************************************
*
*       METRIC OPERTAORS TO BE CALLED ON-THE-FLY
*
*****************************************************************************/
///Base metric operators class that stores functions used in both the volume and on surface.
template <typename real, int dim, int n_faces>  
class metric_operators: public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    metric_operators(
        const int nstate_input,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const bool store_vol_flux_nodes_input = false,
        const bool store_surf_flux_nodes_input = false,
        const bool store_Jacobian_input = false);

    ///Destructor.
    ~metric_operators();

    ///Flag if store metric Jacobian at flux nodes.
    const bool store_Jacobian;

    ///Flag if store metric Jacobian at flux nodes.
    const bool store_vol_flux_nodes;

    ///Flag if store metric Jacobian at flux nodes.
    const bool store_surf_flux_nodes;

    ///Given a physical tensor, return the reference tensor.
    void transform_physical_to_reference(
        const dealii::Tensor<1,dim,real> &phys,
        const dealii::Tensor<2,dim,real> &metric_cofactor,
        dealii::Tensor<1,dim,real> &ref);

    ///Given a reference tensor, return the physical tensor.
    void transform_reference_to_physical(
        const dealii::Tensor<1,dim,real> &ref,
        const dealii::Tensor<2,dim,real> &metric_cofactor,
        dealii::Tensor<1,dim,real> &phys);

    ///Given a physical tensor of vector of points, return the reference tensor of vector.
    void transform_physical_to_reference_vector(
        const dealii::Tensor<1,dim,std::vector<real>> &phys,
        const dealii::Tensor<2,dim,std::vector<real>> &metric_cofactor,
        dealii::Tensor<1,dim,std::vector<real>> &ref);

    ///Given a reference tensor, return the physical tensor.
    void transform_reference_unit_normal_to_physical_unit_normal(
        const unsigned int n_quad_pts,
        const dealii::Tensor<1,dim,real> &ref,
        const dealii::Tensor<2,dim,std::vector<real>> &metric_cofactor,
        std::vector<dealii::Tensor<1,dim,real>> &phys);

    ///Builds just the determinant of the volume metric determinant.
    void build_determinant_volume_metric_Jacobian(
        const unsigned int n_quad_pts,//number volume quad pts
        const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
        const std::array<std::vector<real>,dim> &mapping_support_points,
        mapping_shape_functions<dim,n_faces> &mapping_basis);

    ///Builds the volume metric operators.
    /** Builds and stores volume metric cofactor and determinant of metric Jacobian
    * at the volume cubature nodes. If passed flag to store Jacobian when metric
    * operators is constructed, will also store the JAcobian at flux nodes.
    */
    void build_volume_metric_operators(
        const unsigned int n_quad_pts,//number volume quad pts
        const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
        const std::array<std::vector<real>,dim> &mapping_support_points,
        mapping_shape_functions<dim,n_faces> &mapping_basis,
        const bool use_invariant_curl_form = false);

    ///Builds the facet metric operators.
    /** Builds and stores facet metric cofactor and determinant of metric Jacobian
    * at the facet cubature nodes (one face). If passed flag to store Jacobian when metric
    * operators is constructed, will also store the JAcobian at flux nodes.
    */
    void build_facet_metric_operators(
        const unsigned int iface,
        const unsigned int n_quad_pts,//number facet quad pts
        const unsigned int n_metric_dofs,//dofs of metric basis. NOTE: this is the number of mapping support points
        const std::array<std::vector<real>,dim> &mapping_support_points,
        mapping_shape_functions<dim,n_faces> &mapping_basis,
        const bool use_invariant_curl_form = false);

    ///The volume metric cofactor matrix.
    dealii::Tensor<2,dim,std::vector<real>> metric_cofactor_vol;
    
    ///The facet metric cofactor matrix, for ONE face.
    dealii::Tensor<2,dim,std::vector<real>> metric_cofactor_surf;

    ///The determinant of the metric Jacobian at volume cubature nodes.
    std::vector<real> det_Jac_vol;

    ///The determinant of the metric Jacobian at facet cubature nodes.
    std::vector<real> det_Jac_surf;

    ///Stores the metric Jacobian at flux nodes.
    dealii::Tensor<2,dim,std::vector<real>> metric_Jacobian_vol_cubature;

    ///Stores the physical volume flux nodes.
    dealii::Tensor<1,dim,std::vector<real>> flux_nodes_vol;

    ///Stores the physical facet flux nodes.
    std::array<dealii::Tensor<1,dim,std::vector<real>>,n_faces> flux_nodes_surf;

protected:

    ///Builds the metric Jacobian evaluated at a vector of points.
    /** \f$ \mathbf{J} = [\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3]^T \f$
    * where \f$\mathbf{a}_i = \mathbf{\nabla}^r x_i \f$ are the physical vector bases.
    */
    void build_metric_Jacobian(
        const unsigned int n_quad_pts,//the dim sized n_quad_pts, NOT the 1D
        const std::array<std::vector<real>,dim> &mapping_support_points,
        const dealii::FullMatrix<double> &basis_x_flux_nodes,
        const dealii::FullMatrix<double> &basis_y_flux_nodes,
        const dealii::FullMatrix<double> &basis_z_flux_nodes,
        const dealii::FullMatrix<double> &grad_basis_x_flux_nodes,
        const dealii::FullMatrix<double> &grad_basis_y_flux_nodes,
        const dealii::FullMatrix<double> &grad_basis_z_flux_nodes,
        std::vector<dealii::Tensor<2,dim,real>> &local_Jac);

    ///Assembles the determinant of metric Jacobian.
    /** \f$ \|J^\Omega \| = \mathbf{a}_1 \cdot (\mathbf{a}_2 \otimes \mathbf{a}_3)\f$,
    * where \f$\mathbf{a}_i = \mathbf{\nabla}^r x_i \f$ are the physical vector bases.
    * Pass the 1D mapping shape functions evaluated at flux nodes,
    * and the 1D gradient of mapping shape functions evaluated at flux nodes.
    */
    void build_determinant_metric_Jacobian(
        const unsigned int n_quad_pts,//number volume quad pts
        const std::array<std::vector<real>,dim> &mapping_support_points,
        const dealii::FullMatrix<double> &basis_x_flux_nodes,
        const dealii::FullMatrix<double> &basis_y_flux_nodes,
        const dealii::FullMatrix<double> &basis_z_flux_nodes,
        const dealii::FullMatrix<double> &grad_basis_x_flux_nodes,
        const dealii::FullMatrix<double> &grad_basis_y_flux_nodes,
        const dealii::FullMatrix<double> &grad_basis_z_flux_nodes,
        std::vector<real> &det_metric_Jac);

    ///Called on the fly and returns the metric cofactor at cubature nodes.
    void build_local_metric_cofactor_matrix(
        const unsigned int n_quad_pts,//number volume quad pts
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
        const bool use_invariant_curl_form = false);

    ///Computes local 3D cofactor matrix.
    /**
    *We compute the metric cofactor matrix \f$\mathbf{C}_m\f$ via the conservative curl form of Abe 2014 and Kopriva 2006 by default. Can use invariant curl form by passing flag. To ensure consistent normals, we consider
    * the two cubature sets, grid nodes (mapping-support-points), and flux nodes (quadrature nodes). The metric cofactor matrix is thus:
    * \f[
    * (\mathbf{C})_{ni} = J(\mathbf{a}^i)_n= -\hat{\mathbf{e}}_i \cdot \nabla^r\times\mathbf{\Theta}(\mathbf{\xi}_{\text{flux nodes}}^r)\Big[
             \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_l^{c^T}
       \nabla^r \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_m^{c^T}
        \Big]
             \text{, }\\i=1,2,3\text{, }n=1,2,3\text{ }(n,m,l)\text{ cyclic,}
    * \f] for the conservative curl form, and
    * \f[
    *      (\mathbf{C})_{ni} = J(\mathbf{a}^i)_n= -\frac{1}{2}\hat{\mathbf{e}}_i \cdot \nabla^r\times\mathbf{\Theta}(\mathbf{\xi}_{\text{flux nodes}}^r)\Big[
             \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_l^{c^T}
       \nabla^r \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_m^{c^T}
        -
        \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_m^{c^T}
       \nabla^r \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_l^{c^T}\Big]
             \text{, }\\i=1,2,3\text{, }n=1,2,3\text{ }(n,m,l)\text{ cyclic,}
    * \f] for the invariant curl form.<br>
    * We let \f$\mathbf{\Theta}(\mathbf{\xi}^r)\f$ represent the mapping shape functions.
    */
    void compute_local_3D_cofactor(
        const unsigned int n_metric_dofs,
        const unsigned int n_quad_pts,
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
        const bool use_invariant_curl_form  = false);
};

/************************************************************
*
*       SUMFACTORIZED STATE
*
************************************************************/

///In order to have all state operators be arrays of array, we template by dim, type, nstate, and number of faces. 
/**Note that dofs and quad points aren't templated because they are variable with respect to each polynomial degree. 
*Also I couldn't template by polynomial degree/grid degree since they aren't compile time constant expressions.
*/
template <int dim, int nstate, int n_faces>  
class SumFactorizedOperatorsState : public SumFactorizedOperators<dim,n_faces>
{
public:
    ///Constructor.
    SumFactorizedOperatorsState (
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~SumFactorizedOperatorsState (); 

    ///Stores the one dimensional volume operator.
    std::array<dealii::FullMatrix<double>,nstate>  oneD_vol_state_operator;

    ///Stores the one dimensional surface operator.
    std::array<std::array<dealii::FullMatrix<double>,2>,nstate>  oneD_surf_state_operator;

    ///Stores the one dimensional gradient operator.
    std::array<dealii::FullMatrix<double>,nstate>  oneD_grad_state_operator;

    ///Stores the one dimensional surface gradient operator.
    std::array<std::array<dealii::FullMatrix<double>,2>,nstate>  oneD_surf_grad_state_operator;

};//end of OperatorsBaseState Class

///The basis functions separated by nstate with n shape functions.
template <int dim, int nstate, int n_faces>  
class basis_functions_state : public SumFactorizedOperatorsState<dim,nstate,n_faces>
{
public:
    ///Constructor.
    basis_functions_state (
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~basis_functions_state ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_volume_state_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_gradient_state_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_surface_state_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<0> &face_quadrature);
};

///The FLUX basis functions separated by nstate with n shape functions.
/** The flux basis are collocated on the flux nodes (volume cubature nodes).
* Thus, they are the order of the quadrature set, not by the state!
*/
template <int dim, int nstate, int n_faces>  
class flux_basis_functions_state : public SumFactorizedOperatorsState<dim,nstate,n_faces>
{
public:
    ///Constructor.
    flux_basis_functions_state (
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~flux_basis_functions_state ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    virtual void build_1D_volume_state_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_gradient_state_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<1> &quadrature);

    ///Assembles the one dimensional operator.
    void build_1D_surface_state_operator(
            const dealii::FESystem<1,1> &finite_element,
            const dealii::Quadrature<0> &face_quadrature);
};

///"Stiffness" operator used in DG Strong form.
/**Since the volume integral in strong form uses the flux basis spanning the flux.
*Explicitly, this is integral of basis_functions and the gradient of the flux basis
*\f[
     (\mathbf{S}_{\text{FLUX},\xi})_{ij}  = \int_\mathbf{{\Omega}_r} \mathbf{\chi}_i(\mathbf{\xi}^r) \frac{\mathbf{\chi}_{\text{FLUX},j}(\mathbf{\xi}^r)}{\partial \xi} d\mathbf{\Omega}_r
     \f]
*/
template <int dim, int nstate, int n_faces>  
class local_flux_basis_stiffness : public flux_basis_functions_state<dim,nstate,n_faces>
{
public:
    ///Constructor.
    local_flux_basis_stiffness (
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input);

    ///Destructor.
    ~local_flux_basis_stiffness ();

    ///Stores the degree of the current poly degree.
    unsigned int current_degree;

    ///Assembles the one dimensional operator.
    void build_1D_volume_state_operator(
            const dealii::FESystem<1,1> &finite_element,//pass the finite element of the TEST FUNCTION
            const dealii::Quadrature<1> &quadrature);
};


} /// OPERATOR namespace
} /// PHiLiP namespace

#endif

