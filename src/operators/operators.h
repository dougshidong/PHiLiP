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

template <int dim, typename real, int n_faces>
class OperatorsBase
{
public:
    ///Constructor
    OperatorsBase(
          const Parameters::AllParameters *const parameters_input,
          const int nstate_input,//number of states input
          const unsigned int degree,//degree not really needed at the moment
          const unsigned int max_degree_input,//max poly degree for operators
          const unsigned int grid_degree_input);//max grid degree for operators
    ///Destructor
    ~OperatorsBase();

    /// Input parameters.
    const Parameters::AllParameters *const all_parameters;
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

    /// Makes for cleaner doxygen documentation.
    using MassiveCollectionTuple = std::tuple<
        dealii::hp::FECollection<dim>, // Solution FE
        dealii::hp::QCollection<dim>,  // Volume quadrature
        dealii::hp::QCollection<dim-1>, // Face quadrature
        dealii::hp::QCollection<1>, // 1D quadrature for strong form
        dealii::hp::FECollection<dim> >;  // Lagrange polynomials for strong form
    ///Second constructor call
    /** Since a function is used to generate multiple different objects, a delegated
     *  constructor is used to unwrap the tuple and initialize the collections.
     *
     *  The tuple is built from create_collection_tuple(). */
    OperatorsBase( 
            const Parameters::AllParameters *const parameters_input,
            const int nstate_input,//number of states input
            const unsigned int degree,
            const unsigned int max_degree_input,
            const unsigned int grid_degree_input,
            const MassiveCollectionTuple collection_tuple);

    ///The collection tuple.
    MassiveCollectionTuple create_collection_tuple(const unsigned int max_degree, const int nstate, const Parameters::AllParameters *const parameters_input) const;

    ///The collections of FE (basis) and Quadrature sets.
    const dealii::hp::FECollection<dim>    fe_collection_basis;
    /// Quadrature used to evaluate volume integrals.
    dealii::hp::QCollection<dim>     volume_quadrature_collection;
    /// Quadrature used to evaluate face integrals.
    dealii::hp::QCollection<dim-1>   face_quadrature_collection;
    /// 1D quadrature to generate Lagrange polynomials for the sake of flux interpolation.
    dealii::hp::QCollection<1>       oned_quadrature_collection;
    ///Storing the "Flux" basis (basis collocated on flux volume nodes) previously referred to as the fe_collection_Lagrange.
    const dealii::hp::FECollection<dim>    fe_collection_flux_basis;

    ///Standard function to compute factorial of a number.
    double compute_factorial(double n);
protected:

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

public:

    /*
        * Operators Base to be used in DGBase needs: basis at vol, surf and metric basis at quad nodes.
        * Mass matrix, Flux_Reconstruction oper and vol det Jac which needs the metric basis grad at vol quad nodes.
        * Everything else will be in OperatorsBaseState
        * Also deirvtivep etc, which requires modal derivative oper, which reuiqres local_basis_stiffness
        * FR_mass_inv
        * basically everythign except flux lol
    */

    ///Allocates the volume operators.
    void allocate_volume_operators ();
    ///Allocates the surface operators.
    void allocate_surface_operators ();

     ///Solution basis functions evaluated at volume cubature nodes.
     std::vector<dealii::FullMatrix<real>> basis_at_vol_cubature;
     ///\f$ \mathbf{W}*\mathbf{\chi}(\mathbf{\xi}_v^r) \f$  That is Quadrature Weights multiplies with basis_at_vol_cubature.
     std::vector<dealii::FullMatrix<real>> vol_integral_basis;
     ///This is the solution basis \f$\mathbf{D}_i\f$, the modal differential opertaor commonly seen in DG defined as \f$\mathbf{D}_i=\mathbf{M}^{-1}*\mathbf{S}_i\f$.
    std::vector<std::array<dealii::FullMatrix<real>,dim>> modal_basis_differential_operator;
    ///Local mass matrix without jacobian dependence.
    std::vector<dealii::FullMatrix<real>> local_mass;
    ///Local stiffness matrix without jacobian dependence.
    /**NOTE: this is not used in DG volume integral since that needs to use the derivative of the flux basis and is multiplied by flux at volume cubature nodes this is strictly for consturtcing D operator
   *\f[
        (\mathbf{S}_\xi)_{ij}  = \int_\mathbf{{\Omega}_r} \mathbf{\chi}_i(\mathbf{\xi}^r) \frac{\mathbf{\chi}_{j}(\mathbf{\xi}^r)}{\partial \xi} d\mathbf{\Omega}_r
        \f]
    */
    std::vector<std::array<dealii::FullMatrix<real>,dim>> local_basis_stiffness;
    /// ESFR correction matrix without jac dependence
    std::vector<dealii::FullMatrix<real>> local_Flux_Reconstruction_operator;
    ///ESFR c parameter Flux_Reconstruction operator
    std::vector<real> c_param_FR;
   /// ESFR correction matrix for AUX EQUATION without jac dependence
   /** NOTE Auxiliary equation is a vector in dim, so theres an ESFR correction for each dim -> Flux_Reconstruction_aux also a vector of dim
   * ie/ local_Flux_Reconstruction_operator_aux[degree_index][dimension_index] = Flux_Reconstruction_operator for AUX eaquation in direction dimension_index for
   * polynomial degree of degree_index+1
   */
    std::vector<std::array<dealii::FullMatrix<real>,dim>> local_Flux_Reconstruction_operator_aux;
    ///ESFR k parameter Flux_Reconstruction-auxiliary operator
    std::vector<real> k_param_FR;
    ///\f$ p\f$ -th order modal derivative of basis fuctions, ie/\f$ [D_\xi^p, D_\eta^p, D_\zeta^p]\f$
    std::vector<std::array<dealii::FullMatrix<real>,dim>> derivative_p;
    /// \f$2p\f$-th order modal derivative of basis fuctions, ie/ \f$[D_\xi^p D_\eta^p, D_\xi^p D_\zeta^p, D_\eta^p D_\zeta^p]\f$
    std::vector<std::array<dealii::FullMatrix<real>,dim>> derivative_2p;
    /// \f$3p\f$-th order modal derivative of basis fuctions \f$[D_\xi^p D_\eta^p D_\zeta^p]\f$
    std::vector<dealii::FullMatrix<real>> derivative_3p;

    ///Projection operator corresponding to basis functions onto M-norm (L2).
    std::vector<dealii::FullMatrix<real>> vol_projection_operator;
    ///Projection operator corresponding to basis functions onto \f$(M+K)\f$-norm.
    std::vector<dealii::FullMatrix<real>> vol_projection_operator_FR;

    ///The metric independent inverse of the FR mass matrix \f$(M+K)^{-1}\f$.
    std::vector<dealii::FullMatrix<real>> FR_mass_inv;

    ///Builds basis and flux functions operators and gradient operator.
    void create_vol_basis_operators ();
    ///Constructs a mass matrix on the fly for a single degree NOTE: for Jacobian dependence pass JxW to quad_weights.
    void build_local_Mass_Matrix (
                                const std::vector<real> &quad_weights,
                                const unsigned int n_dofs_cell, const unsigned int n_quad_pts,
                                const int current_fe_index,
                                dealii::FullMatrix<real> &Mass_Matrix);
    ///Constructs local_mass which is a vector of metric Jacobian independent local mass matrices.
    void build_Mass_Matrix_operators ();
    ///Constructs local stiffness operator corresponding to the basis. 
    /**Also it constructs the flux basis stiffness as \f$\int_{\mathbf{\Omega}_r}\chi_i(\mathbf{\xi}^r)\frac{\partial \chi_{\text{FLUX},j}(\mathbf{\xi}^r)}{\partial \xi} d\mathbf{\Omega}_r \f$.
    *Also builds modal_basis_differential_operator since has the stiffness matrix there.
    */
    void build_Stiffness_Matrix_operators ();
    ///FR specific build operator functions
    /**builds the \f$ p,\: 2p,\f$ and \f$ 3p \f$ derivative operators to compute broken Sobolev-space.
    */    
    void get_higher_derivatives ();

    ///Evaluates Huynh's g2 parameter for flux reconstruction.
    /* This parameter recovers Huynh, Hung T. "A flux reconstruction approach to high-order schemes including discontinuous Galerkin methods." 18th AIAA computational fluid dynamics conference. 2007.
    */
    void get_Huynh_g2_parameter (
                                const unsigned int curr_cell_degree,
                                real &c);

    ///Evaluates the spectral difference parameter for flux reconstruction.
    /*Value from Allaneau, Y., and Antony Jameson. "Connections between the filtered discontinuous Galerkin method and the flux reconstruction approach to high order discretizations." Computer Methods in Applied Mechanics and Engineering 200.49-52 (2011): 3628-3636. 
    */
    void get_spectral_difference_parameter (
                                const unsigned int curr_cell_degree,
                                real &c);
    ///Evaluates the flux reconstruction parameter at the bottom limit where the scheme is unstable.
    /*Value from Allaneau, Y., and Antony Jameson. "Connections between the filtered discontinuous Galerkin method and the flux reconstruction approach to high order discretizations." Computer Methods in Applied Mechanics and Engineering 200.49-52 (2011): 3628-3636. 
    */
    void get_c_negative_FR_parameter (
                                const unsigned int curr_cell_degree,
                                real &c);
    ///Evaluates the flux reconstruction parameter at the bottom limit where the scheme is unstable, divided by 2.
    /*Value from Allaneau, Y., and Antony Jameson. "Connections between the filtered discontinuous Galerkin method and the flux reconstruction approach to high order discretizations." Computer Methods in Applied Mechanics and Engineering 200.49-52 (2011): 3628-3636. 
    * Commonly in the lterature we use this value to show the approach to the bottom limit of stability.
    */
    void get_c_negative_divided_by_two_FR_parameter (
                                const unsigned int curr_cell_degree,
                                real &c);
    ///Gets the FR correction parameter corresponding to the maximum timestep.
    /* Note that this parameter is also a good approximation for when the FR scheme begins to
    * lose an order of accuracy, but the original definition is that it corresponds to the maximum timestep.
    * Value from Table 3.4 in Castonguay, Patrice. High-order energy stable flux reconstruction schemes for fluid flow simulations on unstructured grids. Stanford University, 2012.
    */
    void get_c_plus_parameter (
                                const unsigned int curr_cell_degree,
                                real &c);

    ///Gets the FR correction parameter for both primary and auxiliary equations and stores for each degree.
    /**These values are name specified in parameters/all_parameters.h, passed through control file/or test and here converts/stores as value.
    * Please note that in all the functions within this that evaluate the parameter, we divide the value in the literature by 2.0
    * because our basis are contructed by an orthonormal Legendre basis rather than the orthogonal basis in the literature. 
    * Also, we have the additional scaling by pow(pow(2.0,curr_cell_degree),2) because our basis functions are defined on
    * a reference element between [0,1], whereas the values in the literature are based on [-1,1].
    * For further details please refer to Cicchino, Alexander, and Siva Nadarajah. "A new norm and stability condition for tensor product flux reconstruction schemes." Journal of Computational Physics 429 (2021): 110025.
    */
    void get_FR_correction_parameter (
                                    const unsigned int curr_cell_degree,
                                    real &c, real &k);
    ///Constructs the vector of Flux_Reconstruction operators (ESFR correction operator) for each poly degree.
    void build_Flux_Reconstruction_operators ();

   ///Computes a single local Flux_Reconstruction operator (ESFR correction operator) on the fly for a local element.
   /**Note that this is dependent on the Mass Matrix, so for metric Jacobian dependent \f$K_m\f$,
   *pass the metric Jacobian dependent Mass Matrix \f$M_m\f$.
    */
    void build_local_Flux_Reconstruction_operator(
                                const dealii::FullMatrix<real> &local_Mass_Matrix,
                                const unsigned int  n_dofs_cell, const unsigned int degree_index, 
                                dealii::FullMatrix<real> &Flux_Reconstruction_operator);
    ///Similar to above but for the local Flux_Reconstruction operator for the Auxiliary equation.
    void build_local_Flux_Reconstruction_operator_AUX(
                                const dealii::FullMatrix<real> &local_Mass_Matrix,
                                const unsigned int  n_dofs_cell, const unsigned int degree_index, 
                                std::array<dealii::FullMatrix<real>,dim> &Flux_Reconstruction_operator_aux);

    ///Computes the volume projection operators.
    void get_vol_projection_operators();
    ///Computes a single local projection operator on some space (norm).
    void compute_local_vol_projection_operator(
                                const unsigned int degree_index, 
                                const unsigned int n_dofs_cell, 
                                const dealii::FullMatrix<real> &norm_matrix, 
                                dealii::FullMatrix<real> &volume_projection);

    ///Solution basis functions evaluated at facet cubature nodes.
    std::vector<std::array<dealii::FullMatrix<real>,n_faces>> basis_at_facet_cubature;
//    ///Flux basis functions evaluated at facet cubature nodes.
//   std::vector<std::array<std::array<dealii::FullMatrix<real>,n_faces>,nstate>> flux_basis_at_facet_cubature;
    ///The surface integral of test functions.
   /**\f[
   *     \mathbf{W}_f \mathbf{\chi}(\mathbf{\xi}_f^r) 
   * \f]
   *ie/ diag of REFERENCE unit normal times facet quadrature weights times solution basis functions evaluated on that face
   *in DG surface integral would be transpose(face_integral_basis) times flux_on_face
   */
    std::vector<std::array<dealii::FullMatrix<real>,n_faces>> face_integral_basis;
   /// The DG lifting operator is defined as the operator that lifts inner products of polynomials of some order \f$p\f$ onto the L2-space.
   /**In DG lifting operator is \f$L=\mathbf{M}^{-1}*(\text{face_integral_basis})^T\f$.
   *So DG surface is \f$L*\text{flux_interpolated_to_face}\f$.
   *NOTE this doesn't have metric Jacobian dependence, for DG solver
   *we build that using the functions below on the fly!
   */
    std::vector<std::array<dealii::FullMatrix<real>,n_faces>> lifting_operator;
   ///The ESFR lifting operator. 
   /**
   *Consider the broken Sobolev-space \f$W_{\delta}^{dim*p,2}(\mathbf{\Omega}_r)\f$ (which is the ESFR norm)
   * for \f$u \in W_{\delta}^{dim*p,2}(\mathbf{\Omega}_r)\f$,
   * \f[
   * L_{FR}:\: <L_{FR} u,v>_{\mathbf{\Omega}_r} = <u,v>_{\mathbf{\Gamma}_2}, \forall v\in P^p(\mathbf{\Omega}_r)
   * \f].
   */
    std::vector<std::array<dealii::FullMatrix<real>,n_faces>> lifting_operator_FR;


    ///Builds surface basis and flux functions operators and gradient operator.
    void create_surface_basis_operators ();
    ///Builds surface lifting operators.
    void get_surface_lifting_operators ();
    ///Builds the local lifting operator. 
    void build_local_surface_lifting_operator (
                                const unsigned int degree_index, 
                                const unsigned int n_dofs_cell, 
                                const unsigned int face_number, 
                                const dealii::FullMatrix<real> &norm_matrix, 
                                dealii::FullMatrix<real> &lifting);
    
    ///The mapping shape functions evaluated at the volume grid nodes (facet set included in volume grid nodes for consistency).
    std::vector<dealii::FullMatrix<real>> mapping_shape_functions_grid_nodes; 
    ///REFERENCE gradient of the the mapping shape functions evaluated at the volume grid nodes.
    std::vector<std::array<dealii::FullMatrix<real>,dim>> gradient_mapping_shape_functions_grid_nodes; 

    ///Mapping shape functions evaluated at the VOLUME flux nodes (arbitrary, does not have to be on the surface ex/ GL).
    /** FOR Flux Nodes operators there is the grid degree, then the degree of the cubature set it is applied on
    * to handle all general cases. 
    */
    std::vector<std::vector<dealii::FullMatrix<real>>> mapping_shape_functions_vol_flux_nodes; 
    ///Mapping shape functions evaluated at the SURFACE flux nodes. 
    std::vector<std::vector<std::array<dealii::FullMatrix<real>,n_faces>>> mapping_shape_functions_face_flux_nodes; 
   ///Gradient of mapping shape functions evalutated at VOLUME flux nodes.
   /**Note that for a single grid degree it can evaluate at varying degree of fluxnodes 
   *ie allows of sub/super parametric etc and over integration
   *Vector order goes: [Grid_Degree][Flux_Poly_Degree][Dim][n_quad_pts][n_mapping_shape_functions]
    */
    std::vector<std::vector<std::array<dealii::FullMatrix<real>,dim>>> gradient_mapping_shape_functions_vol_flux_nodes; 
    ///Gradient of mapping shape functions evalutated at surface flux nodes.
    /**Is a vector of degree->array of n_faces -> array of dim -> Matrix n_face_quad_pts x n_shape_functions.
    */
    std::vector<std::vector<std::array<std::array<dealii::FullMatrix<real>,dim>,n_faces>>> gradient_mapping_shape_functions_face_flux_nodes; 

    ///Allocates metric shape functions operators.
    void allocate_metric_operators(     
                                    const unsigned int max_grid_degree_local);
    ///Creates metric shape functions operators.
    void create_metric_basis_operators(
                                    const unsigned int max_grid_degree_local);

    ///Builds just the dterminant of the volume metric Jacobian.
    void build_local_vol_determinant_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts, 
                                    const unsigned int n_metric_dofs,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<real> &determinant_Jacobian);
    ///Called on the fly and returns the metric cofactor and determinant of Jacobian at VOLUME cubature nodes.
    /**
    *We compute the metric cofactor matrix \f$\mathbf{C}_m\f$ via the invariant curl form of Abe 2014 and Kopriva 2006. To ensure consistent normals, we consider
    * the two cubature sets, grid nodes (mapping-support-points), and flux nodes (quadrature nodes). The metric cofactor matrix is thus:
    * \f[
    *      (\mathbf{C})_{ni} = J(\mathbf{a}^i)_n= -\frac{1}{2}\hat{\mathbf{e}}_i \cdot \nabla^r\times\mathbf{\Theta}(\mathbf{\xi}_{\text{flux nodes}}^r)\Big[
             \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_l^{c^T}
       \nabla^r \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_m^{c^T}
        -
        \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_m^{c^T}
       \nabla^r \mathbf{\Theta}(\mathbf{\xi}_{\text{grid nodes}}^r)\hat{\mathbf{x}}_l^{c^T}\Big]
             \text{, }\\i=1,2,3\text{, }n=1,2,3\text{ }(n,m,l)\text{ cyclic.}
    * \f]
    * where we let \f$\mathbf{\Theta}(\mathbf{\xi}^r)\f$ represent the mapping shape functions.
    */
    void build_local_vol_metric_cofactor_matrix_and_det_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts, 
                                    const unsigned int n_metric_dofs,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<real> &determinant_Jacobian,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
    ///Called on the fly and returns the metric cofactor and determinant of Jacobian at face cubature nodes.
    void build_local_face_metric_cofactor_matrix_and_det_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int iface,
                                    const unsigned int n_quad_pts, const unsigned int n_metric_dofs,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<real> &determinant_Jacobian,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
    ///Computes local 3D cofactor matrix in VOLUME.
    void compute_local_3D_cofactor_vol(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts,
                                    const unsigned int n_metric_dofs,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
    ///Computes local 3D cofactor matrix on FACE for consistent normals with water-tight mesh.
    void compute_local_3D_cofactor_face(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts,
                                    const unsigned int n_metric_dofs, const unsigned int iface,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
    ///Computes \f$(\mathbf{x}_l * \nabla(\mathbf{x}_m))\f$ at GRID NODES.
    void compute_Xl_grad_Xm(
                                    const unsigned int grid_degree,
                                    const unsigned int n_metric_dofs, 
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<dealii::DerivativeForm<1,dim,dim>> &Xl_grad_Xm);
    ///Computes the cyclic curl loop for metric cofactor matrix.
    /**
    *      This function is currently no longer used, but left in with the conservative curl form commented out
    *      incase we want to compare the forms in the future.
    */
    void do_curl_loop_metric_cofactor(
                                    const unsigned int n_quad_pts,
                                    const std::vector<dealii::DerivativeForm<2,dim,dim>> grad_Xl_grad_Xm,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);

    ///Given a physical tensor, return the reference tensor.
    void compute_physical_to_reference(
                                    const dealii::Tensor<1,dim,real> &phys,
                                    const dealii::FullMatrix<real> &metric_cofactor,
                                    dealii::Tensor<1,dim,real> &ref);
    ///Given a reference tensor, return the physical tensor.
    void compute_reference_to_physical(
                                    const dealii::Tensor<1,dim,real> &ref,
                                    const dealii::FullMatrix<real> &metric_cofactor,
                                    dealii::Tensor<1,dim,real> &phys);
    ///Checks on the fly that the grid hasn't been updated with a higher order. If the grid has been updated, then it recreates the appropriate metric basis.
    void is_the_grid_higher_order_than_initialized(
                                    const unsigned int grid_degree);
                                
    ///Computes a matrix-vector product using sum-factorization. Pass the one-dimensional basis, where x runs the fastest, then y, and z runs the slowest. Also, assume each one-dimensional basis is the same size.
    /** Uses sum-factorization with BLAS techniques to solve the the matrix-vector multiplication, where the matrix is the tensor product of three one-dimensional matrices. We use the standard notation that x runs the fastest, then y, and z runs the slowest.
    */
    void sum_factorization_matrix_vector_mult(
                                    const std::vector<real> &input_vect,
                                    std::vector<real> &output_vect,
                                    const unsigned int rows, const unsigned int columns,
                                    const unsigned int dimension,
                                    const dealii::FullMatrix<real> &basis_x,
                                    const dealii::FullMatrix<real> &basis_y,
                                    const dealii::FullMatrix<real> &basis_z);

    ///Computes the inner prodct between a matrix and a vector multiplied by some weight function.  
    /** That is, we compute \f$ \int Awu d\mathbf{\Omega}_r = \mathbf{A}^T \text{diag}(w) \mathbf{u}^T \f$. When using this function, pass \f$ \mathbf{A} \f$ and NOT it's transpose--the function transposes it in the first few lines.
    */
    void sum_factorization_inner_product(
                                    const std::vector<real> &input_vect,
                                    const std::vector<real> &weight_vect,
                                    std::vector<real> &output_vect,
                                    const unsigned int rows, const unsigned int columns,
                                    const unsigned int dimension,
                                    const dealii::FullMatrix<real> &basis_x,
                                    const dealii::FullMatrix<real> &basis_y,
                                    const dealii::FullMatrix<real> &basis_z);
};///End operator base class.


///In order to have all operators be arrays of array, we template by dim, type, nstate, and number of faces. Note that dofs and quad points aren't templated because they are variable with respect to each polynomial degree. Also I couldn't template by polynomial degree/grid degree since they aren't compile time constant expressions.
template <int dim, typename real, int nstate, int n_faces>  
class OperatorsBaseState : public OperatorsBase<dim, real, n_faces>
{
public:
    ///Constructor
    OperatorsBaseState(
          const Parameters::AllParameters *const parameters_input,//Parameters.
          const unsigned int max_degree_input,//max poly degree for operators
          const unsigned int grid_degree_input);//max grid degree for operators
    ///Destructor
    ~OperatorsBaseState();

public:

    ///Allocates the volume operators for state class.
    void allocate_volume_operators_state ();
    ///Builds basis and flux functions operators and gradient operator for state class.
    void create_vol_basis_operators_state ();
     ///The flux basis functions evaluated at volume cubature nodes.
    /**Flux (over int) basis functions evaluated at volume cubature nodes (should always be identity).
    *NOTE THE FLUX BASIS IS COLLOCATED ON FLUX NODES
    *so it does not have vector state embedded in its degrees of freedom, so we make the operator have a vector by state, ie/ 
    *\f$\text{n_dofs_for_solution_basis} = \text{nstate} *pow(p+1,dim)\f$
    *but \f$\text{n_dofs_flux_basis} = pow(p+1,dim)\f$.
    *Also flux basis has an extra vector by nstate so that we can use .vmult later on with state vectors (in the residuals)
     *So example flux_basis_at_vol_cubature[poly_degree][state_number][test_functions_for_state_number][flux_basis_shape_functions]
        */
     std::vector<std::array<dealii::FullMatrix<real>,nstate>> flux_basis_at_vol_cubature;
     ///Gradient of flux basis functions evaluated at volume cubature nodes.
     /**Note that since it is gradient and not derivative, it is a tensor of dim.
     */
     std::vector<std::array<std::array<dealii::FullMatrix<real>,dim>,nstate>> gradient_flux_basis;
    ///"Stiffness" operator used in DG Strong form.
   /**Since the volume integral in strong form uses the flux basis spanning the flux.
   *Explicitly, this is integral of basis_functions and the gradient of the flux basis
   *\f[
        (\mathbf{S}_{\text{FLUX},\xi})_{ij}  = \int_\mathbf{{\Omega}_r} \mathbf{\chi}_i(\mathbf{\xi}^r) \frac{\mathbf{\chi}_{\text{FLUX},j}(\mathbf{\xi}^r)}{\partial \xi} d\mathbf{\Omega}_r
        \f]
    */
    std::vector<std::array<std::array<dealii::FullMatrix<real>,dim>,nstate>> local_flux_basis_stiffness;
    ///The integration of gradient of solution basis.
   /**Please note that for the weak form volume integral with arbitrary integration strength, use the transpose of the following operator.
   *Note: that it is also a vector of nstate since it will be applied to a flux vector per state.
   *Lastly its gradient of basis functions NOT flux basis because in the weak form the test function is the basis function not the flux basis (technically the flux is spanned by the flux basis at quadrature nodes).
    *   \f[
    *           \mathbf{W}\nabla\Big(\chi_i(\mathbf{\xi}^r)\Big)  
    *   \f]
    */
    std::vector<std::array<std::array<dealii::FullMatrix<real>,dim>,nstate>> vol_integral_gradient_basis;
    ///Allocates the surface operators for the state class.
    void allocate_surface_operators_state ();
    ///Builds surface basis and flux functions operators and gradient operator for the state class.
    void create_surface_basis_operators_state ();
    ///Flux basis functions evaluated at facet cubature nodes.
    std::vector<std::array<std::array<dealii::FullMatrix<real>,n_faces>,nstate>> flux_basis_at_facet_cubature;
    ///Computes the physical gradient operator scaled by the determinant of the metric Jacobian. 
    /**By default we should use the skew-symmetric form for curvilinear elements. For the metric split-form, pass "use_curvilinear_split_form" in parameters. Explicitly, the skew-symmetric form's first component is the gradient operator and the second is the divergence operator. 
    * Using \f$\mathbf{\phi}\f$ as the flux basis collocated on the volume cubature nodes, the output of the function is \f$ D_i = \frac{1}{2} \sum_{j=1}^{d}(J\frac{\partial \xi_j}{\partial x_i})\frac{\partial \phi(\mathbf{\xi}_v^r)}{\partial \xi_j}  + \frac{\partial \phi(\mathbf{\xi}_v^r)}{\partial \xi_j}(J\frac{\partial \xi_j}{\partial x_i})\f$. 
    * Since this operator can also compute either the divergence or the gradoent, pass the option of whether to compute the conservative divergence operator or the gradient operator. For definitions of gradient and divergence operators, please see Equations (10) and (11) in Cicchino, Alexander, et al. "Provably Stable Flux Reconstruction High-Order Methods on Curvilinear Elements." arXiv preprint arXiv:2109.11617 (2021).
    */
    void get_Jacobian_scaled_physical_gradient(
                                    const bool use_conservative_divergence,
                                    const std::array<std::array<dealii::FullMatrix<real>,dim>,nstate> &ref_gradient,
                                    const std::vector<dealii::FullMatrix<real>> &metric_cofactor,
                                    const unsigned int n_quad_pts,
                                    std::array<std::array<dealii::FullMatrix<real>,dim>,nstate> &physical_gradient);

};///End operator base state class.

} /// OPERATOR namespace
} /// PHiLiP namespace

#endif

