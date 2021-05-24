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


#include "mesh/high_order_grid.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"

namespace PHiLiP {
namespace OPERATOR {

template <int dim, int nstate, typename real>
class OperatorBase
{
public:
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::Triangulation<dim>;
#else
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
public:


    OperatorBase(//std::shared_ptr< DGBase<dim, real> > dg_input);//,
          const Parameters::AllParameters *const parameters_input,
          const unsigned int degree,
          const unsigned int max_degree_input,
          const unsigned int grid_degree_input,
          const std::shared_ptr<Triangulation> triangulation_input);
    //Destructor
    ~OperatorBase();


    /// Input parameters.
    const Parameters::AllParameters *const all_parameters;
    const unsigned int max_degree;
    const unsigned int max_grid_degree;


    /// Makes for cleaner doxygen documentation
    using MassiveCollectionTuple = std::tuple<
        dealii::hp::FECollection<dim>, // Solution FE
        dealii::hp::QCollection<dim>,  // Volume quadrature
        dealii::hp::QCollection<dim-1>, // Face quadrature
        dealii::hp::QCollection<1>, // 1D quadrature for strong form
        dealii::hp::FECollection<dim> >;  // Lagrange polynomials for strong form
    /** Since a function is used to generate multiple different objects, a delegated
     *  constructor is used to unwrap the tuple and initialize the collections.
     *
     *  The tuple is built from create_collection_tuple(). */
    OperatorBase( 
            const Parameters::AllParameters *const parameters_input,
            const unsigned int degree,
            const unsigned int max_degree_input,
            const unsigned int grid_degree_input,
            const std::shared_ptr<Triangulation> triangulation_input,
            const MassiveCollectionTuple collection_tuple);

    MassiveCollectionTuple create_collection_tuple(const unsigned int max_degree, const Parameters::AllParameters *const parameters_input) const;

    std::shared_ptr<Triangulation> triangulation; ///< Mesh
//    /// Sets the associated high order grid with the provided one.
//    void set_high_order_grid(std::shared_ptr<HighOrderGrid<dim,real>> new_high_order_grid);

    //The collections of FE (basis) and Quadrature sets
    const dealii::hp::FECollection<dim>    fe_collection_basis;
    /// Quadrature used to evaluate volume integrals.
    dealii::hp::QCollection<dim>     volume_quadrature_collection;
    /// Quadrature used to evaluate face integrals.
    dealii::hp::QCollection<dim-1>   face_quadrature_collection;
    /// 1D quadrature to generate Lagrange polynomials for the sake of flux interpolation.
    dealii::hp::QCollection<1>       oned_quadrature_collection;
    ///Storing the "Flux" basis (basis collocated on flux volume nodes) previously referred to as the fe_collection_Lagrange
    const dealii::hp::FECollection<dim>    fe_collection_flux_basis;

    //allocates the volume operators
    void allocate_volume_operators ();
    //allocates the surface operators
    void allocate_surface_operators ();

    //standard function to compute factorial of a number
    double compute_factorial(double n);
protected:

    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,real>> dg;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

public:

    //OPERATORS

    /*************************************************************************
 *             Operators will follow the format listed here:
 *      Each local operator is a dealii::FullMatrix<real> (which is just a matrix)
 *      then it is a std::vector for every single polynomial degree upto the max degree in the solver
 *      for operators that have a dimenion component, they are std::vector in dim
 *      EXAMPLE:
 *      std::vector<std::vector<dealii::FullMatrix<real>>> gradient_flux_basis;
 *      This is the gradient of the flux basis,
 *      gradient_flux_basis = [  [ d flux_basis / d \xi], [ d flux_basis / d \eta], [ d flux_basis / d \zeta] ;//p=1
 *                               [ d flux_basis / d \xi], [ d flux_basis / d \eta], [ d flux_basis / d \zeta] ;//p=2
 *                               ... 
 *                               [ d flux_basis / d \xi], [ d flux_basis / d \eta], [ d flux_basis / d \zeta] ]//p=max_degree
 *                               
 *      where [d flux_basis / d \xi] is a matrix storing the derivative of all flux_basis functions evaluated at volume cubature nodes 
 *
 *
 * **************************************************************************/    

    /********************************************
 *
 *      VOLUME OPERATORS
 *
 *      *********************************************/

    //solution basis functions evaluated at volume cubature nodes
    std::vector<dealii::FullMatrix<real>> basis_at_vol_cubature;
    //W*Chi  ie/ Quadrature Weights multiplies with basis_at_vol_cubature
    std::vector<dealii::FullMatrix<real>> vol_integral_basis;
    //flux (over int) basis functions evaluated at volume cubature nodes (should always be identity)
    //NOTE THE FLUX BASIS IS COLLOCATED ON FLUX NODES
    //so it does not have vector state, ie/ 
    //n_dofs_for_solution_basis = nstate *pow(p+1,dim)
    //but n_dofs_flux_basis = pow(p+1,dim)
    //Also flux basis has an extra vector by nstate so that we can use .vmult later on with state vectors (in the residuals)
    //So example flux_basis_at_vol_cubature[poly_degree][state_number][test_functions_for_state_number][flux_basis_shape_functions]
    std::vector<std::vector<dealii::FullMatrix<real>>> flux_basis_at_vol_cubature;
    //gradient of flux basis functions evaluated at volume cubature nodes
    //Note that since it is gradient and not derivative, it is a tensor of dim
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> gradient_flux_basis;
    //This is the solution basis D_i, the modal differential opertaor commonly seen in DG defined as D_i = M^{-1}*S_i
    std::vector<std::vector<dealii::FullMatrix<real>>> modal_basis_differential_operator;
    //local mass matrix without jacobian dependence
    std::vector<dealii::FullMatrix<real>> local_mass;
    //local stiffness matrix without jacobian dependence,
    //NOTE: this is not used in DG volume integral since that needs to use the derivative of the flux basis and is multiplied by flux at volume cubature nodes this is strictly for consturtcing D operator
    std::vector<std::vector<dealii::FullMatrix<real>>> local_basis_stiffness;
    //Since the vol integral uses the flux basis this is the operator used in DG strong volume integral
    //That is this is integral of basis_functions and the gradient of the flux basis
    //(local_flux_basis_stiffness[poly_degree])_{ij}  = \int_{\Omega_r} basis_functions_[i] \nabla(flux_basis_functions_j) d\Omega_r
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> local_flux_basis_stiffness;
    //for weak form volume integral with arbitrary integration strength, use the transpose of the following operator
    //think of its transpose as the transpose of the above operator
    //Note that it is also a vector of nstate since it will be applied to a flux vector per state
    //lastly its gradient of basis NOT flux basis because in weak form the test function is the basis function not the flux basis (technically the flux is spanned by the flux basis at quadrature nodes)
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> vol_integral_gradient_basis;
    //ESFR OPERATORS
    // ESFR correction matrix without jac dependence
    std::vector<dealii::FullMatrix<real>> local_K_operator;
    std::vector<real> c_param_FR;
    // ESFR correction matrix for AUX EQUATION without jac dependence
    // NOTE Auxiliary equation is a vector in dim, so theres an ESFR correction for each dim -> K_aux also a vector of dim
    // ie/ local_K_operator_aux[degree_index][dimension_index] = K_operator for AUX eaquation in direction dimension_index for
    // polynomial degree of degree_index+1
    std::vector<std::vector<dealii::FullMatrix<real>>> local_K_operator_aux;
    std::vector<real> k_param_FR;
    // pth order modal derivative of basis fuctions, ie/ [D_\xi^p, D_\eta^p, D_\zeta^p]
    std::vector<std::vector<dealii::FullMatrix<real>>> derivative_p;
    // 2pth order modal derivative of basis fuctions, ie/ [D_\xi^p*D_\eta^p, D_\xi^p*D_\zeta^p, D_\eta^p*D_\zeta^p]
    std::vector<std::vector<dealii::FullMatrix<real>>> derivative_2p;
    // 3pth order modal derivative of basis fuctions [D_\xi^p*D_\eta^p*D_\zeta^p]
    std::vector<dealii::FullMatrix<real>> derivative_3p;

    //projection operator corresponding to basis functions onto M-norm (L2)
    std::vector<dealii::FullMatrix<real>> vol_projection_operator;
    //projection operator corresponding to basis functions onto (M+K)-norm
    std::vector<dealii::FullMatrix<real>> vol_projection_operator_FR;

    /********************************************
 *
 *      VOLUME OPERATOR FUNCTIONS
 *
 *      *********************************************/

    //builds basis/flux functions operators and gradient operator
    void create_vol_basis_operators ();
    //constructs a mass matrix on the fly for a single degree
    //NOTE: for Jacobian dependence pass JxW to quad_weights
    void build_local_Mass_Matrix (
                                const std::vector<real> &quad_weights,
                                const unsigned int n_dofs_cell, const unsigned int n_quad_pts,
                                const int current_fe_index,
                                dealii::FullMatrix<real> &Mass_Matrix);
    //constructs local_mass which is a vector of metric Jacobian independent local mass matrices
    void build_Mass_Matrix_operators ();
    //constructs local stiffness operator corresponding to the basis, 
    //also the flux basis stiffness which inetrgates the gradient of the flux basis with the solution basis functions as the test function (the DG strong over-integrated or flux projected however you want to see it volume integral)
    //also builds modal_basis_differential_operator since has the stiffness matrix there
    void build_Stiffness_Matrix_operators ();
    ///FR specific build operator functions
    //builds the p, 2p, and 3p derivative operators to compute broken Sobolev-space
    void get_higher_derivatives ();
    //gets the FR correction parameter for both primary and auxiliary equations and stores for each degree
    //these values are name specified in parameters/all_parameters.h, passed through control file/or test and here converts/stores as value
    void get_FR_correction_parameter (
                                    const unsigned int curr_cell_degree,
                                    real &c, real &k);
    //constructs the vector of K operators (ESFR correction operator) for each poly degree
    void build_K_operators ();
    //Computes a single local K operator (ESFR correction operator) on the fly for a local element
    //Note that this is dependent on the Mass MAtrix, so for metric Jacobian dependent K_m,
    //pass the metric Jacobian dependent Mass Matrix M_m
    void build_local_K_operator(
                                const dealii::FullMatrix<real> &local_Mass_Matrix,
                                const unsigned int  n_dofs_cell, const unsigned int degree_index, 
                                dealii::FullMatrix<real> &K_operator);
    //Similar to above but for the local K operator for the Auxiliary equation
    void build_local_K_operator_AUX(
                                const dealii::FullMatrix<real> &local_Mass_Matrix,
                                const unsigned int  n_dofs_cell, const unsigned int degree_index, 
                                std::vector<dealii::FullMatrix<real>> &K_operator_aux);

    //computes the volume projection operators
    void get_vol_projection_operators();
    //computes a single local projection operator on some space (norm)
    //defined as the mass matrix
    void compute_local_vol_projection_operator(
                                const unsigned int degree_index, 
                                const unsigned int n_dofs_cell, 
                                const dealii::FullMatrix<real> &norm_matrix, 
                                dealii::FullMatrix<real> &volume_projection);

    /********************************************************
 *
 *              SURFACE OPERATORS
 *
 *      Since surface operators depend on which face they are applied on,
 *      and throughout the code the degree_index and face_index (iface)
 *      is usually knwon, the basic structure is vector of size
 *      max_degree storing for each degree, then vector of size
 *      n_faces = 2.0*dim (2 faces in 1D, 4 faces in 2D, 6 faces in 3D)
 *      which stores the corresponding local matrix
 *              *******************************************/

    //solution basis functions evaluated at facet cubature nodes
    std::vector<std::vector<dealii::FullMatrix<real>>> basis_at_facet_cubature;
    //flux basis functions evaluated at facet cubature nodes
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> flux_basis_at_facet_cubature;
    //diag(\hat{n}^r)*W_f*Chi_f
    //ie/ diag of REFERENCE unit normal times facet quadrature weights times solution basis functions evaluated on that face
    //in DG surface integral would be (face_integral_basis)^T * flux_on_face
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> face_integral_basis;
    // the DG lifting operator is defined as the operator that lifts
    // inner products of polynomials of some order p onto
    // the L2-space
    //in DG lifting operator is L=M^{-1}*(face_integral_basis)^T
    //so DG surface is L*f_face
    //NOTE this doesn't have metric Jacobian dependence, for DG solver
    //we build that using the functions below on the fly!
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> lifting_operator;
    //the ESFR lifting opertaor, below is a proper definition for the
    //broken Sobolev-space W_{\delta}^{dim*p,2}(\bm{\Omega}_r which is the ESFR norm)
    // for u \in W_{\delta}^{dim*p,2}(\bm{\Omega}_r),
    // L_FR: <L_FR *u,v>_\bm{\Omega}_r = <u,v>_{\bm{\Gamma}_2}, for v\in P^p(\bm{\Omega}_r)
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> lifting_operator_FR;


    /**************************************************
 *
 *
 *              SURFACE OPERATOR FUNCTIONS
 *
 *              **********************************************/

    //builds surface basis/flux functions operators and gradient operator
    void create_surface_basis_operators ();
    void get_surface_lifting_operators ();
    void build_local_surface_lifting_operator (
                                const unsigned int degree_index, 
                                const unsigned int n_dofs_cell, 
                                const unsigned int face_number, 
                                const dealii::FullMatrix<real> &norm_matrix, 
                                std::vector<dealii::FullMatrix<real>> &lifting);
    
    /**************************************************
 *
 *
 *              METRIC DEPENDENT OPERATORS
 *
 *              **********************************************/

    //Since we compute the metric cofactor matrix/determinant of the Jacobian
    //on the fly, it is advantageous to store the mapping shape functions
    //The mapping shape functions are basis that are collocated on the mapping
    //support points (also referred to as grid-nodes) and are used to evaluated
    //the metric Jacobian at the flux nodes (volume/facet cubature nodes).
    //For consistent normals, it is imperative that the grid nodes are included
    //on the boundary, hence making the polynomial grid "continuous" finite-elements
    //in a sense. dealii does this by using Gauss-Legendre-Lobatto quadrature points
    //as the grid nodes in the reference element. Thus, the "physical" coordinates of
    //the grid nodes are derived by the manifold of some reference GLL nodal set.
    //The metric cofactor matrix is formulated similar to the paper of Abe et al. 
    // using the invariant-conservative curl form of Kopriva 2006.
    //EXPLICITLY: Let \\bm{\Theta}(\bm{\xi}^r) represent the mapping shape functions
    //Then:  for conservative curl 
    //J(a^i)_n = -\hat{\bm{e}}_i \cdot \nabla^r\times\bm{\Theta}(\bm{\xi}_{{flux nodes}}^r) *
//     \Big[ \bm{\Theta}(\bm{\xi}_{{grid nodes}}^r)\hat{\bm{x}}_l^{c^T}
//                      \nabla^r \bm{\Theta}(\bm{\xi}_{{grid nodes}}^r)\hat{\bm{x}}_m^{c^T} \Big]
//              i=1,2,3{, }n=1,2,3{ }(n,m,l){ cyclic,}
//
//             SO  metric Cofactor matrix transpose corresponds to Jacobian inverse
//             implies (metric_cofactor)_{j,i} = (metric_cofactor^T)_{i,j}= |J| * ( d \xi_i / d x_j ), for i,j = 1,2,3
//
//              LASTLY the operators act on mapping support points of size [dim][n_shape_functions]
//              This is not the same as the "volume_nodes" in the high_order_grid class, 
//              instead use fe_metric[metric_degree].system_to_component_index.first() for dim component, 
//              fe_metric[metric_degree].system_to_component_index.second() for shape function component
//              transform "volume_nodes" to vector[dim][n_shape_functions]
//              as the mapping support points to use these operators


    ///NOTE: these basis are evaluated in the REFERENCE element.
    //the mapping shape functions evaluated at the volume grid nodes (facet set included in volume grid nodes for consistency)
    std::vector<dealii::FullMatrix<real>> mapping_shape_functions_grid_nodes; 
    //REFERENCE gradient of the the mapping shape functions evaluated at the volume grid nodes
    std::vector<std::vector<dealii::FullMatrix<real>>> gradient_mapping_shape_functions_grid_nodes; 
// FOR Flux Nodes operators there is the grid degree, then the degree of the cubature set it is applied on
// to handle all general cases 
//
    //mapping shape functions evaluated at the VOLUME flux nodes (arbitrary, does not have to be on the surface ex/ GL)
    std::vector<std::vector<dealii::FullMatrix<real>>> mapping_shape_functions_vol_flux_nodes; 
    //mapping shape functions evaluated at the SURFACE flux nodes 
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> mapping_shape_functions_face_flux_nodes; 
    //gradient of mapping shape functions evalutated at VOLUME flux nodes
    //Note that for a single grid degree it can evaluate at varying degree of fluxnodes 
    //ie allows of sub/super parametric etc and over integration
    //Vector order goes: [Grid_Degree][Flux_Poly_Degree][Dim][n_quad_pts][n_mapping_shape_functions]
    std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>> gradient_mapping_shape_functions_vol_flux_nodes; 
    //gradient of mapping shape functions evalutated at surface flux nodes
    //got vector of degree->vector of n_faces -> vector of dim -> Matrix n_face_quad_pts x n_shape_functions
    std::vector<std::vector<std::vector<std::vector<dealii::FullMatrix<real>>>>> gradient_mapping_shape_functions_face_flux_nodes; 

    /**************************************************
 *
 *
 *              METRIC DEPENDENT OPERATORS FUNCTIONS
 *
 *              **********************************************/

    //allocates metric shape functions operators
    void allocate_metric_operators();
    //creates metric shape functions operators
    void create_metric_basis_operators ();
    //called on the fly and returns the metric cofactor and determinant of Jacobian at VOLUME cubature nodes
    void build_local_vol_metric_cofactor_matrix_and_det_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts, 
                                    const unsigned int n_metric_dofs,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<real> &determinant_Jacobian,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
    //called on the fly and returns the metric cofactor and determinant of Jacobian at face cubature nodes
    //ONLY DOES COFACTOR FOR dim>=1, if dim==1 pass null to cofactor and just compute determinant of jac
    void build_local_face_metric_cofactor_matrix_and_det_Jac(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int iface,
                                    const unsigned int n_quad_pts, const unsigned int n_metric_dofs,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<real> &determinant_Jacobian,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
    //computes 3D conservative curl cofactor matrix in VOLUME
    void compute_local_3D_cofactor_vol(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts,
                                    const unsigned int n_metric_dofs,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
    //computes 3D conservative curl cofactor matrix on FACE for consistent normals with water-tight mesh
    void compute_local_3D_cofactor_face(
                                    const unsigned int grid_degree, const unsigned int poly_degree,
                                    const unsigned int n_quad_pts,
                                    const unsigned int n_metric_dofs, const unsigned int iface,
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
    //computes (x_l * \nabla(x_m)) at GRID NODES
    void compute_Xl_grad_Xm(
                                    const unsigned int grid_degree,
                                    const unsigned int n_metric_dofs, 
                                    const std::vector<std::vector<real>> &mapping_support_points,
                                    std::vector<dealii::DerivativeForm<1,dim,dim>> &Xl_grad_Xm);
    //computes the cyclic curl loop for metric cofactor matrix
    void do_curl_loop_metric_cofactor(
                                    const unsigned int n_quad_pts,
                                    const std::vector<dealii::DerivativeForm<2,dim,dim>> grad_Xl_grad_Xm,
                                    std::vector<dealii::FullMatrix<real>> &metric_cofactor);
                                

};//end operator base


} // OPERATOR namespace
} // PHiLiP namespace

#endif
