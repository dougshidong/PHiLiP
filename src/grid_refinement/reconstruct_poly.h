
#ifndef __RECONSTRUCT_POLY_H__
#define __RECONSTRUCT_POLY_H__

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/base/polynomial_space.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

#include "physics/manufactured_solution.h"

namespace PHiLiP {

namespace GridRefinement {

/// Enumeration of norms availible to be used in the patchwise reconstruction
enum class NormType{
    H1,
    L2,
    };

// forward declaration of multi-index computation from Dealii
/// Modified implementaiton of polynomial space indexing from Deal.II
/** Computes multi-index of polynomial term for different dimensions. 
  * Based on the Deal.II protected function: 
  * https://www.dealii.org/current/doxygen/deal.II/polynomial__space_8cc_source.html
  */ 
template <int dim>
std::array<unsigned int, dim> compute_index(
    const unsigned int i,
    const unsigned int size);

// funcitons for polynomial reconstruction
template <int dim, int nstate, typename real>
class ReconstructPoly
{

public:
    /// Deleted default constructor
    ReconstructPoly() = delete;

    /// Constructor. Stores required information about the mesh and quadrature rules.
    ReconstructPoly(
        const dealii::hp::DoFHandler<dim>&        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim>& mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim>&      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim>&       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags&                update_flags);         // update flags for for volume fe

    /// Reinitialze the internal vectors 
    /** These vectors are used to store the obtained derivative
      * values and directions at each mesh element.
      */
    void reinit(const unsigned int n);

    /// Select the Norm to be used in reconstruction
    /** Results in a modified local reconstruction process for the patch of neighbouring elements.
      */ 
    void set_norm_type(const NormType norm_type);

    /// Construct directional derivatives along the chords of the cell
    /** $p+1$ (or rel_order) derivatives are constructed and extracted along the specified directions
      * from the existing cell size. Once all polynomial terms on the surrounding patch are approximated
      * (see reconstruct_norm for description), the derivative components are obtained by evaluating this
      * function along a given direction:
      * 
      * \f[
      *     u_{\bar{\bm{x}}, p}(\bm{x}) = 
      *     \sum_{|\bm{\alpha}| \leq p} {
      *         \frac{\partial^{\bm{\alpha}} u(\bar{\bm{x}})}
      *              {\bm{\alpha}!} 
      *         (\bm{x}-\bar{\bm{x}})^{\bm{\alpha}}
      *         }
      * \f]
      * \f[
      *     D^{p+1}_{\bm{\xi}} u(\bar{\bm{x}}) h^{p+1} 
      *     = u_{\bar{\bm{x}}, p+1}(\bm{x}+h\bm{\xi}) 
      *     - u_{\bar{\bm{x}}, p}(\bm{x}+h\bm{\xi})
      *     = \sum_{i=0}^{p+1}{
      *     \frac{1}
      *          {i! (p+1-i)!} 
      *     \frac{\partial^{p+1} u(\bar{\bm{x}})}
      *          {\partial x^i \partial y^{p+1-i}} 
      *     (x-\bar{x})^i (y-\bar{y})^{p+1-i}}
      * \f]
      * 
      * Ordering is based on the internal dealii numbering. Used in cases where the orientation of the element
      * is not controlled by the refinement procedure (e.g. fixed fraction cases). Gives direct prediction of how
      * error will change with modifying the length of these axes.
      */ 
    void reconstruct_chord_derivative(
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,   ///< Solution approximation to be reconstructed
        const unsigned int                                     rel_order); ///< Relative order of the approximation

    /// Construct the set of largest perpendicular directional derivatives
    /** $p+1$ (or rel_order) derivatives are constructed  and the largest values are extracted. In order to approximate 
      * the "worst case" unit-ball of the error in the high-order case, the method of Dolejsi is used where values are extracted
      * from the maximum direction in the next perpendicular hyperplane to existing directions. In this way, a set of orthogonal directions 
      * is chosen with descending largest derivative orders. In 2D, this process can be written as solving the collection of functions
      * to find maximums based on the patchwise reconstruction of the polynomial (obtained from reconstruction):
      * 
      * \f[
      *     A_1(\bar{\bm{x}}, p) &= \max_{\left\lVert{\bm{\xi}}\right\rVert_2=1}{|D^p_{\bm{\xi}} u(\bar{\bm{x}})|}
      * \f]
      * \f[
      *     \bm{\xi}_1(\bar{\bm{x}}, p) &= \operatornamewithlimits{argmax}_{\left\lVert{\bm{\xi}}\right\rVert_2=1}{|D^p_{\bm{\xi}} u(\bar{\bm{x}})|}
      * \f]
      * \f[
      *     \varphi(\bar{\bm{x}}, p) & \in \left[ 0, 2\pi\right) \quad s.t. \text{ } \bm{\xi}_1 = (cos(\varphi), sin(\varphi))
      * \f]
      * \f[
      *     A_2(\bar{\bm{x}}, p) &= |D^p_{\bm{\xi}_2} u(\bar{\bm{x}})|, \quad \text{where } \bm{\xi}_1 \cdot \bm{\xi}_2 = 0.
      * \f]
      * 
      * In the linear case where $p=2$, these directions and values can be directly extracted from the local hessian reconstruction.
      * Otherewise, for the high-order case, this is performed by sampling an approximately equidistributed set of points to give a "good enough"
      * approximation. In 2D, 180 points are distributed radially on $[0,\pi]$ to give a $1^\circ$ accuracy (second half of angles will be equal 
      * or negative depending on even/odd polynomial order). In 3D, the initial sampling is done using a fibbonaci spiral mapped to the unit sphere 
      * (a fibbonaci sphere, see https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012).
      * This provides a good enough apporximation to an even distrubution for this case. $180 \times 180 / 2$ samples are used to give a 
      * similar angular resolution. The second components are extracted from a unit-circle on the perpendicular plane to the largest direction.
      */
    void reconstruct_directional_derivative(
        const dealii::LinearAlgebra::distributed::Vector<real>&solution,   ///< Solution approximation to be reconstructed
        const unsigned int                                     rel_order); ///< Relative order of the approximation

    /// Constructs directional derivates based on the manufactured solution hessian
    /** For $p=2$ only, gets the exact directional derivative components using the spectral decomposition of the hessian:
      * 
      * \f[
      *     H = \left[\begin{matrix} u_{xx} & u_{xy} \\ u_{yx} & u_{yy} \end{matrix}\right] 
      *     = \left[\begin{matrix} \bm{v}_0 & \bm{v}_1 \end{matrix}\right]
      *     \left[\begin{matrix} \lambda_0 & \\ & \lambda_1 \end{matrix}\right]
      *     \left[\begin{matrix} \bm{v}_0^T \\ \bm{v}_1^T \end{matrix}\right]
      * \f]
      * \f[
      *     D^{p=2}_{\bm{\xi}} u(\bar{\bm{x}}) h^{p+1} 
      *     = \bm{\xi}^T H \bm{\xi}
      *     = \sum_{i=0}^{dim} {\lambda_i \left(\bm{\xi}^T \bm{v}_i\right)^2}
      * \f]
      * 
      * Where then $\lambda_i$ (eigenvales) are the directional derivatives and $v_i$ (eigenvectors) are the direction vectors.
      */
    void reconstruct_manufactured_derivative(
        const std::shared_ptr<ManufacturedSolutionFunction<dim,real>>& manufactured_solution, ///< Manufactured solution function
        const unsigned int                                             rel_order);            ///< Relative order of the approximation

private:
    template <typename DoFCellAccessorType>
    dealii::Vector<real> reconstruct_norm(
        const NormType                                          norm_type,
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution);

    template <typename DoFCellAccessorType>
    dealii::Vector<real> reconstruct_H1_norm(
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution);

    template <typename DoFCellAccessorType>
    dealii::Vector<real> reconstruct_L2_norm(
        const DoFCellAccessorType &                             curr_cell,
        const dealii::PolynomialSpace<dim>                      ps,
        const dealii::LinearAlgebra::distributed::Vector<real> &solution);

    template <typename DoFCellAccessorType>
    std::vector<DoFCellAccessorType> get_patch_around_dof_cell(
        const DoFCellAccessorType &cell);

    // member attributes
    const dealii::hp::DoFHandler<dim>&         dof_handler;
    const dealii::hp::MappingCollection<dim> & mapping_collection;
    const dealii::hp::FECollection<dim> &      fe_collection;
    const dealii::hp::QCollection<dim> &       quadrature_collection;
    const dealii::UpdateFlags &                update_flags;

    // controls the norm settings
    NormType norm_type;

public:
    // values for return
    std::vector<std::array<real,dim>>                       derivative_value;
    std::vector<std::array<dealii::Tensor<1,dim,real>,dim>> derivative_direction;

    // returning the directional derivative dealii::Vector for the i^th largest component
    dealii::Vector<real> get_derivative_value_vector_dealii(
        const unsigned int index);
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __RECONSTRUCT_POLY_H__
