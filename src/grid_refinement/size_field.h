#ifndef __SIZE_FIELD_H__
#define __SIZE_FIELD_H__

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

#include "grid_refinement/field.h"

namespace PHiLiP {

namespace GridRefinement {

/// Size Field Static Class
/** This class contains various utilities for computing updated continuous mesh representations.
  * Notably, these are used along with continuous error estimation models to drive unstructured
  * mesh adaptation methods from grid_refinement_continuous.cpp. Depending on the remeshing 
  * targets, mesh type and information availible, different functions are included for isotropic,
  * anisotropic ($h$-adaptation), varying polynomial order ($h$-adaptation), hybrid ($hp$-adaptation)
  * and goal-oriented (adjoint based) cases. The results are used to modify the local continuous element
  * representation of the Field class (with isotropic or anisotropic element definitions) and a vector
  * of local polynomial orders. For the quad meshing case, this anisotropic field is represented by the 
  * frame field:
  * 
  * \f[
  *     f_\bm{x} 
  *     = \left< \bm{v}, \bm{w}, -\bm{v}, -\bm{w} \right>
  *     = V \left< \bm{e}_1, \bm{e}_2, -\bm{e}_1, -\bm{e}_2 \right>
  * \f]
  * 
  * Where the vectors $v$ and $w$ are sized and aligned with the target element axis, and $V$ is the local
  * transformation matrix from the reference element axis $e_i$ to form this physical frame. For the orthogonal 
  * frame-field adaptation case this can be decomposed as
  * 
  * \f[
  *     V 
  *     = \left[\begin{matrix} \bm{v} & \bm{w} \end{matrix}\right]
  *     = h \left[\begin{matrix} 
  *         cos(\thea) & -sin(\theta) \\
  *         sin(\theta) & cos(\theta) \end{matrix}\right] 
  *     \left[\begin{matrix} 
  *         \sqrt{\rho} & \\
  *          & 1/\sqrt{\rho}
  *     \end{matrix}\right]
  * \f]
  * 
  * where $h$ represents the size field, $\theta$ the orientation and $\rho$ the anisotropy.Central to many 
  * of the techniques is the target complexity defined by
  * 
  * \f[
  *     \mathcal{C}(\mathcal{M}) = 
  *     \int_{\Omega} {
  *         \operatorname{det}{(V(\bm{x}))} 
  *         \left(\mathcal{P}(\bm{x})+1\right)^2 
  *     \mathrm{d}\bm{x}}
  * \f]
  * 
  * This uses the frame field transformation $V(x)$ and continuous polynomial field $\mathcal{P})(x)$.
  * Together these provide an approximation of the continuous degrees of freedoms in the adapted target mesh.
  * Note: $p-$ based functionality is still not complete and requires updates in other parts of the code.
  */ 
template <int dim, typename real>
class SizeField
{
public:
    // computes the isotropic size field (h has only 1 component) for a uniform (p-dist constant) input
    static void isotropic_uniform(
        const real &                               complexity,   // (input) complexity target
        const dealii::Vector<real> &               B,            // only one since p is constant
        const dealii::hp::DoFHandler<dim> &        dof_handler,  // dof_handler
        std::unique_ptr<Field<dim,real>> &         h_field,      // (output) size field
        const real &                               poly_degree); // (input)  polynomial degree

    // computes isotropic size field (h has only 1 component) for a non-uniform p_field
    static void isotropic_h(
        const real                                 complexity,            // (input) complexity target
        const dealii::Vector<real> &               B,                     // only one since p is constant
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>> &         h_field,               // (output) size field
        const dealii::Vector<real> &               p_field);              // (input)  poly field

    // computes updated p-field with a constant h-field
    static void isotropic_p(
        const dealii::Vector<real> &               Bm,                    // constant for p-1
        const dealii::Vector<real> &               B,                     // constant for p
        const dealii::Vector<real> &               Bp,                    // constant for p+1
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        const std::unique_ptr<Field<dim,real>> &   h_field,               // (input) size field
        dealii::Vector<real> &                     p_field);              // (output) poly field

    // updates both the h-field and p-field
    static void isotropic_hp(
        const real                                 complexity,            // target complexity
        const dealii::Vector<real> &               Bm,                    // constant for p-1
        const dealii::Vector<real> &               B,                     // constant for p
        const dealii::Vector<real> &               Bp,                    // constant for p+1
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>> &         h_field,               // (output) size field
        dealii::Vector<real> &                     p_field);              // (output) poly field

    // performs adjoint based size-field adaptation with uniform p-field. Based on:
    // Balan, A., Woopen, M., & May, G. (2016). 
    // Adjoint-based hp-adaptivity on anisotropic meshes for high-order compressible flow simulations. 
    // Computers and Fluids, 139. https://doi.org/10.1016/j.compfluid.2016.03.029
    static void adjoint_uniform_balan(
        const real                                 complexity,            // target complexity
        const real                                 r_max,                 // maximum refinement factor
        const real                                 c_max,                 // maximum coarsening factor
        const dealii::Vector<real> &               eta,                   // error indicator (DWR)
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
        const real &                               poly_degree);          // uniform polynomial degree

    // performs adjoint based size-field adaptation with general p-field. Based on:
    // Balan, A., Woopen, M., & May, G. (2016). 
    // Adjoint-based hp-adaptivity on anisotropic meshes for high-order compressible flow simulations. 
    // Computers and Fluids, 139. https://doi.org/10.1016/j.compfluid.2016.03.029
    static void adjoint_h_balan(
        const real                                 complexity,            // target complexity
        const real                                 r_max,                 // maximum refinement factor
        const real                                 c_max,                 // maximum coarsening factor
        const dealii::Vector<real> &               eta,                   // error indicator (DWR)
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
        const dealii::Vector<real> &               p_field);              // polynomial degree vector

    // performs adjoint based size field adaptatation with uniform p-field
    // peforms equidistribution of DWR to sizes based on 2p+1 power of convergence
    static void adjoint_h_equal(
        const real                                 complexity,            // target complexity
        const dealii::Vector<real> &               eta,                   // error indicator (DWR)
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
        const real &                               poly_degree);          // uniform polynomial degree

protected:
    // sets the h_field sizes based on a reference value and DWR distribution
    static void update_h_dwr(
        const real                          tau,          // reference value for settings sizes
        const dealii::Vector<real> &        eta,          // error indicator (DWR)
        const dealii::hp::DoFHandler<dim> & dof_handler,  // dof_handler
        std::unique_ptr<Field<dim,real>>&   h_field,      // (output) target size_field
        const real &                        poly_degree); // uniform polynomial degree

    // given a p-field, redistribute h_field according to B
    static void update_h_optimal(
        const real                          lambda,      // (input) bisection parameter
        const dealii::Vector<real> &        B,           // constant for current p
        const dealii::hp::DoFHandler<dim> & dof_handler, // dof_handler
        std::unique_ptr<Field<dim,real>> &  h_field,     // (output) size field
        const dealii::Vector<real> &        p_field);    // (input)  poly field

    static real evaluate_complexity(
        const dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
        const std::unique_ptr<Field<dim,real>> &   h_field,               // (input) size field
        const dealii::Vector<real> &               p_field);              // (input) poly field  

    static void update_alpha_vector_balan(
        const dealii::Vector<real>&        eta,         ///< Vector of DWR indicator values
        const real                         r_max,       ///< Maximum refinement scaling factor
        const real                         c_max,       ///< Maximum coarsening scaling factor
        const real                         eta_min,     ///< Minimum value of DWR indicator
        const real                         eta_max,     ///< Maximum value of DWR indicator
        const real                         eta_ref,     ///< Threshold value of DWR for deciding between coarsening and refinement
        const dealii::hp::DoFHandler<dim>& dof_handler, ///< DoFHandler describing the mesh
        const dealii::Vector<real>&        I_c,         ///< Vector of cell current cell area measure
        std::unique_ptr<Field<dim,real>>&  h_field);    ///< (Output) Updated size-field 

    /// Determines local $\alpha$ sizing factor (from adjoint estimates)
    /** Uses relative scale of local Dual-Weighted Residual (DWR) relative to global maximum and minimums and
      * a chosen threshold value based on target output complexity. Sizing update is basaed on Eq. 30-33 from
      * Balan et al. "Adjoint-based hp-adaptivity on anisotropic mesh for high-order..." where the scaling is
      * fit quadratically to a predefined range in the logarithmic space:
      * 
      * \f[
      *     \alpha_{k}=\left\{\begin{array}{ll}
      *         \left(\left(r_{max }-1\right) \xi_{k}^{2}+1\right)^{-1}, & 
      *         \eta_{k} \geq \eta_{ref}, \\
      *         \left(\left(c_{max}-1\right) \xi_{k}^{2}+1\right), 
      *         & \eta_{k}<\eta_{ref},
      *     \end{array}\right.
      * \f]
      * 
      * where 
      * 
      * \f[
      *     \xi_{k}=
      *     \left\{\begin{array}{ll}
      *         \frac{\log \left(\eta_{k}\right)-\log \left(\eta_{ref}\right)}{\log \left(\eta_{max }\right)-\log \left(\eta_{ref}\right)}, & 
      *         \eta_{k} \geq \eta_{ref}, \\
      *         \frac{\log \left(\eta_{k}\right)-\log \left(\eta_{ref}\right)}{\log \left(\eta_{min }\right)-\log \left(\eta_{ref}\right)}, & 
      *         \eta_{k}<\eta_{ref}.
      *     \end{array}\right.
      * \f]
      * 
      * This function returns the value $\alpha$ used to update the local size measure.
      */ 
    static real update_alpha_k_balan(
        const real eta_k,   ///< Local value of DWR indicator
        const real r_max,   ///< Maximum refinement scaling factor
        const real c_max,   ///< Maximum coarsening scaling factor
        const real eta_min, ///< Minimum value of DWR indicator
        const real eta_max, ///< Maximum value of DWR indicator
        const real eta_ref);///< Threshold value of DWR for deciding between coarsening and refinement

    /// Bisect function based on starting bounds
    /** Performs bisection on an input lambda function $f(x)$ with starting bounds
      * $x\in\left[a,b\right]$. Assumes that $f(a)$ and $f(b)$ are of opposite sign
      * and iteratively chooses half interval until a suitably accurate approximation
      * of $f(x)\approx 0 $ is found. Stops when either a absolute or relative (to
      * the initial range) function value tolerance is achieved.
      */ 
    static real bisection(
        const std::function<real(real)> func,                 ///< Input lambda function to be solved for $f(x)=0$, takes real value and returns real value 
        real                            lower_bound,          ///< lower bound of the search, $a$
        real                            upper_bound,          ///< upper bound of the search, $b$
        real                            rel_tolerance = 1e-6, ///< Relative tolerance scale, stops search when $\left|f(x_i)\right|<\epsilon \left|f(a)-f(b)\right|$
        real                            abs_tolerance = 1.0); ///< Absolute tolerance scale, stops search when $\left|f(x_i)\right|<\epsilon$
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __SIZE_FIELD_H__
