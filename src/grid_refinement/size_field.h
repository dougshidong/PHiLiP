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
  *     f_\boldsymbol{x} 
  *     = \left< \boldsymbol{v}, \boldsymbol{w}, -\boldsymbol{v}, -\boldsymbol{w} \right>
  *     = V \left< \boldsymbol{e}_1, \boldsymbol{e}_2, -\boldsymbol{e}_1, -\boldsymbol{e}_2 \right>
  * \f]
  * 
  * Where the vectors \f$v\f$ and \f$w\f$ are sized and aligned with the target element axis, and \f$V\f$ is the local
  * transformation matrix from the reference element axis \f$e_i\f$ to form this physical frame. For the orthogonal 
  * frame-field adaptation case this can be decomposed as
  * 
  * \f[
  *     V 
  *     = \left[\begin{matrix} \boldsymbol{v} & \boldsymbol{w} \end{matrix}\right]
  *     = h \left[\begin{matrix} 
  *         cos(\thea) & -sin(\theta) \\
  *         sin(\theta) & cos(\theta) \end{matrix}\right] 
  *     \left[\begin{matrix} 
  *         \sqrt{\rho} & \\
  *          & 1/\sqrt{\rho}
  *     \end{matrix}\right]
  * \f]
  * 
  * where \f$h\f$ represents the size field, \f$\theta\f$ the orientation and \f$\rho\f$ the anisotropy.Central to many 
  * of the techniques is the target complexity defined by
  * 
  * \f[
  *     \mathcal{C}(\mathcal{M}) = 
  *     \int_{\Omega} {
  *         \mathrm{det}{(V(\boldsymbol{x}))} 
  *         \left(\mathcal{P}(\boldsymbol{x})+1\right)^2 
  *     \mathrm{d}\boldsymbol{x}}
  * \f]
  * 
  * This uses the frame field transformation \f$V(x)\f$ and continuous polynomial field \f$\mathcal{P})(x)\f$.
  * Together these provide an approximation of the continuous degrees of freedoms in the adapted target mesh.
  * Note: \f$p-\f$ based functionality is still not complete and requires updates in other parts of the code.
  */ 
template <int dim, typename real>
class SizeField
{
public:
    /// Computes the size field (element scale) for a uniform \f$p\f$- distribution
    /** Unlike other cases, here for a constant value of \f$p\f$-, the contiunuous constrained error minimization
      * problem can be written by the set of integrals involving the mesh density distribution \f$d(x)\f$
      * 
      * \f[
      *     \min_{d} \quad \mathcal{E}(d,\mathcal{P}) = 
      *     \int_{\Omega} {
      *         B(\boldsymbol{x},\mathcal{P}(\boldsymbol{x})) 
      *         d(\boldsymbol{x})^{-\frac{q(\mathcal{P}(\boldsymbol{x})+1)}{2}}
      *     \mathrm{d}\boldsymbol{x}}
      * \f]
      * \f[
      *     \textrm{s.t.} \quad 
      *     \mathcal{N}_{hp}(d, \mathcal{P}) = 
      *     \int_{\Omega} {
      *         d(\boldsymbol{x}) (\mathcal{P}(\boldsymbol{x}))+1)^2
      *     \mathrm{d}\boldsymbol{x}} \leq \mathcal{C}_{t}
      * \f]
      * 
      * where \f$B(x)\f$ is the error distribution based on the approximate quadratic form obtained
      * by reconstruction of the \f$p+1\f$ directional derivatives. This can be solved directly by
      * calculus of variations leading to the global distribution for the scaling as:
      * 
      * \f[
      *     h(\boldsymbol{x})^{dim} = \left(
      *         \frac
      *             {C_t B(\boldsymbol{x})}
      *             {\int_{\Omega}{
      *                 B(\boldsymbol{x})^{\frac{2}{q(p+1)+2}}
      *             \mathrm{d}\boldsymbol{x}}}
      *     \right)^{\frac{-2}{q(p+1)+2}}
      * \f]
      * 
      * The main difference here is that the \f$\lambda\f$ bisection parameter can be plugged back through
      * the complexity constraint and removed the integral giving the simplified final form.
      */
    static void isotropic_uniform(
        const real &                               complexity,   ///< Target global continuous mesh complexity
        const dealii::Vector<real> &               B,            ///< Continuous error model for \f$p+1\f$ directional derivatives
        const dealii::DoFHandler<dim> &            dof_handler,  ///< DoFHandler describing the mesh
        std::unique_ptr<Field<dim,real>> &         h_field,      ///< (Output) Target size-field
        const real &                               poly_degree   ///< (Input) Uniform polynomial degree
        );

    /// Computes the size field (element scale) for a potentially non-uniform \f$p\f$- distribution
    /** Based on the same idea as isotropic_uniform size field method. The continuous constained error minimization
      * problem can again be expressed based on \f$B(x)\f$, the error distribution from the quadratic form of \f$p+1\f$ 
      * directional derivatives:
      * 
      * \f[
      *     \min_{d} \quad \mathcal{E}(d,\mathcal{P}) = 
      *     \int_{\Omega} {
      *         B(\boldsymbol{x},\mathcal{P}(\boldsymbol{x})) 
      *         d(\boldsymbol{x})^{-\frac{q(\mathcal{P}(\boldsymbol{x})+1)}{2}}
      *     \mathrm{d}\boldsymbol{x}}
      * \f]
      * \f[
      *     \textrm{s.t.} \quad 
      *     \mathcal{N}_{hp}(d, \mathcal{P}) = 
      *     \int_{\Omega} {
      *         d(\boldsymbol{x}) (\mathcal{P}(\boldsymbol{x}))+1)^2
      *     \mathrm{d}\boldsymbol{x}} \leq \mathcal{C}_{t}
      * \f]
      * 
      * However, at this point the global problem cannot be solved analytically due to the inclusion of the
      * polynomial distribution \f$\mathcal{P}(\boldsymbol{x})\f$. Instead, it can be rewritten as a lagragian which leads to
      * 
      * \f[
      *     h(\boldsymbol{x})^{dim} = \left(
      *         \frac{2\lambda(\mathcal{P}(\boldsymbol{x})+1)}
      *              {q B(\boldsymbol{x}, \mathcal{P}(\boldsymbol{x}))}
      *     \right)^{\frac{-2}{-q(\mathcal{P}(\boldsymbol{x})+1)+2}}
      * \f]
      * 
      * The overall problem is then solved by performing bisection on this \f$\lambda\f$ parameter until the complexity target 
      * is suitably matched. Thus allowing the updated size distribution to be directly extracted from this expression.
      */ 
    static void isotropic_h(
        const real                                 complexity,            ///< Target global continuous mesh complexity
        const dealii::Vector<real> &               B,                     ///< Continuous error model for \f$p+1\f$ directional derivatives
        const dealii::DoFHandler<dim> &            dof_handler,           ///< DoFHandler describing the mesh
        const dealii::hp::MappingCollection<dim> & mapping_collection,    ///< Element mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         ///< Finite element collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, ///< Quadrature rules collection
        const dealii::UpdateFlags &                update_flags,          ///< Update flags for the volume finite elements
        std::unique_ptr<Field<dim,real>> &         h_field,               ///< (Output) Target size-field
        const dealii::Vector<real> &               p_field                ///< (Input) Current polynomial field  
        );
      
    // computes updated p-field with a constant h-field
    // NOT IMPLEMENTED yet
    /*
    static void isotropic_p(
        const dealii::Vector<real> &               Bm,                    ///< Continuous error model for \f$p\f$ directional derivatives
        const dealii::Vector<real> &               B,                     ///< Continuous error model for \f$p+1\f$ directional derivatives
        const dealii::Vector<real> &               Bp,                    ///< Continuous error model for \f$p+2\f$ directional derivatives
        const dealii::DoFHandler<dim> &            dof_handler,           ///< DoFHandler describing the mesh
        const dealii::hp::MappingCollection<dim> & mapping_collection,    ///< Element mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         ///< Finite element collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, ///< Quadrature rules collection
        const dealii::UpdateFlags &                update_flags,          ///< Update flags for the volume finite elements
        const std::unique_ptr<Field<dim,real>> &   h_field,               ///< (Input) Current size-field
        dealii::Vector<real> &                     p_field);              ///< (Output) Target polynomial field  
    */

    /// Computes the size field (element scale) and updated polynomial distrubition for high-order error function
    /** Based on the method of Dolejsi et al. originally developped for the simplex case. See:
      * Dolejsi et al. 2018 "A continuous hp-mesh model for adaptive discontinuous Galerkin schemes"
      * 
      * Overall, the process involves solving for an optimal \f$h\f$- distribution using the method 
      * outlined in isotropic_h above followed by a sampling of changes to the local polynomial orders.
      * Here, additional error distributions for the \f$p+1\f$ quadratic derivative forms must be provided for
      * an approximation of a coarsened (error is current largest terms) or refined polynomial order 
      * (error is two orders higher). After convergence of the size fields, this amounts to minimizing 
      * the error at a constant complexity level based on the error distribution integrand:
      * 
      * \f[
      *     e_{ref}(\boldsymbol{x}) = \left(
      *         B_{p}(\boldsymbol{x})
      *         h(\boldsymbol{x})^{\frac{dim(p+1)}{2}}
      *     \right)^q
      * \f]
      * \f[
      *     N_{ref}(\boldsymbol{x}) = \left(
      *         \frac{p+1}{h(\boldsymbol{x})}
      *     \right)^{dim}
      * \f]
      * 
      * Based on the equal complexity \f$N_{ref} = N_{p-1} = N_{p-1}\f$ the equivalent local sizes are:
      * 
      * \f[
      *     h_{p-1}(\boldsymbol{x}) = 
      *     \frac{p}
      *          {N_{ref}^{\frac{1}{dim}}} 
      * \f]
      * \f[
      *     h_{p+1}(\boldsymbol{x}) = 
      *     \frac{p+2}
      *          {N_{ref}^{\frac{1}{dim}}} 
      * \f]
      * 
      * Where the lower and higher polynomial distributions lead to error estimates of:
      * 
      * \f[
      *     e_{p-1}(\boldsymbol{x}) = \left(
      *         B_{p-1}(\boldsymbol{x})
      *         h_{p-1}(\boldsymbol{x})^{\frac{dim(p)}{2}}
      *     \right)^q
      * \f]
      * \f[
      *     e_{p+1}(\boldsymbol{x}) = \left(
      *         B_{p+1}(\boldsymbol{x})
      *         h_{p+1}(\boldsymbol{x})^{\frac{dim(p+2)}{2}}
      *     \right)^q
      * \f]
      * 
      * Where the polynomial with lowest error is then chosen for the \f$p\f$- field.
      * Note: not thoroughly tested due to limitations in the \f$p\f$- adaptivity.
      */ 
    static void isotropic_hp(
        const real                                 complexity,            ///< Target global continuous mesh complexity
        const dealii::Vector<real> &               Bm,                    ///< Continuous error model for \f$p\f$ directional derivatives
        const dealii::Vector<real> &               B,                     ///< Continuous error model for \f$p+1\f$ directional derivatives
        const dealii::Vector<real> &               Bp,                    ///< Continuous error model for \f$p+2\f$ directional derivatives
        const dealii::DoFHandler<dim> &            dof_handler,           ///< DoFHandler describing the mesh
        const dealii::hp::MappingCollection<dim> & mapping_collection,    ///< Element mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         ///< Finite element collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, ///< Quadrature rules collection
        const dealii::UpdateFlags &                update_flags,          ///< Update flags for the volume finite elements
        std::unique_ptr<Field<dim,real>> &         h_field,               ///< (Output) Target size-field
        dealii::Vector<real> &                     p_field                ///< (Output) Target polynomial field  
        );

    /// Performs adjoint based size field adaptation for uniform polynomial distribution based on the method of Balan et al.
    /** Works by scaling the local element sizes according to a logarithmic distribution
      * of the error parameter based on the dual-weighted residual (DWR). See adjoint_h_balan
      * for details in the more general \f$p\f$- distribution case.
      */
    static void adjoint_uniform_balan(
        const real                                 complexity,            ///< Target global continuous mesh complexity
        const real                                 r_max,                 ///< Maximum refinement scaling factor
        const real                                 c_max,                 ///< Maximum coarsening scaling factor
        const dealii::Vector<real> &               eta,                   ///< Dual-weighted residual (DWR) error indicator distribution
        const dealii::DoFHandler<dim> &            dof_handler,           ///< DoFHandler describing the mesh
        const dealii::hp::MappingCollection<dim> & mapping_collection,    ///< Element mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         ///< Finite element collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, ///< Quadrature rules collection
        const dealii::UpdateFlags &                update_flags,          ///< Update flags for the volume finite elements
        std::unique_ptr<Field<dim,real>>&          h_field,               ///< (Output) Target size-field
        const real &                               poly_degree            ///< uniform polynomial degree
        );

    /// Performs adjoint based size field adaptation for non-uniform polynomial distribution based on the method of Balan et al.
    /** Works by scaling the local element sizes according to a logarithmic distribution
      * of the error parameter based on the dual-weighted residual (DWR). This works with
      * non-uniform high-order \f$p\f$-distrubtions (although they are not modified). It is recommended
      * to use this to determine only the mesh scale update relative to the previous iteration
      * with the anisotropic targets obtained from the directional derivatives quadratic form.
      * Overall, the problem involves iteratively sampling threshold values \f$\eta_{ref}\f$ to
      * drive the choice of refinement/coarsening choices. Care must be taken in setting the 
      * complexity target, maximum refinement and maximum coarsening range to ensure that a 
      * suitable refinement distribution is maintained throughout grid refinements. See the original
      * paper by Balan et al. for more information "Adjoint-based hp-adaptivity on anisotropic meshes..."
      * and the functions update_alpha_vector_balan and update_alpha_k_balan for implementation details.
      */
    static void adjoint_h_balan(
        const real                                 complexity,            ///< Target global continuous mesh complexity
        const real                                 r_max,                 ///< Maximum refinement scaling factor
        const real                                 c_max,                 ///< Maximum coarsening scaling factor
        const dealii::Vector<real> &               eta,                   ///< Dual-weighted residual (DWR) error indicator distribution
        const dealii::DoFHandler<dim> &            dof_handler,           ///< DoFHandler describing the mesh
        const dealii::hp::MappingCollection<dim> & mapping_collection,    ///< Element mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         ///< Finite element collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, ///< Quadrature rules collection
        const dealii::UpdateFlags &                update_flags,          ///< Update flags for the volume finite elements
        std::unique_ptr<Field<dim,real>>&          h_field,               ///< (Output) Target size-field
        const dealii::Vector<real> &               p_field                ///< polynomial degree vector
        );

    /// Performs adjoint based size field adaptation with a uniform \f$p\f$- distribution based on equidistribution of error
    /** This process assumes that the error in the DWR with locally converge at the optimal \f$2p+1\f$ rate
      * and attempts to select a size field distribution which will equally distribute this error and
      * minimize the global error function at a chosen target complexity. Locally this leads to scaling 
      * the error relative to some reference bisection parameter with a \f$2p+1\f$ exponent. See update_h_dwr
      * for the description of the size field.
      * 
      * Note: This method is not recommended as it leads to large oscillations in the size field
      *       because the error does not predictibly follow this convergence pattern.
      */ 
    static void adjoint_h_equal(
        const real                                 complexity,            ///< Target global continuous mesh complexity
        const dealii::Vector<real> &               eta,                   ///< Dual-weighted residual (DWR) error indicator distribution
        const dealii::DoFHandler<dim> &            dof_handler,           ///< DoFHandler describing the mesh
        const dealii::hp::MappingCollection<dim> & mapping_collection,    ///< Element mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         ///< Finite element collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, ///< Quadrature rules collection
        const dealii::UpdateFlags &                update_flags,          ///< Update flags for the volume finite elements
        std::unique_ptr<Field<dim,real>>&          h_field,               ///< (Output) Target size-field
        const real &                               poly_degree            ///< (Input) Uniform polynomial degree
        );

protected:

    /// Updates the size field based on the DWR disitribution and a reference value, \f$\tau\f$.
    /** This assumes that the error in the DWR will converge at the optimal \f$2p+1\f$ rate and
      * attempts to equally distribute this error over the mesh complexity. Locally, with suitable
      * choice of this parameter this leads to the size field distribution from:
      * 
      * \f[
      *     h(\boldsymbol{x}) = \left(
      *         \frac{\tau}{\eta(\boldsymbol{x})}
      *     \right)^{\frac{1}{2p+1}}
      * \f]
      * 
      * Note: This method is not recommended as it leads to large oscillations in the size field
      *       because the error does not predictibly follow this convergence pattern.
      */ 
    static void update_h_dwr(
        const real                          tau,          ///< Bisection reference value for scaling
        const dealii::Vector<real> &        eta,          ///< Dual-weighted residual (DWR) error indicator distribution
        const dealii::DoFHandler<dim> &     dof_handler,  ///< DoFHandler describing the mesh
        std::unique_ptr<Field<dim,real>>&   h_field,      ///< (Output) Target size-field
        const real &                        poly_degree   ///< (Input) Uniform polynomial degree
        );

    /// Based on a chosen bisection paraemeter, updates the \f$h\f$- field to an optimal distribution
    /** This is based on the solution to the error minimization problem. Overall, given some choice 
      * of paratere \f$\lambda\f$, the size field should take the form of:
      * 
      * \f[
      *     d(\boldsymbol{x}) = \left(
      *     \frac{2\lambda(\mathcal{P}(\boldsymbol{x})+1)}
      *          {q B(\boldsymbol{x}, \mathcal{P}(\boldsymbol{x}))}
      *     \right)^{\frac{2}{-q(\mathcal{P}(\boldsymbol{x})+1)+2}}
      * \f]
      * 
      * where \f$d(\boldsymbol{x})\f$ is the local mesh density (\f$\approx h(\boldsymbol{x})^{-2}\f$ for 2D quad meshing)
      * and \f$B(\boldsymbol{x})\f$ is the quadratic error distribution. The overall process can be controlled
      * by checking the resulting continuous complexity of the result until the error problem 
      * constraint is suitably satisfied.
      */ 
    static void update_h_optimal(
        const real                          lambda,      ///< (Input) Current complexity bisection parameter
        const dealii::Vector<real> &        B,           ///< Continuous error constant for quadratic model of \f$p+1\f$ directional derivatives
        const dealii::DoFHandler<dim> &     dof_handler, ///< DoFHandler describing the mesh
        std::unique_ptr<Field<dim,real>> &  h_field,     ///< (Input) Target size-field
        const dealii::Vector<real> &        p_field      ///< (Input) Current polynomial field  
        );

    /// Evaluates the continuous complexity (approximate degrees of freedom) for the target \f$hp\f$ mesh representation
    /** Discrete complexity approximation is obtained by considering the frame-field target distribution (with
      * either isotropic or anisotropic field distribution) and the vector of \f$C^0\f$ polynomial distribution.
      * Together these fully describe the target \f$hp\f$ mesh and in the continuous space the degrees of freedom
      * can be approximated by:
      * 
      * \f[
      *     \mathcal{C}(\mathcal{M}) = 
      *     \int_{\Omega} {
      *         \mathrm{det}{(V(\boldsymbol{x}))} 
      *         \left(\mathcal{P}(\boldsymbol{x})+1\right)^2 
      *     \mathrm{d}\boldsymbol{x}}
      * \f]
      * 
      * or by a summation based on the elements of the discrete existing mesh:
      * 
      * \f[
      *     \mathcal{C} = 
      *     \sum_{i=0}^{N} {
      *         \mathrm{det}{(V_i^{-1})}
      *         \left(p_i+1\right)^2 
      *     }
      * \f]
      * 
      * This is commonly used as a target parameter for controlling leve of refinement in conjunction with
      * a bisection function as used in many of the methods listed above.
      */     
    static real evaluate_complexity(
        const dealii::DoFHandler<dim> &            dof_handler,           ///< DoFHandler describing the mesh
        const dealii::hp::MappingCollection<dim> & mapping_collection,    ///< Element mapping collection
        const dealii::hp::FECollection<dim> &      fe_collection,         ///< Finite element collection
        const dealii::hp::QCollection<dim> &       quadrature_collection, ///< Quadrature rules collection
        const dealii::UpdateFlags &                update_flags,          ///< Update flags for the volume finite elements
        const std::unique_ptr<Field<dim,real>> &   h_field,               ///< (Input) Current size-field
        const dealii::Vector<real> &               p_field                ///< (Input) Current polynomial field  
        );

    /// Updates the size field for the mesh through the area measure based on reference value
    /** Updates the h-field object based on an input vector or error values and reference parameters.
      * Mainly called as the update for bisection from adjoint_h_balan. Based on the relative local
      * error determines a scaling factor from update_alpha_balan, scales the local h field according 
      * to the growth in cell measure \f$I_k\f$ (length in 1D, area in 2D, volume in 3D):
      * 
      * \f[
      *     I_k = \alpha_k I_k^c
      * \f]
      * 
      * where \f$I_k^c\f$ is the initial coarse cell size. In the 2D quad case for the frame field defintion \
      * this is equivalent to updating the local scaling factors by:
      * 
      * \f[
      *     h = \sqrt{\frac{1}{4} \alpha_k I_k^c}
      * \f]
      * 
      * Overall, this shifts the problem of complexity targetting to sampling different values of \f$\eta_{ref}\f$.
      */ 
    static void update_alpha_vector_balan(
        const dealii::Vector<real>&        eta,         ///< Dual-weighted residual (DWR) error indicator distribution
        const real                         r_max,       ///< Maximum refinement scaling factor
        const real                         c_max,       ///< Maximum coarsening scaling factor
        const real                         eta_min,     ///< Minimum value of DWR indicator
        const real                         eta_max,     ///< Maximum value of DWR indicator
        const real                         eta_ref,     ///< Threshold value of DWR for deciding between coarsening and refinement
        const dealii::DoFHandler<dim>&     dof_handler, ///< DoFHandler describing the mesh
        const dealii::Vector<real>&        I_c,         ///< Vector of cell current cell area measure
        std::unique_ptr<Field<dim,real>>&  h_field      ///< (Output) Updated size-field 
        );

    /// Determines local \f$\alpha\f$ sizing factor (from adjoint estimates)
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
      * This function returns the value \f$\alpha\f$ used to update the local size measure.
      */ 
    static real update_alpha_k_balan(
        const real eta_k,   ///< Local value of DWR indicator
        const real r_max,   ///< Maximum refinement scaling factor
        const real c_max,   ///< Maximum coarsening scaling factor
        const real eta_min, ///< Minimum value of DWR indicator
        const real eta_max, ///< Maximum value of DWR indicator
        const real eta_ref  ///< Threshold value of DWR for deciding between coarsening and refinement
        );

    /// Bisect function based on starting bounds
    /** Performs bisection on an input lambda function \f$f(x)\f$ with starting bounds
      * \f$x\in\left[a,b\right]\f$. Assumes that \f$f(a)\f$ and \f$f(b)\f$ are of opposite sign
      * and iteratively chooses half interval until a suitably accurate approximation
      * of \f$f(x)\approx 0\f$ is found. Stops when either a absolute or relative (to
      * the initial range) function value tolerance is achieved.
      */ 
    static real bisection(
        const std::function<real(real)> func,                 ///< Input lambda function to be solved for \f$f(x)=0\f$, takes real value and returns real value 
        real                            lower_bound,          ///< lower bound of the search, \f$a\f$
        real                            upper_bound,          ///< upper bound of the search, \f$b\f$
        real                            rel_tolerance = 1e-6, ///< Relative tolerance scale, stops search when \f$\left|f(x_i)\right|<\epsilon \left|f(a)-f(b)\right|\f$
        real                            abs_tolerance = 1.0   ///< Absolute tolerance scale, stops search when \f$\left|f(x_i)\right|<\epsilon\f$
        );

};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __SIZE_FIELD_H__
