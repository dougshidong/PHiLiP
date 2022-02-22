#ifndef __GRID_REFINEMENT_FIXED_FRACTION_H__
#define __GRID_REFINEMENT_FIXED_FRACTION_H__

#include "grid_refinement/grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

/// Fixed-Fraction Grid Refinement Class
/** This class offers methods to perform fixed-fraction style refinement using
  * built in Deal.II refinement methods. These methods work by first flagging a
  * chosen fraction of the mesh with largest error indicator to be refined. In 
  * \f$h\f$-refinement, this is applied by splitting each cell into a set of subcells
  * (which may potentially be anisotropic). Methods for \f$p\f$-refinement are also
  * supported which locally increment the polynomial order of the discretization.
  * \f$hp\f$-refinement is not fully supported, but some placeholder functions have been
  * included to facilitate this in the future. 
  * Note: Some functionality and behaviour will vary depending on the choice of triangulation
  *       used for both the refinement step and in other parts of the code.
  */ 
#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_FixedFraction : public GridRefinementBase<dim,nstate,real,MeshType>
{
public:
    using GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase;

    /// Perform call to the grid refinement object of choice
    /** This will automatically select the proper subclass, error indicator
      * and various refinement types based on the grid refinement parameters
      * passed at setup to the grid refinement factor class.
      * 
      * For fixed-fraction refinement, this function first computes the error-indicator
      * and optionally smoothness indicator (if used for anisotropy or \f$hp\f$-refinement)
      * then calls proper refinement function to flag and modify the grid in the desired 
      * manner (locally splitting cells or change polynomial orderes). Also automatically
      * transfers the solution onto the new embedded mesh.
      * 
      * See subclass functions for details of refinement types.
      */
    void refine_grid() override;

protected:

    // specified refinement functions for different cases

    /// Based on error indicator, performs fixed-fraction flagging for element splitting
    /** Based on the distribution of the error indicator, the fraction of cells with the
      * largest values will be marked for refinement and the fraction with the lowest values
      * will be marked for coarsening. At the end of the refine_grid function, when the refinement
      * is executed refined cells will be split into \f$2^{dim}\f$ subcells along with any neighbors
      * needed to maintain 2:1 face connectivity. Coarsened cells will be recombined if neighbor
      * cells have also been marked in such a way that they can be merged. Usually coarsening
      * is less consistently applied and it may be simpler to only target refinements.
      */ 
    void refine_grid_h();

    /// Based on error indicator, performs fixed-fraction flagging for polynomial order enrichment
    /** First uses the refine_grid_h function to perform flagging of worst portion of cells
      * then loops over the mesh and converts each of these flags to instead increment the target
      * finite element polynomial order for the local cell instead. No coarsening is currently supported.
      */  
    void refine_grid_p();

    /// (NOT IMPLEMENTED) Based on error and smoothness indicator, perform fixed-fraction flagging and decision between element splitting and polynomial order enrichment
    /** After first flagging the cells using the refine_grid_h function, this method 
      * loops back over the grid and selectively changes the local cell to target \f$p\f$-refinement
      * if the smoothness indicator is above a specified tolerance. This offers better overall
      * convergence in smooth areas of the flow, but, due to Gibb's phenomenon, \f$h\f$-refinement
      * is better suited for areas with discontinuities (such as shocks).
      */ 
    void refine_grid_hp();   

    /// Flags the domain boundaries for refinement
    /** Used for testing the effect this resolution has on
      * the measurement of the functional output value for 
      * boundary integral based functionals.
      */ 
    void refine_boundary_h();

    /// (NOT IMPLEMENTED) Computes smoothness indicator for \f$hp\f$-refinement decisions
    /** Due to Gibbs phenomenon, even with a high-order method convergence degrades in 
      * regions with discontinuities. Therefore, for capturing shocks and other phenomenon
      * \f$h\f$-refinement is more suitable. However, in smoother areas of the problem,
      * \f$p\f$-refinement is capable of capturing large areas with fewer elements offering
      * potential spectral convergence. Therefore, this placeholder function is intended
      * to approximate the smoothness of the flow for choosing between the refinement types.
      */ 
    void smoothness_indicator();

    /// Performs anisotropic adaptation of the mesh
    /** Based on the choice of aniso_indicator, will call either
      * jump_based or reconstruction_based anisotropy methods.
      * Note: anisotropic adaptation is not currently applicable to all
      *       deal.II mesh types.
      */ 
    void anisotropic_h();

    /// Sets anisotropic refinement flags for cells based on discontinuity between neighbors
    /** If the amount of jump across pairs of face in a given element axes are sufficiently
      * above the average of all faces for the cell, then that axis will be flagged for anisotropic 
      * refinement instead of regular refinement. The anisotropic_threshold_ratio parameter controls
      * how strong an anisotropic behavior triggers the response. This option is only considered on 
      * cells flagged isotropically such that the refinement strategy is modified before
      * the application of the splitting. This is based on Step 30 of the Deal.II examples:
      * https://www.dealii.org/current/doxygen/deal.II/step_30.html
      */ 
    void anisotropic_h_jump_based();

    /// Sets anisotropic refinement flags for cells based on directional derivatives reconstructed along the chord lines
    /** Uses a polynomial reconstruction of the high-order \f$p+1\f$ derivatives to estimate the 
      * effectivness of refinement along a single cell axis instead of isotropically. The indicator
      * along each cell chord (the vector from opposing face center to opposing face center) and
      * compared relative to the anisotropic_threshold_ratio needed to alter the flagging of the cell
      * to be split along a single axis instead. Unlike continuous refinement methods using reconstruction
      * the effectiveness of anisotropy is limited by the initial cell orientation and dimensions.
      */ 
    void anisotropic_h_reconstruction_based();

    /// Performs call to proper error indcator function based on type of error_indicator parameter
    /** Each of these functions operates by filling the indicator member variable
      * vector with a value for each cell in the mesh. These inidcators are then 
      * used to decide which cells of the mesh will be flagged for coarsening and
      * refinement respectively based on the established fractions.
      */ 
    void error_indicator();

    // error distribution for each indicator type

    /// Compute error indicator based on Lq norm relative to exact manufactured solution
    /** For debugging and testing, uses the manufactured solution function to determine
      * innacuracies in the current results. Not useful for any actual refinement as it 
      * requires knowledge about the exact problem solution, \f$u(\boldsymbol{x}\f$:
      * 
      * \f[
      *     E_i = \int_{\Omega_i} {|u_h(\boldsymbol{x}) - u(\boldsymbol{x})|^{Lq} \mathrm{d}\boldsymbol{x}}
      * \f]
      * 
      * Evaluated for each element \f$\Omega_i\f$ in the mesh.
      */ 
    void error_indicator_error();

    /// Compute error indicator based on reconstructed \f$p+1\f$ directional derivatives
    /** Uses the reconstructed enriched solution space to determine which areas of
      * the mesh result in the largest discretization error. It is similar in strategy
      * to existing hessian-based and similar high-order approximations. However, in
      * the current scope it is applied to flag cells for splitting. This is done based
      * on the fact the error approximates the \f$p+1\f$ solution:
      * 
      * \f[
      *     E_i = u_{h,p+1}(\boldsymbol{x}+h\boldsymbol{\xi}) - u_{h,p}(\boldsymbol{x}+h\boldsymbol{\xi}) 
      *         = D_{\boldsymbol{\xi}}^{p+1} u(x) h^{p+1}
      * \f]
      * 
      * based on the taylor series expansion where \f$\boldsymbol{\xi}\f$ is the direction vector used
      * in the largest directional derivative
      */ 
    void error_indicator_hessian();

    /// (NOT IMPLEMENTED) Compute error indicator based on fine grid residual distribution
    /** By reconstructing the current solution on a finer grid and evaluating the magnitude
      * of the local residuals, this strategy approximates how innacurate the current approximation
      * would be in a smoother space or after the local mesh is refined. It is similar to 
      * a goal-oriented approach without adjoint weighting:
      * 
      * \f[
      *     E_i = |\mathcal{R}_h(u_h^H)|
      * \f]
      * 
      * where \f$u_h^H\f$ is the prolongation operator to the fine mesh.
      */
    void error_indicator_residual();

    /// Compute error indicator for goal-oriented approach using a dual-weighted residual
    /** Uses the sensitivity of the local flow solution relative to the functional of interest
      * obtained through the solution of the adjoint problem to weight the fine grid residual.
      * Together this forms the "dual-weighted residual (DWR)" and provides an estimate of the
      * contribution of local errors in the flow to the global goal-oriented functional evaluation
      * erorrs. This can be written globally as:
      * 
      * \f[
      *     |\mathcal{J}(u) - \mathcal{J}_h(u_h^H)|\approx-\phi_h^T\mathcal{R}_h(u_h^H)
      * \f]
      *  
      * where locally, the DWR is evaluated by a sum over fine grid elements:
      * 
      * \f[
      *     E_i = |\sum_{\Omega_k \in \Omega_i} {(\psi_h)^T_k (\mathcal{R}_h(u_h^H))_k}|
      * \f]
      * 
      * which has been implemented as part of the adjoint class using a fine grid adjoint solution.
      */ 
    void error_indicator_adjoint();

    /// Output refinement method dependent results
    /** For the current class this includes the error indicator and smoothness indicator distribtuions.
      */ 
    std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() override;

protected:
    dealii::Vector<real> indicator;
    dealii::Vector<real> smoothness;
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_H__