#ifndef __GRID_REFINEMENT_CONTINUOUS_H__
#define __GRID_REFINEMENT_CONTINUOUS_H__

#include "grid_refinement/grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

/// Continuous Grid Refinement Class
/** This class provides method to update the grid using a combination of continuous mesh and
  * continuous error models. Together these are used to replace the discrete error minimization 
  * problem with one that can be directly optimized globally over the entire domain. Overall, 
  * regardless of the target of the simulation (whether it is feature-based or goal-oriented), 
  * this process reduces to finding an updated target frame-field at every point in the mesh.
  * This representation determines the target element axis at that point which describe the size,
  * anisotropy and orientation of the future cells. Due to the flexibility of the DG methods,
  * a polynomial distribution can also be used to modify the range of shape functions use for
  * each cell.
  * 
  * Once the continuous target mesh representation is determined, an update is obtained by performing
  * a call to an external mesh generation software capable of conforming to these measurements. This
  * is primarily done through GMSH for isotropic all-quad remeshing using the \f$L^\infty\f$ advancing front
  * for quads method, or in the anisotropic case using the BAMG mesh generator recombined via 
  * Blossom-Quad to form a final all-quad output mesh. Tools for writing to an experimental external
  * mesh generator based on \f$L_p\f$-CVT energy minimization to produce an anisotropic all-quad mesh
  * have also been included.
  * 
  * Note: While there are some placeholder functions have been included and certain functionality support
  *       polynomial distributions, \f$p\f$ and \f$hp\f$ adaptation have not been fully implemented or tested.
  */ 
#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinement_Continuous : public GridRefinementBase<dim,nstate,real,MeshType>
{
public:
    /// Deleted default constructor
    GridRefinement_Continuous() = delete;

    // overriding the other constructors to call delegated constructor for this class

    /// Constructor. Stores the adjoint object, physics and parameters
    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    /// Constructor. Storers the dg object, physics, functional and parameters.
    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input);

    /// Constructor. Stores the dg object, physics and parameters
    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    /// Constructor. Stores the dg object and parameters
    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                gr_param_input,
        // PHiLiP::Parameters::AllParameters const *const param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg_input);

protected:
    /// Delegated constructor which handles the various optional inputs and setup.
    GridRefinement_Continuous(
        PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >    adj_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input);

    using GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase;

    /// Perform call to the grid refinement object of choice
    /** This will automatically select the proper subclass, error indicator
      * and various refinement types based on the grid refinement parameters
      * passed at setup to the grid refinement factor class.
      * 
      * For continuous refinement, this function computes an error distribution
      * through calling the field function which is then used to set anisotropy 
      * targets and compute a global optimal size field during refine_grid_h
      * that matches the target complexity for the current iteration. 
      * 
      * See subclass functions for details of refinement types.
      */
    void refine_grid() override;

protected:

    // specified refinement functions for different cases

    /// Performs call to global remeshing method
    /** For continuous methods, this function requires interfacing with
      * an external mesh generation software to remesh the domain based
      * on the updated target continuous mesh model. Primarily, this is
      * handled through GMSH for isotropic \f$L^\infty\f$ advancing front for
      * quads or Blossom-Quad recombination of anistropic BAMG generated 
      * mesh. However, an additional interface to generate a .msh file 
      * suitable for use with an expermental Lp-CVT energy minimzation 
      * all-quad mesh generator is also included.
      */ 
    void refine_grid_h();

    /// Updates the global polynomial distribution based on the target \f$p\f$-field
    /** This requires a field generated by interpolation of the updated polynomial
      * distribution from previous error estimation. Note that this is not currently
      * implemented in field function due to a variety of code complications and is
      * included as a placehold
      */ 
    void refine_grid_p();

    /// (NOT IMPLEMENTED) Performs call to global remeshing method with updated polynomial distribution
    /** Placeholder function for inclusion of eventual \f$hp\f$ continuous mesh model
      * where both the continuous mesh representation and distribution of polynomial
      * orders are updated simulateneously at each iteration. 
      */ 
    void refine_grid_hp();    

    /// Output refinement method dependent results
    /** For the current class this includes current and target h and p fields,
      * the anisotropic ratio and set magnitude of \f$p+1\f$ directional derivatives 
      * used in the continuous error model
      */ 
    std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() override;

    // getting the size or tensor fields based on indicator

    /// Updates the continuous mesh and polynomial distribution
    /** Based on setup parameters, globally updates the target anisotropic mesh 
      * representation stored in h_field and polynomial distribution stored in 
      * p_field. Together these describe the continuous \f$hp\f$-mesh model. This 
      * is generated based on a variety of techniques using optimization of a 
      * corresponding continuous error model.
      */ 
    void field();

    // based on exact error function from manufactured solution

    /// Generate mesh distribution based on manufactured solution 
    /** For the \f$p=1\f$ case only, uses the hessian-based error estimate involving the
      * exact manufactured solution hessian to derive an adaptation metric from (known)
      * second derivatives of the solution at each point. This leads to a standard
      * hessian-based representation for the mesh generator which is expressed in
      * general high-order form but matches the definition of Loseille et. al 2010
      * "Fully anisotropic goal-oriented mesh adaptation for 3D steady Euler equations".
      * This leads to the error distribution:
      * 
      * \f[
      *     B(\boldsymbol{x}) = (\mathrm{tr}{(|H_u|)})^{\frac{Lq}{2}}
      * \f]
      * 
      */ 
    void field_h_error();
    /// (NOT IMPLEMENTED) Generate polynomial distribution based on manufactured solution
    void field_p_error();
    /// (NOT IMPLEMENTED) Generate mesh and polynomial distribution based on manufactured solution
    void field_hp_error();

    // based on hessian or feature-based error

    /// Generate mesh distribution based on \f$p+1\f$ directional derivatives
    /** This method is based on the high-order continuous error model of Dolejsi et al. 2018
      * "A continuous hp-mesh model for adaptive discontinuous galerkin Schemes".
      * Here it has been generalized to work based on the concept of "frame-fields"
      * where the target mesh for all-quad remeshing is represented by a set of coupled
      * vector fields. In this model, the local error is approximated through a quadratic
      * form. See the associated size_field.h function descriptions for details.
      * This leads to the error distribution:
      * 
      * \f[
      *     B(\boldsymbol{x},p) = 
      *     2^{\frac{q(p+1)+2}{2}} 
      *     \left(\frac{2 \pi}{q(p+1)+2}\right) 
      *     \left(A_1(\boldsymbol{x}, p) A_2(\boldsymbol{x}, p)\right)^{\frac{q}{2}}
      * \f]
      * 
      * where the constant is ignored and \f$A_1\f$ and \f$A_2\f$ are the
      * largest and orthogonal directional derivative components. 
      * The anisotropy and orientation are also used.
      */ 
    void field_h_hessian();
    /// (NOT IMPLEMENTED) Generate polynomial distribution based on \f$p+1\f$ directinal derivatives
    void field_p_hessian();
    /// (NOT IMPLEMENTED) Generate mesh and polynomial distribution based on \f$p+1\f$ directional derivatives
    void field_hp_hessian();

    // based on high-order solution residual

    /// (NOT IMPLEMENTED) Generate mesh distribution based on fine-grid residual distribution
    void field_h_residual();
    /// (NOT IMPLEMENTED) Generate polynomial distribution based on fine-grid residual distribution
    void field_p_residual();
    /// (NOT IMPLEMENTED) Generate mesh and polynomial distribution based on fine-grid residual distribution
    void field_hp_residual();

    // based on adjoint solution and dual-weighted residual

    /// Generate mesh distribution based on dual-weighted residual distribution
    /** For goal-oriented size-field adaptation, the updated mesh size is generated
      * by using the dual-weighted residual (DWR) which through the use of the adjoint solution
      * (local senstivity to the functional) it approximates the contribution of local errors
      * to the global functional evaluation error. Here the relative distribution of these
      * values is used to produce relative changes in the size of the local mesh. This has
      * been found to be more stable than directly optimizing on the distribution. The method
      * itself is based on the approach of Balan et al. 2016 "Adjoint-based hp-adaptivity on
      * anisotropic meshes..." and similarily uses hessian of the flow is used to determine
      * local mesh anisotropy here as well. See associated size_field.h functions for more details.
      * This is based on the DWR solved locally on each element from adjoint.h using the fine grid
      * adjoint and residual functions:
      * 
      * \f[
      *     \eta_i = |\sum_{\Omega_k \in \Omega_i} {(\psi_h)^T_k (\mathcal{R}_h(u_h^H))_k}|
      * \f]
      * 
      * where \f$\Omega_k\f$ are the fine grid elements and dofs obtained by enrichment of \f$\Omega_i\f$.
      */ 
    void field_h_adjoint();
    /// (NOT IMPLEMENTED) Generate polynomial distribution based on dual-weighted residual distribution
    void field_p_adjoint();
    /// (NOT IMPLEMENTED) Generate mesh and polynomial distribution based on dual-weighted residual distribution
    void field_hp_adjoint();
    
    // performing output to appropriate mesh generator

    /// Generates a new mesh based on GMSH using .pos and .geo files for i/o
    /** Here for isotropic remeshing, the advancing front for quads method is used
      * where a size field is specified from h_field. For the anisotropic case, BAMG 
      * is used to first generate an anisotropic tri mesh which is recombined using
      * Blossom-Quad. A .pos file is used to pass these field distribution of either 
      * scalar or tensor value information between the programs. For more information 
      * see gmsh_out.h. 
      * 
      * Note: Depending on the version and input to GMSH it may fail to fully recombine
      *       all elements. In this case Deal.II will fail due to the inclusion of other
      *       element types. Unfortunately, this will stop the run, but may sometimes be
      *       avoided by using slightly different initial values. 
      */ 
    void refine_grid_gmsh();

    /// Generates an output .msh file with matrix information about the target frame field
    /** For use with the external Lp-CVT mesh generator, this function will generate
      * a .msh file with associated matrix valued information on each element representing
      * the target mesh metric. See msh_out.h for more details on the output format. For the 
      * frame-field generation case this is written as the inverse linear transformation from 
      * the reference element to the physical element:
      * 
      * \f[
      *     f_\boldsymbol{x} 
      *     = \left< \boldsymbol{v}, \boldsymbol{w}, -\boldsymbol{v}, -\boldsymbol{w} \right>
      *     = V \left< \boldsymbol{e}_1, \boldsymbol{e}_2, -\boldsymbol{e}_1, -\boldsymbol{e}_2 \right>
      * \f]
      * 
      * Where \f$V(x)\f$ is the passed to the Lp-CVT mesh generator and used in the node movement energy
      * functional to help produce a well-aligned distribution for recombination to all-quad mesh:
      * 
      * \f[
      *     E_{L_p} (\boldsymbol{x}) 
      *     = \sum_{i} 
      *     \int_{\Omega_i \cap \Omega} 
      *         \left\lVert
      *         V(\boldsymbol{y})^{-1} (\boldsymbol{y}-\boldsymbol{x}_i)
      *         \right\rVert_p^p 
      *     \mathrm{d}\boldsymbol{y}
      * \f]
      * 
      * Note: After writing the target mesh description, the program will terminate as the interface is not yet fully automated.
      */ 
    void refine_grid_msh();

    // scheduling of complexity growth

    /// Evaluates the current complexity of the mesh
    /** The continuous complexity usually represents an integral approximation
      * of the degrees of freedom using the target size and polynomial distributions.
      * However, for the current mesh, this reduces to a simple sum of the degrees of 
      * freedom from each element (for the varying \f$p\f$ case):
      * 
      * \f[
      *     C = \Sum_{i} {(p_i+1)^2}
      * \f]
      * 
      * for the high-order quad mesh case.
      */
    real current_complexity();

    /// Updates the complexity target based on the current refinement iteration
    /** This is set based on the input complexity vector if there are entries availible,
      * otherwise, new values are added by scaling and adding to the previous last entry:
      * 
      * \f[
      *     C_{i+1} = \alpha \times C_{i} + \beta
      * \f]
      */ 
    void target_complexity();

    real              complexity_initial; ///< Initial mesh complexity at construction
    real              complexity_target;  ///< Current complexity target
    std::vector<real> complexity_vector;  ///< Vector of complexity target goals for each iteration

    /// Evaluates the mesh size and ansitropy description for the current mesh
    /** Here the scale is set based on the cell area measure. If the continuous 
      * grid refinement is anisotropic, the frame-field vector direction and scale
      * are determined based on the current chord (opposing face-center to face-center)
      * for each element direction.
      */ 
    void get_current_field_h();

    /// Evaluates the polynomial distribution for the current mesh
    /** Extracts the polynomial order based on the current select Finite Element
      * index chosen in each celll of the mesh.
      */
    void get_current_field_p();

    std::unique_ptr<Field<dim,real>> h_field; ///< Continuous representation of the mesh size and anisotropy distribution
    dealii::Vector<real>             p_field; ///< Continuous representation of the polynomial distribution
};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_CONTINUOUS_H__