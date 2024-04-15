#ifndef __PHILIP_EXTRACTION_FUNCTIONAL_H__
#define __PHILIP_EXTRACTION_FUNCTIONAL_H__

#include "functional.h"
#include "physics/model.h"
#include "physics/reynolds_averaged_navier_stokes.h"
#include "physics/negative_spalart_allmaras_rans_model.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class ExtractionFunctional : public Functional<dim, nstate, real, MeshType>
{
public:
    /// @brief Switch between displacement and momentum thickness.
    enum Functional_types { displacement_thickness, momentum_thickness };
private:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real,MeshType>::evaluate_volume_integrand;

public:
    /// @brief Extraction start point on solid boundary surface.
    dealii::Point<dim,real> start_point;

    /// @brief Normal vector (points into fluid domain) on the solid boundary surface at extraction start point.
    dealii::Tensor<1,dim,real> start_point_normal_vector;

    /// @brief Tangential vector (points to flow direction) on the solid boundary surface at extraction start point. 
    dealii::Tensor<1,dim,real> start_point_tangential_vector;

    /// @brief Interpolated solutions at extraction start point.
    //std::array<real,nstate> soln_at_start_point;

    /// @brief Interpolated solution gradients at extraction start point.
    //std::array<dealii::Tensor<1,dim,real>,nstate> soln_grad_at_start_point;

    /// @brief Extraction end point. 
    dealii::Point<dim,real> end_point;

    /// @brief Interpolated solutions at extraction end point.
    //std::array<real,nstate> soln_at_end_point;

    /// @brief Interpolated solution gradients at extraction end point.
    //std::array<dealii::Tensor<1,dim,real>,nstate> soln_grad_at_end_point;

    /// @brief Number of sampling points over the extraction line.
    const int number_of_sampling;

    /// @brief Number of total sampling points over the extraction line, including start and end points.
    const int number_of_total_sampling;

    /// @brief Coordinate of sampling points over extraction line.
    //std::vector<dealii::Point<dim,real>> coord_of_sampling;

    /// @brief Interpolated solutions at each sampling points over extraction line.
    //std::vector<std::array<real,nstate>> soln_of_sampling;

    /// @brief Interpolated solution gradients at each sampling points over extraction line.
    //std::vector<std::array<dealii::Tensor<1,dim,real>,nstate>> soln_grad_of_sampling;

    /// @brief Length of extraction line 
    real length_of_sampling;

    /// @brief Converged speed of free-stream flow over extraction line.
    //real U_inf;

    /// @brief Converged density of free-stream flow over extraction line.
    //real density_inf;

    /// @brief Navier-Stokes physics reference.
    const Physics::NavierStokes<dim,dim+2,FadType> &navier_stokes_fad;

    /// @brief Reynolds-averaged Navier-Stokes negative SA model pointer.
    std::shared_ptr <Physics::ReynoldsAveragedNavierStokes_SAneg<dim,dim+3,FadType>> rans_sa_neg_fad;

public:
    /// Constructor
    ExtractionFunctional(
        std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
        const dealii::Point<dim,real> start_point_input,
        const int number_of_sampling);
    /// Destructor
    ~ExtractionFunctional(){};

    real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

public:
    //std::vector<dealii::Point<dim,real>> evaluate_straight_line_sampling_point_coord();
    
    /// Function to interpolate solutions and solution gradients on sampling points over extraction line.
    //void evaluate_straight_line_sampling_point_soln();

    /// Function to interpolate solutions and solution gradients on extraction start and end points.
    //void evaluate_start_end_point_soln();

    void evaluate_extraction_start_point_coord();

    void evaluate_extraction_start_point_normal_tangential_vector();

    void evaluate_extraction_end_point_coord();

    std::vector<dealii::Point<dim,real>> evaluate_straight_line_sampling_point_coord();

    std::vector<dealii::Point<dim,real>> evaluate_straight_line_total_sampling_point_coord();

    std::vector<std::pair<dealii::DoFHandler<dim>::active_cell_iterator,dealii::Point<dim,real>>> find_active_cell_around_points(
        const dealii::hp::MappingCollection<dim> mapping_collection,
        const dealii::DoFHandler<dim> dof_handler,
        const std::vector<dealii::Point<dim,real>> coord_of_total_sampling) const;

    template <typename real2>
    std::array<real2,nstate> point_value(
        const dealii::Point<dim,real> &coord_of_sampling,
        const dealii::hp::MappingCollection<dim> &mapping_collection,
        const dealii::hp::FECollection<dim> &fe_collection,
        const std::pair<dealii::DoFHandler<dim>::active_cell_iterator,dealii::Point<dim,real>> &cell_index_and_ref_point_of_sampling,
        const std::vector<real2> &soln_coeff,
        const std::vector<dealii::types::global_dof_index> &cell_soln_dofs_indices) const;

    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> point_gradient(
        const dealii::Point<dim,real> &coord_of_sampling,
        const dealii::hp::MappingCollection<dim> &mapping_collection,
        const dealii::hp::FECollection<dim> &fe_collection,
        const std::pair<dealii::DoFHandler<dim>::active_cell_iterator,dealii::Point<dim,real>> &cell_index_and_ref_point_of_sampling,
        const std::vector<real2> &soln_coeff,
        const std::vector<dealii::types::global_dof_index> &cell_soln_dofs_indices) const;

    //template <typename real2>
    //std::vector<std::array<real2,nstate>> evaluate_straight_line_sampling_point_soln(
    //    const std::vector<dealii::Point<dim,real>> &coord_of_sampling,
    //    const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector) const;

    //template <typename real2>
    //std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> evaluate_straight_line_sampling_point_soln_grad(
    //    const std::vector<dealii::Point<dim,real>> &coord_of_sampling,
    //    const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector) const;

    //template <typename real2>
    //std::array<real2,nstate> evaluate_start_point_soln(
    //    const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector) const;

    //template <typename real2>
    //std::array<real2,nstate> evaluate_end_point_soln(
    //    const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector) const;

    //template <typename real2>
    //std::array<dealii::Tensor<1,dim,real2>,nstate> evaluate_start_point_soln_grad(
    //    const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector) const;

    //template <typename real2>
    //std::array<dealii::Tensor<1,dim,real2>,nstate> evaluate_end_point_soln_grad(
    //    const dealii::LinearAlgebra::distributed::Vector<real2> &solution_vector) const;

    /// Function to evaluate straight line integral.
    template<typename real2>
    real2 evaluate_straight_line_integral(
        const Functional_types &functional_type,
        const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const;

    /// Function to evaluate integrand of straight line integral.
    template<typename real2>
    real2 evaluate_straight_line_integrand(
        const Functional_types &functional_type,
        const std::pair<real,real> &values_free_stream,
        const dealii::Tensor<1,dim,real> &tangential,
        const std::array<real2,nstate> &soln_at_q) const;

    /// Function to evaluate integrand for displacement thickness.
    template<typename real2>
    real2 evaluate_displacement_thickness_integrand(
        const std::pair<real,real> &values_free_stream,
        const dealii::Tensor<1,dim,real> &tangential,
        const std::array<real2,nstate> &soln_at_q) const;

    /// Function to evaluate integrand for momentum thickness.
    template<typename real2>
    real2 evaluate_momentum_thickness_integrand(
        const std::pair<real,real> &values_free_stream,
        const dealii::Tensor<1,dim,real> &tangential,
        const std::array<real2,nstate> &soln_at_q) const;

    /// Function to evaluate kinematic viscosity.
    template<typename real2>
    real2 evaluate_kinematic_viscosity(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const;

    /// Function to evaluate displacement thickness.
    template<typename real2>
    real2 evaluate_displacement_thickness(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const;

    /// Function to evaluate momentum thickness.
    template<typename real2>
    real2 evaluate_momentum_thickness(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const;

    /// Function to evaluate edge velocity of boundary layer.
    template<typename real2>
    real evaluate_edge_velocity(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const;

    /// Function to evaluate shear stress at extraction location.
    template<typename real2>
    real2 evaluate_shear_stress(
        const std::array<real2,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_grad_at_q) const;

    /// Function to evaluate wall shear stress at extraction location.
    template<typename real2>
    real2 evaluate_wall_shear_stress(
        const std::vector<std::array<real2,nstate>> &soln_of_total_sampling,
        const std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> &soln_grad_of_total_sampling) const;

    /// Function to evaluate maximum shear stress over extraction line.
    template<typename real2>
    real evaluate_maximum_shear_stress(
        const std::vector<std::array<real2,nstate>> &soln_of_total_sampling,
        const std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> &soln_grad_of_total_sampling) const;

    /// Function to evaluate friction velocity at extraction location.
    template<typename real2>
    real2 evaluate_friction_velocity(
        const std::vector<std::array<real2,nstate>> &soln_of_total_sampling,
        const std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> &soln_grad_of_total_sampling) const;

    /// Function to evaluate boundary layer thickness at extraction location.
    template<typename real2>
    real evaluate_boundary_layer_thickness(
        const std::vector<dealii::Point<dim,real>> &coord_of_total_sampling,
        const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const;

    /// Function to evaluate pressure gradient over tangential direction on solid body at extraction location.
    template<typename real2>
    real2 evaluate_pressure_gradient_tangential(
        const std::vector<std::array<real2,nstate>> &soln_of_total_sampling,
        const std::vector<std::array<dealii::Tensor<1,dim,real2>,nstate>> &soln_grad_of_total_sampling) const;

    /// Function to evaluate converged free-stream values, i.e. free-stream velocity and density, over extraction line.
    template<typename real2>
    std::pair<real,real> evaluate_converged_free_stream_values(const std::vector<std::array<real2,nstate>> &soln_of_total_sampling) const;
};

} // PHiLiP namespace

#endif