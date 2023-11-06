#ifndef PHILIP_DG_STRONG_LES_HPP
#define PHILIP_DG_STRONG_LES_HPP

#include "strong_dg.hpp"
#include "physics/large_eddy_simulation.h"

namespace PHiLiP {

/// DGStrongLES class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGStrongLES: public DGStrong<dim, nstate, real, MeshType>
{
protected:
    /// Alias to base class Triangulation.
    using Triangulation = typename DGStrong<dim,nstate,real,MeshType>::Triangulation;

public:
    /// Constructor
    DGStrongLES(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

    /// Destructor
    ~DGStrongLES();

    /// Contains the large eddy simulation object
    std::shared_ptr < Physics::LargeEddySimulationBase<dim, nstate, real > > pde_model_les_double;

    /// Allocate the necessary variables declared in src/physics/model.h
    virtual void allocate_model_variables() override;

    /// Update the necessary variables declared in src/physics/model.h
    void update_model_variables() override;

protected:
    /// Update the cellwise volume and polynomial degree
    void update_cellwise_volume_and_poly_degree();

    /// Update the cellwise mean quantities
    virtual void update_cellwise_mean_quantities();

    // const bool do_compute_filtered_solution; ///< Flag to compute the filtered solution
    // const bool apply_modal_high_pass_filter_on_filtered_solution; ///< Flag to apply modal high pass filter on the filtered solution
    // const unsigned int poly_degree_max_large_scales; ///< For filtered solution; lower bound of high pass filter

    using DGBase<dim,real,MeshType>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    
}; // end of DGStrongLES class

/// DGStrongLES_ShearImproved class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGStrongLES_ShearImproved: public DGStrongLES<dim, nstate, real, MeshType>
{
protected:
    /// Alias to base class Triangulation.
    using Triangulation = typename DGStrongLES<dim,nstate,real,MeshType>::Triangulation;

public:
    /// Constructor
    DGStrongLES_ShearImproved(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

    /// Destructor
    ~DGStrongLES_ShearImproved();

    /// Allocate the necessary variables declared in src/physics/model.h
    void allocate_model_variables() override;

protected:
    /// Update the cellwise mean quantities
    void update_cellwise_mean_quantities() override;

    // const bool do_compute_filtered_solution; ///< Flag to compute the filtered solution
    // const bool apply_modal_high_pass_filter_on_filtered_solution; ///< Flag to apply modal high pass filter on the filtered solution
    // const unsigned int poly_degree_max_large_scales; ///< For filtered solution; lower bound of high pass filter

    using DGBase<dim,real,MeshType>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    
}; // end of DGStrongLES_ShearImproved class

/// DGStrongLES_ShearImproved class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGStrong_ChannelFlow: public DGStrong<dim, nstate, real, MeshType>
{
protected:
    /// Alias to base class Triangulation.
    using Triangulation = typename DGStrongLES<dim,nstate,real,MeshType>::Triangulation;

public:
    /// Constructor
    DGStrong_ChannelFlow(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

    /// Destructor
    ~DGStrong_ChannelFlow();

protected:
    // TO DO: reduce these
    const double channel_height; ///< Channel height
    const double half_channel_height; ///< Half channel height
    const double channel_friction_velocity_reynolds_number; ///< Channel Reynolds number based on wall friction velocity
    const int number_of_cells_x_direction; ///< Number of cells in x-direction
    const int number_of_cells_y_direction; ///< Number of cells in y-direction
    const int number_of_cells_z_direction; ///< Number of cells in z-direction
    const double pi_val; ///< Value of pi
    const double domain_length_x; ///< Domain length in x-direction
    const double domain_length_y; ///< Domain length in y-direction
    const double domain_length_z; ///< Domain length in z-direction
    const double domain_volume; ///< Domain volume

    /** 
     * Bulk velocity Reynolds number computed from friction velocity based Reynolds numbers (Empirical relation)
     * Reference:
     *  - R. B. Dean, "Reynolds Number Dependence of Skin Friction and Other Bulk
     *    Flow Variables in Two-Dimensional Rectangular Duct Flow", 
     *    Journal of Fluids Engineering, 1978 
     * */
    const double channel_bulk_velocity_reynolds_number;

    /** 
     * Centerline velocity Reynolds number computed from friction velocity based Reynolds numbers (Empirical relation)
     * Reference:
     *  - R. B. Dean, "Reynolds Number Dependence of Skin Friction and Other Bulk
     *    Flow Variables in Two-Dimensional Rectangular Duct Flow", 
     *    Journal of Fluids Engineering, 1978 
     * */
    const double channel_centerline_velocity_reynolds_number;

public:
    /// Contains the large eddy simulation object
    std::shared_ptr < Physics::LargeEddySimulationBase<dim, nstate, real > > pde_model_les_double;

    /// Allocate the necessary variables declared in src/physics/model.h
    void allocate_model_variables() override;

    /// Update the necessary variables declared in src/physics/model.h
    void update_model_variables() override;

protected:

    using DGBase<dim,real,MeshType>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    
}; // end of DGStrong_ChannelFlow class

} // PHiLiP namespace

#endif
