#ifndef __PERIODIC_TURBULENCE_H__
#define __PERIODIC_TURBULENCE_H__

#include "periodic_cube_flow.h"
#include "dg/dg.h"
#include "physics/navier_stokes.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class PeriodicTurbulence : public PeriodicCubeFlow<dim,nstate>
{
    /** Number of different computed quantities
     *  Corresponds to the number of items in IntegratedQuantitiesEnum
     * */
    static const int NUMBER_OF_INTEGRATED_QUANTITIES = 4;

public:
    /// Constructor.
    PeriodicTurbulence(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~PeriodicTurbulence() {};
    
    /// Computes the integrated quantities over the domain simultaneously and updates the array storing them
    void compute_and_update_integrated_quantities(DGBase<dim, double> &dg);

    /** Gets the nondimensional integrated kinetic energy given a DG object from dg->solution
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_integrated_kinetic_energy() const;

    /** Gets the nondimensional integrated enstrophy given a DG object from dg->solution
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_integrated_enstrophy() const;

    /** Gets non-dimensional theoretical vorticity tensor based dissipation rate 
     *  Note: For incompressible flows or when dilatation effects are negligible 
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_vorticity_based_dissipation_rate() const;

    /** Evaluate non-dimensional theoretical pressure-dilatation dissipation rate
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_pressure_dilatation_based_dissipation_rate () const;

    /** Gets non-dimensional theoretical deviatoric strain-rate tensor based dissipation rate 
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_deviatoric_strain_rate_tensor_based_dissipation_rate() const;

protected:
    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    /// Pointer to Navier-Stokes physics object for computing things on the fly
    std::shared_ptr< Physics::NavierStokes<dim,dim+2,double> > navier_stokes_physics;

    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.
    bool is_viscous_flow = true; ///< Identifies if viscous flow; initialized as true.

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

    /// Function to compute the constant time step
    double get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

    /// List of possible integrated quantities over the domain
    enum IntegratedQuantitiesEnum {
        kinetic_energy,
        enstrophy,
        pressure_dilatation,
        deviatoric_strain_rate_tensor_magnitude_sqr
    };
    /// Array for storing the integrated quantities; done for computational efficiency
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrated_quantities;
};

} // Tests namespace
} // PHiLiP namespace
#endif