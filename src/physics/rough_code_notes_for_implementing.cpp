template <int dim, int nstate, typename real>
void NavierStokes<dim,nstate,real>
::boundary_wall (
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // The given
}


/// Evaluates boundary values and gradients on the other side of the face.
void boundary_face_values (
    const int /*boundary_type*/,
    const dealii::Point<dim, real> &/*pos*/,
    const dealii::Tensor<1,dim,real> &/*normal*/,
    const std::array<real,nstate> &/*soln_int*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
    std::array<real,nstate> &/*soln_bc*/,
    std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
    // from PhysicsBase
    // Turbulent velocity profile using Reichart's law of the wall
    // -- apply initial condition symmetrically w.r.t. the top/bottom walls of the channel
    const real dist_from_wall = this->get_distance_from_wall(point);

    // Get the nondimensional (w.r.t. freestream) friction velocity
    const real viscosity_coefficient = this->navier_stokes_physics.compute_viscosity_coefficient_from_temperature(temperature);
    const real friction_velocity = viscosity_coefficient*this->channel_friction_velocity_reynolds_number/(density*this->half_channel_height*this->navier_stokes_physics.reynolds_number_inf);

    
    // Reichardt law of the wall (provides a smoothing between the linear and the log regions)
    // References: 
    /*  Frere, Carton de Wiart, Hillewaert, Chatelain, and Winckelmans 
        "Application of wall-models to discontinuous Galerkin LES", Phys. Fluids 29, 2017

        (Original paper) J. M.  Osterlund, A. V. Johansson, H. M. Nagib, and M. H. Hites, “A note
        on the overlap region in turbulent boundary layers,” Phys. Fluids 12, 1–4, (2000).
    */
    const real kappa = 0.38; // von Karman's constant
    const real C = 4.1;
    const real y_plus = density*friction_velocity*dist_from_wall/viscosity_coefficient;
    const real u_plus = (1.0/kappa)*log(1.0+kappa*y_plus) + (C - (1.0/kappa)*log(kappa))*(1.0 - exp(-y_plus/11.0) - (y_plus/11.0)*exp(-y_plus/3.0));
    const real x_velocity = u_plus*friction_velocity;
}


const real kappa = 0.38; // von Karman's constant
const real C = 4.1;
const real y_plus = density*friction_velocity*dist_from_wall/viscosity_coefficient;
const real u_plus = (1.0/kappa)*log(1.0+kappa*y_plus) + (C - (1.0/kappa)*log(kappa))*(1.0 - exp(-y_plus/11.0) - (y_plus/11.0)*exp(-y_plus/3.0));
const real x_velocity = u_plus*friction_velocity;



double GetVibTemp(double *Qvec, double *Sp_Density, double r0)
{
    /* Newton's method for root finding */
    register double r1 = r0;
    register double err = 1.1;
    register double dr;
    register int i = 0;
    register int maxIter = 5000;
    
    while(err > RootFindingTolerance_Vib)
    {
        r0 = r1;
        dr = f_GetVibTemp_Newton(Qvec, Sp_Density, r0)/df_GetVibTemp_Newton(Sp_Density, r0);
        r1 = r0 - dr;
        err = abs(r1-r0);
        i += 1;
        
        // if(i%100 == 0)
        // {
        //  cout << "\t i = " << i << "\t error = " << err << endl;
        // }

        if(i>maxIter)
        {
            cout << "ERROR: GetVibTemp() reached maximum number of iterations." << endl;
            break;  
        }
    }
    return r1;
}

