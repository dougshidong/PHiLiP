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

// real get_wall_parallel_velocity (
//     const dealii::Tensor<1,dim,real> &/*normal*/,
//     const std::array<real,nstate>    &conservative_soln)
// {
//     // Currently only works for channel flow case
//     // Make use of the normal vector in the future
//     const dealii::Tensor<1,dim,real> vel = this->navier_stokes_physics.compute_velocities(conservative_soln); 
//     return vel[0]; // x-velocity for turbulent channel case
//     // TO DO: actually, this can be part of navier-stokes
//     // there is already a code snippet for it there in the 
//     // compute_wall_shear_stress formula
// }

// real get_velocity_gradients_from_wall_shear_stress()
// {
//     real2 velocity_gradient_of_parallel_velocity_in_the_direction_normal_to_wall = 0.0;

// }

template <int dim, int nstate, typename real>
inline real WallModel_TurbulentChannelFlow<dim, nstate, real>
::get_distance_from_wall(const dealii::Point<dim,real> &point,
                         const dealii::Tensor<1,dim,real> &/*normal*/) const
{   // TO DO: Move this function somewhere or create a factory of wall distance calculators? 
    // or take advantage of the parameters object but still create a wall dist factory
    // based on the flow case; there are constants in here that need to be initialized

    // Get closest wall normal distance
    real y = point[1]; // y-coordinate of position
    real dist_from_wall = 0.0; // represents distance normal to top/bottom wall (which ever is closer); y-domain bounds are [-half_channel_height, half_channel_height]
    if(y > 0.0){
        // top wall
        dist_from_wall = half_channel_height - y; // distance from top wall
    } else if(y < 0.0) {
        dist_from_wall = y - half_channel_height; // distance from bottom wall
    }
    return dist_from_wall;
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
    // (1) Get the friction velocity
    // Get the nondimensional (w.r.t. freestream) friction velocity
    const real viscosity_coefficient = this->navier_stokes_physics.constant_viscosity;
    const real wall_distance = wall_geometry.get_distance_from_wall(point,normal); // TO DO: Update this line
    const real density = soln_int[0];
    const real wall_parallel_velocity = this->get_wall_parallel_velocity(normal,soln_int);
    const real friction_velocity = get_friction_velocity(density,
                                                         wall_distance,
                                                         wall_parallel_velocity,
                                                         initial_guess_for_friction_velocity)
    // (2) Get wall shear stress
    const real wall_shear_stress = density*friction_velocity*friction_velocity;
    const real gradient_of_parallel_velocity_in_direction_normal_to_wall 
    = wall_shear_stress/viscosity_coefficient;

    // Now look at navier stokes how to apply the rest of the BC
    // use the gradient_of_parallel_velocity_in_direction_normal_to_wall 
    // to assign the value to that vel grad component then convert 
    // the primitive gradients to conservative using navier stokes
}


// const real kappa = 0.38; // von Karman's constant
// const real C = 4.1;
// const real y_plus = density*friction_velocity*dist_from_wall/viscosity_coefficient;
// const real u_plus = (1.0/kappa)*log(1.0+kappa*y_plus) + (C - (1.0/kappa)*log(kappa))*(1.0 - exp(-y_plus/11.0) - (y_plus/11.0)*exp(-y_plus/3.0));
// const real x_velocity = u_plus*friction_velocity;

real dimensionless_velocity_parallel_to_wall(const real density,
            const real viscosity_coefficient, 
            const real wall_distance, 
            const real friction_velocity) const 
{
    // Reichardt law of the wall (provides a smoothing between the linear and the log regions)
    // References: 
    /*  Frere, Carton de Wiart, Hillewaert, Chatelain, and Winckelmans 
        "Application of wall-models to discontinuous Galerkin LES", Phys. Fluids 29, 2017

        (Original paper) J. M.  Osterlund, A. V. Johansson, H. M. Nagib, and M. H. Hites, “A note
        on the overlap region in turbulent boundary layers,” Phys. Fluids 12, 1–4, (2000).
    */

    // expression for u^+
    // returns the dimensionless mean streamwise velocity parallel to the wall
    const real kappa = 0.38; // von Karman's constant
    const real C = 4.1;
    const real y_plus = density*friction_velocity*wall_distance/viscosity_coefficient;
    const real u_plus = (1.0/kappa)*log(1.0+kappa*y_plus) + (C - (1.0/kappa)*log(kappa))*(1.0 - exp(-y_plus/11.0) - (y_plus/11.0)*exp(-y_plus/3.0));
    return u_plus;
}

real f_root_finding(const real density,
                    const real viscosity_coefficient, 
                    const real wall_distance, 
                    const real wall_parallel_velocity, 
                    const real friction_velocity) const
{
    const real u_plus = dimensionless_velocity_parallel_to_wall(density,
                                                                viscosity_coefficient,
                                                                wall_distance,
                                                                friction_velocity);
    const real f = (wall_parallel_velocity/friction_velocity)-u_plus;
    return f;
}

real df_root_finding(const real density,
                     const real viscosity_coefficient, 
                     const real wall_distance, 
                     const real wall_parallel_velocity, 
                     const real friction_velocity) const
{
    // derivative wrt to friction_velocity
    real df = -(wall_parallel_velocity/(friction_velocity*friction_velocity));
    y_plus = density*friction_velocity*wall_distance/viscosity_coefficient;
    dy_plus = density*wall_distance/viscosity_coefficient;
    df -= (1.0/(1.0+kappa*y_plus))*dy_plus
            +(C - (1.0/kappa)*log(kappa))*(
                (1.0/11.0)*exp(-y_plus/11.0)*dy_plus
                + (-1.0/11.0)*dy_plus*exp(-y_plus/3.0)
                + (y_plus/33.0)*exp(-y_plus/3.0)*dy_plus
                );
    return df;
}


double get_friction_velocity(const real density,
                             const real viscosity_coefficient, 
                             const real wall_distance, 
                             const real wall_parallel_velocity,
                             const real initial_guess_for_friction_velocity) const
{
    /* Newton's method for root finding */
    double r1 = initial_guess_for_friction_velocity;
    double err = 1.1;
    double dr;
    int i = 0;
    int maxIter = 5000;
    const double tolerance = 1.0e-12;
    
    while(err > tolerance)
    {
        r0 = r1;
        dr = this->f_root_finding(density,
                            viscosity_coefficient,
                            wall_distance,
                            r0)/this->df_root_finding(density,
                                                viscosity_coefficient,
                                                wall_distance,
                                                wall_parallel_velocity,
                                                r0);
        r1 = r0 - dr;
        err = abs(r1-r0);
        i += 1;
        
        // if(i%100 == 0)
        // {
        //  cout << "\t i = " << i << "\t error = " << err << endl;
        // }

        if(i>maxIter)
        {
            pcout << "ERROR: get_friction_velocity() reached maximum number of iterations..." << std::endl;
            pcout << "Aborting." << std::endl;
            std::abort();
        }
    }
    return r1;
}

