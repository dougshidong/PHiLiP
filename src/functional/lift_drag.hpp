#ifndef __PHILIP_LIFT_DRAG_H__
#define __PHILIP_LIFT_DRAG_H__

#include "functional.h"
#include "parameters/all_parameters.h"
#include "physics/physics_factory.h"
#include "physics/navier_stokes.h"

namespace PHiLiP {

/** Target boundary values.
 *  Simply zero out the default volume contribution.
 */
#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class LiftDragFunctional : public Functional<dim, nstate, real, MeshType>
{
public:
    /// @brief Switch between lift and drag functional types.
    enum Functional_types { lift, drag };
private:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real,MeshType>::evaluate_volume_integrand;

    /// @brief Switches between lift and drag.
    const Functional_types functional_type;

    /// @brief Casts DG's physics into an Euler physics reference.
    const Physics::Euler<dim,dim+2,FadFadType> &euler_fad_fad;
    /// @brief Angle of attack retrieved from euler_fad_fad.
    const double angle_of_attack;
    /// @brief Rotation matrix based on angle of attack.
    const dealii::Tensor<2,dim,double> rotation_matrix;
    /// @brief Lift force scaling based on the rotation matrix applied on a [0 1]^T vector.
    /// Assumes that the lift is the force in the positive y-direction.
    const dealii::Tensor<1,dim,double> lift_vector;
    /// @brief Drag force scaling based on the rotation matrix applied on a [1 0]^T vector.
    /// Assumes that the drag is the force in the positive x-direction.
    const dealii::Tensor<1,dim,double> drag_vector;

    /// Used force scaling vector depending whether this functional represents lift or drag.
    dealii::Tensor<1,dim,double> force_vector;

    /// Pressure induced drag is given by
    /**
     *  \f[
     *      C_D = \frac{2}{L \rho_{\infty} V_{\infty}^2} \int_{S} p (\mathbf{n} \cdot \text{DragVector}) ds
     *  \f]
     *  \f[
     *      C_D = \text{force_dimensionalization_factor} \int_{S} p (\mathbf{n} \cdot \text{DragVector}) ds
     *  \f]
     *
     */
    const double force_dimensionalization_factor;

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Compute force dimensionalization factor.
    double initialize_force_dimensionalization_factor();

    /// Initialize rotation matrix based on given angle of attack.
    dealii::Tensor<2,dim,double> initialize_rotation_matrix(const double angle_of_attack);

    /// Initialize lift vector with given rotation matrix based on angle of attack.
    /** Use convention that lift is in the positive y-direction.
     */
    dealii::Tensor<1,dim,double> initialize_lift_vector(const dealii::Tensor<2,dim,double> &rotation_matrix);

    /// Initialize drag vector with given rotation matrix based on angle of attack.
    /** Use convention that drag is in the positive x-direction.
     */
    dealii::Tensor<1,dim,double> initialize_drag_vector(const dealii::Tensor<2,dim,double> &rotation_matrix);

public:
    /// Constructor
    LiftDragFunctional(
        std::shared_ptr<DGBase<dim,real,MeshType>> dg_input,
        const Functional_types functional_type);
    /// Destructor
    ~LiftDragFunctional(){};

    real evaluate_functional( const bool compute_dIdW = false, const bool compute_dIdX = false, const bool compute_d2I = false) override;

public:
    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    template<typename real2>
    real2 evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/*physics*/,
        const unsigned int boundary_id,
        const dealii::Point<dim,real2> &/*phys_coord*/,
        const dealii::Tensor<1,dim,real2> &normal,
        const std::array<real2,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_grad_at_q) const
    {
        if (boundary_id == 1001) {
            assert(soln_at_q.size() == dim+2);

            /// Pointer to Navier-Stokes physics object
            using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
            std::shared_ptr< Physics::NavierStokes<dim,dim+2,real2> > navier_stokes_physics = dynamic_pointer_cast<Physics::NavierStokes<dim,dim+2,real2>> (Physics::PhysicsFactory<dim,dim+2,real2>::create_Physics(this->all_parameters, PDE_enum::navier_stokes, nullptr));

            // Compute pressure (same as Euler physics)
            const real2 pressure = navier_stokes_physics->compute_pressure (soln_at_q);
			
            // Initialize
			dealii::Tensor<1,dim,real2> viscous_tensor_times_normal;
			for(int i=0; i<dim; i++){
				viscous_tensor_times_normal[i] = 0;
			}
            // add viscous stress tensor contribution if viscous (i.e. not Euler)
            if(this->all_parameters->pde_type != PDE_enum::euler) {
                // Compute viscous stress tensor
                const dealii::Tensor<2,dim,real2> viscous_stress_tensor = navier_stokes_physics->compute_viscous_stress_tensor_from_conservative_templated(soln_at_q, soln_grad_at_q);
                // std::cout<<"Norm of viscous stress tensor = "<<  viscous_stress_tensor[0][0]<<std::endl;
                for (int i=0;i<dim;i++){
                    for (int j=0;j<dim;j++){
                        viscous_tensor_times_normal[i]+= viscous_stress_tensor[i][j]*normal[j];
                    }
                }
            }

		    //std::cout << " force_dimensionalization_factor: " << force_dimensionalization_factor
            //          << " pressure: " << pressure
            //          << " normal*force_vector: " << normal*force_vector
            //          << std::endl;

            return force_dimensionalization_factor * (pressure * (normal * force_vector) -  viscous_tensor_times_normal*force_vector);
        } 
        return (real2) 0.0;
    }

    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,real> &phys_coord,
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_at_q) const override
    {
        return evaluate_boundary_integrand<real>(
            physics,
            boundary_id,
            phys_coord,
            normal,
            soln_at_q,
            soln_grad_at_q);
    }

    /// Virtual function for Sacado computation of cell boundary functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual FadFadType evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,FadFadType> &phys_coord,
        const dealii::Tensor<1,dim,FadFadType> &normal,
        const std::array<FadFadType,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &soln_grad_at_q) const override
    {
        return evaluate_boundary_integrand<FadFadType>(
            physics,
            boundary_id,
            phys_coord,
            normal,
            soln_at_q,
            soln_grad_at_q);
    }

    /// Virtual function for computation of cell volume functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    virtual real evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &/*physics*/,
        const dealii::Point<dim,real> &/*phys_coord*/,
        const std::array<real,nstate> &/*soln_at_q*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_at_q*/) const
    { return (real) 0.0; }

    /// Virtual function for Sacado computation of cell volume functional term and derivatives
    /** Used only in the computation of evaluate_dIdw(). If not overriden returns 0. */
    virtual FadFadType evaluate_volume_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,FadFadType> &/*physics*/,
        const dealii::Point<dim,FadFadType> &/*phys_coord*/, const std::array<FadFadType,nstate> &/*soln_at_q*/,
        const std::array<dealii::Tensor<1,dim,FadFadType>,nstate> &/*soln_grad_at_q*/) const
    { return (FadFadType) 0.0; }


};

// template <int dim, int nstate, typename real>
// class TargetLiftDragFunctional : public LiftDragFunctional<dim,nstate,real>
// {
// private:
//     /// Constructor
//     TargetLiftDragFunctional(
//         std::shared_ptr<DGBase<dim,real>> dg_input,
//         const Functional_types functional_type
//         const double target_value = -1e200
//         : LiftDragFunctional(dg_input, functional_type)
//         , target_value(target_value)
//     { }
// 
// 
//     real evaluate_functional(
//         const bool compute_dIdW,
//         const bool compute_dIdX,
//         const bool compute_d2I)
//     {
//         real value = LiftDragFunctional<dim,nstate,real>::evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
// 
//         return value - target_value
//     }
// 
// };
// 
// template <int dim, int nstate, typename real>
// class QuadraticPenaltyTargetLiftDragFunctional : public TargetLiftDragFunctional<dim,nstate,real>
// {
// public:
// 
//     double penalty;
// 
//     /// Constructor
//     QuadraticPenaltyTargetLiftDragFunctional(
//         std::shared_ptr<DGBase<dim,real>> dg_input,
//         const Functional_types functional_type
//         const double target_value = -1e200
//         const double penalty = 0
//         : TargetLiftDragFunctional(dg_input, functional_type, target_value)
//         , penalty(penalty)
//     { }
// 
// 
//     real evaluate_functional(
//         const bool compute_dIdW,
//         const bool compute_dIdX,
//         const bool compute_d2I)
//     {
//         real value = TargetLiftDragFunctional<dim,nstate,real>::evaluate_functional((compute_dIdW || compute_d2I), (compute_dIdX || compute_d2I), compute_d2I);
// 
//         if (compute_dIdW) {
//             const real scaling = 2.0*value;
//             this->dIdw *= scaling;
//         }
// 
//         if (compute_dIdX) {
//             const real scaling = 2.0*value;
//             this->dIdX *= scaling;
//         }
// 
//         if (compute_d2I) {
//             const real scaling = 2.0*value;
//             this->dIdX *= scaling;
//         }
// 
//         return penalty * value * value;
//     }
// 
// };

} // PHiLiP namespace

#endif
