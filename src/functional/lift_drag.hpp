#ifndef __PHILIP_LIFT_DRAG_H__
#define __PHILIP_LIFT_DRAG_H__

#include "functional.h"

namespace PHiLiP {

/** Target boundary values.
 *  Simply zero out the default volume contribution.
 */
template <int dim, int nstate, typename real>
class LiftDragFunctional : public Functional<dim, nstate, real>
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
    using Functional<dim,nstate,real>::evaluate_volume_integrand;

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

    /// Compute force dimensionalization factor.
    double initialize_force_dimensionalization_factor()
    {
        const double ref_length  = euler_fad_fad.ref_length;
        const double dynamic_pressure_inf  = euler_fad_fad.dynamic_pressure_inf;

        return 1.0 / (ref_length * dynamic_pressure_inf);
    }

    /// Initialize rotation matrix based on given angle of attack.
    dealii::Tensor<2,dim,double> initialize_rotation_matrix (const double angle_of_attack)
    {
        dealii::Tensor<2,dim,double> rotation_matrix;
        if constexpr (dim == 1) {
            assert(false);
        }

        rotation_matrix[0][0] = cos(angle_of_attack);
        rotation_matrix[0][1] = -sin(angle_of_attack);
        rotation_matrix[1][0] = sin(angle_of_attack);
        rotation_matrix[1][1] = cos(angle_of_attack);

        if constexpr (dim == 3) {
            rotation_matrix[0][2] = 0.0;
            rotation_matrix[1][2] = 0.0;

            rotation_matrix[2][0] = 0.0;
            rotation_matrix[2][1] = 0.0;
            rotation_matrix[2][2] = 1.0;
        }

        return rotation_matrix;
    }

    /// Initialize lift vector with given rotation matrix based on angle of attack.
    /** Use convention that lift is in the positive y-direction.
     */
    dealii::Tensor<1,dim,double> initialize_lift_vector (const dealii::Tensor<2,dim,double> &rotation_matrix)
    {
        dealii::Tensor<1,dim,double> lift_direction;
        lift_direction[0] = 0.0;
        lift_direction[1] = 1.0;

        if constexpr (dim == 1) {
            assert(false);
        }
        if constexpr (dim == 3) {
            lift_direction[2] = 0.0;
        }

        dealii::Tensor<1,dim,double> vec;
        vec = rotation_matrix * lift_direction;

        return vec;
    }

    /// Initialize drag vector with given rotation matrix based on angle of attack.
    /** Use convention that drag is in the positive x-direction.
     */
    dealii::Tensor<1,dim,double> initialize_drag_vector (const dealii::Tensor<2,dim,double> &rotation_matrix)
    {
        dealii::Tensor<1,dim,double> drag_direction;
        drag_direction[0] = 1.0;
        drag_direction[1] = 0.0;

        if constexpr (dim == 1) {
            assert(false);
        }
        if constexpr (dim == 3) {
            drag_direction[2] = 0.0;
        }

        dealii::Tensor<1,dim,double> vec;
        vec = rotation_matrix * drag_direction;

        return vec;
    }
   

public:
    /// Constructor
    LiftDragFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const Functional_types functional_type)
        : Functional<dim,nstate,real>(dg_input)
        , functional_type(functional_type)
        , euler_fad_fad(dynamic_cast< Physics::Euler<dim,dim+2,FadFadType> &>(*(this->physics_fad_fad)))
        , angle_of_attack(euler_fad_fad.angle_of_attack)
        , rotation_matrix(initialize_rotation_matrix(angle_of_attack))
        , lift_vector(initialize_lift_vector(rotation_matrix))
        , drag_vector(initialize_drag_vector(rotation_matrix))
        , force_dimensionalization_factor(initialize_force_dimensionalization_factor())
    {
        switch(functional_type) {
            case Functional_types::lift : force_vector = lift_vector; break;
            case Functional_types::drag : force_vector = drag_vector; break;
            default: break;
        }
    }

    real evaluate_functional( const bool compute_dIdW = false, const bool compute_dIdX = false, const bool compute_d2I = false) override
    {
        double value = Functional<dim,nstate,real>::evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I);

        if (functional_type == Functional_types::lift) {
            this->pcout << "Lift value: " << value << "\n";
            //std::cout << "Lift value: " << value << std::cout;
            //std::cout << "Lift value: " << value << std::cout;
        }
        if (functional_type == Functional_types::drag) {
            this->pcout << "Drag value: " << value << "\n";
        }

        return value;
    }

public:
    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    template<typename real2>
    real2 evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,real2> &/*phys_coord*/,
        const dealii::Tensor<1,dim,real2> &normal,
        const std::array<real2,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/) const
    {
        if (boundary_id == 1001) {
            assert(soln_at_q.size() == dim+2);
            const Physics::Euler<dim,dim+2,real2> &euler = dynamic_cast< const Physics::Euler<dim,dim+2,real2> &> (physics);

            real2 pressure = euler.compute_pressure (soln_at_q);

            //std::cout << " force_dimensionalization_factor: " << force_dimensionalization_factor
            //          << " pressure: " << pressure
            //          << " normal*force_vector: " << normal*force_vector
            //          << std::endl;

            return force_dimensionalization_factor * pressure * (normal * force_vector);
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
