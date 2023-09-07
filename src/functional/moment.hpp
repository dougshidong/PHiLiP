#ifndef __PHILIP_MOMENT_H__
#define __PHILIP_MOMENT_H__

#include "functional.h"

namespace PHiLiP {

/** Target boundary values.
 *  Simply zero out the default volume contribution.
 */
template <int dim, int nstate, typename real>
class ZMomentFunctional : public Functional<dim, nstate, real>
{
private:
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

    /// Avoid warning that the function was hidden [-Woverloaded-virtual].
    /** The compiler would otherwise hide Functional::evaluate_volume_integrand, which is fine for 
     *  us, but is a typical bug that other people have. This 'using' imports the base class function
     *  to our derived class even though we don't need it.
     */
    using Functional<dim,nstate,real>::evaluate_volume_integrand;


    /// @brief Origin used to evaluate moment coefficients.
	const dealii::Point<dim,double> moment_origin;

    /// @brief Casts DG's physics into an Euler physics reference.
    const Physics::Euler<dim,dim+2,FadFadType> &euler_fad_fad;

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
    ZMomentFunctional(
        std::shared_ptr<DGBase<dim,real>> dg_input,
		const dealii::Point<dim,double> moment_origin)
        : Functional<dim,nstate,real>(dg_input)
		, moment_origin(moment_origin)
        , euler_fad_fad(dynamic_cast< Physics::Euler<dim,dim+2,FadFadType> &>(*(this->physics_fad_fad)))
        , force_dimensionalization_factor(initialize_force_dimensionalization_factor())
    { }

    real evaluate_functional( const bool compute_dIdW = false, const bool compute_dIdX = false, const bool compute_d2I = false) override
    {
        //if(Functional<dim,nstate,real>::dg->get_residual_l2norm() > 1e-9) return 1.7e199;
        double value = Functional<dim,nstate,real>::evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I);

		//this->pcout << "ZMoment value: " << value << "\n";
        return value;
    }

public:
    /// Virtual function for computation of cell boundary functional term
    /** Used only in the computation of evaluate_function(). If not overriden returns 0. */
    template<typename real2>
    real2 evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,real2> &phys_coord,
        const dealii::Tensor<1,dim,real2> &normal,
        const std::array<real2,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/) const
    {
        if (boundary_id == 1001) {
            assert(soln_at_q.size() == dim+2);
            const Physics::Euler<dim,dim+2,real2> &euler = dynamic_cast< const Physics::Euler<dim,dim+2,real2> &> (physics);

            real2 pressure = euler.compute_pressure (soln_at_q);

			dealii::Tensor<1,dim,real2> distance_vector; 
			for (int d = 0; d < dim; ++d) {
			    distance_vector[d] = phys_coord[d] - moment_origin[d];
			}
			const dealii::Tensor<1,dim,real2> force = pressure * normal; // Area accounted for in integration
			dealii::Tensor<1,3,real2> distance3D;
			dealii::Tensor<1,3,real2> force3D;
			if constexpr (dim == 1) { 
			    std::abort();
			} else if constexpr (dim == 2) { 
				distance3D = dealii::Tensor<1,3,real2> ({ distance_vector[0], distance_vector[1], 0.0 });
				force3D = dealii::Tensor<1,3,real2> ({ force[0], force[1], 0.0 });
			} else if constexpr (dim == 3) {
			    distance3D = distance_vector;
				force3D = force;
			}
			const dealii::Tensor<1,3,real2> moment3D = dealii::cross_product_3d(distance3D, force3D);

			const int Zmoment_index = 2;
            return force_dimensionalization_factor * moment3D[Zmoment_index];
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

} // PHiLiP namespace

#endif
