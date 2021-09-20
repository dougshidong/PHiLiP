#ifndef __BURGERS_REWIENSKI__
#define __BURGERS_REWIENSKI__

#include "burgers.h"

namespace PHiLiP {
    namespace Physics {
/// Burgers Rewienski equation. Derived from Burgers, which is derived from PhysicsBase.
        template <int dim, int nstate, typename real>
        class BurgersRewienski : public Burgers <dim, nstate, real>
        {
        public:
            /// Constructor
            BurgersRewienski(
                    const bool                                                convection,
                    const bool                                                diffusion,
                    const dealii::Tensor<2,3,double>                          input_diffusion_tensor,
                    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function)
                    : Burgers<dim, nstate, real>(convection,
                                                 diffusion,
                                                 input_diffusion_tensor,
                                                 manufactured_solution_function)
            {
                static_assert(nstate==dim, "Physics::Burgers() should be created with nstate==dim");
            };

            /// Destructor
            ~BurgersRewienski () {};

            /// Source term is zero or depends on manufactured solution
            std::array<real,nstate> source_term (
                    const dealii::Point<dim,real> &pos,
                    const std::array<real,nstate> &solution) const override;

            /// If diffusion is present, assign Dirichlet boundary condition
            /** Using Neumann boundary conditions might need to modify the functional
             *  in order to obtain the optimal 2p convergence of the functional error
             */
            void boundary_face_values (
                    const int /*boundary_type*/,
                    const dealii::Point<dim, real> &/*pos*/,
                    const dealii::Tensor<1,dim,real> &/*normal*/,
                    const std::array<real,nstate> &/*soln_int*/,
                    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
                    std::array<real,nstate> &/*soln_bc*/,
                    std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

        };
    } // Physics namespace
} // PHiLiP namespace

#endif
