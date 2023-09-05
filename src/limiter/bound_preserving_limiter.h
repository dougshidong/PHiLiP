#ifndef __BOUND_PRESERVING_LIMITER__
#define __BOUND_PRESERVING_LIMITER__

#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1_eulerian.h>


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/fe_values.h>

#include "dg/dg.h"
#include "physics/physics.h"

namespace PHiLiP {
        template<int dim, typename real>
        class BoundPreservingLimiter
        {
        public:
            /// Constructor
            BoundPreservingLimiter(
                const int nstate_input,//number of states input
                const Parameters::AllParameters* const parameters_input);//pointer to parameters

            /// Destructor
            ~BoundPreservingLimiter() {};

            ///Number of states
            const int nstate;

            /// Pointer to parameters object
            const Parameters::AllParameters* const all_parameters;

            /// Function to limit the solution
            virtual void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection) = 0;

        }; // End of BoundPreservingLimiter Class

        template<int dim, int nstate, typename real>
        class TVBLimiter : public BoundPreservingLimiter <dim, real>
        {
        public:
            /// Constructor
            TVBLimiter(
                const Parameters::AllParameters* const parameters_input);

            /// Destructor
            ~TVBLimiter() {};

            /// Applies total variation bounded limiter to the solution.
            /** Using Chen,Shu September 2017 Thm3.7 we apply a limiter on the solution
            */
            void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection);

        }; // End of TVBLimiter Class

        template<int dim, int nstate, typename real>
        class MaximumPrincipleLimiter : public BoundPreservingLimiter <dim, real>
        {
        public:
            /// Constructor
            MaximumPrincipleLimiter(
                const Parameters::AllParameters* const parameters_input);

            /// Destructor
            ~MaximumPrincipleLimiter() {};

            /// Maximum of initial solution in domain.
            std::vector<real> global_max;
            /// Minimum of initial solution in domain.
            std::vector<real> global_min;

            /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
            std::shared_ptr<BoundPreservingLimiter<dim, real>> tvbLimiter;

            /// Function to obtain the maximum and minimum of the initial solution
            void get_global_max_and_min_of_solution(
                dealii::LinearAlgebra::distributed::Vector<double>      solution,
                const dealii::DoFHandler<dim>&                          dof_handler,
                const dealii::hp::FECollection<dim>&                    fe_collection);

            /// Applies maximum-principle-satisfying limiter to the solution.
            /** Using Zhang,Shu May 2010 Eq 3.8 and 3.9 we apply a limiter on the global solution
            */
            void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection);

        }; // End of MaximumPrincipleLimiter Class

        template<int dim, int nstate, typename real>
        class PositivityPreservingLimiter_Zhang2010 : public BoundPreservingLimiter <dim, real>
        {
        public:
            /// Constructor
            PositivityPreservingLimiter_Zhang2010(
                const Parameters::AllParameters* const parameters_input);

            /// Destructor
            ~PositivityPreservingLimiter_Zhang2010() {};

            /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
            std::shared_ptr<BoundPreservingLimiter<dim, real>> tvbLimiter;
            // Euler physics pointer. Used to compute pressure.
            std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

            /// Applies positivity-preserving limiter to the solution.
            /** Using Zhang,Shu November 2010 Eq 3.14-3.19 we apply a limiter on the global solution
            */
            void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection);

        }; // End of PositivityPreservingLimiter_Zhang2010 Class

        template<int dim, int nstate, typename real>
        class PositivityPreservingLimiter_Wang2012 : public BoundPreservingLimiter <dim, real>
        {
        public:
            /// Constructor
            PositivityPreservingLimiter_Wang2012(
                const Parameters::AllParameters* const parameters_input);

            /// Destructor
            ~PositivityPreservingLimiter_Wang2012() {};

            /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
            std::shared_ptr<BoundPreservingLimiter<dim, real>> tvbLimiter;
            // Euler physics pointer. Used to compute pressure.
            std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

            /// Applies positivity-preserving limiter to the solution.
            /** Using Wang,Shu January 2012 Eq 3.2-3.7 we apply a limiter on the global solution
            */
            void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection);

        }; // End of PositivityPreservingLimiter_Wang2012 Class
} // PHiLiP namespace

#endif

