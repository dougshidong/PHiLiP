
#include "advection.h"
namespace PHiLiP
{
    using namespace dealii;

    // Constructors definition
    template <int dim, typename real>
    PDE<dim, real>::PDE()
        : mapping(1)
        , fe(1)
        , dof_handler(triangulation)
    {}
    template PDE<1, double>::PDE();
    template PDE<2, double>::PDE();
    template PDE<3, double>::PDE();


    template <int dim, typename real>
    PDE<dim, real>::PDE(const unsigned int polynomial_order)
        : mapping1(polynomial_order+1)
        , fe(polynomial_order)
        , dof_handler(triangulation)
    {}
    template PDE<1, double>::PDE(const unsigned int);
    template PDE<2, double>::PDE(const unsigned int);
    template PDE<3, double>::PDE(const unsigned int);


} // end of PHiLiP namespace
