#ifndef __OPERATORS_FACTORY_H__
#define __OPERATORS_FACTORY_H__

#include "parameters/all_parameters.h"
#include "operators.h"

namespace PHiLiP {
namespace OPERATOR {

/// This class creates a new Operators object
/** This allows the Operators to not be templated on the number of state variables,
  * and number of faces
  * while allowing the volume, face, and metric operators to be templated on them */
template <int dim, typename real>
class OperatorsFactory
{
public:
    /// Creates a derived object Operators, but returns it as OperatorsBase.
    /** That way, the caller is agnostic to the number of state variables,
      * poly degree, dofs, etc.*/
    static std::shared_ptr< OperatorsBase<dim,real,2*dim> >
    create_operators(
        const Parameters::AllParameters *const parameters_input,
        const int nstate_input,//number of states input
        const unsigned int degree,//degree not really needed at the moment
        const unsigned int max_degree_input,//max poly degree for operators
        const unsigned int grid_degree_input);//max grid degree for operators

};

} // OPERATOR namespace
} // PHiLiP namespace

#endif
