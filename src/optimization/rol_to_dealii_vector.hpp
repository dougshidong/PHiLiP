#ifndef __ROLTODEALIIVECTOR_H__
#define __ROLTODEALIIVECTOR_H__

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/optimization/rol/vector_adaptor.h>

namespace PHiLiP {

/// Access the read-write deali.II Vector stored within the ROL::Vector.
/** Note that the ROL::Vector<double> @p x should actually be a
 *  dealii::Rol::VectorAdaptor, which is derived from ROL::Vector<double>.
 *  @p x is dynamically downcasted into the VectorAdaptor, which in turn
 *  returns a reference to the stored dealii::LinearAlgebra::distributed::Vector<double>.
 */
const dealii::LinearAlgebra::distributed::Vector<double> &
ROL_vector_to_dealii_vector_reference(const ROL::Vector<double> &x);

/// Access the read-only deali.II Vector stored within the ROL::Vector.
/** Note that the ROL::Vector<double> @p x should actually be a
 *  dealii::Rol::VectorAdaptor, which is derived from ROL::Vector<double>.
 *  @p x is dynamically downcasted into the VectorAdaptor, which in turn
 *  returns a reference to the stored dealii::LinearAlgebra::distributed::Vector<double>.
 */
dealii::LinearAlgebra::distributed::Vector<double> &
ROL_vector_to_dealii_vector_reference(ROL::Vector<double> &x);

} // PHiLiP namespace

#endif
