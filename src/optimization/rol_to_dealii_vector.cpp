#include "rol_to_dealii_vector.hpp"

namespace PHiLiP {

const dealii::LinearAlgebra::distributed::Vector<double> &
ROL_vector_to_dealii_vector_reference(const ROL::Vector<double> &x)
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorAdaptor = dealii::Rol::VectorAdaptor<VectorType>;

    return *(Teuchos::dyn_cast<const VectorAdaptor>(x)).getVector();
}

dealii::LinearAlgebra::distributed::Vector<double> &
ROL_vector_to_dealii_vector_reference(ROL::Vector<double> &x)
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorAdaptor = dealii::Rol::VectorAdaptor<VectorType>;

    return *(Teuchos::dyn_cast<VectorAdaptor>(x)).getVector();
}

} // PHiLiP namespace
