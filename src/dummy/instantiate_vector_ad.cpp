#include <Sacado.hpp>
namespace dealii{
namespace numbers{
    bool is_finite(const Sacado::Fad::DFad<double> &x) {
        (void) x;
        return true;
    }
}
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.templates.h>


int main (int /*argc*/, char * /*argv*/[])
{
    using namespace dealii;
    dealii::LinearAlgebra::distributed::Vector<double> vector_double;

    using ADtype = Sacado::Fad::DFad<double>;
    dealii::LinearAlgebra::distributed::Vector<ADtype> vector_ad;

    return 0;
}

