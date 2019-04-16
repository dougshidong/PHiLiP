#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_product_types.h>
#include <deal.II/base/tensor.h>

int main (int argc, char *argv[])
{
    using namespace dealii;


    Tensor <1, 3, Sacado::Fad::DFad<double>> velocity_field;
    Tensor <1, 3, Sacado::Fad::DFad<double>> conv_flux;
    Sacado::Fad::DFad<double> solution_ad = 1.0;
    solution_ad.diff(0, 1);

    velocity_field[0] = 1.0;
    velocity_field[1] = 1.0;
    velocity_field[2] = 1.0;
    
    conv_flux = velocity_field * solution_ad;
    return 0;
}

