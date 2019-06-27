// hello
#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_product_types.h>
#include <deal.II/base/tensor.h>

int main (int /*argc*/, char * /*argv*/[])
{
    using namespace dealii;

    using ADtype = Sacado::Fad::DFad<double>;
    using ADADtype = Sacado::Fad::DFad<ADtype>;

    ADtype scalarA_AD = 1.0;

    Tensor <1, 3, ADtype> vectorA_AD;
    vectorA_AD[0] = 1*scalarA_AD;
    vectorA_AD[1] = 2*scalarA_AD;
    vectorA_AD[2] = 3*scalarA_AD;

    scalarA_AD.diff(0, 4);
    vectorA_AD[0].diff(1, 4);
    vectorA_AD[1].diff(2, 4);
    vectorA_AD[2].diff(3, 4);
    
    Tensor <1, 3, ADtype> vector_scalar_AD;
    vector_scalar_AD = vectorA_AD * scalarA_AD;

    Tensor <1, 3, double> vectorB_double;
    vectorB_double[0] = 1.0;
    vectorB_double[1] = 2.0;
    vectorB_double[2] = 3.0;

    
    ADtype vector_vector_AD;
    vector_vector_AD = vectorA_AD * vectorB_double;

    std::cout<<"Should be 2 AD is " << vector_scalar_AD[0].dx(0) << std::endl;
    std::cout<<"Should be 4 AD is " << vector_scalar_AD[1].dx(0) << std::endl;
    std::cout<<"Should be 6 AD is " << vector_scalar_AD[2].dx(0) << std::endl;

    std::cout<<"Should be 1 AD is " << vector_scalar_AD[0].dx(1) << std::endl;
    std::cout<<"Should be 0 AD is " << vector_scalar_AD[1].dx(1) << std::endl;
    std::cout<<"Should be 0 AD is " << vector_scalar_AD[2].dx(1) << std::endl;

    std::cout<<"Should be 0 AD is " << vector_scalar_AD[0].dx(2) << std::endl;
    std::cout<<"Should be 1 AD is " << vector_scalar_AD[1].dx(2) << std::endl;
    std::cout<<"Should be 0 AD is " << vector_scalar_AD[2].dx(2) << std::endl;

    std::cout<<"Should be 0 AD is " << vector_scalar_AD[0].dx(3) << std::endl;
    std::cout<<"Should be 0 AD is " << vector_scalar_AD[1].dx(3) << std::endl;
    std::cout<<"Should be 1 AD is " << vector_scalar_AD[2].dx(3) << std::endl;

    std::cout<<"Should be 0 AD is " << vector_vector_AD.dx(0) << std::endl;
    std::cout<<"Should be 1 AD is " << vector_vector_AD.dx(1) << std::endl;
    std::cout<<"Should be 2 AD is " << vector_vector_AD.dx(2) << std::endl;
    std::cout<<"Should be 3 AD is " << vector_vector_AD.dx(3) << std::endl;


    //vectorA_AD[0] = 1*scalarA_AD*scalarA_AD;
    //vectorA_AD[1] = 2*scalarA_AD*scalarA_AD*scalarA_AD;
    //vectorA_AD[3] = 3*scalarA_AD*scalarA_AD*scalarA_AD*scalarA_AD;

    //Tensor <1, 3, ADADtype> vectorA_ADAD;
    //vectorA_ADAD[0] = 1*scalarA_AD*scalarA_AD;
    //vectorA_ADAD[1] = 2*scalarA_AD*scalarA_AD*scalarA_AD;
    //vectorA_ADAD[3] = 3*scalarA_AD*scalarA_AD*scalarA_AD*scalarA_AD;

    //std::cout<<"Should be 2 AD is " << vector_scalar_AD[0].dx(0).dx(0) << std::endl;
    //std::cout<<"Should be 4 AD is " << vector_scalar_AD[1].dx(0).dx(0) << std::endl;
    //std::cout<<"Should be 6 AD is " << vector_scalar_AD[2].dx(0).dx(0) << std::endl;


    return 0;
}

