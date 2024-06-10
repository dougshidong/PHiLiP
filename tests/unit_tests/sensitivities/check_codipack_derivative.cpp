#include "ADTypes.hpp"

template<typename adtype>
class TapeHelper2 : public codi::TapeHelper<adtype>
{
    public:
        std::vector<typename adtype::GradientData>& getinputValues() {return this->inputValues;} 
};

template<typename adtype>
adtype f1(adtype x)
{
    return x*x;
}

template<typename adtype>
void f2(adtype x, adtype y1)
{
    using TH = TapeHelper2<adtype>;
    TH th;
    th.getinputValues().push_back(x.getGradientData());
    adtype y2 = x*x*x + y1;
    th.registerOutput(y2);
    typename TH::JacobianType& jac = th.createJacobian();
    th.evalJacobian(jac);
    std::cout<<"Derivative dy2/dx = "<<jac(0,0)<<std::endl; 
    th.deleteJacobian(jac);
    
    typename TH::HessianType& hes = th.createHessian();
    th.evalHessian(hes);
    std::cout<<"Hessian d^2 y2/dx^2 = "<<hes(0,0,0)<<std::endl; 
    th.deleteHessian(hes);
}

template<typename adtype>
void f3(adtype x, adtype y1)
{
    using TH = TapeHelper2<adtype>;
    TH th;
    th.getinputValues().push_back(x.getGradientData());
    adtype y3 = 30.0*pow(x,1) + y1;
    th.registerOutput(y3);
    typename TH::JacobianType& jac = th.createJacobian();
    th.evalJacobian(jac);
    std::cout<<"Derivative dy3/dx = "<<jac(0,0)<<std::endl; 
    th.deleteJacobian(jac);
    
    typename TH::HessianType& hes = th.createHessian();
    th.evalHessian(hes);
    std::cout<<"Hessian d^2 y2/dx^2 = "<<hes(0,0,0)<<std::endl; 
    th.deleteHessian(hes);
}
int main()
{
    //using adtype = typename codi_JacobianComputationType;
    //using adtype = codi::RealReverseIndexVec<1>; ///< Reverse mode type for Jacobian computation using TapeHelper.
    using adtype  = codi::RealReversePrimalIndexGen< codi::RealForwardVec<1>,
                                                      codi::Direction< codi::RealForwardVec<1>, 1>
                                                    >; ///< Nested reverse-forward mode type for Jacobian and Hessian computation using TapeHelper.
    using Tape = typename adtype::TapeType;
    using Position = typename Tape::Position;

    adtype x = 10.0;

    Tape &tape = adtype::getGlobalTape();
    tape.setActive();
    tape.registerInput(x);
    adtype y1 = f1(x); 
    Position pos = tape.getPosition();
    
    tape.reset(pos);
    tape.clearAdjoints();
    f2(x,y1);
    
    tape.reset(pos);
    tape.clearAdjoints();
    f3(x,y1);
    
    return 0;
}
