#include <iostream>
#include <CoDiPack/include/codi.hpp>
using namespace std;
typedef codi::RealForwardGen<double> t1s;
typedef codi::RealForwardGen<t1s>    t2s;
typedef codi::RealReverseGen<t1s>    r2s;
template<typename T>
T func(const T* x) {
  T t =   13.0*x[0]*x[0]
        + 23.0*x[1]*x[1]
        + 33.0*x[0]*x[1]
        + x[0]
        + x[1];
  return t * 3.0;
}

template<typename T>
T func_dx1(const T* x) {
  T t =   26.0*x[0]
        + 33.0*x[1]
        + 1.0;
  return t * 3.0;
}
template<typename T>
T func_dx2(const T* x) {
  T t = 46.0*x[1]
        + 33.0*x[0]
        + 1.0;
  return t * 3.0;
}

template<typename T>
T func_dx1_dx1(const T* /*x*/) {
  T t =   26.0;
  return t * 3.0;
}

template<typename T>
T func_dx1_dx2(const T* /*x*/) {
  T t =   33.0;
  return t * 3.0;
}

template<typename T>
T func_dx2_dx2(const T* /*x*/) {
  T t = 46.0;
  return t * 3.0;
}

int main(int argc, char * argv[]) {
    (void) argc;
    const int n = atoi(argv[1]);

    std::vector<double> x_double(n);
    std::vector<t2s> x(n);

    for (int i=0; i<n; ++i) {
        x_double[i] = 2.0 + i;
        x[i].value() = 2.0 + i;
    }
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            x[j].value().gradient() = 0.0;
            x[j].gradient().value() = 0.0;
        }
        x[i].value().gradient() = 1.0;
        x[i].gradient().value() = 1.0;
        //#x[i].gradient().value() = 1.0;

        t2s cFor2 = func(&(x[0]));
        cout << "t0s:   " << cFor2.value().value() << std::endl;
        cout << "t1_1s: " << cFor2.value().gradient() << std::endl;
        cout << "t1_2s: " << cFor2.gradient().value() << std::endl;
        cout << "t2s:   " << cFor2.gradient().gradient() << std::endl;

        cout << std::endl;

        cout << "Exact : " << std::endl;
        cout << "t0s:   " << func(&(x_double[0])) << std::endl;
        cout << "t1_1s: " << func_dx1(&(x_double[0])) << std::endl;
        cout << "t1_2s: " << func_dx2(&(x_double[0])) << std::endl;
        cout << "dx1_dx2 " << func_dx1_dx2(&(x_double[0])) << std::endl;
        cout << "dx1_dx1 " << func_dx1_dx1(&(x_double[0])) << std::endl;
        cout << "dx2_dx2 " << func_dx2_dx2(&(x_double[0])) << std::endl;

        cout << std::endl;
        cout << std::endl;

    }
    for (int j=0; j<n; ++j) {
        x[j].value().gradient() = 0.0;
        x[j].gradient().value() = 0.0;
    }
    x[0].value().gradient() = 1.0;
    x[1].gradient().value() = 1.0;
    //#x[i].gradient().value() = 1.0;

    t2s cFor2 = func(&(x[0]));
    cout << "t0s:   " << cFor2.value().value() << std::endl;
    cout << "t1_1s: " << cFor2.value().gradient() << std::endl;
    cout << "t1_2s: " << cFor2.gradient().value() << std::endl;
    cout << "t2s:   " << cFor2.gradient().gradient() << std::endl;

    cout << std::endl;

    cout << "Exact : " << std::endl;
    cout << "t0s:   " << func(&(x_double[0])) << std::endl;
    cout << "t1_1s: " << func_dx1(&(x_double[0])) << std::endl;
    cout << "t1_2s: " << func_dx2(&(x_double[0])) << std::endl;
    cout << "dx1_dx2 " << func_dx1_dx2(&(x_double[0])) << std::endl;
    cout << "dx1_dx1 " << func_dx1_dx1(&(x_double[0])) << std::endl;
    cout << "dx2_dx2 " << func_dx2_dx2(&(x_double[0])) << std::endl;

    cout << std::endl;
    cout << std::endl;


//   {
//     r6s::TapeType& tape = r6s::getGlobalTape();
//     r6s aRev = 2.0;
//     // set all first order directions on the primal value
//     aRev.value().value().value().value().value().gradient() = 1.0;
//     aRev.value().value().value().value().gradient().value() = 1.0;
//     aRev.value().value().value().gradient().value().value() = 1.0;
//     aRev.value().value().gradient().value().value().value() = 1.0;
//     aRev.value().gradient().value().value().value().value() = 1.0;
//     tape.setActive();
//     tape.registerInput(aRev);
//     r6s cRev = func(aRev);
//     tape.registerOutput(cRev);
//     // set all first order directions on the adjoint value
//     cRev.gradient().value().value().value().value().value() = 1.0;
//     tape.setPassive();
//     tape.evaluate();
//     cout << "r0s: " << cRev << std::endl;
//     cout << "r6s: " << aRev.gradient().gradient().gradient().gradient().gradient().gradient() << std::endl;
//   }
  return 0;
}
