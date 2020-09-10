#include <CoDiPack/include/codi.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
template<typename Real>
void dotWithNorms(Real const* a, Real const* b, size_t n, Real& alpha, Real& aNorm, Real& bNorm) {
  alpha = Real(); // Dot product is accumulated in alpha
  aNorm = Real();
  bNorm = Real();
  Real coefficient_to_waste_time = 1.0;
  for(size_t i = 0; i < 100*n; i += 1) {
    coefficient_to_waste_time *= 1.0 + 1e-7 * a[0];
    coefficient_to_waste_time *= 1.0 + 1e-7 * a[1];
  }
  for(size_t i = 0; i < n; i += 1) {
    alpha += coefficient_to_waste_time + a[i] * b[i];
    aNorm += a[i] * a[i];
    bNorm += b[i] * b[i];
  }
  aNorm = sqrt(aNorm);
  bNorm = sqrt(bNorm);
  alpha = acos(alpha / (aNorm * bNorm));
}
template<typename Vec>
void printVector(std::string const& name, Vec const& v, size_t length, size_t offset) {
  std::cout << "Vector " << name << ": {";
  for(size_t i = 0; i < length; i += 1) {
    if(i != 0) {
      std::cout << ", ";
    }
    std::cout << v[offset + i];
  }
  std::cout << "}" << std::endl;
}
template<typename Vec>
void printVectorDim(std::string const& name, Vec const& v, size_t length, size_t offset, size_t dim) {
  std::cout << "Vector " << name << ": {";
  for(size_t i = 0; i < length; i += 1) {
    if(i != 0) {
      std::cout << ", ";
    }
    std::cout << v[offset + i][dim];
  }
  std::cout << "}" << std::endl;
}
template<typename Jac>
void printJacCol(std::string const& text, Jac const &jac, size_t col) {
  std::cout << text <<": {";
  for(size_t j = 0; j < jac.getN(); j += 1) {
    if(j != 0) {
      std::cout << ", ";
    }
    std::cout << jac(col, j);
  }
  std::cout << "}" << std::endl;
}
template<typename Hes>
void printHesForOutput(std::string const& text, Hes const &hes, size_t output) {
  std::cout << text <<": {\n";
  for(size_t j = 0; j < hes.getN(); j += 1) {
    std::cout << "  ";
    for(size_t k = 0; k < hes.getN(); k += 1) {
      if(k != 0) {
        std::cout << ", ";
      }
      std::cout << hes(output, j, k);
    }
    std::cout << "\n";
  }
  std::cout << "}" << std::endl;
}
template<int forwardDim, int reverseDim>
int ad(const int mode, const size_t n)
{
  using HessType = codi::RealReversePrimalIndexGen<
                        codi::RealForwardVec<forwardDim>,
                        codi::Direction< codi::RealForwardVec<forwardDim>, reverseDim>>;
  //using JacType = codi::RealReverseIndexVec<forwardDim>;
  //using HessType = codi::HessianComputationType;
  //using TH = codi::TapeHelper<HessType, JacType>;
  using TH = codi::TapeHelper<HessType>;
  TH th;
  std::vector<HessType> a(n);
  std::vector<HessType> b(n);
  for(size_t i = 0; i < n; i += 1) {
    a[i] = i;
    b[i] = pow(-1, i);
  }
  th.startRecording();
  for(size_t i = 0; i < n; i += 1) {
    th.registerInput(a[i]);
  }
  for(size_t i = 0; i < n; i += 1) {
    th.registerInput(b[i]);
  }
  HessType alpha, aNorm, bNorm;
  dotWithNorms(a.data(), b.data(), n, alpha, aNorm, bNorm);
  th.registerOutput(alpha);
  //th.registerOutput(aNorm);
  //th.registerOutput(bNorm);
  th.stopRecording();
  //HessType::getGlobalTape().printStatistics();
  typename TH::JacobianType& jac = th.createJacobian();
  typename TH::HessianType& hes = th.createHessian();
  if(1 == mode) {
    //th.evalJacobian(jac);
    th.evalHessian(hes);
  } else {
    th.evalHessian(hes, jac);
  }
  //printVector("a", a, n, 0);
  //printVector("b", b, n, 0);
  //std::cout << std::endl;
  //printJacCol("Jacobian with respect to alpha: ", jac, 0);
  //printJacCol("Jacobian with respect to aNorm: ", jac, 1);
  //printJacCol("Jacobian with respect to bNorm: ", jac, 2);
  //std::cout << std::endl;
  //printHesForOutput("Hessian with respect to alpha: ", hes, 0);
  //printHesForOutput("Hessian with respect to aNorm: ", hes, 1);
  //printHesForOutput("Hessian with respect to bNorm: ", hes, 2);
  // Evaluate at different position
  typename TH::Real* x = th.createPrimalVectorInput();
  typename TH::Real* y = th.createPrimalVectorOutput();
  for(size_t i = 0; i < n; i += 1) {
    x[0 + i] = i * i;
    x[n + i] = pow(-1, i + 1);
  }
  if(1 == mode) {
    //th.evalJacobianAt(x, jac, y);
    // Jacobian evaluation already shifted the point for the evaluation. No second ...At call is required here
    th.evalHessian(hes);
  } else {
    th.evalHessianAt(x, hes, y, jac);
  }
  //printVector("a", a, n, 0);
  //printVector("b", b, n, 0);
  //std::cout << std::endl;
  //printJacCol("Jacobian with respect to alpha: ", jac, 0);
  //printJacCol("Jacobian with respect to aNorm: ", jac, 1);
  //printJacCol("Jacobian with respect to bNorm: ", jac, 2);
  //std::cout << std::endl;
  //printHesForOutput("Hessian with respect to alpha: ", hes, 0);
  //printHesForOutput("Hessian with respect to aNorm: ", hes, 1);
  //printHesForOutput("Hessian with respect to bNorm: ", hes, 2);
  // Evaluate gradient
  //typename TH::GradientValue* x_b = th.createGradientVectorInput();
  //typename TH::GradientValue* y_b = th.createGradientVectorOutput();
  //y_b[0] = {1.0, 0.0, 0.0, 0.0};
  //y_b[1] = {0.0, 1.0, 0.0, 0.0};
  //y_b[2] = {0.0, 0.0, 1.0, 0.0};
  //th.evalReverse(y_b, x_b);
  //std::cout << "Reverse evaluation for alpha_b:" << std::endl;
  //printVectorDim("a_b", x_b, n, 0, 0);
  //printVectorDim("b_b", x_b, n, n, 0);
  //std::cout << std::endl;
  //std::cout << "Reverse evaluation for aNorm_b:" << std::endl;
  //printVectorDim("a_b", x_b, n, 0, 1);
  //printVectorDim("b_b", x_b, n, n, 1);
  //std::cout << std::endl;
  //std::cout << "Reverse evaluation for bNorm_b:" << std::endl;
  //printVectorDim("a_b", x_b, n, 0, 2);
  //printVectorDim("b_b", x_b, n, n, 2);
  //th.deleteGradientVector(x_b);
  //th.deleteGradientVector(y_b);
  // Clean up vectors
  th.deletePrimalVector(x);
  th.deletePrimalVector(y);
  th.deleteJacobian(jac);
  th.deleteHessian(hes);
  return 0;
}

template<int forwardDim, int reverseDim>
void timing_ad(int mode, unsigned int max_independent, unsigned int n_assemblies)
{
  std::cout << "Forward-mode vector-size: " << forwardDim << std::endl;
  std::cout << "Reverse-mode vector-size: " << reverseDim << std::endl;
  for (unsigned int isize = 32; isize <= max_independent; isize*=2) {
      std::clock_t c_start = std::clock();
      for (unsigned int i = 0; i < n_assemblies; ++i) {
          (void) ad<forwardDim, reverseDim>(mode,isize);
      }
      std::clock_t c_end = std::clock();
      std::cout << "Number of independent variables: " << isize << " Timing: "
              << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n";
  }
}

int main(int nargs, char** args) {
  int mode = 1;
  if(2 <= nargs) {
    mode = std::atoi(args[1]);
    if(mode < 1 || 2 < mode) {
      std::cerr << "Error: Please enter a mode from 1 to 2, it was '" << mode << "'." << std::endl;
      std::cerr << "  Mode  1: separate evaluation of Hessian and Jacobian" << std::endl;
      std::cerr << "  Mode  2: combined evaluation of Hessian and Jacobian" << std::endl;
      exit(-1);
    }
  }
  size_t n = 5;
  if(3 <= nargs) {
      n = atoi(args[2]);
  }

  const int n_assemblies = 10;
  const int reverseDim = 1;
  timing_ad<1,reverseDim>(mode, n, n_assemblies);
  timing_ad<2,reverseDim>(mode, n, n_assemblies);
  timing_ad<8,reverseDim>(mode, n, n_assemblies);


  return 0;
}
