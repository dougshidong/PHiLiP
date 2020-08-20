#include <CoDiPack/include/codi.hpp>
#include <iostream>

#include <initializer_list>
#include <vector>
#include <cassert>

/**
 * @brief Global namespace for CoDiPack - Code Differentiation Package
 */
namespace codi {

  /**
   * @brief The vector for direction of the forward mode or the reverse mode.
   *
   * The direction is an array with a static size that can be added and multiplied by
   * a scalar value.
   *
   * @tparam Real  The scalar value type that is used by the array.
   * @tparam  dim  The dimension of the array.
   */
  template<typename Real>
  class DirectionVar {
    private:
      std::vector<Real> vec; /**< The data vector with the given dimension */
      size_t dim;

    public:

      /**
       * @brief Creates a zero direction.
       */
      // CODI_INLINE DirectionVar()
      // : vec(100)
      // , dim(100)
      // { }

      /**
       * @brief Creates a direction with the same value in every component.
       *
       * @param[in] s  The value that is set to all components.
       */
      CODI_INLINE DirectionVar(const Real& s, const size_t n)
      // : DirectionVar()
      {
        this->resize(n);
        for(size_t i = 0; i < dim; ++i) {
          vec[i] = s;
        }
      }

      /**
       * @brief Copy constructor.
       *
       * @param[in] d  The other direction.
       */
      CODI_INLINE DirectionVar(const DirectionVar<Real>& d)
      // : DirectionVar()
      {
        this->resize(d.size());
        for(size_t i = 0; i < dim; ++i) {
          vec[i] = d.vec[i];
        }
      }

      /**
       * @brief The direction is initialized with the values from the initializer list.
       *
       * If the list is to small, then only the first m elements are set.
       * If the list is to large, then only the first dim elements are set.
       *
       * @param[in] l  The list with the values for the direction.
       */
      CODI_INLINE DirectionVar(std::initializer_list<Real> l)
      // : DirectionVar()
      {
        size_t size = std::min(dim, l.size());
        const Real* array = l.begin(); // this is possible because the standard requires an array storage
        for(size_t i = 0; i < size; ++i) {
          vec[i] = array[i];
        }
      }

      /**
       * @brief Get number of directions.
       *
       * @return The number of directions.
       */
      CODI_INLINE size_t size() const {
        return dim;
      }
      /**
       * @brief Changes the number of directions.
       *
       * @param[in] new_size  The new number of directions.
       */
      CODI_INLINE void resize(size_t new_size) {
        dim = new_size;
        vec.resize(new_size);
      }

      /**
       * @brief Get the i-th element of the direction.
       *
       * No bounds checks are performed.
       *
       * @param[in] i  The index for the vector.
       *
       * @return The element from the vector.
       */
      CODI_INLINE Real& operator[] (const size_t& i) {
        return vec[i];
      }

      /**
       * @brief Get the i-th element of the direction.
       *
       * No bounds checks are performed.
       *
       * @param[in] i  The index for the vector.
       *
       * @return The element from the vector.
       */
      CODI_INLINE const Real& operator[] (const size_t& i) const {
        return vec[i];
      }

      /**
       * @brief Assign operator for the direction.
       *
       * @param[in] v  The values from the direction are set to the values of this direction object.
       *
       * @return Reference to this object.
       */
      CODI_INLINE DirectionVar<Real>& operator = (const DirectionVar<Real>& v) {
        this->resize(v.size());
        for(size_t i = 0; i < dim; ++i) {
          this->vec[i] = v.vec[i];
        }

        return *this;
      }

      /**
       * @brief Update operator for the direction.
       *
       * @param[in] v  The values from the direction are added to the values of this direction object.
       *
       * @return Reference to this object.
       */
      CODI_INLINE DirectionVar<Real>& operator += (const DirectionVar<Real>& v) {
        //assert(vec.size() == v.size());
        for(size_t i = 0; i < dim; ++i) {
          this->vec[i] += v.vec[i];
        }

        return *this;
      }

      /**
       * @brief Checks if all entries in the direction are also a total zero.
       *
       * @return true if all entries are a total zero.
       */
      CODI_INLINE bool isTotalZero() const {
        for(size_t i = 0; i < dim; ++i) {
          if( !codi::isTotalZero(vec[i])) {
            return false;
          }
        }

        return true;
      }
  };

  /**
   * @brief Tests if all elements of the given direction are finite.
   *
   * Calls on all elements codi::isfinite.
   *
   * @param[in] d  The direction vector that is tested.
   * @return true if all elements are finite.
   * @tparam Real  The computation type of the direction vector.
   * @tparam  dim  The dimension of the direction vector.
   */
  template<typename Real>
  bool isfinite(const DirectionVar<Real>& d) {
    bool finite = true;

    for(size_t i = 0; i < d.size(); ++i) {
      finite &= codi::isfinite(d[i]);
    }

    return finite;
  }

  /**
   * @brief Scalar multiplication of a direction.
   *
   * Performs the operation w = s * v
   *
   * @param[in] s  The scalar value for the multiplication.
   * @param[in] v  The direction that is multiplied.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator * (const Real& s, const DirectionVar<Real>& v) {
    DirectionVar<Real> r;
    r.resize(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
      r[i] = s * v[i];
    }

    return r;
  }

  /**
   * \copydoc operator*(const Real& s, const DirectionVar<Real, dim>& v)
   */
  template<typename Real, typename = typename std::enable_if<!std::is_same<Real, typename TypeTraits<Real>::PassiveReal>::value>::type>
  CODI_INLINE DirectionVar<Real> operator * (const typename TypeTraits<Real>::PassiveReal& s, const DirectionVar<Real>& v) {
    DirectionVar<Real> r;
    r.resize(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
      r[i] = s * v[i];
    }

    return r;
  }

  /**
   * @brief Scalar multiplication of a direction.
   *
   * Performs the operation w = v * s
   *
   * @param[in] v  The direction that is multiplied.
   * @param[in] s  The scalar value for the multiplication.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator * (const DirectionVar<Real>& v, const Real& s) {
    return s * v;
  }

  /**
   * \copydoc operator*(const DirectionVar<Real, dim>& v, const Real& s)
   */
  template<typename Real, typename , typename = typename std::enable_if<!std::is_same<Real, typename TypeTraits<Real>::PassiveReal>::value>::type>
  CODI_INLINE DirectionVar<Real> operator * (const DirectionVar<Real>& v, const typename TypeTraits<Real>::PassiveReal& s) {
    return s * v;
  }

  /**
   * @brief Scalar division of a direction.
   *
   * Performs the operation w = v / s
   *
   * @param[in] v  The direction that is divided.
   * @param[in] s  The scalar value for the division.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator / (const DirectionVar<Real>& v, const Real& s) {
    DirectionVar<Real> r;
    r.resize(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
      r[i] = v[i] / s;
    }

    return r;
  }

  /**
   * @brief Addition of two directions.
   *
   * Performs the operation w = v1 + v2
   *
   * @param[in] v1  The first direction that is added.
   * @param[in] v2  The second direction that is added.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator + (const DirectionVar<Real>& v1, const DirectionVar<Real>& v2) {
    DirectionVar<Real> r;
    r.resize(v1.size());
    for(size_t i = 0; i < v1.size(); ++i) {
      r[i] = v1[i] + v2[i];
    }

    return r;
  }

  /**
   * @brief Subtraction of two directions.
   *
   * Performs the operation w = v1 - v2
   *
   * @param[in] v1  The first direction that is added.
   * @param[in] v2  The second direction that is subtracted.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator - (const DirectionVar<Real>& v1, const DirectionVar<Real>& v2) {
    DirectionVar<Real> r;
    r.resize(v1.size());
    for(size_t i = 0; i < v1.size(); ++i) {
      r[i] = v1[i] - v2[i];
    }

    return r;
  }

  /**
   * @brief Negation of a direction.
   *
   * Performs the negation on all elements.
   *
   * @param[in] v  The first direction that is added.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator - (const DirectionVar<Real>& v) {
    DirectionVar<Real> r;
    r.resize(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
      r[i] = -v[i];
    }

    return r;
  }

  /**
   * @brief Check if at least one component of the direction is not equal to s.
   *
   * The operator returns false if all the components of the direction v are equal to s,
   * true otherwise.
   *
   * @param[in] s  The scalar value that is checked against the components of the direction
   * @param[in] v  The direction that is compared with the scalar value.
   *
   * @return true if at least one component of v is not equal to s, false otherwise.
   *
   * @tparam    A  The type of the scalar value.
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename A, typename Real>
  CODI_INLINE bool operator != (const A& s, const DirectionVar<Real>& v) {
    for(size_t i = 0; i < v.size(); ++i) {
      if( s != v[i] ) {
        return true;
      }
    }

    return false;
  }

  /**
   * @brief Check if at least one component of the direction is not equal to s.
   *
   * The operator returns false if all the components of the direction v are equal to s,
   * true otherwise.
   *
   * @param[in] v  The direction that is compared with the scalar value.
   * @param[in] s  The scalar value that is checked against the components of the direction
   *
   * @return true if at least one component of v is not equal to s, false otherwise.
   *
   * @tparam    A  The type of the scalar value.
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename A, typename Real>
  CODI_INLINE bool operator != (const DirectionVar<Real>& v, const A& s) {
    return s != v;
  }

  /**
   * @brief Output the direction to a stream.
   *
   * The output format is: {v[0], v[1], ..., v[dim - 1]}
   *
   * @param[in,out] os  The output stream that is used for the writing.
   * @param[in]      v  The direction that is written to the stream.
   *
   * @return The output stream os.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  std::ostream& operator<<(std::ostream& os, const DirectionVar<Real>& v){
    os << "{";
    for(size_t i = 0; i < v.size(); ++i) {
      if(i != 0) {
        os << ", ";
      }
      os << v[i];
    }
    os << "}";

    return os;
  }
}


template<typename Real>
void func(const Real* x, size_t l, Real* y) {
  y[0] = 0.0;
  y[1] = 1.0;
  for(size_t i = 0; i < l; ++i) {
    y[0] += x[i];
    y[1] *= x[i];
  }
}
template<typename Real>
void func_2(const std::vector<Real> &x, const std::vector<double> &psi, int nx, int ny, std::vector<Real> &y, Real &psi_dot_y) {
  std::cout << "x[0].gradient" << std::endl;
  std::cout << x[0].gradient() << std::endl;

  std::cout << "y[0].gradient" << std::endl;
  std::cout << y[0].gradient() << std::endl;
  for (int j = 0; j < ny; ++j) {
      y[j] = x[0] + 0.3 + 0.1*j;
      y[j] = x[0] + 0.3 + 0.1*j;
      y[j] = x[0];//+ 0.3 + 0.1*j;
  }
  std::cout << "y[0].gradient" << std::endl;
  std::cout << y[0].gradient() << std::endl;
  for(int i = 0; i < nx; ++i) {
    y[0] += x[i];
  std::cout << "y[0].gradient" << std::endl;
  std::cout << y[0].gradient() << std::endl;
    for (int j = 1; j < ny; ++j) {
  std::cout << "y[i].gradient" << std::endl;
  std::cout << y[i].gradient() << std::endl;
        y[j] *= x[i] * x[i] * sin(x[i]);
    }
  }
  psi_dot_y = 0;
  for (int j = 1; j < ny; ++j) {
     psi_dot_y += psi[j] * y[j];
  }
}

template<int dim, typename codiType>
std::unique_ptr< codi::TapeVectorHelperInterface<codiType> > create_TapeVectorHelper(const int target_dim)
{
    if(dim == target_dim) {
        return std::make_unique<codi::TapeVectorHelper<codiType, codi::Direction<double, dim> >>();
    } else if constexpr (dim > 1) {
        return create_TapeVectorHelper<dim-1, codiType>(target_dim);
    } else {
        return nullptr;
    }
}

template<int nd>
int n_direction(const int target_dir) {
    return (nd == target_dir || nd == 0) ? nd : n_direction<nd-1>(target_dir);
}

void vectorHelperInterface(int nx, int ny) {

    //using codiType = codi::RealForward;
    using codiType = codi::RealReverse;
    std::cout << "codiType( vector helper interface):" << std::endl;
    std::vector<double> xR_double(nx);
    std::vector<double> dual(ny);
    for (int i = 0; i < nx; ++i) {
        xR_double[i] = 0.333*i + 1.0;
    }
    for (int j = 0; j < ny; ++j) {
        dual[j] = 0.75/(1+j);
    }

    // Reverse vector mode
    std::vector<codiType> xR(nx);
    std::vector<codiType> yR(ny);
    for (int i = 0; i < nx; ++i) {
      xR[i] = xR_double[i];
    }
    codiType::TapeType& tape = codiType::getGlobalTape();
    tape.setActive();
    for(int i = 0; i < nx; ++i) {
      tape.registerInput(xR[i]);
    }

    codiType psi_dot_y = 0;
    func_2(xR, dual, nx, ny, yR, psi_dot_y);

    for(int i = 0; i < ny; ++i) {
        tape.registerOutput(yR[i]);
    }
    tape.setPassive();
    //codi::TapeVectorHelperInterface<codiType>* vh = new codi::TapeVectorHelper<codiType, codi::Direction<double, 2> >();

    std::unique_ptr<codi::TapeVectorHelperInterface<codiType>> vh = create_TapeVectorHelper<10,codiType>(ny);
    codi::AdjointInterface<codiType::Real, codiType::GradientData>* ai = vh->getAdjointInterface();
    for(int j = 0; j < ny; ++j) {
        ai->updateAdjoint(yR[j].getGradientData(), j, 1.0);
    }
    vh->evaluate();

    std::vector<double> jacobiR(nx*ny); // default construction
    std::cout << ai->getVectorSize() << std::endl;
    for(int i = 0; i < nx; ++i) {
      for(int j = 0; j < ny; ++j) {
        jacobiR[i*ny+j] = ai->getAdjoint(xR[i].getGradientData(), j);
      }
    }

    std::cout << "Reverse vector mode:" << std::endl;
    std::cout << "f(1 .. nx) = (" << yR[0];
    for(int j = 1; j < ny; ++j) {
        std::cout << ", " << yR[j];
    }
    std::cout << ")" << std::endl;
    for(int i = 0; i < nx; ++i) {
      std::cout << "df/dx_" << (i + 1) << " (1 .. nx) = (" << jacobiR[i*ny+0];
      for(int j = 1; j < ny; ++j) {
          std::cout << ", " << jacobiR[i*ny+j];
      }
      std::cout << ")" << std::endl;
    }

    // const double eps = 1e-6;
    // std::vector<double> yR_p(ny);
    // std::vector<double> yR_m(ny);
    // double psi_dot_y_double = 0;
    // for (int i = 0; i < nx; ++i) {
    //     double old_x = xR_double[i];

    //     xR_double[i] = old_x;
    //     xR_double[i] += eps;
    //     func_2(xR_double, dual, nx, ny, yR_p, psi_dot_y_double);

    //     xR_double[i] = old_x;
    //     xR_double[i] -= eps;
    //     func_2(xR_double, dual, nx, ny, yR_m, psi_dot_y_double);

    //     for (int j = 0; j < ny; ++j) {
    //         jacobiR[i*ny + j] = yR_p[j] - yR_m[j];
    //         jacobiR[i*ny + j] /= 2.0*eps;
    //     }
    // }
    // std::cout << "FD mode:" << std::endl;
    // std::cout << "f(1 .. nx) = (" << yR[0];
    // for(int j = 1; j < ny; ++j) {
    //     std::cout << ", " << yR[j];
    // }
    // std::cout << ")" << std::endl;
    // for(int i = 0; i < nx; ++i) {
    //   std::cout << "df/dx_" << (i + 1) << " (1 .. nx) = (" << jacobiR[i*ny+0];
    //   for(int j = 1; j < ny; ++j) {
    //       std::cout << ", " << jacobiR[i*ny+j];
    //   }
    //   std::cout << ")" << std::endl;
    // }

}

void forward(int nx, int ny) {

    //using codiType = codi::RealForward;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Vector Forward mode:" << std::endl;
    std::vector<double> xR_double(nx);
    std::vector<double> dual(ny);
    for (int i = 0; i < nx; ++i) {
        xR_double[i] = 0.333*i + 1.0;
    }
    for (int j = 0; j < ny; ++j) {
        dual[j] = 0.75/(1+j);
    }

    // Reverse vector mode

    using codiType = codi::RealForwardGen<double, codi::DirectionVar<double>>;
    std::vector<codiType> xR(nx);
    std::vector<codiType> yR(ny);
    for (int i = 0; i < nx; ++i) {
      xR[i] = xR_double[i];

      xR[i].gradient().resize(nx);

      xR[i].gradient() = codi::DirectionVar(0.0,nx);
      xR[i].gradient()[i] = 1.0;
      std::cout << xR[i].gradient() << std::endl;
    }
    codiType result;
    result = xR[0];
    result.gradient() = codi::DirectionVar(0.0,nx);
    std::cout << " result: " << std::endl;
    std::cout << result << std::endl;
    std::cout << result.gradient() << std::endl;
    for (int i = 0; i < nx; ++i) {
        result = result + xR[i]*xR[i];
    }
    std::cout << " result: " << std::endl;
    std::cout << result << std::endl;
    std::cout << result.gradient() << std::endl;

    for (int i = 0; i < ny; ++i) {
      yR[i].gradient() = codi::DirectionVar(0.0,nx);
      yR[i] = 0;
    }
    //std::cout << xR[1].gradient() << std::endl;

    codiType psi_dot_y;
    psi_dot_y.gradient() = codi::DirectionVar(0.0,nx);
    func_2(xR, dual, nx, ny, yR, psi_dot_y);

    std::vector<double> jacobiR(nx*ny); // default construction
    for(int i = 0; i < nx; ++i) {
      for(int j = 0; j < ny; ++j) {
        std::cout << yR[j].gradient() << std::endl;
        jacobiR[i*ny+j] = yR[j].gradient()[i];
      }
    }

    std::cout << "Forward vector mode:" << std::endl;
    std::cout << "f(1 .. nx) = (" << yR[0];
    for(int j = 1; j < ny; ++j) {
        std::cout << ", " << yR[j];
    }
    std::cout << ")" << std::endl;
    for(int i = 0; i < nx; ++i) {
      std::cout << "df/dx_" << (i + 1) << " (1 .. nx) = (" << jacobiR[i*ny+0];
      for(int j = 1; j < ny; ++j) {
          std::cout << ", " << jacobiR[i*ny+j];
      }
      std::cout << ")" << std::endl;
    }

    // const double eps = 1e-6;
    // std::vector<double> yR_p(ny);
    // std::vector<double> yR_m(ny);
    // double psi_dot_y_double = 0;
    // for (int i = 0; i < nx; ++i) {
    //     double old_x = xR_double[i];

    //     xR_double[i] = old_x;
    //     xR_double[i] += eps;
    //     func_2(xR_double, dual, nx, ny, yR_p, psi_dot_y_double);

    //     xR_double[i] = old_x;
    //     xR_double[i] -= eps;
    //     func_2(xR_double, dual, nx, ny, yR_m, psi_dot_y_double);

    //     for (int j = 0; j < ny; ++j) {
    //         jacobiR[i*ny + j] = yR_p[j] - yR_m[j];
    //         jacobiR[i*ny + j] /= 2.0*eps;
    //     }
    // }
    // std::cout << "FD mode:" << std::endl;
    // std::cout << "f(1 .. nx) = (" << yR[0];
    // for(int j = 1; j < ny; ++j) {
    //     std::cout << ", " << yR[j];
    // }
    // std::cout << ")" << std::endl;
    // for(int i = 0; i < nx; ++i) {
    //   std::cout << "df/dx_" << (i + 1) << " (1 .. nx) = (" << jacobiR[i*ny+0];
    //   for(int j = 1; j < ny; ++j) {
    //       std::cout << ", " << jacobiR[i*ny+j];
    //   }
    //   std::cout << ")" << std::endl;
    // }

}
int main(int nargs, char** args) {
  (void) nargs;
  const int ny = atoi(args[1]);
  codi::RealReverse::getGlobalTape().reset();
  vectorHelperInterface(5, ny);

  forward(5, ny);
  return 0;
}
